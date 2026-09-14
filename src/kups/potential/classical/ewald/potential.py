# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Ewald energies, caches and potential construction.

Incremental values use cached structure factors, while derivatives always
include every current particle.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Literal

import einops
import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import Periodic3D
from kups.core.constants import BOHR, HARTREE
from kups.core.data import Index, Table, WithIndices
from kups.core.lens import Lens, View, lens
from kups.core.neighborlist import (
    EmptyNeighborList,
    NeighborList,
    all_connected_neighborlist,
)
from kups.core.patch import Accept, IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY,
    EMPTY_LENS,
    Energy,
    Potential,
    PotentialOut,
    ScaledPotential,
    SummedPotential,
)
from kups.core.typing import (
    ExclusionId,
    HasCell,
    InclusionId,
    ParticleId,
    SystemId,
)
from kups.core.utils.functools import pipe
from kups.core.utils.jax import (
    dataclass,
    field,
    no_jax_tracing,
    tree_zeros_like,
)
from kups.core.utils.kahan import KahanSummand
from kups.core.utils.math import triangular_3x3_matmul
from kups.core.utils.ops import where_broadcast_last
from kups.core.utils.segment import segment_sum
from kups.potential.classical.coulomb import _pairwise_coulomb_energy
from kups.potential.common.energy import (
    EnergyFunction,
    FullSumComposer,
    LocalSumComposer,
    PotentialFromEnergy,
    Sum,
    SumComposer,
    Summand,
)
from kups.potential.common.graph import (
    GraphConstructor,
    GraphInputConstructor,
    GraphPotentialInput,
    IsGraphProbe,
    PointCloud,
)

from .parameters import EwaldParameters, IsEwaldPointData, ReciprocalGridBound
from .reciprocal import _structure_factor_full, _use_grid_response

TO_STANDARD_UNITS = HARTREE * BOHR


@dataclass
class EwaldCache[Gradient, Hessian]:
    """Compensated structure factors and component outputs for incremental updates."""

    structure_factor: KahanSummand[Array]  # (n_groups, n_kvecs, 2)
    short_range: KahanSummand[PotentialOut[Gradient, Hessian]]
    long_range: KahanSummand[PotentialOut[Gradient, Hessian]]
    self_interaction: KahanSummand[PotentialOut[Gradient, Hessian]]
    exclusion: KahanSummand[PotentialOut[Gradient, Hessian]]

    @classmethod
    @no_jax_tracing
    def make[G, H](
        cls, n_sys: int, n_kvecs: int, gradient: G = EMPTY, hessian: H = EMPTY
    ) -> EwaldCache[G, H]:
        """Create zeroed caches with the supplied derivative shapes.

        Args:
            n_sys: Number of systems.
            n_kvecs: Number of reciprocal vectors per system, including padding.
            gradient: Template defining cached gradient shapes.
            hessian: Template defining cached Hessian shapes.

        Returns:
            Compensated structure-factor and component-output caches initialized to zero.
        """
        out = PotentialOut(
            Table.arange(jnp.zeros(n_sys, dtype=float), label=SystemId),
            gradient,
            hessian,
        )
        return EwaldCache(
            KahanSummand.init(jnp.zeros((n_sys, n_kvecs, 2), dtype=float)),
            KahanSummand.init(tree_zeros_like(out)),
            KahanSummand.init(tree_zeros_like(out)),
            KahanSummand.init(tree_zeros_like(out)),
            KahanSummand.init(tree_zeros_like(out)),
        )


@dataclass
class EwaldCachePatch[State, Gradient, Hessian](Patch[State]):
    """Accept or reject each system's structure factor, including its compensation."""

    new_structure_factor: KahanSummand[Array]
    system_idx: Index[SystemId]
    lens: Lens[State, EwaldCache[Gradient, Hessian]] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        """Apply structure-factor updates for accepted systems.

        Args:
            state: State containing the Ewald cache.
            accept: Per-system acceptance mask.

        Returns:
            State with accepted structure factors and their compensation updated.
        """
        mask = accept[self.system_idx]
        new_sf = self.new_structure_factor
        return self.lens.focus(lambda cache: cache.structure_factor).apply(
            state,
            lambda old_sf: jax.tree.map(
                lambda new, old: where_broadcast_last(mask, new, old),
                new_sf,
                old_sf,
            ),
        )


type EwaldShortRangeInput = GraphPotentialInput[
    EwaldParameters, IsEwaldPointData, HasCell[Periodic3D], Literal[2]
]

type EwaldSelfInput = GraphPotentialInput[
    EwaldParameters, IsEwaldPointData, HasCell[Periodic3D], Literal[0]
]


@dataclass
class EwaldLongRangeInput[State]:
    """Current particles and optional previous particle values for a cached update."""

    point_cloud: PointCloud[IsEwaldPointData, HasCell[Periodic3D]]
    parameters: EwaldParameters
    cache: EwaldCache[Any, Any] | None = None
    cache_lens: Lens[State, EwaldCache[Any, Any]] | None = None
    changes_from_prev: WithIndices[ParticleId, IsEwaldPointData] | None = None

    @property
    def volume(self) -> Array:
        """Cell volumes in Å³, shaped ``(n_systems,)``."""
        return self.point_cloud.systems.data.cell.volume

    @property
    def kvecs(self) -> Array:
        """Convert the stored integer shifts to Cartesian reciprocal vectors.

        The result is in 1/Å, shaped ``(n_systems, n_kvecs, 3)``.
        """
        sys_idx = self.point_cloud.systems.index
        return triangular_3x3_matmul(
            self.point_cloud.systems.data.cell.inverse_vectors.mT[:, None] * 2 * jnp.pi,
            self.parameters.reciprocal_lattice_shifts[sys_idx],
            lower=False,
        )


def ewald_self_interaction_energy(
    inp: EwaldSelfInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Subtract each charge's interaction with its own Gaussian screening cloud.

    Args:
        inp: Particle graph and per-system screening parameters.

    Returns:
        Per-system self-interaction energies in eV and an identity patch.
    """
    sys_idx = inp.graph.systems.index
    energies = (
        -segment_sum(
            inp.graph.particles.data.charges**2,
            inp.graph.particles.data.system.indices,
            inp.graph.batch_size,
            mode="drop",
        )
        * inp.parameters.alpha[sys_idx]
        / jnp.sqrt(jnp.pi)
    )
    energies *= TO_STANDARD_UNITS
    return WithPatch(Table.arange(energies, label=SystemId), IdPatch[Any]())


def ewald_short_range_energy(
    inp: EwaldShortRangeInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Real-space (short-range) screened Coulomb energy.

    Math: ``E_sr = 1/2 * TO_STANDARD_UNITS * sum_{i<j} q_i*q_j * erfc(alpha*r_ij) / r_ij``.

    The ``erfc(alpha*r)`` damping ensures convergence within the cutoff.
    Factor 1/2 corrects for double-counted pairs from the radius graph edges.
    Positions in Ang, charges in e, energy in eV.
    """
    edg = inp.graph.particles[inp.graph.edges.indices]
    qij = edg.charges[:, 0] * edg.charges[:, 1]
    dists = jnp.linalg.norm(inp.graph.edge_shifts[:, 0], axis=-1)
    edge_systems = inp.graph.edge_batch_mask
    erfc = jax.scipy.special.erfc(inp.parameters.alpha[edge_systems] * dists)
    energies = qij * erfc / dists
    mask = dists < inp.parameters.cutoff[edge_systems]
    energies *= mask
    total = inp.graph.edge_batch_mask.sum_over(energies) / 2 * TO_STANDARD_UNITS
    return WithPatch(total, IdPatch[Any]())


def long_range(inp: EwaldLongRangeInput[Any], structure_factor: Array) -> Energy:
    """Compute the unscaled reciprocal sum ``sum_k P(k) |S(k)|²``.

    Args:
        inp: System cells and reciprocal-space parameters.
        structure_factor: Real/imaginary structure factors, shaped ``(n_systems, n_kvecs, 2)``.

    Returns:
        Per-system reciprocal sums before ``TO_STANDARD_UNITS`` scaling,
        excluding the neutralizing-background correction.
    """
    return einops.einsum(
        prefactor(inp),
        structure_factor,
        structure_factor,
        "batch_size kvecs, batch_size kvecs two, batch_size kvecs two -> batch_size",
    )


def prefactor(inp: EwaldLongRangeInput[Any]) -> Array:
    """Weights ``2π/V exp(-k²/(4α²))/k²``, zero outside the reciprocal cutoff.

    Double contributions with nonzero first lattice coefficient for the omitted
    half-space; the zero plane already contains both signs.

    Args:
        inp: System cells, screening parameters and reciprocal cutoffs.

    Returns:
        Weights shaped ``(n_systems, n_kvecs)``; zero for k=0 and masked vectors.
    """
    sys_idx = inp.point_cloud.systems.index
    alpha = inp.parameters.alpha[sys_idx]
    shifts = inp.parameters.reciprocal_lattice_shifts[sys_idx]
    kv = inp.kvecs
    k_squared = einops.einsum(
        kv, kv, "batch_size kvecs dim, batch_size kvecs dim -> batch_size kvecs"
    )
    mask = k_squared > 0
    mask &= k_squared <= inp.parameters.k_max[sys_idx][:, None] ** 2
    k_squared = jnp.where(mask, k_squared, 1)
    result = (
        (2 * jnp.pi)
        / inp.volume[:, None]
        * jnp.exp(-k_squared / (4 * alpha[:, None] ** 2))
        / k_squared
    )
    leading_zero = shifts[..., 0] == 0
    result = (2 - leading_zero) * result
    return jnp.where(mask, result, 0.0)


def _changed_particle_rows(
    positions: Array,
    charges: Array,
    batch_mask: Index[SystemId],
    previous: WithIndices[ParticleId, IsEwaldPointData],
) -> tuple[Array, Array, Index[SystemId]]:
    """Pair changed current rows with negated previous charges; drop invalid probes."""
    idx = previous.indices.indices
    valid = (idx >= 0) & (idx < len(positions))
    idx = jnp.where(valid, idx, len(positions))
    positions = jnp.concatenate(
        (positions.at[idx].get(mode="fill", fill_value=0), previous.data.positions)
    )
    charges = jnp.concatenate(
        (charges.at[idx].get(mode="fill", fill_value=0), -previous.data.charges)
    )
    ns = batch_mask.num_labels
    system_ids = jnp.concatenate(
        (
            batch_mask.indices.at[idx].get(mode="fill", fill_value=ns),
            jnp.where(valid, previous.data.system.indices, ns),
        )
    )
    return positions, charges, Index(batch_mask.keys, system_ids)


@functools.partial(
    jax.custom_jvp,
    nondiff_argnames=(
        "batch_mask",
        "bound",
        "particle_chunk_size",
        "k_chunk_size",
        "cache",
        "changes",
    ),
)
def _structure_factor(
    positions: Array,
    charges: Array,
    kvecs: Array,
    inverse_vectors: Array,
    batch_mask: Index[SystemId],
    bound: ReciprocalGridBound,
    particle_chunk_size: int,
    k_chunk_size: int,
    cache: EwaldCache[Any, Any] | None,
    changes: WithIndices[ParticleId, IsEwaldPointData] | None,
) -> KahanSummand[Array]:
    """Evaluate S(k), or add the signed new-minus-old contribution to the cache.

    Changes to cells or reciprocal parameters require a full evaluation.
    """
    if changes is not None:
        positions, charges, batch_mask = _changed_particle_rows(
            positions, charges, batch_mask, changes
        )
    sf = _structure_factor_full(
        positions,
        charges,
        kvecs,
        inverse_vectors,
        batch_mask=batch_mask,
        bound=bound,
        particle_chunk_size=particle_chunk_size,
        k_chunk_size=k_chunk_size,
    )
    if changes is None:
        return KahanSummand.init(sf)
    assert cache is not None, "Cache required for structure factor update"
    return cache.structure_factor + sf


@functools.partial(_structure_factor.defjvp, symbolic_zeros=True)
def _structure_factor_jvp(
    batch_mask: Index[SystemId],
    bound: ReciprocalGridBound,
    particle_chunk_size: int,
    k_chunk_size: int,
    cache: EwaldCache[Any, Any] | None,
    changes: WithIndices[ParticleId, IsEwaldPointData] | None,
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[KahanSummand[Array], KahanSummand[Array]]:
    """Differentiate all current particles, even when values use an incremental cache.

    Each kernel uses either k-vectors or inverse cells, counting cell dependence
    once. Kahan compensation has zero tangent.
    """
    full = functools.partial(
        _structure_factor_full,
        batch_mask=batch_mask,
        bound=bound,
        particle_chunk_size=particle_chunk_size,
        k_chunk_size=k_chunk_size,
        tiled=not _use_grid_response(bound),
    )
    # Absent tangents must not trigger charge or cell derivative work.
    active = tuple(
        i
        for i, t in enumerate(tangents)
        if not isinstance(t, jax.custom_derivatives.SymbolicZero)
    )

    def active_full(*args: Array) -> Array:
        inputs = list(primals)
        for i, value in zip(active, args):
            inputs[i] = value
        return full(*inputs)

    value, tangent = jax.jvp(
        active_full,
        tuple(primals[i] for i in active),
        tuple(tangents[i] for i in active),
    )
    # Reuse the full CPU grid value; retain cached and direct GPU values.
    if changes is None and _use_grid_response(bound):
        sf = KahanSummand.init(value)
    else:
        sf = _structure_factor(
            *primals,
            batch_mask,
            bound,
            particle_chunk_size,
            k_chunk_size,
            cache,
            changes,
        )
    return sf, KahanSummand(tangent, jnp.zeros_like(tangent))


def structure_factor[State](
    inp: EwaldLongRangeInput[State],
) -> tuple[KahanSummand[Array], Patch[State]]:
    """Return the full or incrementally updated structure factor and cache patch.

    Args:
        inp: Current particles and optional cache and previous particle values.

    Returns:
        Compensated structure factors shaped ``(n_systems, n_kvecs, 2)`` and
        an acceptance patch, or an identity patch when no cache lens is supplied.
    """
    particles = inp.point_cloud.particles.data
    params = inp.parameters
    sk = _structure_factor(
        particles.positions,
        particles.charges,
        inp.kvecs,
        inp.point_cloud.systems.data.cell.inverse_vectors,
        particles.system,
        params.reciprocal_shift_bound,
        params.reciprocal_particle_chunk_size,
        params.reciprocal_k_chunk_size,
        inp.cache,
        inp.changes_from_prev,
    )
    patch = (
        EwaldCachePatch(sk, inp.point_cloud.systems.index, inp.cache_lens)
        if inp.cache_lens is not None
        else IdPatch[State]()
    )
    return sk, patch


def ewald_net_charge_energy(inp: EwaldLongRangeInput[Any]) -> Table[SystemId, Energy]:
    """Uniform neutralizing-background correction ``-π Q² / (2 V α²)`` per system.

    Args:
        inp: Particle charges, system volumes and screening parameters.

    Returns:
        Table of per-system neutralizing-background energies in eV.
    """
    particles = inp.point_cloud.particles.data
    sys_idx = inp.point_cloud.systems.index
    net_charge = segment_sum(
        particles.charges,
        particles.system.indices,
        inp.point_cloud.batch_size,
        mode="drop",
    )
    alpha = inp.parameters.alpha[sys_idx]
    energies = -jnp.pi / (2 * inp.volume * alpha**2) * net_charge**2 * TO_STANDARD_UNITS
    return Table.arange(energies, label=SystemId)


def ewald_long_range_energy[State](
    inp: EwaldLongRangeInput[State],
) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
    """Reciprocal energy plus neutralizing background, with an acceptance cache patch.

    Args:
        inp: Current particles, reciprocal parameters and optional incremental cache.

    Returns:
        Per-system energies in eV with the structure-factor acceptance patch.
    """
    structure_out, patch = structure_factor(inp)
    energy = long_range(inp, structure_out.total)
    assert energy.shape == (inp.point_cloud.batch_size,), (
        f"Expected energy shape {(inp.point_cloud.batch_size,)} but got {energy.shape}."
    )
    energy = energy * TO_STANDARD_UNITS
    total = ewald_net_charge_energy(inp).map_data(lambda e_net: e_net + energy)
    return WithPatch(total, patch)


@dataclass
class EwaldLongRangeComposer[
    State,
    Ptch: Patch[Any],
]:
    """Build current particles and retain previous values for incremental updates."""

    particles: View[State, Table[ParticleId, IsEwaldPointData]] = field(static=True)
    systems: View[State, Table[SystemId, HasCell[Periodic3D]]] = field(static=True)
    probe: Probe[State, Ptch, WithIndices[ParticleId, IsEwaldPointData]] | None = field(
        static=True
    )
    parameters: Lens[State, EwaldParameters] = field(static=True)
    cache: Lens[State, EwaldCache[Any, Any]] | None = field(static=True)

    def __call__(
        self, state: State, patch: Ptch | None
    ) -> Sum[EwaldLongRangeInput[State]]:
        """Compose the reciprocal-energy input for a full evaluation or proposal.

        Args:
            state: State providing particles, systems, parameters and optional cache.
            patch: Optional proposed state patch to inspect with the particle probe.

        Returns:
            A single-summand input with current particles and any probed previous values.
        """
        particles = self.particles(state)
        previous = None
        if patch is not None and self.probe is not None:
            updates = self.probe(state, patch)
            previous = WithIndices(updates.indices, particles[updates.indices])
            particles = particles.update(updates.indices, updates.data)
        inp = EwaldLongRangeInput(
            PointCloud(particles, self.systems(state)),
            self.parameters(state),
            self.cache.get(state) if self.cache else None,
            self.cache,
            previous,
        )
        return Sum(Summand(inp))


@dataclass
class EwaldPotential[State, Gradients, Hessians, P: Patch[Any]](
    SummedPotential[State, Gradients, Hessians, P]
):
    """Complete Ewald potential with named access to each component term."""

    @property
    def short_range(self) -> Potential[State, Gradients, Hessians, P]:
        """Real-space screened Coulomb potential."""
        return self.potentials[0]

    @property
    def long_range(self) -> Potential[State, Gradients, Hessians, P]:
        """Reciprocal potential including the neutralizing-background correction."""
        return self.potentials[1]

    @property
    def self_interaction(self) -> Potential[State, Gradients, Hessians, P]:
        """Subtract each charge's interaction with its screening cloud."""
        return self.potentials[2]

    @property
    def exclusion_correction(self) -> Potential[State, Gradients, Hessians, P]:
        """Subtract excluded pair interactions; requires exclusions enabled."""
        return self.potentials[3]


def make_ewald_short_range_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    parameter_view: View[State, EwaldParameters],
    probe: Probe[State, Ptch, IsGraphProbe[IsEwaldPointData, Literal[2]]] | None,
    gradient_lens: Lens[PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create the Ewald real-space (short-range) potential."""

    return PotentialFromEnergy(
        energy_fn=ewald_short_range_energy,
        composer=LocalSumComposer(
            GraphInputConstructor(
                graph_constructor=GraphConstructor(
                    particles=particles_view,
                    systems=systems_view,
                    neighborlist=neighborlist_view,
                    probe=probe,
                ),
                parameter_view=parameter_view,
            )
        ),
        gradient_lens=lens(lambda x: x.graph).nest(gradient_lens),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )


def make_ewald_long_range_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    parameter_lens: Lens[State, EwaldParameters],
    cache_lens: Lens[State, EwaldCache[Gradients, Hessians]] | None,
    probe: Probe[State, Ptch, WithIndices[ParticleId, IsEwaldPointData]] | None = None,
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[State, Hessians] = EMPTY_LENS,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create the Ewald reciprocal-space (long-range) potential.

    Args:
        particles_view: View of the particle table.
        systems_view: View of the system table and periodic cells.
        parameter_lens: Lens to Ewald parameters.
        cache_lens: Optional lens to structure-factor and component-output caches.
        probe: Optional probe returning proposed particle updates.
        gradient_lens: Lens selecting point-cloud variables to differentiate.
        hessian_lens: Lens selecting gradient entries to differentiate again.
        hessian_idx_view: View supplying Hessian row and column indices.
        patch_idx_view: Optional view supplying indices for output-cache updates.

    Returns:
        Reciprocal potential with optional incremental structure-factor updates.
    """
    return PotentialFromEnergy(
        energy_fn=ewald_long_range_energy,
        composer=EwaldLongRangeComposer(
            particles=particles_view,
            systems=systems_view,
            probe=probe,
            parameters=parameter_lens,
            cache=cache_lens,
        ),
        gradient_lens=lens(lambda x: x.point_cloud).nest(gradient_lens),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens.focus(lambda x: x.long_range) if cache_lens else None,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )


def make_ewald_self_interaction_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    parameter_view: View[State, EwaldParameters],
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[State, Hessians] = EMPTY_LENS,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Recompute the inexpensive charge-only sum to avoid drift from cached deltas.

    Args:
        particles_view: View of the particle table.
        systems_view: View of the system table and periodic cells.
        parameter_view: View of Ewald parameters.
        gradient_lens: Lens selecting point-cloud variables to differentiate.
        hessian_lens: Lens selecting gradient entries to differentiate again.
        hessian_idx_view: View supplying Hessian row and column indices.
        patch_idx_view: Optional view supplying indices for output-cache updates.
        cache_lens: Optional lens to the compensated self-interaction output cache.

    Returns:
        Self-interaction potential recomputed from all current charges.
    """
    return PotentialFromEnergy(
        energy_fn=ewald_self_interaction_energy,
        composer=FullSumComposer(
            GraphInputConstructor(
                graph_constructor=GraphConstructor(
                    particles=particles_view,
                    systems=systems_view,
                    neighborlist=lambda _: EmptyNeighborList[Literal[0]](),
                    probe=None,
                ),
                parameter_view=parameter_view,
            )
        ),
        gradient_lens=lens(lambda x: x.graph).nest(gradient_lens),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )


def make_ewald_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasCell[Periodic3D]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    parameter_lens: Lens[State, EwaldParameters],
    cache_lens: Lens[State, EwaldCache[Gradients, Hessians]] | None,
    probe: Probe[State, Ptch, IsGraphProbe[IsEwaldPointData, Literal[2]]] | None,
    gradient_lens: Lens[PointCloud[IsEwaldPointData, HasCell[Periodic3D]], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    include_exclusion_mask: bool = False,
) -> EwaldPotential[State, Gradients, Hessians, Ptch]:
    """Create the complete Ewald potential combining all component terms.

    Implements the Ewald decomposition:
    ``E_total = E_sr + E_lr - E_self - E_excl``
    where each term is computed independently and cached for incremental
    MC updates. Short-range and exclusion use radius graphs (real-space
    pairs), long-range uses point clouds (reciprocal space), and
    self-interaction is per-particle.

    Internally converts ``_ParticleData`` adding ``inclusion`` and
    ``exclusion`` fields for the neighbor list:

    - sr/lr/self: ``inclusion=system`` (all particles in same system
      interact), ``exclusion=particle_id`` (self-exclusion).
    - exclusion correction: ``inclusion=group`` (only same-molecule
      pairs), ``exclusion=particle_id``.

    Args:
        particles_view: Indexed particle data (positions, charges, system index).
        systems_view: Indexed system data (cell).
        neighborlist_view: Cutoff-bound neighbor list.
        parameter_lens: Lens to EwaldParameters.
        cache_lens: Lens to EwaldCache, or ``None``.
        probe: Probe for incremental updates, or ``None``.
        gradient_lens: Specifies gradients to compute.
        hessian_lens: Specifies Hessians to compute.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure (optional).
        include_exclusion_mask: Whether to include the exclusion correction.

    Returns:
        Complete Ewald potential (sum of three or four components).
    """
    # The definition of the ewald potential is only correct when computing the total
    # energy. One cannot directly exclude energy terms, thus, we compute the total coulomb
    # energy and subtract the excluded interactions later.

    @dataclass
    class _ParticleData:
        positions: Array
        charges: Array
        system: Index[SystemId]
        inclusion: Index[InclusionId]
        exclusion: Index[ExclusionId]

    def _convert_particles(
        indexed: Table[ParticleId, IsEwaldPointData],
        inclusion_fn: Callable[[IsEwaldPointData], Index[InclusionId]],
    ) -> Table[ParticleId, _ParticleData]:
        """Convert particles, deriving inclusion from `inclusion_fn`."""
        p = indexed.data
        excl = Index.arange(len(indexed), label=ExclusionId)
        return Table(
            indexed.keys,
            _ParticleData(
                p.positions,
                p.charges,
                p.system,
                inclusion=inclusion_fn(p),
                exclusion=excl,
            ),
        )

    def _make_probe(
        inclusion_fn: Callable[[IsEwaldPointData], Index[InclusionId]],
        neighborlist_override: NeighborList[Literal[2]] | None = None,
    ) -> Probe[State, Ptch, IsGraphProbe[_ParticleData, Literal[2]]] | None:
        """Wrap `probe` to convert particle data with `inclusion_fn`.

        Args:
            inclusion_fn: Extracts the inclusion Index from particle data.
            neighborlist_override: If set, replaces the probe's neighbor lists
                (e.g., ``AllConnectedNeighborList`` for exclusion correction).
        """
        if probe is None:
            return None
        _p = probe

        @dataclass
        class _ProbeResult:
            particles: WithIndices[ParticleId, _ParticleData]
            neighborlist_after: NeighborList[Literal[2]]
            neighborlist_before: NeighborList[Literal[2]]

        def _wrapper(state: State, patch: Ptch) -> _ProbeResult:
            result = _p(state, patch)
            p = result.particles
            d = p.data
            excl = Index(p.indices.keys, p.indices.indices, _cls=p.indices.cls)
            data = _ParticleData(
                d.positions,
                d.charges,
                d.system,
                inclusion=inclusion_fn(d).to_cls(InclusionId),
                exclusion=excl.to_cls(ExclusionId),
            )
            nn_after = neighborlist_override or result.neighborlist_after
            nn_before = neighborlist_override or result.neighborlist_before
            return _ProbeResult(WithIndices(p.indices, data), nn_after, nn_before)

        return _wrapper

    def _make_particles_probe(
        inclusion_fn: Callable[[IsEwaldPointData], Index[InclusionId]],
    ) -> Probe[State, Ptch, WithIndices[ParticleId, _ParticleData]] | None:
        """Wrap `probe` returning only WithIndices (no neighborlists)."""
        full = _make_probe(inclusion_fn)
        if full is None:
            return None

        def _wrapper(
            state: State, patch: Ptch
        ) -> WithIndices[ParticleId, _ParticleData]:
            return full(state, patch).particles

        return _wrapper

    # Atomic view: inclusion = system
    def _system_inclusion(d: IsEwaldPointData) -> Index[InclusionId]:
        return d.system.to_cls(InclusionId)

    atomic_view = pipe(
        particles_view, lambda p: _convert_particles(p, _system_inclusion)
    )
    atomic_probe = _make_probe(_system_inclusion)
    atomic_particles_probe = _make_particles_probe(_system_inclusion)

    sr_potential = make_ewald_short_range_potential(
        particles_view=atomic_view,
        systems_view=systems_view,
        neighborlist_view=neighborlist_view,
        parameter_view=parameter_lens,
        probe=atomic_probe,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
        cache_lens=cache_lens.focus(lambda x: x.short_range) if cache_lens else None,
    )
    lr_potential = make_ewald_long_range_potential(
        particles_view=atomic_view,
        systems_view=systems_view,
        parameter_lens=parameter_lens,
        cache_lens=cache_lens,
        probe=atomic_particles_probe,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    self_potential = make_ewald_self_interaction_potential(
        particles_view=atomic_view,
        systems_view=systems_view,
        parameter_view=parameter_lens,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
        cache_lens=cache_lens.focus(lambda x: x.self_interaction)
        if cache_lens
        else None,
    )

    # Exclusion view: inclusion = exclusion group
    def _excl_inclusion(d: IsEwaldPointData) -> Index[InclusionId]:
        return d.exclusion.to_cls(InclusionId)

    excl_view = pipe(particles_view, lambda p: _convert_particles(p, _excl_inclusion))
    excl_probe = _make_probe(_excl_inclusion, all_connected_neighborlist)

    excl_rg = GraphConstructor(
        particles=excl_view,
        systems=systems_view,
        neighborlist=lambda _: all_connected_neighborlist,
        probe=excl_probe,
    )
    exclusion_correction = PotentialFromEnergy(
        energy_fn=_pairwise_coulomb_energy,
        composer=LocalSumComposer(GraphInputConstructor(excl_rg, lambda x: None)),
        gradient_lens=lens(lambda x: x.graph).nest(gradient_lens),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens.focus(lambda x: x.exclusion) if cache_lens else None,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    exclusion_correction = ScaledPotential(exclusion_correction, -1)
    if include_exclusion_mask:
        return EwaldPotential(
            (sr_potential, lr_potential, self_potential, exclusion_correction)
        )
    return EwaldPotential((sr_potential, lr_potential, self_potential))


if TYPE_CHECKING:
    _lr: EnergyFunction[Any, EwaldLongRangeInput[Any]] = ewald_long_range_energy
    _si: EnergyFunction[Any, EwaldSelfInput] = ewald_self_interaction_energy
    _sr: EnergyFunction[Any, EwaldShortRangeInput] = ewald_short_range_energy

    def _check_composer(c: EwaldLongRangeComposer[Any, Any]) -> None:
        _: SumComposer[Any, Any, Any] = c
