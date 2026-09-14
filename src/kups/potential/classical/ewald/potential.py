# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Ewald summation for long-range electrostatics in periodic systems.

Splits the Coulomb potential into short-range (real-space), long-range
(reciprocal-space), and self-interaction terms. Supports incremental
updates via cached structure factors for efficient Monte Carlo.
"""

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
)

import einops
import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import Periodic3D
from kups.core.constants import BOHR, HARTREE
from kups.core.data import Index, Table, WithIndices
from kups.core.lens import Lens, View, bind, lens
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

from .parameters import EwaldParameters, IsEwaldPointData
from .reciprocal import _frequency_response, _structure_factor_full

TO_STANDARD_UNITS = HARTREE * BOHR
"""Conversion factor from atomic units to standard energy units."""


@dataclass
class EwaldCache[Gradient, Hessian]:
    """Cached structure factors and per-component outputs for incremental updates.

    Attributes:
        structure_factor: Compensated accumulator over complex structure factors,
            each of shape `(n_groups, n_kvecs, 2)`.
        short_range: Cached real-space short-range output.
        long_range: Cached reciprocal-space long-range output.
        self_interaction: Cached self-interaction correction output.
        exclusion: Cached bonded-pair exclusion correction output.
    """

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
    """Patch for updating Ewald structure factors on Monte Carlo accept/reject.

    Attributes:
        new_structure_factor: Updated structure factor accumulator to apply on
            acceptance; its Kahan compensation is carried over with the value.
        lens: Lens to the ``EwaldCache`` in the state.
    """

    new_structure_factor: KahanSummand[Array]
    system_idx: Index[SystemId]
    lens: Lens[State, EwaldCache[Gradient, Hessian]] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        mask = accept[self.system_idx]
        new_sf = self.new_structure_factor
        return self.lens.apply(
            state,
            lambda cache: EwaldCache(
                structure_factor=jax.tree.map(
                    lambda new, old: where_broadcast_last(mask, new, old),
                    new_sf,
                    cache.structure_factor,
                ),
                short_range=cache.short_range,
                long_range=cache.long_range,
                self_interaction=cache.self_interaction,
                exclusion=cache.exclusion,
            ),
        )


type EwaldShortRangeInput = GraphPotentialInput[
    EwaldParameters, IsEwaldPointData, HasCell[Periodic3D], Literal[2]
]
"""Input type for the real-space short-range Ewald energy."""

type EwaldSelfInput = GraphPotentialInput[
    EwaldParameters, IsEwaldPointData, HasCell[Periodic3D], Literal[0]
]
"""Input type for the Ewald self-interaction correction."""


@dataclass
class EwaldLongRangeInput[State]:
    """Input for the reciprocal-space (long-range) Ewald energy.

    Attributes:
        point_cloud: Particle and system data.
        parameters: Ewald convergence parameters and k-vectors.
        cache: Cached structure factors for incremental updates; ``None`` for full computation.
        cache_lens: Lens to the ``EwaldCache`` in the state; ``None`` disables cache patching.
        changes_from_prev: Changed particles for incremental structure factor updates.
    """

    point_cloud: PointCloud[IsEwaldPointData, HasCell[Periodic3D]]
    parameters: EwaldParameters
    cache: EwaldCache[Any, Any] | None = None
    cache_lens: Lens[State, EwaldCache[Any, Any]] | None = None
    changes_from_prev: WithIndices[ParticleId, IsEwaldPointData] | None = None

    @property
    def volume(self) -> Array:
        return self.point_cloud.systems.data.cell.volume

    @property
    def kvecs(self) -> Array:
        sys_idx = self.point_cloud.systems.index
        return triangular_3x3_matmul(
            self.point_cloud.systems.data.cell.inverse_vectors.mT[:, None] * 2 * jnp.pi,
            self.parameters.reciprocal_lattice_shifts[sys_idx],
            lower=False,
        )


def ewald_self_interaction_energy(
    inp: EwaldSelfInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Self-interaction correction for Ewald summation.

    Removes the artificial interaction of each charge with its own Gaussian
    cloud introduced by the Ewald splitting.

    Math: ``E_self = -alpha / sqrt(pi) * sum_i q_i^2 * TO_STANDARD_UNITS``.

    Summed per system via segment_sum. Positions in Ang, charges in e,
    energy in eV.
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
    """Reciprocal-space energy from structure factors.

    Math: ``E_lr = sum_k P(k) * |S(k)|^2`` where ``P(k)`` is the prefactor
    and ``S(k)`` the structure factor.
    """
    return einops.einsum(
        prefactor(inp),
        structure_factor,
        structure_factor,
        "batch_size kvecs, batch_size kvecs two, batch_size kvecs two -> batch_size",
    )


def prefactor(inp: EwaldLongRangeInput[Any]) -> Array:
    """Reciprocal-space prefactor for each k-vector.

    Math: ``P(k) = 2*pi/V * exp(-k^2 / (4*alpha^2)) / k^2`` for k != 0,
    zero for k = 0. The ``(2 - leading_zero)`` factor accounts for the
    Hermitian symmetry optimization: only half the k-vectors are stored
    (k and -k give conjugate contributions).

    Returns:
        Prefactor array, shape ``(batch_size, n_kvecs)``.
    """
    sys_idx = inp.point_cloud.systems.index
    alpha = inp.parameters.alpha[sys_idx]
    rls = inp.parameters.reciprocal_lattice_shifts[sys_idx]
    kv = inp.kvecs
    k_squared = einops.einsum(
        kv, kv, "batch_size kvecs dim, batch_size kvecs dim -> batch_size kvecs"
    )
    mask = k_squared > 0
    k_squared = jnp.where(mask, k_squared, 1)
    result = (
        (2 * jnp.pi)
        / inp.volume[:, None]
        * jnp.exp(-k_squared / (4 * alpha[:, None] ** 2))
        / k_squared
    )
    leading_zero = rls[..., 0] == 0
    result = (2 - leading_zero) * result  # correct for half the k-vectors being dropped
    return jnp.where(mask, result, 0.0)


@functools.partial(
    jax.custom_jvp,
    nondiff_argnames=("batch_mask", "cache", "changes"),
)
def _structure_factor_update(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
    cache: EwaldCache[Any, Any],
    changes: WithIndices[ParticleId, IsEwaldPointData],
) -> KahanSummand[Array]:
    """Incremental structure factor update.

    Math: ``S'(k) = S(k) + dS(k)`` where
    ``dS = sum_changed [rho_new(k) - rho_old(k)]``.

    Adds the contribution of changed particles and subtracts their old
    contribution, using the cached ``S(k)`` from the previous step. The delta is
    folded in with Kahan compensation so the low-order bits dropped when a small
    ``dS(k)`` meets a large ``S(k)`` are carried into the next update instead of
    accumulating as drift over a long chain of moves.
    """
    idx = changes.indices
    idx_data = idx.indices
    updates = changes.data
    new_response = _frequency_response(
        positions[idx_data], charges[idx_data], kvecs, batch_mask[idx_data]
    )
    old_response = _frequency_response(
        updates.positions,
        updates.charges,
        kvecs,
        updates.system,
    )
    sk_delta = segment_sum(
        new_response,
        batch_mask.indices[idx_data],
        batch_mask.num_labels,
        mode="drop",
    ) - segment_sum(
        old_response,
        updates.system.indices,
        updates.system.num_labels,
        mode="drop",
    )
    return cache.structure_factor + sk_delta


@functools.partial(_structure_factor_update.defjvp, symbolic_zeros=True)
def _structure_factor_update_jvp(
    batch_mask: Index[SystemId],
    cache: EwaldCache[Any, Any],
    changes: WithIndices[ParticleId, IsEwaldPointData],
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, Array],
):
    """Custom JVP for ``_structure_factor_update``.

    Computes the full structure factor JVP (not incremental delta) because
    the cached structure factor is treated as a constant -- only the current
    positions/charges/kvecs contribute tangents. This ensures correct
    gradients through the incremental update path. The Kahan compensation is
    exactly zero in real arithmetic and hence carries a zero tangent.
    """
    positions, charges, kvecs = primals
    d_positions, d_charges, d_kvecs = tangents
    sk = _structure_factor_update(
        positions,
        charges,
        kvecs,
        batch_mask,
        cache,
        changes,
    )
    full_response = _frequency_response(positions, charges, kvecs, batch_mask)
    full_response_dot = jnp.zeros_like(full_response)
    if not isinstance(d_positions, jax.custom_derivatives.SymbolicZero):
        full_response_dot += einops.einsum(
            d_positions,
            kvecs[batch_mask.indices],
            "particles dim, particles shifts dim -> particles shifts",
        )[..., None]
    if not isinstance(d_kvecs, jax.custom_derivatives.SymbolicZero):
        full_response_dot += einops.einsum(
            d_kvecs[batch_mask.indices],
            positions,
            "particles shifts dim, particles dim -> particles shifts",
        )[..., None]
    if not isinstance(
        d_positions, jax.custom_derivatives.SymbolicZero
    ) or not isinstance(d_kvecs, jax.custom_derivatives.SymbolicZero):
        full_response_dot *= full_response[..., ::-1] * jnp.array([-1, 1])
    if not isinstance(d_charges, jax.custom_derivatives.SymbolicZero):
        full_response_dot += einops.einsum(
            d_charges,
            charges,
            full_response,
            "particles, particles, particles shifts two -> particles shifts two",
        )
    sk_dot = segment_sum(
        full_response_dot,
        batch_mask.indices,
        batch_mask.num_labels,
        mode="drop",
    )
    return sk, KahanSummand(sk_dot, jnp.zeros_like(sk_dot))


def structure_factor[State](
    inp: EwaldLongRangeInput[State],
) -> tuple[KahanSummand[Array], Patch[State]]:
    """Compute the structure factor, dispatching between full and incremental.

    Uses ``_structure_factor_full`` when no ``changes_from_prev`` is available,
    otherwise ``_structure_factor_update`` for incremental MC updates.

    Returns:
        Tuple of the structure factor accumulator and a cache patch. A full
        recomputation starts a fresh accumulator with zero compensation.
    """
    if inp.changes_from_prev is None:
        sk = KahanSummand.init(
            _structure_factor_full(
                inp.point_cloud.particles.data.positions,
                inp.point_cloud.particles.data.charges,
                inp.kvecs,
                inp.point_cloud.particles.data.system,
            )
        )
    else:
        assert inp.cache is not None, "Cache required for structure factor update"
        sk = _structure_factor_update(
            inp.point_cloud.particles.data.positions,
            inp.point_cloud.particles.data.charges,
            inp.kvecs,
            inp.point_cloud.particles.data.system,
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
    """Neutralizing-background correction for systems with nonzero net charge.

    Math: ``E_net = -(pi / (2 * V * alpha^2)) * Q^2 * TO_STANDARD_UNITS`` where
    ``Q = sum_i q_i`` is the per-system net charge and ``V`` the cell volume.

    Replaces the omitted (divergent) ``k = 0`` term of the reciprocal sum with a
    uniform neutralizing background, restoring independence of the total energy
    from ``alpha``. Vanishes for charge-neutral systems. Position-independent (no
    forces) but volume-dependent, so it contributes to the virial/pressure.
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
    """Reciprocal-space (long-range) Ewald energy.

    Math: ``E_lr = TO_STANDARD_UNITS * sum_k P(k) * |S(k)|^2 + E_net``.

    Wraps ``structure_factor`` + ``long_range``, adds the neutralizing-background
    correction (``ewald_net_charge_energy``) for nonzero net charge, and returns a
    cache patch for structure factor updates on MC accept/reject. ``S(k)`` is read
    off the accumulator with its compensation applied.
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
    """Composer for the long-range Ewald potential.

    Without a patch, builds a single full point cloud for the structure
    factor computation. With a patch, builds a point cloud containing
    the proposed changes and stores previous particle data for incremental
    structure factor updates.
    """

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
        ewald_parameters = self.parameters(state)
        particles = self.particles(state)
        systems = self.systems(state)
        cache = self.cache.get(state) if self.cache else None

        # Build PointCloud from separate components
        point_cloud = PointCloud(particles=particles, systems=systems)

        inp = EwaldLongRangeInput(
            point_cloud,
            ewald_parameters,
            cache,
            self.cache,
        )
        if patch is not None and self.probe is not None:
            particle_updates = self.probe(state, patch)
            indices = particle_updates.indices
            previous_values = (
                bind(particle_updates).focus(lambda x: x.data).set(particles[indices])
            )
            patched_particles = particles.update(indices, particle_updates.data)
            point_cloud = PointCloud(patched_particles, systems)
            inp = EwaldLongRangeInput(
                point_cloud,
                ewald_parameters,
                cache,
                self.cache,
                previous_values,
            )
        return Sum(Summand(inp))


@dataclass
class EwaldPotential[State, Gradients, Hessians, P: Patch[Any]](
    SummedPotential[State, Gradients, Hessians, P]
):
    """Complete Ewald potential with named access to each component term."""

    @property
    def short_range(self) -> Potential[State, Gradients, Hessians, P]:
        """Real-space short-range potential component."""
        return self.potentials[0]

    @property
    def long_range(self) -> Potential[State, Gradients, Hessians, P]:
        """Reciprocal-space long-range potential component."""
        return self.potentials[1]

    @property
    def self_interaction(self) -> Potential[State, Gradients, Hessians, P]:
        """Self-interaction correction term."""
        return self.potentials[2]

    @property
    def exclusion_correction(self) -> Potential[State, Gradients, Hessians, P]:
        """Exclusion correction: subtracts vacuum Coulomb energy for bonded/excluded pairs."""
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
    """Create the Ewald reciprocal-space (long-range) potential."""
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
    """Create the Ewald self-interaction correction potential.

    The self energy depends only on the charges present, not on their positions,
    and costs a single sum over particles. It is therefore recomputed in full on
    every call rather than accumulated from per-move deltas: accumulating it
    would form each delta as the difference of two full-system sums, whose
    rounding error then compounds over the Monte Carlo chain.
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
