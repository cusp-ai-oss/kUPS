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
    Protocol,
    overload,
)

import einops
import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.special
from jax import Array
from scipy.special import erfc

from kups.core.constants import BOHR, HARTREE
from kups.core.data import Index, Table, WithIndices
from kups.core.lens import Lens, NestedLens, SimpleLens, View, bind
from kups.core.neighborlist import NearestNeighborList, all_connected_neighborlist
from kups.core.patch import Accept, IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY,
    EMPTY_LENS,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    SummedPotential,
    empty_patch_idx_view,
)
from kups.core.typing import (
    ExclusionId,
    HasCache,
    HasCharges,
    HasUnitCell,
    InclusionId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field, no_jax_tracing, tree_zeros_like
from kups.core.utils.math import triangular_3x3_matmul
from kups.core.utils.ops import where_broadcast_last
from kups.potential.classical.coulomb import coulomb_vacuum_energy
from kups.potential.common.energy import (
    EnergyFunction,
    PositionAndUnitCell,
    PotentialFromEnergy,
    Sum,
    SumComposer,
    Summand,
    position_and_unitcell_idx_view,
)
from kups.potential.common.graph import (
    GraphPotentialInput,
    IsRadiusGraphPoints,
    IsRadiusGraphProbe,
    LocalGraphSumComposer,
    PointCloud,
    PointCloudConstructor,
    RadiusGraphConstructor,
)

TO_STANDARD_UNITS = HARTREE * BOHR
"""Conversion factor from atomic units to standard energy units."""


@dataclass
class EwaldCache[Gradient, Hessian]:
    """Cached structure factors and per-component outputs for incremental updates.

    Attributes:
        structure_factor: Complex structure factors, shape `(n_groups, n_kvecs, 2)`.
        short_range: Cached real-space short-range output.
        long_range: Cached reciprocal-space long-range output.
        self_interaction: Cached self-interaction correction output.
        exclusion: Cached bonded-pair exclusion correction output.
    """

    structure_factor: Array  # (n_groups, n_kvecs, 2)
    short_range: PotentialOut[Gradient, Hessian]
    long_range: PotentialOut[Gradient, Hessian]
    self_interaction: PotentialOut[Gradient, Hessian]
    exclusion: PotentialOut[Gradient, Hessian]

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
            jnp.zeros((n_sys, n_kvecs, 2), dtype=float),
            tree_zeros_like(out),
            tree_zeros_like(out),
            tree_zeros_like(out),
            tree_zeros_like(out),
        )


@dataclass
class EwaldCachePatch[State, Gradient, Hessian](Patch[State]):
    """Patch for updating Ewald structure factors on Monte Carlo accept/reject.

    Attributes:
        new_structure_factor: Updated structure factors to apply on acceptance.
        lens: Lens to the ``EwaldCache`` in the state.
    """

    new_structure_factor: Array
    system_idx: Index[SystemId]
    lens: Lens[State, EwaldCache[Gradient, Hessian]] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        mask = accept[self.system_idx]
        new_sf = self.new_structure_factor
        return self.lens.apply(
            state,
            lambda cache: EwaldCache(
                structure_factor=where_broadcast_last(
                    mask, new_sf, cache.structure_factor
                ),
                short_range=cache.short_range,
                long_range=cache.long_range,
                self_interaction=cache.self_interaction,
                exclusion=cache.exclusion,
            ),
        )


@dataclass
class EwaldParameters:
    """Ewald summation parameters: convergence settings and reciprocal lattice vectors.

    Attributes:
        alpha: Ewald screening parameter [1/Ang], shape `(n_graphs,)`.
        cutoff: Real-space cutoff radius [Ang], shape `(n_graphs,)`.
        reciprocal_lattice_shifts: Integer k-vector coefficients,
            shape `(n_graphs, n_kvecs, 3)`.
    """

    alpha: Table[SystemId, Array]  # (n_graphs,)
    cutoff: Table[SystemId, Array]  # (n_graphs,)
    reciprocal_lattice_shifts: Table[
        SystemId, Array
    ]  # (n_graphs, n_rec_shifts, 3) integers

    @classmethod
    @no_jax_tracing
    def make(
        cls,
        charges: Table[ParticleId, IsEwaldPointData],
        unitcell: Table[SystemId, HasUnitCell],
        epsilon_total: float = 1e-8,
        real_cutoff: float | None = None,
    ) -> EwaldParameters:
        """Estimate Ewald parameters from indexed particles and systems.

        Splits particles by system index, estimates per-system parameters,
        and zero-pads k-vectors to the maximum count across systems.

        Args:
            charges: Indexed particles with charges and system assignment.
            unitcell: Indexed systems with unit cells.
            epsilon_total: Target total accuracy for the Ewald sum.
            real_cutoff: Optional real-space cutoff override; estimated if not given.

        Returns:
            ``EwaldParameters`` with estimated physics parameters.
        """
        # Unpack Indexed into per-system lists
        n_systems = len(unitcell.keys)
        sys_idx = charges.data.system.indices
        charges_list = [charges.data.charges[sys_idx == i] for i in range(n_systems)]
        unitcell_list = [unitcell.data.unitcell[i] for i in range(n_systems)]

        estimates_list = [
            estimate_ewald_parameters(
                c, u, real_cutoff=real_cutoff, epsilon_total=epsilon_total
            )
            for c, u in zip(charges_list, unitcell_list)
        ]
        shifts_list = [
            kvecs_from_kmax(u, est.k_max)
            for u, est in zip(unitcell_list, estimates_list)
        ]
        max_n_kvecs = max(len(s) for s in shifts_list)
        padded_shifts = jnp.stack(
            [jnp.pad(s, [(0, max_n_kvecs - len(s)), (0, 0)]) for s in shifts_list]
        )
        return cls(
            alpha=Table(
                unitcell.keys,
                jnp.asarray([est.alpha for est in estimates_list]),
            ),
            cutoff=Table(
                unitcell.keys,
                jnp.asarray([est.real_cutoff for est in estimates_list]),
            ),
            reciprocal_lattice_shifts=Table(unitcell.keys, padded_shifts),
        )


class IsEwaldPointData(HasCharges, IsRadiusGraphPoints, Protocol):
    """Particle data required by Ewald: charges, positions, system/inclusion/exclusion indices."""

    ...


type EwaldShortRangeInput = GraphPotentialInput[
    EwaldParameters, IsEwaldPointData, HasUnitCell, Literal[2]
]
"""Input type for the real-space short-range Ewald energy."""

type EwaldSelfInput = GraphPotentialInput[
    EwaldParameters, IsEwaldPointData, HasUnitCell, Any
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

    point_cloud: PointCloud[IsEwaldPointData, HasUnitCell]
    parameters: EwaldParameters
    cache: EwaldCache | None = None
    cache_lens: Lens[State, EwaldCache[Any, Any]] | None = None
    changes_from_prev: WithIndices[ParticleId, IsEwaldPointData] | None = None

    @property
    def volume(self) -> Array:
        return self.point_cloud.systems.data.unitcell.volume

    @property
    def kvecs(self) -> Array:
        sys_idx = Index.new(list(self.point_cloud.systems.keys))
        return triangular_3x3_matmul(
            self.point_cloud.systems.data.unitcell.inverse_lattice_vectors.mT[:, None]
            * 2
            * jnp.pi,
            self.parameters.reciprocal_lattice_shifts[sys_idx],
            lower=False,
        )


def ewald_self_interaction_energy(
    inp: EwaldSelfInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """Self-interaction correction for Ewald summation.

    Removes the artificial interaction of each charge with its own Gaussian
    cloud introduced by the Ewald splitting.

    Math: ``E_self = -alpha / sqrt(pi) * sum_i q_i^2 * TO_STANDARD_UNITS``.

    Summed per system via segment_sum. Positions in Ang, charges in e,
    energy in eV.
    """
    sys_idx = Index.new(list(inp.graph.systems.keys))
    energies = (
        -jax.ops.segment_sum(
            inp.graph.particles.data.charges**2,
            inp.graph.particles.data.system.indices,
            inp.graph.batch_size,
            mode="drop",
        )
        * inp.parameters.alpha[sys_idx]
        / jnp.sqrt(jnp.pi)
    )
    energies *= TO_STANDARD_UNITS
    return WithPatch(Table.arange(energies, label=SystemId), IdPatch())


def ewald_short_range_energy(
    inp: EwaldShortRangeInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
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
    return WithPatch(total, IdPatch())


def exclusion_correction_energy(
    inp: EwaldShortRangeInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """Correction for excluded pairs (e.g., bonded atoms).

    Math: ``E_excl = E_sr(excluded) - E_vacuum(excluded)``.

    Subtracts the vacuum Coulomb energy from the short-range Ewald energy
    for excluded pairs, ensuring they have zero net Coulomb interaction
    in the total Ewald sum.
    """
    return WithPatch(
        jax.tree.map(
            jnp.subtract,
            ewald_short_range_energy(inp).data,
            coulomb_vacuum_energy(inp).data,
        ),
        IdPatch(),
    )


def long_range(inp: EwaldLongRangeInput, structure_factor: Array) -> Energy:
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


def prefactor(inp: EwaldLongRangeInput) -> Array:
    """Reciprocal-space prefactor for each k-vector.

    Math: ``P(k) = 2*pi/V * exp(-k^2 / (4*alpha^2)) / k^2`` for k != 0,
    zero for k = 0. The ``(2 - leading_zero)`` factor accounts for the
    Hermitian symmetry optimization: only half the k-vectors are stored
    (k and -k give conjugate contributions).

    Returns:
        Prefactor array, shape ``(batch_size, n_kvecs)``.
    """
    sys_idx = Index.new(inp.point_cloud.systems.keys)
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


def _frequency_response(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
) -> Array:
    """Per-particle response in reciprocal space.

    Math: ``rho_i(k) = q_i * [cos(k . r_i), sin(k . r_i)]``.

    Returns:
        Response array, shape ``(n_particles, n_kvecs, 2)`` for cos/sin.
    """
    exponent = einops.einsum(
        kvecs[batch_mask.indices],
        positions,
        "particles shifts dim, particles dim->particles shifts",
    )
    # particles x shifts x 2
    response = charges[:, None, None] * jnp.stack(
        [jnp.cos(exponent), jnp.sin(exponent)], axis=-1
    )
    return response


def _structure_factor_full(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
) -> Array:
    """Full structure factor computation.

    Math: ``S(k) = sum_i rho_i(k)`` summed per system via segment_sum.

    Returns:
        Structure factor, shape ``(n_systems, n_kvecs, 2)``.
    """
    response = _frequency_response(positions, charges, kvecs, batch_mask)
    structure_factor = jax.ops.segment_sum(
        response,
        batch_mask.indices,
        batch_mask.num_labels,
        mode="drop",
    )
    return structure_factor


@functools.partial(
    jax.custom_jvp,
    nondiff_argnames=("batch_mask", "cache", "changes"),
)
def _structure_factor_update(
    positions: Array,
    charges: Array,
    kvecs: Array,
    batch_mask: Index[SystemId],
    cache: EwaldCache,
    changes: WithIndices[ParticleId, IsEwaldPointData],
):
    """Incremental structure factor update.

    Math: ``S'(k) = S(k) + dS(k)`` where
    ``dS = sum_changed [rho_new(k) - rho_old(k)]``.

    Adds the contribution of changed particles and subtracts their old
    contribution, using the cached ``S(k)`` from the previous step.
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
    sk_delta = jax.ops.segment_sum(
        new_response,
        batch_mask.indices[idx_data],
        batch_mask.num_labels,
        mode="drop",
    ) - jax.ops.segment_sum(
        old_response,
        updates.system.indices,
        updates.system.num_labels,
        mode="drop",
    )
    new_sk = cache.structure_factor + sk_delta
    return new_sk


@functools.partial(_structure_factor_update.defjvp, symbolic_zeros=True)
def _structure_factor_update_jvp(
    batch_mask: Index[SystemId],
    cache: EwaldCache,
    changes: WithIndices[ParticleId, IsEwaldPointData],
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, Array],
):
    """Custom JVP for ``_structure_factor_update``.

    Computes the full structure factor JVP (not incremental delta) because
    the cached structure factor is treated as a constant -- only the current
    positions/charges/kvecs contribute tangents. This ensures correct
    gradients through the incremental update path.
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
    sk_dot = jnp.zeros_like(sk)
    full_response = _frequency_response(positions, charges, kvecs, batch_mask)
    full_response_dot = jnp.zeros_like(full_response)
    if not isinstance(d_positions, jax.custom_derivatives.SymbolicZero):
        full_response_dot += einops.einsum(
            d_positions,
            kvecs[batch_mask.indices],
            "particles dim, particles shifts dim, -> particles shifts",
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
    sk_dot = jax.ops.segment_sum(
        full_response_dot,
        batch_mask.indices,
        batch_mask.num_labels,
        mode="drop",
    )
    return sk, sk_dot


def structure_factor[State](
    inp: EwaldLongRangeInput[State],
) -> tuple[Array, Patch[State]]:
    """Compute the structure factor, dispatching between full and incremental.

    Uses ``_structure_factor_full`` when no ``changes_from_prev`` is available,
    otherwise ``_structure_factor_update`` for incremental MC updates.

    Returns:
        Tuple of structure factor array and a cache patch.
    """
    if inp.changes_from_prev is None:
        sk = _structure_factor_full(
            inp.point_cloud.particles.data.positions,
            inp.point_cloud.particles.data.charges,
            inp.kvecs,
            inp.point_cloud.particles.data.system,
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
        EwaldCachePatch(sk, Index.new(inp.point_cloud.systems.keys), inp.cache_lens)
        if inp.cache_lens is not None
        else IdPatch()
    )
    return sk, patch


def ewald_long_range_energy[State](
    inp: EwaldLongRangeInput[State],
) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
    """Reciprocal-space (long-range) Ewald energy.

    Math: ``E_lr = TO_STANDARD_UNITS * sum_k P(k) * |S(k)|^2``.

    Wraps ``structure_factor`` + ``long_range`` and returns a cache patch
    for structure factor updates on MC accept/reject.
    """
    structure_out, patch = structure_factor(inp)
    energy = long_range(inp, structure_out)
    assert energy.shape == (inp.point_cloud.batch_size,), (
        f"Expected energy shape {(inp.point_cloud.batch_size,)} but got {energy.shape}."
    )
    energy = energy * TO_STANDARD_UNITS
    return WithPatch(Table.arange(energy, label=SystemId), patch)


@dataclass
class EwaldLongRangeComposer[
    State,
    Ptch: Patch,
]:
    """Composer for the long-range Ewald potential.

    Without a patch, builds a single full point cloud for the structure
    factor computation. With a patch, builds a point cloud containing
    the proposed changes and stores previous particle data for incremental
    structure factor updates.
    """

    particles: View[State, Table[ParticleId, IsEwaldPointData]] = field(static=True)
    systems: View[State, Table[SystemId, HasUnitCell]] = field(static=True)
    probe: Probe[State, Ptch, WithIndices[ParticleId, IsEwaldPointData]] | None = field(
        static=True
    )
    parameters: Lens[State, EwaldParameters] = field(static=True)
    cache: Lens[State, EwaldCache] | None = field(static=True)

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
class EwaldPotential[State, Gradients, Hessians, P: Patch](
    SummedPotential[State, Gradients, Hessians, P]
):
    """Complete Ewald potential with named access to each component term."""

    @property
    def short_range(self):
        """Real-space short-range potential component."""
        return self.potentials[0]

    @property
    def long_range(self):
        """Reciprocal-space long-range potential component."""
        return self.potentials[1]

    @property
    def self_interaction(self):
        """Self-interaction correction term."""
        return self.potentials[2]

    @property
    def exclusion_correction(self):
        """Exclusion correction: subtracts vacuum Coulomb energy for bonded/excluded pairs."""
        return self.potentials[3]


def make_ewald_short_range_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    neighborlist_view: View[State, NearestNeighborList],
    parameter_view: View[State, EwaldParameters],
    probe: Probe[State, Ptch, IsRadiusGraphProbe[IsEwaldPointData]] | None,
    gradient_lens: Lens[PointCloud[IsEwaldPointData, HasUnitCell], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Any]:
    """Create the Ewald real-space (short-range) potential."""

    return PotentialFromEnergy(
        energy_fn=ewald_short_range_energy,
        composer=LocalGraphSumComposer(
            graph_constructor=RadiusGraphConstructor(
                particles=particles_view,
                systems=systems_view,
                cutoffs=pipe(parameter_view, lambda p: p.cutoff),
                neighborlist=neighborlist_view,
                probe=probe,
            ),
            parameter_view=parameter_view,
        ),
        gradient_lens=NestedLens(
            SimpleLens[GraphPotentialInput, PointCloud](lambda state: state.graph),
            gradient_lens,
        ),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )


def make_ewald_long_range_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_lens: Lens[State, EwaldParameters],
    cache_lens: Lens[State, EwaldCache] | None,
    probe: Probe[State, Ptch, WithIndices[ParticleId, IsEwaldPointData]] | None = None,
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasUnitCell], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[State, Hessians] = EMPTY_LENS,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Any]:
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
        gradient_lens=NestedLens(
            SimpleLens[EwaldLongRangeInput, PointCloud](
                lambda state: state.point_cloud
            ),
            gradient_lens,
        ),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens.focus(lambda x: x.long_range) if cache_lens else None,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )


def make_ewald_self_interaction_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, EwaldParameters],
    probe: Probe[State, Ptch, WithIndices[ParticleId, IsEwaldPointData]] | None = None,
    gradient_lens: Lens[
        PointCloud[IsEwaldPointData, HasUnitCell], Gradients
    ] = EMPTY_LENS,
    hessian_lens: Lens[Gradients, Hessians] = EMPTY_LENS,
    hessian_idx_view: View[State, Hessians] = EMPTY_LENS,
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Any]:
    """Create the Ewald self-interaction correction potential."""
    return PotentialFromEnergy(
        energy_fn=ewald_self_interaction_energy,
        composer=LocalGraphSumComposer(
            graph_constructor=PointCloudConstructor(
                particles=particles_view,
                systems=systems_view,
                probe_particles=probe,
            ),
            parameter_view=parameter_view,
        ),
        gradient_lens=NestedLens(
            SimpleLens[GraphPotentialInput, PointCloud](lambda state: state.graph),
            gradient_lens,
        ),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )


def make_ewald_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsEwaldPointData]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    neighborlist_view: View[State, NearestNeighborList],
    parameter_lens: Lens[State, EwaldParameters],
    cache_lens: Lens[State, EwaldCache] | None,
    probe: Probe[State, Ptch, IsRadiusGraphProbe[IsEwaldPointData]] | None,
    gradient_lens: Lens[PointCloud[IsEwaldPointData, HasUnitCell], Gradients],
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
        systems_view: Indexed system data (unit cell).
        neighborlist_view: Full neighbor list.
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
        inclusion_fn: Callable[[IsEwaldPointData], Index[Any]],
    ) -> Table[ParticleId, _ParticleData]:
        """Convert particles, deriving inclusion from `inclusion_fn`."""
        p = indexed.data
        particle_ids = tuple(indexed.keys)
        excl = Index(particle_ids, jnp.arange(len(particle_ids)), _cls=indexed.cls)
        return Table(
            indexed.keys,
            _ParticleData(
                p.positions,
                p.charges,
                p.system,
                inclusion=inclusion_fn(p).to_cls(InclusionId),
                exclusion=excl.to_cls(ExclusionId),
            ),
        )

    def _make_probe(
        inclusion_fn: Callable[[IsEwaldPointData], Index[Any]],
        neighborlist_override: NearestNeighborList | None = None,
    ) -> Probe[State, Ptch, IsRadiusGraphProbe[_ParticleData]] | None:
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
            neighborlist_after: NearestNeighborList
            neighborlist_before: NearestNeighborList

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
        probe=atomic_particles_probe,
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

    excl_rg = RadiusGraphConstructor(
        particles=excl_view,
        systems=systems_view,
        cutoffs=pipe(parameter_lens, lambda p: p.cutoff),
        neighborlist=lambda _: all_connected_neighborlist,
        probe=excl_probe,
    )
    exclusion_correction = PotentialFromEnergy(
        energy_fn=exclusion_correction_energy,
        composer=LocalGraphSumComposer(excl_rg, parameter_lens),
        gradient_lens=NestedLens(
            SimpleLens[GraphPotentialInput, PointCloud](lambda state: state.graph),
            gradient_lens,
        ),
        hessian_lens=hessian_lens,
        cache_lens=cache_lens.focus(lambda x: x.exclusion) if cache_lens else None,
        hessian_idx_view=hessian_idx_view,
        patch_idx_view=patch_idx_view,
    )
    if include_exclusion_mask:
        return EwaldPotential(
            (sr_potential, lr_potential, self_potential, exclusion_correction)
        )
    return EwaldPotential((sr_potential, lr_potential, self_potential))


class IsEwaldState[Params](Protocol):
    """Protocol for states providing all inputs for the Ewald potential."""

    @property
    def particles(self) -> Table[ParticleId, IsEwaldPointData]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def neighborlist(self) -> NearestNeighborList: ...
    @property
    def ewald_parameters(self) -> Params: ...


@overload
def make_ewald_from_state[
    State,
    InState: IsEwaldState[EwaldParameters | HasCache[EwaldParameters, Any]],
](
    state: Lens[State, InState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
    include_exclusion_mask: bool = False,
) -> EwaldPotential[State, EmptyType, EmptyType, Patch]: ...


@overload
def make_ewald_from_state[
    State,
    InState: IsEwaldState[EwaldParameters | HasCache[EwaldParameters, Any]],
](
    state: Lens[State, InState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
    include_exclusion_mask: bool = False,
) -> EwaldPotential[State, PositionAndUnitCell, EmptyType, Patch]: ...


@overload
def make_ewald_from_state[
    State,
    InState: IsEwaldState[HasCache[EwaldParameters, EwaldCache[EmptyType, EmptyType]]],
    P: Patch,
](
    state: Lens[State, InState],
    probe: Probe[State, P, IsRadiusGraphProbe[IsEwaldPointData]],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
    include_exclusion_mask: bool = False,
) -> EwaldPotential[State, EmptyType, EmptyType, P]: ...


@overload
def make_ewald_from_state[
    State,
    InState: IsEwaldState[
        HasCache[EwaldParameters, EwaldCache[PositionAndUnitCell, EmptyType]]
    ],
    P: Patch,
](
    state: Lens[State, InState],
    probe: Probe[State, P, IsRadiusGraphProbe[IsEwaldPointData]],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
    include_exclusion_mask: bool = False,
) -> EwaldPotential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_ewald_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
    include_exclusion_mask: bool = False,
) -> Any:
    """Create an Ewald potential from a typed state, optionally with incremental updates.

    When ``probe`` is ``None``, builds a static potential by extracting
    components directly from the state.  When a probe is provided, the
    potential supports incremental (cached) evaluation via the probe's
    patch mechanism.

    Args:
        state: Lens focusing on the Ewald state (particles, systems,
            neighborlist, and ewald_parameters).
        probe: Probe for incremental updates. ``None`` for a static
            potential.
        compute_position_and_unitcell_gradients: When ``True``, the
            returned potential computes gradients w.r.t. particle
            positions and lattice vectors (for forces / stress).
            Gradient type becomes ``PositionAndUnitCell``.
        include_exclusion_mask: Whether to include the exclusion
            correction term in the returned potential.

    Returns:
        An ``EwaldPotential`` combining short-range, long-range,
        self-energy, and (optionally) exclusion-correction terms.
        Gradient type is ``PositionAndUnitCell`` when gradients are
        requested, ``EmptyType`` otherwise.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view = empty_patch_idx_view
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[PointCloud, PositionAndUnitCell](
            lambda pc: PositionAndUnitCell(
                pc.particles.map_data(lambda p: p.positions),
                pc.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    param_view = state.focus(
        lambda x: (
            x.ewald_parameters.data
            if isinstance(x.ewald_parameters, HasCache)
            else x.ewald_parameters
        )
    )
    cache_view = None
    if probe is not None:
        param_view = state.focus(lambda x: x.ewald_parameters.data)
        cache_view = state.focus(lambda x: x.ewald_parameters.cache)
    return make_ewald_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        state.focus(lambda x: x.neighborlist),
        param_view,
        cache_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        include_exclusion_mask=include_exclusion_mask,
    )


@dataclass
class EwaldParameterEstimates:
    """Estimated optimal Ewald parameters for given accuracy."""

    alpha: float
    real_cutoff: float
    k_max: float
    error_real: float
    error_recip: float
    kvecs: Array


@no_jax_tracing
def estimate_ewald_parameters(
    charges: Array,
    unitcell: UnitCell,
    /,
    real_cutoff: float | None = None,
    alpha: float | None = None,
    epsilon_total: float = 1e-8,
) -> EwaldParameterEstimates:
    """Estimate optimal Ewald parameters for target accuracy.

    Not JAX-compatible (uses scipy); call before JIT compilation.
    Only works on single systems, not batched.

    Args:
        charges: Particle charges [e], shape `(n_particles,)`.
        unitcell: Unit cell parameters.
        real_cutoff: Real-space cutoff [Ang]; optimized if ``None``.
        alpha: Screening parameter [1/Ang]; optimized if ``None``.
        epsilon_total: Target total error (split equally between real/reciprocal).

    Returns:
        Optimized Ewald summation parameters.
    """
    # Note: only runs on a single system, not a batch of systems.
    # Input validation
    charges_np = np.asarray(charges)
    volume = np.asarray(unitcell.volume)
    Q2 = np.vdot(charges_np, charges_np)
    N = charges.size

    # smallest side length spanned by the unit cell
    max_radius = np.min(unitcell.perpendicular_lengths, axis=0) / 2

    # Split error budget equally
    eps_target = epsilon_total / 2

    # Length of a box shaped like the unit cell
    lattice_length = volume ** (1 / 3)

    def minimize(
        f: Callable[[float], float], bounds: tuple[float, float], n: int = 200
    ) -> float:
        attempts = np.linspace(bounds[0], bounds[1], n)  # type: ignore
        return attempts[np.argmin([f(x) for x in attempts])]

    def real_space_error(rc, alpha):
        """Standard Ewald real space error estimate"""
        if rc <= 0 or alpha <= 0:
            return np.inf
        return (Q2 / np.sqrt(N)) * (erfc(alpha * rc) / rc)

    def recip_space_error(kc, alpha):
        """Standard Ewald reciprocal space error estimate"""
        if kc <= 0 or alpha <= 0:
            return np.inf
        exp_arg = -(kc**2) / (4 * alpha**2)
        # Prevent numerical overflow/underflow
        if exp_arg < -700:  # exp(-700) ≈ 0
            return 0.0
        return (Q2 * alpha / (np.sqrt(N) * np.pi)) * np.exp(exp_arg)

    def optimal_rc(alpha):
        # Solve for rc s.t. real_space_error(rc, alpha) ≈ eps_target
        def real_error_diff(rc):
            return abs(real_space_error(rc, alpha) - eps_target)

        # Use reasonable bounds for rc based on system size
        if real_cutoff is None:
            rc_max = min(20.0, max_radius)  # Increased max rc
            rc_bounds = (0.01, rc_max)
            rc_opt = minimize(real_error_diff, rc_bounds)
        else:
            rc_opt = real_cutoff
        if real_space_error(rc_opt, alpha) > eps_target * 2:
            return np.inf
        return rc_opt

    def optimal_kc(alpha):
        """Find kc that gives the target reciprocal space error"""

        def recip_error_diff(kc):
            # Also minimize kc if all things being equal
            return abs(recip_space_error(kc, alpha) - eps_target)  # + kc * eps_target

        # Reasonable bounds for kc
        kc_bounds = (0.1, 5.0)  # Allow larger kc values
        kc_opt = minimize(recip_error_diff, kc_bounds)
        if recip_space_error(kc_opt, alpha) > eps_target * 2:
            return np.inf
        return kc_opt

    def total_cost(alpha: float) -> float:
        if alpha <= 0:
            return np.inf

        rc_opt = optimal_rc(alpha)
        if rc_opt == np.inf:
            return np.inf

        kc_opt = optimal_kc(alpha)
        if kc_opt == np.inf:
            return np.inf

        # Cost function: computational effort scales with rc^3 for real space
        # and with number of k-vectors for reciprocal space
        # Simple cost model: real space scales as rc^3, reciprocal as kc^3
        n_kvecs = (2 * kc_opt * lattice_length / 2 / np.pi + 1) ** 3
        rc_cost = rc_opt**3 * N / volume * 4 / 3 * np.pi
        cost_per_particle = rc_cost + n_kvecs
        return float(cost_per_particle)

    # Fast path without optimization
    if real_cutoff is not None:
        # First we solve for alpha given the real space cutoff by solving
        # erfc(alpha * rc) = rc * eps_target
        target_erfc = real_cutoff * eps_target
        alpha_result: scipy.optimize.OptimizeResult = scipy.optimize.minimize_scalar(  # type: ignore
            lambda z: abs(scipy.special.erfc(z) - target_erfc)
        )
        if not alpha_result.success:
            raise ValueError("Failed to find alpha for given real space cutoff.")
        alpha_opt: float = alpha_result.x / real_cutoff
        # For this alpha, we solve for kmax via the reciprocal space error estimate
        # exp(-k^2/4alpha^2) <= eps_target -> k >= 2 alpha sqrt(-ln(eps_target))
        kmax = 2 * alpha_opt * np.sqrt(-np.log(eps_target))
        return EwaldParameterEstimates(
            alpha=alpha_opt,
            real_cutoff=real_cutoff,
            k_max=kmax,
            error_real=real_space_error(real_cutoff, alpha_opt),
            error_recip=recip_space_error(kmax, alpha_opt),
            kvecs=kvecs_from_kmax(unitcell, kmax),
        )

    if alpha is None:
        alpha_opt = minimize(total_cost, (0.001, 2), n=400)
    else:
        alpha_opt = alpha
    rc_opt = optimal_rc(alpha_opt)
    kc_opt = optimal_kc(alpha_opt)

    # Verify the cutoffs are reasonable
    if rc_opt <= 0 or kc_opt <= 0 or rc_opt == np.inf or kc_opt == np.inf:
        raise ValueError("Invalid cutoff values computed")

    return EwaldParameterEstimates(
        alpha=alpha_opt,
        real_cutoff=rc_opt,
        k_max=kc_opt,
        error_real=real_space_error(rc_opt, alpha_opt),
        error_recip=recip_space_error(kc_opt, alpha_opt),
        kvecs=kvecs_from_kmax(unitcell, kc_opt),
    )


@no_jax_tracing
def kvecs_from_kmax(unitcell: UnitCell, kmax: float) -> Array:
    """Generate integer k-vector coefficients within a sphere of radius ``kmax``.

    Args:
        unitcell: Unit cell defining the reciprocal lattice.
        kmax: Maximum k-vector magnitude cutoff.

    Returns:
        Integer k-vector coefficients, shape ``(n_kvecs, 3)``.
    """
    rvecs = unitcell.inverse_lattice_vectors.mT * 2 * jnp.pi
    min_length = jnp.min(jnp.linalg.svd(rvecs)[1])
    n = jnp.ceil(kmax / min_length).astype(int)
    lattice = (jnp.arange(0, n + 1), jnp.arange(-n, n + 1), jnp.arange(-n, n + 1))
    vecs = jnp.stack(jnp.meshgrid(*lattice), axis=-1).reshape(-1, 3)
    kvecs = einops.einsum(vecs, rvecs, "kvecs dim1, dim1 dim2 -> kvecs dim2")
    return vecs[jnp.linalg.norm(kvecs, axis=-1) <= kmax]


if TYPE_CHECKING:
    _lr: EnergyFunction = ewald_long_range_energy
    _si: EnergyFunction = ewald_self_interaction_energy
    _sr: EnergyFunction = ewald_short_range_energy

    def _check_composer(c: EwaldLongRangeComposer):
        _: SumComposer = c
