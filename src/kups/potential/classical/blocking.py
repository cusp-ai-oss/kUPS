# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Blocking sphere potential for excluded volume constraints.

This module implements hard-sphere repulsion using blocking spheres that create
infinite energy barriers. Useful for preventing particle overlap with framework
atoms in porous materials (e.g., zeolites, MOFs) or enforcing geometric constraints.

Particles inside blocking spheres experience infinite repulsion, automatically
rejecting Monte Carlo moves that violate spatial constraints.
"""

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import Lens, SimpleLens, View
from kups.core.neighborlist import Edges, NearestNeighborList
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import (
    ExclusionId,
    HasExclusionIndex,
    HasInclusionIndex,
    HasMotifIndex,
    HasPositionsAndSystemIndex,
    HasUnitCell,
    InclusionId,
    MotifId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell
from kups.core.utils.jax import dataclass, field
from kups.potential.common.energy import (
    EnergyFunction,
    PositionAndUnitCell,
    PotentialFromEnergy,
    Sum,
    SumComposer,
    Summand,
    position_and_unitcell_idx_view,
)


@dataclass
class BlockingSpheresParameters:
    """Parameters defining blocking sphere positions and radii.

    Attributes:
        radii: Sphere radii, shape `(n_spheres,)`
        positions: Sphere centers, shape `(n_spheres, 3)`
        system: System assignment per sphere
        motif: Motif assignment per sphere
    """

    radii: Array
    positions: Array
    system: Index[SystemId]
    motif: Index[MotifId]

    def __post_init__(self):
        if isinstance(self.radii, Array):
            assert (*self.radii.shape, 3) == self.positions.shape


class _BlockingParticles(
    HasPositionsAndSystemIndex,
    HasMotifIndex,
    HasInclusionIndex,
    HasExclusionIndex,
    Protocol,
): ...


@dataclass
class _BlockingSpherePoints:
    """Wraps blocking sphere centers as NeighborListPoints for neighborlist calls."""

    positions: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


class IsBlockingSpheresProbe(Protocol):
    """Probe result for blocking spheres incremental updates.

    Bundles changed particle indices with the updated neighbor list,
    enabling efficient re-evaluation when only a subset of particles move.
    """

    @property
    def changed_particle_idx(self) -> Array: ...
    @property
    def neighborlist(self) -> NearestNeighborList: ...


@dataclass
class BlockingSpheresPotentialInput[
    Points: _BlockingParticles,
    MaybeUnitCell: UnitCell | None,
]:
    """Input for blocking spheres energy calculation.

    Attributes:
        parameters: Blocking sphere positions and radii
        particles: Indexed particle data with positions, system, and motif index
        unitcell: Optional unit cell for periodic boundary conditions
        edges: Particle-sphere pairs to check for blocking
    """

    parameters: BlockingSpheresParameters
    particles: Table[ParticleId, Points]
    unitcell: MaybeUnitCell
    edges: Edges[Literal[2]]


def blocking_spheres_energy[
    Points: _BlockingParticles,
    UC: UnitCell | None,
](
    inp: BlockingSpheresPotentialInput[Points, UC],
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """Calculate blocking spheres potential energy.

    Returns infinite energy for particles inside blocking spheres.

    Args:
        inp: Potential input containing particles, spheres, and edges

    Returns:
        Energy and patch with infinite energy for blocked particles.
    """
    edge_idx = inp.edges.indices.indices
    particle_idx, sph_idx = edge_idx[:, 0], edge_idx[:, 1]
    batch_data = inp.particles.data.system.indices[particle_idx]
    diffs = (
        inp.particles.data.positions[particle_idx] - inp.parameters.positions[sph_idx]
    )
    if inp.unitcell is not None:
        cell = inp.unitcell[batch_data]
        diffs = cell.wrap(diffs)
    dists = jnp.linalg.norm(diffs, axis=-1)
    radii = inp.parameters.radii[sph_idx]
    raw_energies = jnp.where(
        (dists < radii)
        & (
            inp.particles.data.motif.indices[particle_idx]
            == inp.parameters.motif.indices[sph_idx]
        ),
        jnp.inf,
        0.0,
    )
    batch_idx = Index(inp.particles.data.system.keys, batch_data)
    energies = batch_idx.sum_over(raw_energies)
    return WithPatch(energies, IdPatch())


@dataclass
class BlockingSpheresSumComposer[
    State,
    Ptch: Patch,
    S: HasUnitCell,
    Points: _BlockingParticles,
](SumComposer[State, BlockingSpheresPotentialInput[Points, UnitCell], Ptch]):
    """Composer for blocking spheres potential in energy summation.

    Attributes:
        particles_view: Extracts indexed particle data from state
        systems_view: Extracts indexed systems from state
        parameters_view: Extracts blocking sphere parameters from state
        neighborlist_view: Extracts neighbor list instance from state
        probe: Probe providing a IsBlockingSpheresProbe
    """

    particles_view: View[State, Table[ParticleId, Points]] = field(static=True)
    systems_view: View[State, Table[SystemId, S]] = field(static=True)
    parameters_view: View[State, BlockingSpheresParameters] = field(static=True)
    neighborlist_view: View[State, NearestNeighborList] = field(static=True)
    probe: Probe[State, Ptch, IsBlockingSpheresProbe] | None = field(static=True)

    def __call__(self, state: State, patch: Ptch | None):  # type: ignore[reportReturnType]
        particles = self.particles_view(state)
        systems = self.systems_view(state)
        parameters = self.parameters_view(state)
        neighborlist = self.neighborlist_view(state)

        if patch is not None and self.probe is not None:
            n_sys = particles.data.system.num_labels
            patched_state = patch(
                state, systems.set_data(jnp.ones((n_sys,), dtype=jnp.bool_))
            )
            probe_result = self.probe(state, patch)
            neighborlist = probe_result.neighborlist
            particles = self.particles_view(patched_state)

        # Build cutoffs: remap sphere system indices into systems index space
        seg_ids = parameters.system.indices_in(tuple(systems.keys))
        max_radii = jax.ops.segment_max(parameters.radii, seg_ids, len(systems.keys))
        cutoffs = Table(systems.keys, max_radii)

        # Build sphere rh as Indexed[ParticleId, _BlockingSpherePoints]
        p = parameters.positions.shape[0]
        sphere_inclusion = Index(
            tuple(InclusionId(lab) for lab in parameters.system.keys),
            parameters.system.indices,
        )
        sphere_exclusion = Index(
            tuple(ExclusionId(-1 - i) for i in range(p)),
            jnp.arange(p),
        )
        spheres = Table.arange(
            _BlockingSpherePoints(
                positions=parameters.positions,
                system=parameters.system,
                inclusion=sphere_inclusion,
                exclusion=sphere_exclusion,
            ),
            label=ParticleId,
        )

        edges = neighborlist(particles, spheres, systems, cutoffs)
        unitcell = systems.data.unitcell
        return Sum(
            Summand(
                BlockingSpheresPotentialInput(parameters, particles, unitcell, edges)
            )
        )


def make_blocking_spheres_potential[
    State,
    Gradients,
    Hessians,
    Ptch: Patch,
    Points: _BlockingParticles,
    S: HasUnitCell,
](
    particles_view: View[State, Table[ParticleId, Points]],
    systems_view: View[State, Table[SystemId, S]],
    parameters_view: View[State, BlockingSpheresParameters],
    neighborlist_view: View[State, NearestNeighborList],
    probe: Probe[State, Ptch, IsBlockingSpheresProbe] | None,
    gradient_lens: Lens[BlockingSpheresPotentialInput[Points, UnitCell], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> PotentialFromEnergy[
    State,
    BlockingSpheresPotentialInput[Points, UnitCell],
    Gradients,
    Hessians,
    Ptch,
]:
    """Create blocking sphere potential for excluded volume constraints.

    Args:
        particles_view: Extracts indexed particle data from state
        systems_view: Extracts indexed systems from state
        parameters_view: Extracts blocking sphere parameters (positions, radii)
        neighborlist_view: Extracts neighbor list instance
        probe: Probe returning a IsBlockingSpheresProbe; ``None`` for full recomputation
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure

    Returns:
        Blocking sphere potential.
    """
    return PotentialFromEnergy(
        blocking_spheres_energy,
        BlockingSpheresSumComposer(
            particles_view=particles_view,
            systems_view=systems_view,
            parameters_view=parameters_view,
            neighborlist_view=neighborlist_view,
            probe=probe,
        ),
        hessian_idx_view=hessian_idx_view,
        hessian_lens=hessian_lens,
        gradient_lens=gradient_lens,
        patch_idx_view=patch_idx_view,
        cache_lens=None,
    )


class IsBlockingSpheresState(Protocol):
    """Protocol for states providing all inputs for the blocking spheres potential."""

    @property
    def particles(self) -> Table[ParticleId, _BlockingParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def blocking_spheres_parameters(self) -> BlockingSpheresParameters: ...
    @property
    def blocking_spheres_neighborlist(self) -> NearestNeighborList: ...


@overload
def make_blocking_spheres_from_state[State](
    state: Lens[State, IsBlockingSpheresState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Any]: ...


@overload
def make_blocking_spheres_from_state[State](
    state: Lens[State, IsBlockingSpheresState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Any]: ...


@overload
def make_blocking_spheres_from_state[State, P: Patch](
    state: Lens[State, IsBlockingSpheresState],
    probe: Probe[State, P, IsBlockingSpheresProbe],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_blocking_spheres_from_state[State, P: Patch](
    state: Lens[State, IsBlockingSpheresState],
    probe: Probe[State, P, IsBlockingSpheresProbe],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_blocking_spheres_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a blocking spheres potential, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, systems, blocking sphere
            parameters, and neighbor list.
        probe: Probe returning a IsBlockingSpheresProbe; ``None`` for full
            recomputation.
        compute_position_and_unitcell_gradients: When ``True``, compute gradients
            w.r.t. particle positions and lattice vectors.

    Returns:
        Configured blocking spheres Potential.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[BlockingSpheresPotentialInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.particles.map_data(lambda p: p.positions),
                Table(x.particles.data.system.keys, x.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    if probe is not None:
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_blocking_spheres_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        state.focus(lambda x: x.blocking_spheres_parameters),
        state.focus(lambda x: x.blocking_spheres_neighborlist),
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
    )


if TYPE_CHECKING:
    _: EnergyFunction = blocking_spheres_energy
