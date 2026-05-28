# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Blocking sphere potential for excluded volume constraints.

This module implements hard-sphere repulsion using blocking spheres that create
infinite energy barriers. Useful for preventing particle overlap with framework
atoms in porous materials (e.g., zeolites, MOFs) or enforcing geometric constraints.

Particles inside blocking spheres experience infinite repulsion, automatically
rejecting Monte Carlo moves that violate spatial constraints.
"""

from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, Sequence, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import Cell
from kups.core.data import Index, Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import Edges, NeighborList
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
    GroupId,
    HasCell,
    HasGroupIndex,
    HasMotifIndex,
    HasPositionsAndSystemIndex,
    InclusionId,
    MotifId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.potential.common.energy import (
    EnergyFunction,
    PotentialFromEnergy,
    Sum,
    SumComposer,
    Summand,
)


class BlockingSpheresConfig(Protocol):
    """Protocol for a single blocking sphere description (center and radius)."""

    @property
    def center(self) -> tuple[float, float, float]: ...
    @property
    def radius(self) -> float: ...


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
        if not isinstance(self.radii, Array):
            return
        assert (*self.radii.shape, 3) == self.positions.shape, (
            f"Positions shape {self.positions.shape} must match radii shape {self.radii.shape} with last dimension 3"
        )
        assert self.radii.shape == self.system.shape == self.motif.shape, (
            f"Radii, system, and motif must have the same shape: {self.radii.shape}, {self.system.shape}, {self.motif.shape}"
        )

    @staticmethod
    def from_data(data: Sequence[Sequence[Sequence[BlockingSpheresConfig]]]):
        """Build parameters from a nested sequence of sphere configurations.

        Args:
            data: Nested sequence indexed as ``data[system_idx][motif_idx][sphere_idx]``,
                yielding the sphere configurations for each (system, motif) pair.

        Returns:
            BlockingSpheresParameters with flattened arrays of radii, positions,
            system assignments, and motif assignments.
        """
        radii = []
        positions = []
        system = []
        motif = []
        for sys_idx, sys_spheres in enumerate(data):
            for motif_idx, motif_spheres in enumerate(sys_spheres):
                for sphere in motif_spheres:
                    radii.append(sphere.radius)
                    positions.append(sphere.center)
                    system.append(SystemId(sys_idx))
                    motif.append(MotifId(motif_idx))
        radii = jnp.array(radii)
        positions = jnp.array(positions).reshape(-1, 3)
        system = Index.new(system, label=SystemId).populate_max_count()
        motif = Index.new(motif, label=MotifId).populate_max_count()
        return BlockingSpheresParameters(radii, positions, system, motif)


class _BlockingParticles(HasPositionsAndSystemIndex, HasGroupIndex, Protocol): ...


@dataclass
class _BlockingSpherePoints:
    """Wraps blocking sphere centers as NeighborListPoints for neighborlist calls."""

    positions: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


type BlockingSpheresNeighborListFactory = Callable[
    [Table[SystemId, Array]], NeighborList[Literal[2]]
]


class IsBlockingSpheresProbe(Protocol):
    """Probe result for blocking spheres incremental updates.

    Bundles changed particle indices with the updated neighbor list,
    enabling efficient re-evaluation when only a subset of particles move.
    """

    @property
    def changed_particle_idx(self) -> Array: ...
    @property
    def neighborlist(self) -> NeighborList[Literal[2]]: ...


@dataclass
class BlockingSpheresPotentialInput:
    """Input for blocking spheres energy calculation.

    Attributes:
        parameters: Blocking sphere positions and radii
        particles: Indexed particle data with positions, system, and group index
        groups: Indexed group data providing the motif index for each group
        cell: Per-system cell used for periodic boundary wrapping
        edges: Particle-sphere pairs to check for blocking
    """

    parameters: BlockingSpheresParameters
    particles: Table[ParticleId, _BlockingParticles]
    groups: Table[GroupId, HasMotifIndex]
    cell: Table[SystemId, Cell]
    edges: Edges[Literal[2]]


def blocking_spheres_energy(
    inp: BlockingSpheresPotentialInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """Calculate blocking spheres potential energy.

    Returns infinite energy for particles inside blocking spheres.

    Args:
        inp: Potential input containing particles, spheres, and edges

    Returns:
        Energy and patch with infinite energy for blocked particles.
    """
    particle_motif_idx = inp.edges.indices[:, 0]
    sph_idx = inp.edges.indices[:, 1].indices
    particles = inp.particles[particle_motif_idx]
    particle_sys = particles.system
    diffs = particles.positions - inp.parameters.positions[sph_idx]
    diffs = inp.cell[particle_sys].wrap(diffs)
    dists = jnp.linalg.norm(diffs, axis=-1)
    radii = inp.parameters.radii[sph_idx]
    sph_motif = inp.parameters.motif[sph_idx]
    # Compare motifs in a merged keyspace so that groups whose motif has no
    # sphere (and OOB-sentinel groups, e.g. buffered empty slots) never match
    # any sphere motif and therefore are not blocked.
    group_motif_idx, sph_motif_idx = Index.match(
        inp.groups[particles.group].motif, sph_motif
    )
    raw_energies = jnp.where(
        (dists < radii) & (group_motif_idx == sph_motif_idx), jnp.inf, 0.0
    )
    energies = particle_sys.sum_over(raw_energies)
    return WithPatch(energies, IdPatch())


@dataclass
class BlockingSpheresSumComposer[State, Ptch: Patch](
    SumComposer[State, BlockingSpheresPotentialInput, Ptch]
):
    """Composer for blocking spheres potential in energy summation.

    Attributes:
        particles_view: Extracts indexed particle data from state
        groups_view: Extracts indexed group data (motif assignments) from state
        systems_view: Extracts indexed systems from state
        parameters_view: Extracts blocking sphere parameters from state
        neighborlist_view: Extracts a factory that binds cutoffs to a neighbor list
        probe: Probe providing a IsBlockingSpheresProbe
    """

    particles_view: View[State, Table[ParticleId, _BlockingParticles]] = field(
        static=True
    )
    groups_view: View[State, Table[GroupId, HasMotifIndex]] = field(static=True)
    systems_view: View[State, Table[SystemId, HasCell]] = field(static=True)
    parameters_view: View[State, BlockingSpheresParameters] = field(static=True)
    neighborlist_view: View[State, BlockingSpheresNeighborListFactory] = field(
        static=True
    )
    probe: Probe[State, Ptch, IsBlockingSpheresProbe] | None = field(static=True)

    def __call__(self, state: State, patch: Ptch | None):  # type: ignore[reportReturnType]
        particles = self.particles_view(state)
        systems = self.systems_view(state)
        parameters = self.parameters_view(state)
        neighborlist_factory = self.neighborlist_view(state)
        probe_neighborlist = None

        if patch is not None and self.probe is not None:
            n_sys = particles.data.system.num_labels
            patched_state = patch(
                state, systems.set_data(jnp.ones((n_sys,), dtype=jnp.bool_))
            )
            probe_result = self.probe(state, patch)
            probe_neighborlist = probe_result.neighborlist
            particles = self.particles_view(patched_state)

        # Build cutoffs: remap sphere system indices into systems index space
        seg_ids = parameters.system.indices_in(tuple(systems.keys))
        max_radii = jax.ops.segment_max(parameters.radii, seg_ids, len(systems.keys))
        cutoffs = Table(systems.keys, max_radii)

        # NNList particles
        nnlist_particles = particles.map_data(
            lambda p: _BlockingSpherePoints(
                positions=p.positions,
                system=(sys := p.system.apply_mask(p.group.valid_mask)),
                inclusion=sys.to_cls(InclusionId),
                exclusion=Index.arange(len(sys), label=ExclusionId),
            )
        )

        # Build sphere rh as Indexed[ParticleId, _BlockingSpherePoints]
        p = parameters.positions.shape[0]
        sphere_inclusion = parameters.system.to_cls(InclusionId)
        # Let's just pick negative exclusion IDs for spheres to avoid any possible overlap with particle exclusion IDs
        sphere_exclusion = Index.new(tuple(ExclusionId(-1 - i) for i in range(p)))
        spheres = Table.arange(
            _BlockingSpherePoints(
                positions=parameters.positions,
                system=parameters.system,
                inclusion=sphere_inclusion,
                exclusion=sphere_exclusion,
            ),
            label=ParticleId,
        )

        neighborlist = probe_neighborlist or neighborlist_factory(cutoffs)
        edges = neighborlist(nnlist_particles, spheres, systems)
        cell = systems.map_data(lambda s: s.cell)
        groups = self.groups_view(state)
        return Sum(
            Summand(
                BlockingSpheresPotentialInput(
                    parameters, particles, groups, cell, edges
                )
            )
        )


def make_blocking_spheres_potential[State, Gradients, Hessians, Ptch: Patch](
    particles_view: View[State, Table[ParticleId, _BlockingParticles]],
    groups_view: View[State, Table[GroupId, HasMotifIndex]],
    systems_view: View[State, Table[SystemId, HasCell]],
    parameters_view: View[State, BlockingSpheresParameters],
    neighborlist_view: View[State, BlockingSpheresNeighborListFactory],
    probe: Probe[State, Ptch, IsBlockingSpheresProbe] | None,
    gradient_lens: Lens[BlockingSpheresPotentialInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> PotentialFromEnergy[
    State, BlockingSpheresPotentialInput, Gradients, Hessians, Ptch
]:
    """Create blocking sphere potential for excluded volume constraints.

    Args:
        particles_view: Extracts indexed particle data from state
        groups_view: Extracts indexed group data (motif assignments) from state
        systems_view: Extracts indexed systems from state
        parameters_view: Extracts blocking sphere parameters (positions, radii)
        neighborlist_view: Extracts a factory that binds cutoffs to a neighbor list
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
            groups_view=groups_view,
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
    def groups(self) -> Table[GroupId, HasMotifIndex]: ...
    @property
    def systems(self) -> Table[SystemId, HasCell]: ...
    @property
    def blocking_spheres_parameters(self) -> BlockingSpheresParameters: ...
    def blocking_spheres_neighborlist(
        self, cutoffs: Table[SystemId, Array]
    ) -> NeighborList[Literal[2]]: ...


@overload
def make_blocking_spheres_from_state[State](
    state: Lens[State, IsBlockingSpheresState],
    probe: None = None,
) -> Potential[State, EmptyType, EmptyType, Any]: ...


@overload
def make_blocking_spheres_from_state[State, P: Patch](
    state: Lens[State, IsBlockingSpheresState],
    probe: Probe[State, P, IsBlockingSpheresProbe],
) -> Potential[State, EmptyType, EmptyType, P]: ...


def make_blocking_spheres_from_state(state: Any, probe: Any = None) -> Any:
    """Create a blocking spheres potential, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, groups, systems,
            blocking sphere parameters, and neighbor list.
        probe: Probe returning a IsBlockingSpheresProbe; ``None`` for full
            recomputation.

    Returns:
        Configured blocking spheres Potential.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if probe is not None:
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_blocking_spheres_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.groups),
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
