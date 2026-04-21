# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Monte Carlo moves for molecular simulations.

This module provides Monte Carlo move proposals for grand canonical (GCMC) and
canonical (NVT) ensemble simulations. Moves include particle translations, rotations,
molecular insertions/deletions, and grand canonical exchange moves.

Key Components:

- **[MonteCarloMove][kups.mcmc.moves.MonteCarloMove]**: Base class for MC move proposals
- **[ParticleTranslationMove][kups.mcmc.moves.ParticleTranslationMove]**: Single particle displacement
- **[GroupTranslationMove][kups.mcmc.moves.GroupTranslationMove]**: Rigid body translation of molecular groups
- **[GroupRotationMove][kups.mcmc.moves.GroupRotationMove]**: Rigid body rotation of molecular groups
- **[ReinsertionMove][kups.mcmc.moves.ReinsertionMove]**: Random reinsertion with rotation
- **[ExchangeMove][kups.mcmc.moves.ExchangeMove]**: GCMC insertion/deletion of molecules
All moves operate on indexed particle data and support batched parallel systems.
Moves generate proposals compatible with [MCMCPropagator][kups.core.propagator.MCMCPropagator].
"""

from __future__ import annotations

import abc
from typing import Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.capacity import Capacity
from kups.core.data import Buffered, Index, Table, WithIndices, subselect
from kups.core.lens import Lens, View, bind
from kups.core.parameter_scheduler import (
    ParameterSchedulerState,
    acceptance_target_schedule,
)
from kups.core.patch import Patch
from kups.core.propagator import (
    ChangesFn,
    LogProbabilityRatio,
    LogProbabilityRatioFn,
    MCMCPropagator,
    PatchFn,
    Propagator,
    propose_mixed,
)
from kups.core.schedule import PropertyScheduler
from kups.core.typing import (
    DTypeLike,
    GroupId,
    HasGroupIndex,
    HasMotifAndSystemIndex,
    HasPositions,
    HasPositionsAndSystemIndex,
    HasSystemIndex,
    HasUnitCell,
    MotifId,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field, key_chain, tree_map
from kups.core.utils.math import triangular_3x3_matmul
from kups.core.utils.position import (
    center_of_mass,
    to_absolute_positions,
    to_relative_positions,
)
from kups.core.utils.quaternion import Quaternion


class HasGroupSystemIndex(HasGroupIndex, HasSystemIndex, Protocol):
    """Combined group and system index for particles."""

    ...


class HasPositionsGroupSystem(HasPositions, HasGroupIndex, HasSystemIndex, Protocol):
    """Combined positions, group and system index for particles."""

    ...


type BatchedPositions = Table[ParticleId, HasPositionsGroupSystem]
"""Type alias for particle positions with group and system indexing."""


class MonteCarloMove[State, Changes](ChangesFn[State, Changes], abc.ABC):
    """Base class for Monte Carlo move proposals that satisfy ``ChangesFn``."""

    @abc.abstractmethod
    def __call__(
        self, key: Array, state: State, /
    ) -> tuple[Changes, LogProbabilityRatio]: ...


class SymmetricTranslationDistribution(Protocol):
    """Protocol for symmetric translation distributions."""

    def __call__(
        self, key: Array, shape: tuple[int, ...], dtype: DTypeLike = float
    ) -> Array: ...


@dataclass
class ParticlePositionChanges:
    """Description of particle position updates.

    Attributes:
        particle_ids: Indices of particles to update
        new_positions: New position coordinates for specified particles
    """

    particle_ids: Index[ParticleId]
    new_positions: Array


def random_select_groups(
    key: Array,
    groups: Table[GroupId, HasSystemIndex],
    particles: Table[ParticleId, HasGroupSystemIndex],
    capacity: Capacity[int],
) -> Index[ParticleId]:
    """Randomly select one molecular group per simulation system.

    Args:
        key: JAX PRNG key.
        groups: Indexed groups with system assignment.
        particles: Indexed particle data with group and system indices.
        capacity: Capacity constraints for array operations.

    Returns:
        Index of particle IDs belonging to the selected groups.
    """
    particle_data = particles.data
    groups_to_move = groups.data.system.select_per_label(
        jax.random.bits(key, (particle_data.system.num_labels,), dtype=jnp.uint32)
    )
    groups_index = Index(particle_data.group.keys, groups_to_move)
    return Index(
        particles.keys,
        particle_data.group.where_flat(groups_index, capacity=capacity),
    )


@dataclass
class _RotPositions:
    """Internal: positions with group index for rotation helpers."""

    positions: Array
    group: Index[GroupId]


def random_rotate_groups(
    key: Array,
    particles: Table[ParticleId, HasPositionsGroupSystem],
    systems: Table[SystemId, HasUnitCell],
    step_width: Array,
) -> Array:
    """Rotate molecular groups around their centers of mass.

    Args:
        key: JAX PRNG key.
        particles: Indexed particles with positions, group, and system indices.
        systems: Indexed systems with unit cell data.
        step_width: Rotation step size (0=no rotation, 1=full random rotation).

    Returns:
        Rotated particle positions with center of mass preserved.
    """
    positions = particles.data.positions
    system_ids = particles.data.system.indices
    unitcell = systems.data.unitcell
    chain = key_chain(key)
    n_sys = len(systems)
    rotations = Quaternion.random(next(chain), (n_sys,)) ** step_width
    group_index = Index(
        tuple(GroupId(i) for i in range(n_sys)),
        system_ids,
    )
    data = _RotPositions(positions=positions, group=group_index)
    rot_particles = Table.arange(data, label=ParticleId)
    center_of_masses = center_of_mass(rot_particles, unitcell)
    rel_positions = to_relative_positions(rot_particles, unitcell, center_of_masses)
    rel_positions = rel_positions @ rotations[system_ids]
    new_data = _RotPositions(positions=rel_positions, group=group_index)
    new_rot_particles = Table.arange(new_data, label=ParticleId)
    new_abs_positions = to_absolute_positions(
        new_rot_particles,
        unitcell,
        center_of_masses,
    )
    return new_abs_positions


def translate_groups(
    translations: Table[SystemId, Array],
    particles: Table[ParticleId, HasPositionsAndSystemIndex],
    systems: Table[SystemId, HasUnitCell],
) -> Array:
    """Apply rigid body translations to particles with periodic wrapping.

    Args:
        translations: Per-system translation vectors, shape `(n_systems, 3)`.
        particles: Indexed particles with positions and system index.
        systems: Indexed systems with unit cell data.

    Returns:
        Translated and wrapped particle positions.
    """
    system_ids = particles.data.system
    new_positions = particles.data.positions + translations[system_ids]
    batched_unitcells = systems[system_ids].unitcell
    new_positions = batched_unitcells.wrap(new_positions)
    return new_positions


def propose_group_translation(
    key: Array,
    particles: BatchedPositions,
    groups: Table[GroupId, HasSystemIndex],
    systems: Table[SystemId, HasUnitCell],
    step_width: Table[SystemId, Array],
    capacity: Capacity[int],
    distribution: SymmetricTranslationDistribution = jax.random.normal,
) -> ParticlePositionChanges:
    """Propose a random rigid-body translation of one group per system."""
    chain = key_chain(key)
    n_sys = particles.data.system.num_labels
    selected = random_select_groups(next(chain), groups, particles, capacity)
    selected_data = particles[selected]
    selected_particles = Table.arange(selected_data, label=ParticleId)
    sys_idx = Index.new(systems.keys)
    width = step_width[sys_idx]
    translations = Table(
        systems.keys,
        distribution(next(chain), (n_sys, 3)) * width[:, None],
    )
    new_positions = translate_groups(translations, selected_particles, systems)
    return ParticlePositionChanges(particle_ids=selected, new_positions=new_positions)


def propose_group_rotation(
    key: Array,
    particles: BatchedPositions,
    groups: Table[GroupId, HasSystemIndex],
    systems: Table[SystemId, HasUnitCell],
    step_width: Table[SystemId, Array],
    capacity: Capacity[int],
) -> ParticlePositionChanges:
    """Propose a random rigid-body rotation of one group per system."""
    chain = key_chain(key)
    sys_idx = Index.new(systems.keys)
    width = step_width[sys_idx]
    selected = random_select_groups(next(chain), groups, particles, capacity)
    selected_data = particles[selected]
    selected_particles = Table.arange(selected_data, label=ParticleId)
    new_positions = random_rotate_groups(
        next(chain), selected_particles, systems, width
    )
    return ParticlePositionChanges(particle_ids=selected, new_positions=new_positions)


def propose_reinsertion(
    key: Array,
    particles: BatchedPositions,
    groups: Table[GroupId, HasSystemIndex],
    systems: Table[SystemId, HasUnitCell],
    capacity: Capacity[int],
) -> ParticlePositionChanges:
    """Propose a random reinsertion (new position + rotation) of one group per system."""
    chain = key_chain(key)
    n_sys = particles.data.system.num_labels
    selected = random_select_groups(next(chain), groups, particles, capacity)
    selected_data = particles[selected]
    selected_particles = Table.arange(selected_data, label=ParticleId)
    rotated_positions = random_rotate_groups(
        next(chain), selected_particles, systems, jnp.ones((n_sys,))
    )
    rotated_particles = (
        bind(selected_particles)
        .focus(lambda x: x.data.positions)
        .set(rotated_positions)
    )
    rel_offsets = jax.random.uniform(next(chain), shape=(n_sys, 3))
    abs_offsets = Table(
        systems.keys,
        triangular_3x3_matmul(systems.data.unitcell.lattice_vectors, rel_offsets),
    )
    new_positions = translate_groups(abs_offsets, rotated_particles, systems)
    return ParticlePositionChanges(particle_ids=selected, new_positions=new_positions)


def propose_particle_translation(
    key: Array,
    particles: BatchedPositions,
    systems: Table[SystemId, HasUnitCell],
    step_width: Table[SystemId, Array],
    distribution: SymmetricTranslationDistribution = jax.random.normal,
) -> ParticlePositionChanges:
    """Propose a random single-particle translation (one particle per system)."""
    chain = key_chain(key)
    n_sys = particles.data.system.num_labels
    random_ints = jax.random.bits(next(chain), shape=(n_sys,), dtype=jnp.uint32)
    raw_particle_ids = particles.data.system.select_per_label(random_ints)
    particle_ids = Index(particles.keys, raw_particle_ids)
    sys_idx = Index.new(systems.keys)
    width = step_width[sys_idx]
    translation = (
        distribution(
            next(chain), shape=(n_sys, 3), dtype=particles.data.positions.dtype
        )
        * width[:, None]
    )
    selected_data = particles[particle_ids]
    new_positions = selected_data.positions + translation
    cells = systems[selected_data.system].unitcell
    new_positions = cells.wrap(new_positions)
    return ParticlePositionChanges(
        particle_ids=particle_ids, new_positions=new_positions
    )


@dataclass
class ParticleTranslationMove[State](MonteCarloMove[State, ParticlePositionChanges]):
    """Single particle translation move. Satisfies ``ChangesFn``.

    Attributes:
        positions: Lens to particle positions in state.
        systems: Lens to indexed systems with unit cells.
        step_width: Lens to maximum displacement magnitude (tunable).
        distribution: Symmetric distribution for displacements (default: normal).
    """

    positions: View[State, BatchedPositions] = field(static=True)
    systems: View[State, Table[SystemId, HasUnitCell]] = field(static=True)
    step_width: View[State, Table[SystemId, Array]] = field(static=True)
    distribution: SymmetricTranslationDistribution = field(
        static=True, default=jax.random.normal
    )

    def __call__(
        self, key: Array, state: State, /
    ) -> tuple[ParticlePositionChanges, LogProbabilityRatio]:
        particles = self.positions(state)
        n_sys = particles.data.system.num_labels
        changes = propose_particle_translation(
            key,
            particles,
            self.systems(state),
            self.step_width(state),
            self.distribution,
        )
        return changes, Table.arange(jnp.zeros((n_sys,)), label=SystemId)


@dataclass
class GroupTranslationMove[State](MonteCarloMove[State, ParticlePositionChanges]):
    """Rigid body translation of molecular groups. Satisfies ``ChangesFn``.

    Attributes:
        particles: Lens to particle positions.
        groups: Lens to groups eligible for moves.
        systems: Lens to indexed systems with unit cells.
        step_width: Lens to maximum translation magnitude.
        capacity: Lens to capacity constraints.
        distribution: Symmetric distribution for displacements (default: normal).
    """

    particles: View[State, BatchedPositions] = field(static=True)
    groups: View[State, Table[GroupId, HasSystemIndex]] = field(static=True)
    systems: View[State, Table[SystemId, HasUnitCell]] = field(static=True)
    step_width: View[State, Table[SystemId, Array]] = field(static=True)
    capacity: View[State, Capacity[int]] = field(static=True)
    distribution: SymmetricTranslationDistribution = field(
        static=True, default=jax.random.normal
    )

    def __call__(
        self, key: Array, state: State, /
    ) -> tuple[ParticlePositionChanges, LogProbabilityRatio]:
        n_sys = self.particles(state).data.system.num_labels
        changes = propose_group_translation(
            key,
            self.particles(state),
            self.groups(state),
            self.systems(state),
            self.step_width(state),
            self.capacity(state),
            self.distribution,
        )
        return changes, Table.arange(jnp.zeros((n_sys,)), label=SystemId)


@dataclass
class GroupRotationMove[State](MonteCarloMove[State, ParticlePositionChanges]):
    """Rigid body rotation of molecular groups. Satisfies ``ChangesFn``.

    Attributes:
        particles: Lens to particle positions.
        groups: Lens to groups eligible for moves.
        systems: Lens to indexed systems with unit cell data.
        step_width: Lens to rotation magnitude (0=no rotation, 1=full).
        capacity: Lens to capacity constraints.
    """

    particles: View[State, BatchedPositions] = field(static=True)
    groups: View[State, Table[GroupId, HasSystemIndex]] = field(static=True)
    systems: View[State, Table[SystemId, HasUnitCell]] = field(static=True)
    step_width: View[State, Table[SystemId, Array]] = field(static=True)
    capacity: View[State, Capacity[int]] = field(static=True)

    def __call__(
        self, key: Array, state: State, /
    ) -> tuple[ParticlePositionChanges, LogProbabilityRatio]:
        n_sys = self.particles(state).data.system.num_labels
        changes = propose_group_rotation(
            key,
            self.particles(state),
            self.groups(state),
            self.systems(state),
            self.step_width(state),
            self.capacity(state),
        )
        return changes, Table.arange(jnp.zeros((n_sys,)), label=SystemId)


@dataclass
class ReinsertionMove[State](MonteCarloMove[State, ParticlePositionChanges]):
    """Random reinsertion of molecular groups. Satisfies ``ChangesFn``.

    Attributes:
        positions: Lens to particle positions.
        groups: Lens to groups eligible for moves.
        systems: Lens to indexed systems with unit cell data.
        capacity: Lens to capacity constraints.
    """

    positions: View[State, BatchedPositions] = field(static=True)
    groups: View[State, Table[GroupId, HasSystemIndex]] = field(static=True)
    systems: View[State, Table[SystemId, HasUnitCell]] = field(static=True)
    capacity: View[State, Capacity[int]] = field(static=True)

    def __call__(
        self, key: Array, state: State, /
    ) -> tuple[ParticlePositionChanges, LogProbabilityRatio]:
        n_sys = self.positions(state).data.system.num_labels
        changes = propose_reinsertion(
            key,
            self.positions(state),
            self.groups(state),
            self.systems(state),
            self.capacity(state),
        )
        return changes, Table.arange(jnp.zeros((n_sys,)), label=SystemId)


@runtime_checkable
class IsMotifData(Protocol):
    """Motif template data: positions indexed by motif assignment.

    Attributes:
        positions: Template particle positions, shape ``(n_template_particles, 3)``.
        motif: Per-particle motif assignment.
    """

    @property
    def positions(self) -> Array: ...
    @property
    def motif(self) -> Index[MotifId]: ...


@dataclass
class SystemIndex:
    """Index metadata for systems."""

    system: Index[SystemId]


@dataclass
class ExchangeParticleData:
    """Per-particle payload for GCMC exchange moves (no identity info).

    The ``motif`` field indexes into the motif template
    (``Indexed[MotifParticleId, MotifParticles]``) so the patch function
    can look up masses, charges, labels, etc.

    Attributes:
        new_positions: Proposed coordinates, shape ``(n, 3)``.
        group: Group assignment for each particle.
        system: System assignment for each particle.
        motif: Motif-template index for each particle.
    """

    new_positions: Array
    group: Index[GroupId]
    system: Index[SystemId]
    motif: Index[MotifParticleId]


@dataclass
class ExchangeGroupData:
    """Per-group payload for GCMC exchange moves (no identity info).

    Attributes:
        motif: Molecular species of each group.
        system: System assignment of each group.
    """

    motif: Index[MotifId]
    system: Index[SystemId]


@dataclass
class ExchangeChanges:
    """Combined particle and group changes for a GCMC exchange move.

    Each field pairs an ``Index`` of target slot ids with a ``Buffered``
    holding the new data and an occupation mask.  For insertions the
    occupation is ``True`` (slots become occupied); for deletions it is
    ``False`` (slots are freed).

    Attributes:
        particles: Target particle slots and buffered particle data.
        groups: Target group slots and buffered group data.
    """

    particles: WithIndices[ParticleId, Buffered[ParticleId, ExchangeParticleData]]
    groups: WithIndices[GroupId, Buffered[GroupId, ExchangeGroupData]]


def exchange_changes_from_position_changes(
    changes: ParticlePositionChanges,
    particles: Buffered[ParticleId, IsExchangeParticles],
    groups: Buffered[GroupId, HasMotifAndSystemIndex],
    n_sys: int,
) -> ExchangeChanges:
    """Convert a ``ParticlePositionChanges`` into ``ExchangeChanges``.

    Produces the same shape as ``insert_random_motif`` / ``delete_random_motif``
    (one group entry per system) so that ``jax.lax.select_n`` can mix them.

    Args:
        changes: Proposed particle position changes.
        particles: Current buffered particle positions.
        groups: Current buffered group metadata.
        n_sys: Number of systems (determines group output size).
    """
    selected = particles[changes.particle_ids]

    p_data = ExchangeParticleData(
        new_positions=changes.new_positions,
        group=selected.group,
        system=selected.system,
        motif=selected.motif,
    )
    p_buf = Buffered.arange(p_data, label=ParticleId)
    p_changes = WithIndices(changes.particle_ids, p_buf)

    # One group per system (same shape as insert/delete) — just re-assert existing groups
    group_idx = Index(groups.keys, jnp.zeros(n_sys, dtype=int))
    grps = groups[group_idx]
    g_data = ExchangeGroupData(motif=grps.motif, system=grps.system)
    g_buf = Buffered.arange(g_data, label=GroupId)
    g_changes = WithIndices(group_idx, g_buf)
    return ExchangeChanges(p_changes, g_changes)


def insert_random_motif(
    key: Array,
    motifs: Table[MotifParticleId, IsMotifData],
    particles: Buffered[ParticleId, HasPositionsGroupSystem],
    groups: Buffered[GroupId, HasMotifAndSystemIndex],
    unitcell: Table[SystemId, UnitCell],
    capacity: Capacity[int],
) -> ExchangeChanges:
    """Generate a GCMC insertion move for random molecular motifs.

    Args:
        key: JAX PRNG key.
        motifs: Indexed molecular templates for insertion.
        particles: Current buffered particle positions.
        groups: Current buffered group metadata.
        unitcell: Per-system unit cell parameters.
        capacity: Capacity constraints for state arrays.

    Returns:
        Exchange changes describing the insertion.
    """
    n_sys = particles.data.system.num_labels
    max_sys_count = particles.data.system.max_count
    n_motifs = motifs.data.motif.num_labels
    motif_max_seg = motifs.data.motif.max_count
    n_motif_particles = len(motifs)
    chain = key_chain(key)
    selected_motifs = jax.random.choice(
        next(chain), jnp.arange(n_motifs), shape=(n_sys,)
    )
    ins_system_ids, particle_idx = subselect(
        selected_motifs,
        motifs.data.motif.indices,
        output_buffer_size=capacity,
        num_segments=n_motifs,
    )
    # Gather the motifs
    new_positions = motifs.data.positions[particle_idx]
    # Rotate and translate
    rel_offsets = jax.random.uniform(next(chain), shape=(n_sys, 3))
    abs_offsets = triangular_3x3_matmul(unitcell.data.lattice_vectors, rel_offsets)
    rotations = Quaternion.random(next(chain), (n_sys,))
    sys_idx = Index(unitcell.keys, ins_system_ids)
    new_positions = (
        new_positions @ rotations[ins_system_ids] + abs_offsets[ins_system_ids]
    )
    new_positions = unitcell[sys_idx].wrap(new_positions)

    # Find free particle slots using Buffered.select_free
    n_free_particles = (~particles.occupation).sum()
    runtime_assert(
        n_free_particles >= capacity.size,
        f"Array size insufficient, requested {capacity.size} free entries while available {{available}}.",
        fmt_args={"available": n_free_particles},
    )
    free_particle_idx = particles.select_free(capacity.size).indices
    free_particle_idx = jnp.where(
        ins_system_ids < n_sys, free_particle_idx, len(particles)
    )

    # Find free group slots using Buffered.select_free
    n_free_groups = (~groups.occupation).sum()
    runtime_assert(
        n_free_groups >= n_sys,
        f"Array size insufficient, requested {n_sys} free entries while available {{available}}.",
        fmt_args={"available": n_free_groups},
    )
    free_group_index = groups.select_free(n_sys)
    free_group_idx = free_group_index.indices

    p_idx = Index(particles.keys, free_particle_idx)
    ins_particle_data = ExchangeParticleData(
        new_positions=new_positions,
        system=Index.integer(
            ins_system_ids, n=n_sys, label=SystemId, max_count=max_sys_count
        ),
        group=Index.integer(
            free_group_idx[ins_system_ids],
            n=len(groups),
            label=GroupId,
            max_count=motif_max_seg,
        ),
        motif=Index.integer(particle_idx, n=n_motif_particles, label=MotifParticleId),
    )
    ins_particles = WithIndices(
        p_idx,
        Buffered.arange(ins_particle_data, label=ParticleId),
    )

    ins_group_data = ExchangeGroupData(
        motif=Index.integer(selected_motifs, n=n_motifs, label=MotifId),
        system=Index.integer(
            jnp.arange(n_sys),
            n=n_sys,
            label=SystemId,
            max_count=groups.data.system.max_count,
        ),
    )
    ins_groups = WithIndices(
        free_group_index, Buffered.arange(ins_group_data, label=GroupId)
    )

    return ExchangeChanges(ins_particles, ins_groups)


def delete_random_motif(
    key: Array,
    motifs: Table[MotifParticleId, IsMotifData],
    particles: Buffered[ParticleId, HasPositionsGroupSystem],
    groups: Buffered[GroupId, HasMotifAndSystemIndex],
    capacity: Capacity[int],
) -> ExchangeChanges:
    """Generate a GCMC deletion move removing a random molecular group.

    Args:
        key: JAX PRNG key.
        motifs: Indexed molecular templates (for metadata consistency).
        particles: Current buffered particle positions.
        groups: Current buffered group metadata.
        capacity: Capacity constraints for state arrays.

    Returns:
        Exchange changes describing the deletion.
    """
    chain = key_chain(key)
    n_sys = particles.data.system.num_labels
    max_sys_count = particles.data.system.max_count
    n_motifs = motifs.data.motif.num_labels
    motif_max_seg = motifs.data.motif.max_count
    n_motif_particles = len(motifs)

    # Randomly select a motif to delete for each system
    motifs_to_delete = jax.random.choice(next(chain), n_motifs, shape=(n_sys,))
    # Mark groups whose label matches selected motif and belong to correct system
    possible_group_ids = jnp.where(
        groups.data.motif.indices == motifs_to_delete[groups.data.system.indices],
        groups.data.system.indices,
        n_sys,
    )
    possible_group_sys = Index.integer(
        possible_group_ids,
        n=n_sys,
        label=SystemId,
        max_count=groups.data.system.max_count,
    )

    possible_groups = Table(groups.keys, SystemIndex(system=possible_group_sys))

    # Select a random group belonging to the selected motifs from each system
    selected = random_select_groups(next(chain), possible_groups, particles, capacity)
    selected_data = particles[selected]

    group_ids_to_delete = selected_data.group.indices
    del_system_ids = selected_data.system.indices

    deleted_group_ids = jnp.full(n_sys, len(groups), dtype=group_ids_to_delete.dtype)
    deleted_group_ids = deleted_group_ids.at[del_system_ids].set(group_ids_to_delete)

    n_deleted = selected.indices.shape[0]

    del_particle_data = ExchangeParticleData(
        new_positions=jnp.zeros((n_deleted, 3), dtype=particles.data.positions.dtype),
        system=Index.integer(
            jnp.full(n_deleted, n_sys),
            n=n_sys,
            label=SystemId,
            max_count=max_sys_count,
        ),
        group=Index.integer(
            jnp.full(n_deleted, len(groups)),
            n=len(groups),
            label=GroupId,
            max_count=motif_max_seg,
        ),
        motif=Index.integer(
            jnp.full(n_deleted, n_motif_particles),
            n=n_motif_particles,
            label=MotifParticleId,
        ),
    )
    del_particles = WithIndices(
        selected,
        Buffered.arange(del_particle_data, num_occupied=0, label=ParticleId),
    )

    g_idx = Index(groups.keys, deleted_group_ids)
    del_group_data = ExchangeGroupData(
        motif=Index.integer(jnp.full(n_sys, n_motifs), n=n_motifs, label=MotifId),
        system=Index.integer(
            jnp.full(n_sys, n_sys),
            n=n_sys,
            label=SystemId,
            max_count=groups.data.system.max_count,
        ),
    )
    del_groups = WithIndices(
        g_idx,
        Buffered.arange(del_group_data, num_occupied=0, label=GroupId),
    )

    return ExchangeChanges(del_particles, del_groups)


@dataclass
class ExchangeMove[State](MonteCarloMove[State, ExchangeChanges]):
    """Grand canonical Monte Carlo (GCMC) insertion/deletion move. Satisfies ``ChangesFn``.

    Randomly proposes either insertion or deletion of a molecular group
    with 50% probability each via ``propose_mixed``.

    Attributes:
        positions: Lens to buffered particle positions.
        groups: Lens to buffered molecular groups.
        motifs: Lens to molecular templates available for insertion.
        unitcell: Lens to unit cell parameters.
        capacity: Lens to capacity constraints.
    """

    positions: View[State, Buffered[ParticleId, HasPositionsGroupSystem]] = field(
        static=True
    )
    groups: View[State, Buffered[GroupId, HasMotifAndSystemIndex]] = field(static=True)
    motifs: View[State, Table[MotifParticleId, IsMotifData]] = field(static=True)
    unitcell: View[State, Table[SystemId, UnitCell]] = field(static=True)
    capacity: View[State, Capacity[int]] = field(static=True)

    def _zero_ratio(self, state: State) -> LogProbabilityRatio:
        n_sys = self.positions(state).data.system.num_labels
        return Table.arange(jnp.zeros((n_sys,)), label=SystemId)

    def _propose_insertion(
        self, key: Array, state: State, /
    ) -> tuple[ExchangeChanges, LogProbabilityRatio]:
        changes = insert_random_motif(
            key,
            self.motifs(state),
            self.positions(state),
            self.groups(state),
            self.unitcell(state),
            self.capacity(state),
        )
        return changes, self._zero_ratio(state)

    def _propose_deletion(
        self, key: Array, state: State, /
    ) -> tuple[ExchangeChanges, LogProbabilityRatio]:
        changes = delete_random_motif(
            key,
            self.motifs(state),
            self.positions(state),
            self.groups(state),
            self.capacity(state),
        )
        return changes, self._zero_ratio(state)

    def __call__(
        self, key: Array, state: State, /
    ) -> tuple[ExchangeChanges, LogProbabilityRatio]:
        chain = key_chain(key)
        changes, log_ratio, _ = propose_mixed(
            next(chain), state, (self._propose_insertion, self._propose_deletion)
        )
        return changes, log_ratio


class IsMCMCMoveState(Protocol):
    """Base state protocol for MCMC moves with particles, groups, systems, and capacity."""

    @property
    def particles(self) -> BatchedPositions: ...
    @property
    def groups(self) -> Table[GroupId, HasSystemIndex]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def move_capacity(self) -> Capacity[int]: ...


class IsExchangeParticles(HasPositionsGroupSystem, Protocol):
    @property
    def motif(self) -> Index[MotifParticleId]: ...


class IsTranslationState(IsMCMCMoveState, Protocol):
    """State protocol for group translation MCMC moves."""

    @property
    def translation_params(self) -> Table[SystemId, ParameterSchedulerState]: ...


class IsRotationState(IsMCMCMoveState, Protocol):
    """State protocol for group rotation MCMC moves."""

    @property
    def rotation_params(self) -> Table[SystemId, ParameterSchedulerState]: ...


class IsReinsertionState(IsMCMCMoveState, Protocol):
    """State protocol for particle reinsertion MCMC moves."""

    @property
    def reinsertion_params(self) -> Table[SystemId, ParameterSchedulerState]: ...


class IsDisplacementState(
    IsTranslationState, IsRotationState, IsReinsertionState, Protocol
):
    """State protocol for the combined translation/rotation/reinsertion move."""

    ...


class IsGCMCState(IsDisplacementState, Protocol):
    """State protocol for the combined displacement + exchange move."""

    @property
    def particles(self) -> Buffered[ParticleId, IsExchangeParticles]: ...
    @property
    def groups(self) -> Buffered[GroupId, HasMotifAndSystemIndex]: ...
    @property
    def motifs(self) -> Table[MotifParticleId, IsMotifData]: ...
    @property
    def exchange_params(self) -> Table[SystemId, ParameterSchedulerState]: ...


class IsExchangeState(IsMCMCMoveState, Protocol):
    """State protocol for particle exchange (grand canonical) MCMC moves."""

    @property
    def particles(self) -> Buffered[ParticleId, HasPositionsGroupSystem]: ...
    @property
    def groups(self) -> Buffered[GroupId, HasMotifAndSystemIndex]: ...
    @property
    def motifs(self) -> Table[MotifParticleId, IsMotifData]: ...
    @property
    def exchange_params(self) -> Table[SystemId, ParameterSchedulerState]: ...


def _sched(params_lens: Lens) -> PropertyScheduler:
    """Create a PropertyScheduler for acceptance-based step-width tuning."""
    return PropertyScheduler(params_lens, Table.transform(acceptance_target_schedule))


def make_group_translation_mcmc_propagator[State, Move: Patch](
    state: Lens[State, IsTranslationState],
    patch_fn: PatchFn[State, ParticlePositionChanges, Move],
    probability_fn: LogProbabilityRatioFn[State, Move],
) -> MCMCPropagator[State, ParticlePositionChanges, Move]:
    """Build an MCMC propagator for random group translation moves."""
    move = GroupTranslationMove(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.groups),
        state.focus(lambda x: x.systems),
        pipe(
            state,
            lambda s: Table(s.translation_params.keys, s.translation_params.data.value),
        ),
        state.focus(lambda x: x.move_capacity),
    )
    return MCMCPropagator(
        patch_fn,
        (move,),
        probability_fn,
        (_sched(state.focus(lambda x: x.translation_params)),),
    )


def make_group_rotation_mcmc_propagator[State, Move: Patch](
    state: Lens[State, IsRotationState],
    patch_fn: PatchFn[State, ParticlePositionChanges, Move],
    probability_fn: LogProbabilityRatioFn[State, Move],
) -> MCMCPropagator[State, ParticlePositionChanges, Move]:
    """Build an MCMC propagator for rigid-body group rotation moves."""
    move = GroupRotationMove(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.groups),
        state.focus(lambda x: x.systems),
        pipe(
            state, lambda s: Table(s.rotation_params.keys, s.rotation_params.data.value)
        ),
        state.focus(lambda x: x.move_capacity),
    )
    return MCMCPropagator(
        patch_fn,
        (move,),
        probability_fn,
        (_sched(state.focus(lambda x: x.rotation_params)),),
    )


def make_reinsertion_mcmc_propagator[State, Move: Patch](
    state: Lens[State, IsReinsertionState],
    patch_fn: PatchFn[State, ParticlePositionChanges, Move],
    probability_fn: LogProbabilityRatioFn[State, Move],
) -> MCMCPropagator[State, ParticlePositionChanges, Move]:
    """Build an MCMC propagator for random group reinsertion moves."""
    move = ReinsertionMove(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.groups),
        state.focus(lambda x: x.systems),
        state.focus(lambda x: x.move_capacity),
    )
    return MCMCPropagator(
        patch_fn,
        (move,),
        probability_fn,
        (_sched(state.focus(lambda x: x.reinsertion_params)),),
    )


def make_displacement_mcmc_propagator[State, Move: Patch](
    state: Lens[State, IsDisplacementState],
    patch_fn: PatchFn[State, ParticlePositionChanges, Move],
    probability_fn: LogProbabilityRatioFn[State, Move],
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    reinsertion_weight: float = 1.0,
) -> MCMCPropagator[State, ParticlePositionChanges, Move]:
    """Build an MCMC propagator that randomly picks translation, rotation, or reinsertion.

    Only the scheduler corresponding to the selected move is updated each step.
    """
    propose_fns: list[ChangesFn[State, ParticlePositionChanges]] = []
    wts: list[float] = []
    schedulers: list[PropertyScheduler] = []

    if translation_weight > 0:
        propose_fns.append(
            GroupTranslationMove(
                state.focus(lambda x: x.particles),
                state.focus(lambda x: x.groups),
                state.focus(lambda x: x.systems),
                pipe(
                    state,
                    lambda s: Table(
                        s.translation_params.keys, s.translation_params.data.value
                    ),
                ),
                state.focus(lambda x: x.move_capacity),
            )
        )
        wts.append(translation_weight)
        schedulers.append(_sched(state.focus(lambda x: x.translation_params)))

    if rotation_weight > 0:
        propose_fns.append(
            GroupRotationMove(
                state.focus(lambda x: x.particles),
                state.focus(lambda x: x.groups),
                state.focus(lambda x: x.systems),
                pipe(
                    state,
                    lambda s: Table(
                        s.rotation_params.keys, s.rotation_params.data.value
                    ),
                ),
                state.focus(lambda x: x.move_capacity),
            )
        )
        wts.append(rotation_weight)
        schedulers.append(_sched(state.focus(lambda x: x.rotation_params)))

    if reinsertion_weight > 0:
        propose_fns.append(
            ReinsertionMove(
                state.focus(lambda x: x.particles),
                state.focus(lambda x: x.groups),
                state.focus(lambda x: x.systems),
                state.focus(lambda x: x.move_capacity),
            )
        )
        wts.append(reinsertion_weight)
        schedulers.append(_sched(state.focus(lambda x: x.reinsertion_params)))

    return MCMCPropagator(
        patch_fn,
        tuple(propose_fns),
        probability_fn,
        tuple(schedulers),
        weights=tuple(wts),
    )


def make_exchange_mcmc_propagator[State, Move: Patch](
    state: Lens[State, IsExchangeState],
    patch_fn: PatchFn[State, ExchangeChanges, Move],
    probability_fn: LogProbabilityRatioFn[State, Move],
) -> MCMCPropagator[State, ExchangeChanges, Move]:
    """Build an MCMC propagator for grand-canonical insertion/deletion moves."""
    move = ExchangeMove(
        positions=state.focus(lambda x: x.particles),
        groups=state.focus(lambda x: x.groups),
        motifs=state.focus(lambda x: x.motifs),
        unitcell=state.focus(lambda x: x.systems.map_data(lambda d: d.unitcell)),
        capacity=state.focus(lambda x: x.move_capacity),
    )
    return MCMCPropagator(
        patch_fn,
        (move,),
        probability_fn,
        (_sched(state.focus(lambda x: x.exchange_params)),),
    )


def make_gcmc_mcmc_propagator[State, Move: Patch](
    state: Lens[State, IsGCMCState],
    patch_fn: PatchFn[State, ExchangeChanges, Move],
    probability_fn: LogProbabilityRatioFn[State, Move],
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    reinsertion_weight: float = 1.0,
    exchange_weight: float = 3.0,
) -> Propagator[State]:
    """Build an MCMC propagator combining displacement and exchange moves.

    Randomly picks translation, rotation, reinsertion, or exchange (insert/delete)
    at each step. Displacement proposals are lifted to ``ExchangeChanges`` via
    ``exchange_changes_from_position_changes`` so all four branches share the
    same Changes type.

    Args:
        state: Lens into the sub-state satisfying ``IsGCMCState``.
        patch_fn: Converts exchange changes to a state patch.
        probability_fn: Log probability ratio for acceptance/rejection.
        translation_weight: Unnormalized selection weight for translation moves.
        rotation_weight: Unnormalized selection weight for rotation moves.
        reinsertion_weight: Unnormalized selection weight for reinsertion moves.
        exchange_weight: Unnormalized selection weight for exchange (insert/delete) moves.

    Returns:
        Configured MCMCPropagator with adaptive step-width scheduling for
        all four move types.
    """

    def _lift_to_exchange(
        propose_pos: ChangesFn[State, ParticlePositionChanges],
    ) -> ChangesFn[State, ExchangeChanges]:
        def wrapper(
            key: Array, s: State, /
        ) -> tuple[ExchangeChanges, LogProbabilityRatio]:
            inner = state(s)
            pos_changes, log_ratio = propose_pos(key, s)
            return exchange_changes_from_position_changes(
                pos_changes,
                inner.particles,
                inner.groups,
                inner.particles.data.system.num_labels,
            ), log_ratio

        return wrapper

    def _zero_ratio(s: State) -> LogProbabilityRatio:
        n = state(s).particles.data.system.num_labels
        return Table.arange(jnp.zeros((n,)), label=SystemId)

    def _symmetric[C](fn: Callable[[Array, State], C]) -> ChangesFn[State, C]:
        def wrapper(key: Array, s: State, /) -> tuple[C, LogProbabilityRatio]:
            return fn(key, s), _zero_ratio(s)

        return wrapper  # type: ignore[return-value]

    def _sched(params_lens: Lens) -> PropertyScheduler:
        return PropertyScheduler(
            params_lens, Table.transform(acceptance_target_schedule)
        )

    propose_fns: list[ChangesFn[State, ExchangeChanges]] = []
    wts: list[float] = []
    schedulers: list[PropertyScheduler] = []

    # Displacement moves (symmetric, lifted to ExchangeChanges)
    _moves: list[tuple[float, Callable, Lens]] = []
    if translation_weight > 0:
        _moves.append(
            (
                translation_weight,
                lambda key, s: propose_group_translation(
                    key,
                    (i := state(s)).particles,
                    i.groups,
                    i.systems,
                    Table(i.translation_params.keys, i.translation_params.data.value),
                    i.move_capacity,
                ),
                state.focus(lambda x: x.translation_params),
            )
        )
    if rotation_weight > 0:
        _moves.append(
            (
                rotation_weight,
                lambda key, s: propose_group_rotation(
                    key,
                    (i := state(s)).particles,
                    i.groups,
                    i.systems,
                    Table(i.rotation_params.keys, i.rotation_params.data.value),
                    i.move_capacity,
                ),
                state.focus(lambda x: x.rotation_params),
            )
        )
    if reinsertion_weight > 0:
        _moves.append(
            (
                reinsertion_weight,
                lambda key, s: propose_reinsertion(
                    key,
                    (i := state(s)).particles,
                    i.groups,
                    i.systems,
                    i.move_capacity,
                ),
                state.focus(lambda x: x.reinsertion_params),
            )
        )
    for w, fn, params_lens in _moves:
        propose_fns.append(_lift_to_exchange(_symmetric(fn)))
        wts.append(w)
        schedulers.append(_sched(params_lens))

    # Exchange move (insert/delete, also symmetric)
    if exchange_weight > 0:

        def _propose_exchange(key: Array, s: State) -> ExchangeChanges:
            inner = state(s)
            c = key_chain(key)
            ins = insert_random_motif(
                next(c),
                inner.motifs,
                inner.particles,
                inner.groups,
                inner.systems.map_data(lambda d: d.unitcell),
                inner.move_capacity,
            )
            deletion = delete_random_motif(
                next(c),
                inner.motifs,
                inner.particles,
                inner.groups,
                inner.move_capacity,
            )
            w = jax.random.randint(next(c), (), 0, 2)
            return tree_map(lambda a, b: jax.lax.select_n(w, a, b), ins, deletion)

        propose_fns.append(_symmetric(_propose_exchange))
        wts.append(exchange_weight)
        schedulers.append(_sched(state.focus(lambda x: x.exchange_params)))

    return MCMCPropagator(
        patch_fn,
        tuple(propose_fns),
        probability_fn,
        tuple(schedulers),
        weights=tuple(wts),
    )
