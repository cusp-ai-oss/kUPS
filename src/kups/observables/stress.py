# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Stress tensor calculations via virial theorem and lattice vector gradients."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import Cell, Periodic3D
from kups.core.data import Index, Table
from kups.core.typing import (
    GroupId,
    HasCell,
    HasGroupIndex,
    HasPositions,
    HasSystemIndex,
    IsState,
    ParticleId,
    SystemId,
)
from kups.core.utils.math import triangular_3x3_matmul


@runtime_checkable
class IsVirialParticles(HasPositions, HasSystemIndex, Protocol):
    """Particles with position gradients ∂U/∂r."""

    @property
    def position_gradients(self) -> Array: ...


@runtime_checkable
class IsVirialSystems(HasCell[Periodic3D], Protocol):
    """Systems with cell gradients ∂U/∂h."""

    @property
    def cell_gradients(self) -> Cell[Periodic3D]: ...


@runtime_checkable
class IsMolecularVirialParticles(HasPositions, HasGroupIndex, HasSystemIndex, Protocol):
    """Particles with position gradients, group and system assignment."""

    @property
    def position_gradients(self) -> Array: ...


class IsMolecularVirialState(
    IsState[IsMolecularVirialParticles, IsVirialSystems], Protocol
):
    """State with groups for molecular virial stress."""

    @property
    def groups(self) -> Table[GroupId, HasSystemIndex]: ...


def _stress_via_lattice_vector_gradients(vectors_grad: Array, volume: Array) -> Array:
    """σ = -∂U/∂h / V."""
    return -vectors_grad / volume


def _stress_via_virial_theorem(
    position_gradients: Array,
    vector_gradients: Array,
    positions: Array,
    vectors: Array,
    system: Index[SystemId],
) -> Array:
    """σ = -1/V (Σ_i ∂U/∂r_i ⊗ r_i + h^T · ∂U/∂h)."""
    stress = -system.sum_over(position_gradients[:, None] * positions[..., None]).data
    stress -= triangular_3x3_matmul(vectors.mT, vector_gradients, lower=False)
    stress /= jnp.abs(jnp.linalg.det(vectors))
    return stress


def _molecular_stress_via_virial_theorem(
    position_gradients: Array,
    vector_gradients: Array,
    positions: Array,
    group: Index[GroupId],
    group_cells: Cell[Periodic3D],
    system: Index[SystemId],
    system_vectors: Array,
    system_volume: Array,
) -> Array:
    """Molecular virial stress using center-of-mass positions (RASPA convention)."""
    num_groups = group.num_labels
    batched_cells = group_cells[group.indices]
    ref_idx = (
        jnp.zeros(num_groups, dtype=int)
        .at[group.indices]
        .set(jnp.arange(group.indices.shape[0]), mode="drop")
    )
    offsets = positions[ref_idx]
    rel = batched_cells.wrap(positions - offsets[group.indices])
    com = jax.ops.segment_sum(rel, group.indices, num_groups)
    counts = jnp.bincount(group.indices, length=num_groups)[:, None]
    com = com / jnp.maximum(counts, 1) + offsets
    com = group_cells.wrap(com)
    rel_pos = batched_cells.wrap(
        positions - com.at[group.indices].get(mode="fill", fill_value=0)
    )
    stress = -system.sum_over(
        position_gradients[:, None] * (positions - rel_pos)[..., None]
    ).data
    stress = 0.5 * (stress + stress.mT)
    stress -= system_vectors.mT @ vector_gradients
    stress /= system_volume[:1][:, None, None]
    return stress


def stress_via_lattice_vector_gradients(
    systems: Table[SystemId, IsVirialSystems],
) -> Table[SystemId, Array]:
    """Compute stress from energy gradients w.r.t. lattice vectors.

    Args:
        systems: Per-system cell and cell gradients.

    Returns:
        Stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    stress = _stress_via_lattice_vector_gradients(
        systems.data.cell_gradients.vectors,
        systems.data.cell.volume,
    )
    return Table(systems.keys, stress)


def stress_via_virial_theorem(
    particles: Table[ParticleId, IsVirialParticles],
    systems: Table[SystemId, IsVirialSystems],
) -> Table[SystemId, Array]:
    """Compute atomic-level virial stress tensor.

    Args:
        particles: Per-particle positions, system index, and position gradients.
        systems: Per-system cell and cell gradients.

    Returns:
        Stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    stress = _stress_via_virial_theorem(
        particles.data.position_gradients,
        systems.data.cell_gradients.vectors,
        particles.data.positions,
        systems.data.cell.vectors,
        particles.data.system,
    )
    return Table(systems.keys, stress)


def molecular_stress_via_virial_theorem(
    particles: Table[ParticleId, IsMolecularVirialParticles],
    groups: Table[GroupId, HasSystemIndex],
    systems: Table[SystemId, IsVirialSystems],
) -> Table[SystemId, Array]:
    """Compute molecular virial stress tensor (RASPA convention).

    The stress tensor is symmetrized: σ = (σ + σᵀ)/2.

    Args:
        particles: Per-particle positions, group/system index, and gradients.
        groups: Per-group system assignment.
        systems: Per-system cell and cell gradients.

    Returns:
        Symmetrized stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    group_cells = systems[groups.data.system].cell
    stress = _molecular_stress_via_virial_theorem(
        particles.data.position_gradients,
        systems.data.cell_gradients.vectors,
        particles.data.positions,
        particles.data.group,
        group_cells,
        particles.data.system,
        systems.data.cell.vectors,
        systems.data.cell.volume,
    )
    return Table(systems.keys, stress)


def lattice_vector_stress_from_state(
    key: Array, state: IsState[IsVirialParticles, IsVirialSystems]
) -> Table[SystemId, Array]:
    """Compute stress from lattice vector gradients from a state."""
    del key
    return stress_via_lattice_vector_gradients(state.systems)


def virial_stress_from_state(
    key: Array, state: IsState[IsVirialParticles, IsVirialSystems]
) -> Table[SystemId, Array]:
    """Compute atomic virial stress from a state."""
    del key
    return stress_via_virial_theorem(state.particles, state.systems)


def molecular_virial_stress_from_state(
    key: Array, state: IsMolecularVirialState
) -> Table[SystemId, Array]:
    """Compute molecular virial stress from a state."""
    del key
    return molecular_stress_via_virial_theorem(
        state.particles, state.groups, state.systems
    )
