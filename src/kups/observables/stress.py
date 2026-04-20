# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Stress tensor calculations via virial theorem and lattice vector gradients."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.typing import (
    GroupId,
    HasGroupIndex,
    HasPositions,
    HasSystemIndex,
    HasUnitCell,
    IsState,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell
from kups.core.utils.math import triangular_3x3_matmul


@runtime_checkable
class IsVirialParticles(HasPositions, HasSystemIndex, Protocol):
    """Particles with position gradients ∂U/∂r."""

    @property
    def position_gradients(self) -> Array: ...


@runtime_checkable
class IsVirialSystems(HasUnitCell, Protocol):
    """Systems with unit cell gradients ∂U/∂h."""

    @property
    def unitcell_gradients(self) -> UnitCell: ...


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


def _stress_via_lattice_vector_gradients(
    lattice_vectors_grad: Array, volume: Array
) -> Array:
    """σ = -∂U/∂h / V."""
    return -lattice_vectors_grad / volume


def _stress_via_virial_theorem(
    position_gradients: Array,
    lattice_vector_gradients: Array,
    positions: Array,
    lattice_vectors: Array,
    system: Index[SystemId],
) -> Array:
    """σ = -1/V (Σ_i ∂U/∂r_i ⊗ r_i + h^T · ∂U/∂h)."""
    stress = -system.sum_over(position_gradients[:, None] * positions[..., None]).data
    stress -= triangular_3x3_matmul(
        lattice_vectors.mT, lattice_vector_gradients, lower=False
    )
    stress /= jnp.abs(jnp.linalg.det(lattice_vectors))
    return stress


def _molecular_stress_via_virial_theorem(
    position_gradients: Array,
    lattice_vector_gradients: Array,
    positions: Array,
    group: Index[GroupId],
    group_unitcells: UnitCell,
    system: Index[SystemId],
    system_lattice_vectors: Array,
    system_volume: Array,
) -> Array:
    """Molecular virial stress using center-of-mass positions (RASPA convention)."""
    num_groups = group.num_labels
    batched_unitcells = group_unitcells[group.indices]
    ref_idx = (
        jnp.zeros(num_groups, dtype=int)
        .at[group.indices]
        .set(jnp.arange(group.indices.shape[0]), mode="drop")
    )
    offsets = positions[ref_idx]
    rel = batched_unitcells.wrap(positions - offsets[group.indices])
    com = jax.ops.segment_sum(rel, group.indices, num_groups)
    counts = jnp.bincount(group.indices, length=num_groups)[:, None]
    com = com / jnp.maximum(counts, 1) + offsets
    com = group_unitcells.wrap(com)
    rel_pos = batched_unitcells.wrap(
        positions - com.at[group.indices].get(mode="fill", fill_value=0)
    )
    stress = -system.sum_over(
        position_gradients[:, None] * (positions - rel_pos)[..., None]
    ).data
    stress = 0.5 * (stress + stress.mT)
    stress -= system_lattice_vectors.mT @ lattice_vector_gradients
    stress /= system_volume[:1][:, None, None]
    return stress


def stress_via_lattice_vector_gradients(
    systems: Table[SystemId, IsVirialSystems],
) -> Table[SystemId, Array]:
    """Compute stress from energy gradients w.r.t. lattice vectors.

    Args:
        systems: Per-system unit cell and unit cell gradients.

    Returns:
        Stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    stress = _stress_via_lattice_vector_gradients(
        systems.data.unitcell_gradients.lattice_vectors,
        systems.data.unitcell.volume,
    )
    return Table(systems.keys, stress)


def stress_via_virial_theorem(
    particles: Table[ParticleId, IsVirialParticles],
    systems: Table[SystemId, IsVirialSystems],
) -> Table[SystemId, Array]:
    """Compute atomic-level virial stress tensor.

    Args:
        particles: Per-particle positions, system index, and position gradients.
        systems: Per-system unit cell and unit cell gradients.

    Returns:
        Stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    stress = _stress_via_virial_theorem(
        particles.data.position_gradients,
        systems.data.unitcell_gradients.lattice_vectors,
        particles.data.positions,
        systems.data.unitcell.lattice_vectors,
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
        systems: Per-system unit cell and unit cell gradients.

    Returns:
        Symmetrized stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    group_unitcells = systems[groups.data.system].unitcell
    stress = _molecular_stress_via_virial_theorem(
        particles.data.position_gradients,
        systems.data.unitcell_gradients.lattice_vectors,
        particles.data.positions,
        particles.data.group,
        group_unitcells,
        particles.data.system,
        systems.data.unitcell.lattice_vectors,
        systems.data.unitcell.volume,
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
