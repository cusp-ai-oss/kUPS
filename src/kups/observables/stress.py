# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Stress tensor calculations via the virial theorem.

Stress is the symmetric (3, 3) tensor

    σ = -1/V sym[Σ_i r_i ⊗ ∂U/∂r_i + h^T · ∂U/∂h]

Only the 6 lower-triangular entries of ``∂U/∂h`` are stored (the cell's
parameter DoF; see :class:`kups.core.cell.TriclinicFrame`). For
lower-triangular ``h``, the lower triangle of ``h^T · ∂U/∂h`` depends only
on the lower triangle of ``∂U/∂h``: since ``h[k, i] = 0`` for ``k < i``,
the sum ``(h^T·g)[i, j] = Σ_{k≥i} h[k, i]·g[k, j]`` for ``j ≤ i`` touches
only ``g[k, j]`` with ``k ≥ i ≥ j``, i.e. lower-triangular entries. The
upper triangle of ``σ`` is filled by symmetry — the full 3×3 cell virial
is never materialized.
"""

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


@runtime_checkable
class IsVirialParticles(HasPositions, HasSystemIndex, Protocol):
    """Particles with position gradients ∂U/∂r."""

    @property
    def position_gradients(self) -> Array: ...


@runtime_checkable
class IsVirialSystems(HasCell[Periodic3D], Protocol):
    """Systems with cell gradients ∂U/∂h (stored lower-triangular)."""

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


def _symmetrize_from_lower(lower: Array) -> Array:
    """Build a symmetric (..., 3, 3) tensor from its lower triangle.

    ``lower`` is assumed to have a zero (or arbitrary, ignored) upper
    triangle. Returns ``lower + lower^T − diag(lower)`` so the diagonal
    is not double-counted.
    """
    return lower + lower.mT - lower * jnp.eye(3)


def _lower_sym_cell_virial(vectors: Array, vector_gradients: Array) -> Array:
    """Lower triangle of ``S = h^T · ∂U/∂h`` from lower-triangular ``h`` and
    the lower-triangular projection of ``∂U/∂h``.

    For lower-triangular ``h``, the lower triangle of ``h^T · g`` depends
    only on the lower triangle of ``g`` (the upper-triangular entries of
    ``g`` are not parameters and are not stored). The upper triangle is not
    materialized; the final position-plus-cell virial is symmetrized later.
    """
    return jnp.tril(vectors.mT @ vector_gradients)


def _stress_via_virial_theorem(
    position_gradients: Array,
    vector_gradients: Array,
    positions: Array,
    vectors: Array,
    system: Index[SystemId],
) -> Array:
    """σ = −1/V sym[Σ_i r_i ⊗ ∂U/∂r_i + h^T · ∂U/∂h]."""
    pos_outer = system.sum_over(position_gradients[:, None] * positions[..., None]).data
    pos_lower = jnp.tril(pos_outer)
    cell_lower = _lower_sym_cell_virial(vectors, vector_gradients)
    volume = jnp.abs(jnp.linalg.det(vectors))[..., None, None]
    sigma_lower = -(pos_lower + cell_lower) / volume
    return _symmetrize_from_lower(sigma_lower)


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
    pos_outer = system.sum_over(
        position_gradients[:, None] * (positions - rel_pos)[..., None]
    ).data
    pos_lower = jnp.tril(pos_outer)
    cell_lower = _lower_sym_cell_virial(system_vectors, vector_gradients)
    volume = system_volume[..., None, None]
    sigma_lower = -(pos_lower + cell_lower) / volume
    return _symmetrize_from_lower(sigma_lower)


def stress_via_virial_theorem(
    particles: Table[ParticleId, IsVirialParticles],
    systems: Table[SystemId, IsVirialSystems],
) -> Table[SystemId, Array]:
    """Compute atomic-level virial stress tensor.

    Args:
        particles: Per-particle positions, system index, and position gradients.
        systems: Per-system cell and cell gradients (lower-triangular).

    Returns:
        Symmetric stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    cell = systems.data.cell
    stress = _stress_via_virial_theorem(
        particles.data.position_gradients,
        cell.frame.vectors_gradient(systems.data.cell_gradients.frame),
        particles.data.positions,
        cell.vectors,
        particles.data.system,
    )
    return Table(systems.keys, stress)


def molecular_stress_via_virial_theorem(
    particles: Table[ParticleId, IsMolecularVirialParticles],
    groups: Table[GroupId, HasSystemIndex],
    systems: Table[SystemId, IsVirialSystems],
) -> Table[SystemId, Array]:
    """Compute molecular virial stress tensor (RASPA convention).

    Args:
        particles: Per-particle positions, group/system index, and gradients.
        groups: Per-group system assignment.
        systems: Per-system cell and cell gradients (lower-triangular).

    Returns:
        Symmetric stress tensor per system, shape ``(n_systems, 3, 3)``.
    """
    group_cells = systems[groups.data.system].cell
    cell = systems.data.cell
    stress = _molecular_stress_via_virial_theorem(
        particles.data.position_gradients,
        cell.frame.vectors_gradient(systems.data.cell_gradients.frame),
        particles.data.positions,
        particles.data.group,
        group_cells,
        particles.data.system,
        cell.vectors,
        cell.volume,
    )
    return Table(systems.keys, stress)


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
