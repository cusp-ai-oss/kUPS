# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Stress tensor calculations via the virial theorem.

Stress is the symmetric (3, 3) tensor

    σ = -1/V sym[Σ_i r_i ⊗ ∂U/∂r_i + h^T · ∂U/∂h]

Only the lower-triangular entries of ``∂U/∂h`` are stored -- the parameter
degrees of freedom of a lower-triangular cell ``h`` -- so the full 3×3 cell
virial is never materialized; the upper triangle of ``σ`` is filled by symmetry.

Per-axis periodicity is honoured: components touching a non-periodic
(vacuum/bounding-box) axis are zeroed, so an isolated cluster has zero stress
and a slab keeps only its in-plane block. The volume divisor is the full cell
volume ``|det h|`` (LAMMPS convention: a slab's stress is diluted by its vacuum
extent).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Index, Table
from kups.core.lens import bind
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
from kups.core.utils.jax import tree_map


@runtime_checkable
class IsVirialParticles(HasPositions, HasSystemIndex, Protocol):
    """Particles with position gradients ∂U/∂r."""

    @property
    def position_gradients(self) -> Array: ...


@runtime_checkable
class IsVirialSystems(HasCell[AnyPeriodicity], Protocol):
    """Systems with a cell (any periodicity) and cell gradients ∂U/∂h
    (stored lower-triangular)."""

    @property
    def cell_gradients(self) -> Cell[AnyPeriodicity]: ...


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


def _periodic_mask(cell: Cell[AnyPeriodicity]) -> Array:
    """Outer product of the per-axis periodicity flags, shape ``(3, 3)``.

    ``σ_ab`` is conjugate to a strain deforming axis ``a`` along axis ``b``,
    a lattice strain only when both axes are periodic; the whole row and
    column of a non-periodic axis are zeroed. The mask is symmetric, so it
    preserves the symmetry of ``σ``.
    """
    mask = jnp.array(cell.periodic)
    return mask[:, None] * mask[None, :]


def _stress_via_virial_theorem(
    position_gradients: Array,
    vector_gradients: Array,
    positions: Array,
    cell: Cell[AnyPeriodicity],
    system: Index[SystemId],
) -> Array:
    """σ = −1/V sym[Σ_i r_i ⊗ ∂U/∂r_i + h^T · ∂U/∂h], zeroed on non-periodic axes."""
    pos_outer = system.sum_over(position_gradients[:, None] * positions[..., None]).data
    pos_lower = jnp.tril(pos_outer)
    cell_lower = _lower_sym_cell_virial(cell.vectors, vector_gradients)
    volume = cell.volume[..., None, None]
    sigma = _symmetrize_from_lower(-(pos_lower + cell_lower) / volume)
    return sigma * _periodic_mask(cell)


def _molecular_stress_via_virial_theorem(
    position_gradients: Array,
    vector_gradients: Array,
    positions: Array,
    group: Index[GroupId],
    group_cells: Cell[AnyPeriodicity],
    system: Index[SystemId],
    system_cell: Cell[AnyPeriodicity],
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
    cell_lower = _lower_sym_cell_virial(system_cell.vectors, vector_gradients)
    volume = system_cell.volume[..., None, None]
    sigma = _symmetrize_from_lower(-(pos_lower + cell_lower) / volume)
    return sigma * _periodic_mask(system_cell)


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
        cell,
        particles.data.system,
    )
    return Table(systems.keys, stress)


def total_lattice_gradient[C: Cell[AnyPeriodicity]](
    positions: Array,
    position_gradients: Array,
    cell: Table[SystemId, C],
    partial_lattice_gradient: Table[SystemId, C],
    system: Index[SystemId],
) -> Table[SystemId, C]:
    """Total lattice gradient ``∂E/∂h|_r + h⁻ᵀ·Σ_i r_i ⊗ ∂E/∂r_i``, in frame parameters.

    A potential reports the *partial* gradient ``∂E/∂h|_r``, taken at fixed
    Cartesian positions. Variable-cell relaxation needs the *total* derivative,
    with atoms riding the cell at fixed fractional coordinates; this adds the
    position-virial term ``h⁻ᵀ·Σ_i r_i ⊗ ∂E/∂r_i``. Stress is unchanged either way.

    Non-periodic axes carry no atoms -- a slab/vacuum basis vector is a
    bounding-box edge, not a translation -- so their coupling rows are dropped via
    the cell's [`periodic`][kups.core.cell.Cell] mask.

    Args:
        positions: Real-space positions ``r_i``, shape ``(n, 3)``.
        position_gradients: ``∂E/∂r_i``, shape ``(n, 3)``.
        cell: Per-system cells ``h`` (supply ``h⁻¹``, the Jacobian, and per-axis
            periodicity); fixes the returned cell type.
        partial_lattice_gradient: ``∂E/∂h|_r`` as per-system gradient cells.
        system: Per-particle system index, replicating ``cell`` to particles and
            summing the position virial per system.

    Returns:
        Total lattice gradient as ``Table[SystemId, C]`` of ``cell``'s type.
    """

    @Table.transform
    def compose_gradient(cell: C, coupling: Array, partial: C) -> C:
        # A non-periodic basis vector is a bounding-box edge, not a translation,
        # so its row carries no atoms -- drop the coupling there.
        coupling = coupling * jnp.array(cell.periodic)[:, None]
        coupling_gradient = cell.frame.parameter_gradient(coupling)
        return bind(partial, lambda c: c.frame).apply(
            lambda f: tree_map(jnp.add, f, coupling_gradient)
        )

    outer = positions[:, :, None] * position_gradients[:, None, :]  # r_i ⊗ ∂E/∂r_i
    coupling = system.sum_over(cell[system].inverse_vectors.mT @ outer)
    return compose_gradient(cell, coupling, partial_lattice_gradient)


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
        cell,
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
