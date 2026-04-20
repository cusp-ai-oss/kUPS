# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing particle positions and center of mass in periodic systems.

This module provides functions for handling particle positions under periodic
boundary conditions, including center-of-mass calculations for indexed particle
groups.
"""

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.typing import HasPositionsAndGroupIndex, HasWeights, ParticleId
from kups.core.unitcell import UnitCell
from kups.core.utils.jax import jit


@jit
def center_of_mass[P: HasPositionsAndGroupIndex](
    particles: Table[ParticleId, P], unitcells: UnitCell
) -> Array:
    """Compute center of mass for indexed particle groups.

    Calculates the center of mass for each group of particles defined by
    group index, properly handling periodic boundary conditions. The computation
    ensures that wrapped particles are unwrapped relative to a reference particle
    in each group before averaging.

    Warning:
        Assumes each molecular structure is smaller than half the unit cell size.
        Molecules spanning more than half the box may yield incorrect results.

    Args:
        particles: Indexed particles, optionally supporting `HasWeights` for mass weighting.
        unitcells: Unit cell(s) defining periodic boundary conditions. Must have
            one unit cell per group.

    Returns:
        Shape `(num_groups, 3)` containing center-of-mass positions for each group.

    Example:
        ```python
        # Compute COM for each molecule
        com = center_of_mass(molecules, unit_cell)
        ```
    """
    group_ids = particles.data.group.indices
    num_groups = particles.data.group.num_labels
    # TODO: This function assumes that the structure is less than half the size of the unit cell!
    assert unitcells.lattice_vectors.shape[0] == num_groups, (
        f"Unit cells must match the number of groups. Got {unitcells.lattice_vectors.shape[0]} for {num_groups} groups."
    )
    batched_unitcells = unitcells[group_ids]
    # Index any particle in each group
    ref_idx = (
        jnp.zeros((num_groups,), dtype=int)
        .at[group_ids]
        .set(jnp.arange(len(group_ids)), mode="drop")
    )
    positions = particles.data.positions
    if isinstance(particles.data, HasWeights):
        weight = particles.data.weights[:, None]
    else:
        weight = jnp.ones_like(positions[:, 0])[:, None]
    offsets = positions[ref_idx]
    rel_positions = positions - offsets[group_ids]
    rel_positions = batched_unitcells.wrap(rel_positions)
    center_of_masses = jax.ops.segment_sum(
        rel_positions * weight,
        group_ids,
        num_groups,
    )
    total_mass = jax.ops.segment_sum(
        weight,
        group_ids,
        num_groups,
    )
    center_of_masses /= total_mass
    center_of_masses += offsets
    center_of_masses = unitcells.wrap(center_of_masses)
    return center_of_masses


@jit
def to_relative_positions[P: HasPositionsAndGroupIndex](
    particles: Table[ParticleId, P],
    unitcells: UnitCell,
    center_of_masses: Array | None = None,
) -> Array:
    """Calculate particle positions relative to their group's center of mass.

    Transforms absolute particle positions to positions relative to each group's
    center of mass, properly accounting for periodic boundary conditions.

    Args:
        particles: Indexed particles with position and group index data. Supports
            `HasWeights` if center of mass needs to be computed.
        unitcells: Unit cell(s) defining periodic boundaries.
        center_of_masses: Optional precomputed centers of mass. If `None`,
            will be computed automatically.

    Returns:
        Shape `(N, 3)` containing positions relative to group COMs.

    Example:
        ```python
        rel_pos = to_relative_positions(molecules, unit_cell)
        ```
    """
    group_ids = particles.data.group.indices
    if center_of_masses is None:
        center_of_masses = center_of_mass(particles, unitcells)
    positions = particles.data.positions
    rel_positions = positions - center_of_masses.at[group_ids].get(
        mode="fill", fill_value=0
    )
    rel_positions = unitcells[group_ids].wrap(rel_positions)
    return rel_positions


@jit
def to_absolute_positions[P: HasPositionsAndGroupIndex](
    particles: Table[ParticleId, P],
    unitcells: UnitCell,
    center_of_masses: Array,
) -> Array:
    """Calculate absolute positions from relative positions and group COMs.

    Inverse operation of `to_relative_positions`. Converts positions defined
    relative to group centers of mass back to absolute coordinates, applying
    periodic boundary conditions.

    Args:
        particles: Indexed particles with relative positions and group index.
        unitcells: Unit cell(s) defining periodic boundaries.
        center_of_masses: Centers of mass for each group, shape `(num_groups, 3)`.

    Returns:
        Shape `(N, 3)` containing absolute particle positions.

    Example:
        ```python
        abs_pos = to_absolute_positions(rel_molecules, unit_cell, com)
        ```
    """
    group_ids = particles.data.group.indices
    positions = particles.data.positions
    abs_positions = positions + center_of_masses.at[group_ids].get(
        mode="fill", fill_value=0
    )
    abs_positions = unitcells[group_ids].wrap(abs_positions)
    return abs_positions
