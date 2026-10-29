# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing particle positions and center of mass.

Functions in this module honor the cell's per-axis ``periodic`` mask via
``cell.wrap`` — periodic axes fold into the primary cell, non-periodic
axes pass through unchanged. The same code path covers fully-periodic
crystals, vacuum clusters, and slab / wire geometries.
"""

import jax.numpy as jnp

from kups.core import units
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.typing import (
    HasMasses,
    HasPositionsAndGroupIndex,
    HasWeights,
    ParticleId,
)
from kups.core.utils.jax import jit
from kups.core.utils.units import magnitudes


@jit
def center_of_mass[P: HasPositionsAndGroupIndex](
    particles: Table[ParticleId, P], cells: Cell[AnyPeriodicity]
) -> units.Length:
    """Compute center of mass for indexed particle groups.

    Calculates the center of mass for each group of particles defined by
    group index, honoring the cell's per-axis ``periodic`` mask. The
    computation ensures that wrapped particles are unwrapped relative to a
    reference particle in each group before averaging.

    Warning:
        On periodic axes, assumes each molecular structure is smaller than
        half the cell size. Structures spanning more than half the box may
        yield incorrect results.

    Args:
        particles: Indexed particles, optionally supporting `HasWeights` (or
            `HasMasses`) for mass weighting; unweighted otherwise.
        cells: Cell(s) carrying lattice geometry and per-axis periodicity.
            Must have one cell per group.

    Returns:
        Shape `(num_groups, 3)` containing center-of-mass positions for each group.

    Example:
        ```python
        # Compute COM for each molecule
        com = center_of_mass(molecules, cell)
        ```
    """
    group = particles.data.group
    num_groups = group.num_labels
    # TODO: This function assumes that the structure is less than half the size of the cell!
    assert cells.vectors.shape[0] == num_groups, (
        f"Cells must match the number of groups. Got {cells.vectors.shape[0]} for {num_groups} groups."
    )
    # Particles gather the materialized inverses instead of recomputing them.
    cells = cells.materialize()
    batched_cells = Table(group.keys, cells)[group]
    # Index the first particle of each group
    ref_idx = group.select_per_label(jnp.zeros(num_groups, dtype=int))
    positions = particles.data.positions
    # A weight only sets a particle's relative importance and divides out again,
    # so its magnitudes are enough and an unweighted table can pass plain ones.
    if isinstance(particles.data, HasWeights):
        weight = magnitudes(particles.data.weights)[:, None]
    elif isinstance(particles.data, HasMasses):
        weight = magnitudes(particles.data.masses)[:, None]
    else:
        weight = jnp.ones_like(magnitudes(positions)[:, 0])[:, None]
    offsets = positions[ref_idx]
    rel_positions = positions - Table(group.keys, offsets)[group]
    rel_positions = batched_cells.wrap(rel_positions)
    center_of_masses = group.sum_over(rel_positions * weight).data
    center_of_masses /= group.sum_over(weight).data
    center_of_masses += offsets
    center_of_masses = cells.wrap(center_of_masses)
    return center_of_masses


@jit
def to_relative_positions[P: HasPositionsAndGroupIndex](
    particles: Table[ParticleId, P],
    cells: Cell[AnyPeriodicity],
    center_of_masses: units.Length | None = None,
) -> units.Length:
    """Calculate particle positions relative to their group's center of mass.

    Transforms absolute particle positions to positions relative to each
    group's center of mass, honoring the cell's per-axis ``periodic`` mask
    (periodic axes wrap; non-periodic axes pass through unchanged).

    Args:
        particles: Indexed particles with position and group index data. Supports
            `HasWeights` or `HasMasses` if center of mass needs to be computed.
        cells: Cell(s) carrying lattice geometry and per-axis periodicity.
        center_of_masses: Optional precomputed centers of mass. If `None`,
            will be computed automatically.

    Returns:
        Shape `(N, 3)` containing positions relative to group COMs.

    Example:
        ```python
        rel_pos = to_relative_positions(molecules, cell)
        ```
    """
    group = particles.data.group
    cells = cells.materialize()
    if center_of_masses is None:
        center_of_masses = center_of_mass(particles, cells)
    rel_positions = (
        particles.data.positions - Table(group.keys, center_of_masses)[group]
    )
    return Table(group.keys, cells)[group].wrap(rel_positions)


@jit
def to_absolute_positions[P: HasPositionsAndGroupIndex](
    particles: Table[ParticleId, P],
    cells: Cell[AnyPeriodicity],
    center_of_masses: units.Length,
) -> units.Length:
    """Calculate absolute positions from relative positions and group COMs.

    Inverse operation of `to_relative_positions`. Converts positions defined
    relative to group centers of mass back to absolute coordinates, honoring
    the cell's per-axis ``periodic`` mask via ``wrap`` (identity on
    non-periodic axes).

    Args:
        particles: Indexed particles with relative positions and group index.
        cells: Cell(s) carrying lattice geometry and per-axis periodicity.
        center_of_masses: Centers of mass for each group, shape `(num_groups, 3)`.

    Returns:
        Shape `(N, 3)` containing absolute particle positions.

    Example:
        ```python
        abs_pos = to_absolute_positions(rel_molecules, cell, com)
        ```
    """
    group = particles.data.group
    cells = cells.materialize()
    abs_positions = (
        particles.data.positions + Table(group.keys, center_of_masses)[group]
    )
    return Table(group.keys, cells)[group].wrap(abs_positions)
