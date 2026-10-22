# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Symmetric pair lists as one row of neighbors per particle.

[`neighbor_rows`][kups.core.neighborlist.rows.neighbor_rows] groups a build into
[`NeighborRows`][kups.core.neighborlist.rows.NeighborRows] for kernels that run one
program per particle.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.data import Table
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import NeighborListPoints, NeighborListSystems
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit
from kups.core.utils.subselect import argsort_indices

# Z-order bits per cell vector: 1024 bins, finer than an atom spacing in cells below
# about 1000 Å; the 30-bit code fits int32.
ORDER_BITS = 10


@dataclass
class NeighborRows:
    """Pairs grouped by particle, particles in spatial order.

    Row ``k`` belongs to particle ``order[k]`` and holds slots
    ``offsets[k]:offsets[k + 1]``; slots after ``offsets[-1]`` are padding with
    center and neighbor ``n``. Centers and neighbors are named by their row, so
    kernels gather from arrays permuted by ``order``. Shifts are int8: a build fails
    an assertion if a cell is narrower than about cutoff / 127, or if a pair's
    particles lie more than 127 cells apart (unwrapped input).

    Attributes:
        order: Particle of each row, ``(n,)`` int32.
        rank: Row of each particle, ``(n,)`` int32.
        offsets: First slot of each row, then the end, ``(n + 1,)`` int32.
        centers: Row of each slot, ``(n_slots,)`` int32.
        neighbors: Row of each slot's neighbor, ``(n_slots,)`` int32.
        shifts: Image shift of each slot from its center to its neighbor, in cell
            vectors, ``(n_slots, 3)`` int8.
    """

    order: Array
    rank: Array
    offsets: Array
    centers: Array
    neighbors: Array
    shifts: Array


@jit
def neighbor_rows(
    edges: Edges[Literal[2]],
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> NeighborRows:
    """Rows of a symmetric pair list, particles in spatial order.

    Args:
        edges: Symmetric pairs over ``particles`` with integer image shifts.
        particles: Particles at the build.
        systems: Systems with their cells.

    Returns:
        The rows.
    """
    n = particles.size
    order = _z_order(particles, systems)
    rank = jnp.zeros_like(order).at[order].set(jnp.arange(n, dtype=jnp.int32))
    pairs = edges.indices.indices_in(particles.keys)  # (E, 2), padding n
    neighbor, center = pairs[:, 0], pairs[:, 1]
    shifts = jnp.round(edges.shifts[:, 0])
    runtime_assert(
        jnp.all(jnp.abs(shifts) <= 127),
        "Image shifts exceed int8: a cell narrower than cutoff / 127, or particles "
        "more than 127 cells apart (unwrapped input).",
    )
    # stable, so each row keeps the build's order; padding goes last
    key = rank.at[center].get(mode="fill", fill_value=n)
    slots = argsort_indices(key)
    counts = edges.indices[:, 1].counts.data[order].astype(jnp.int32)
    offsets = jnp.cumulative_sum(counts, include_initial=True, dtype=jnp.int32)
    return NeighborRows(
        order,
        rank,
        offsets,
        key[slots],
        rank.at[neighbor[slots]].get(mode="fill", fill_value=n),
        # edge (j, i, s) is r = x_i - x_j + s h, so i reaches j at shift -s
        (-shifts[slots]).astype(jnp.int8),
    )


def _z_order(
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> Array:
    """Particles by system, then along a Z-order curve of their folded fractional
    coordinates."""
    system = particles.data.system
    frames = systems.map_data(lambda s: s.cell.frame.materialize())
    cells = systems.map_data(lambda s: s.cell)
    frac, _ = cells[system].fold(frames[system].to_fractional(particles.data.positions))
    q = jnp.clip(jnp.floor(frac * 2**ORDER_BITS), 0, 2**ORDER_BITS - 1)
    q = q.astype(jnp.int32)
    code = _spread(q[:, 0]) | (_spread(q[:, 1]) << 1) | (_spread(q[:, 2]) << 2)
    rows = jnp.arange(particles.size, dtype=jnp.int32)
    ids = system.indices_in(systems.keys).astype(jnp.int32)
    return jax.lax.sort((ids, code, rows), num_keys=3)[2]


def _spread(v: Array) -> Array:
    """Move bit ``b`` of a 10-bit ``v`` to bit ``3 b``."""
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    return (v | (v << 2)) & 0x09249249
