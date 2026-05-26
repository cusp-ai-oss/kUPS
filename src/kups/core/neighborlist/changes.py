# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Incremental neighbor list updates for Monte Carlo / patch-style moves.

[`neighborlist_changes`][kups.core.neighborlist.changes.neighborlist_changes]
runs a single neighbor list query that simultaneously discovers edges removed
by replacing a subset of particles and edges added at their new positions,
returning both as
[`NeighborListChangesResult`][kups.core.neighborlist.changes.NeighborListChangesResult].
"""

from __future__ import annotations

from functools import partial
from typing import Literal, NamedTuple

import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.data import Index, Table
from kups.core.data.wrappers import WithIndices
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import (
    NearestNeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import isin, jit
from kups.core.utils.ops import where_broadcast_last


class NeighborListChangesResult(NamedTuple):
    added: Edges[Literal[2]]
    removed: Edges[Literal[2]]


@partial(jit, static_argnames=("compaction",))
def neighborlist_changes(
    neighborlist: NearestNeighborList,
    lh: Table[ParticleId, NeighborListPoints],
    rh: WithIndices[ParticleId, Table[ParticleId, NeighborListPoints]],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    compaction: float = 0.5,
) -> NeighborListChangesResult:
    """Compute added/removed edges from a particle change in a single call.

    Appends proposed positions to the particle array and queries both old
    and new interactions at once, then splits the result by filtering
    edge indices into ``removed`` (before) and ``added`` (after) sets.

    Args:
        neighborlist: Neighbor list implementation.
        lh: Full original particle table.
        rh: Proposed changes — ``rh.indices`` maps entries to particle IDs
            in ``lh``, ``rh.data`` holds the new particle data.
        systems: Per-system data (cells, etc.).
        cutoffs: Per-system cutoff distances.
        compaction: Fraction of total edges allocated per output (0–1).
            0.5 means each of added/removed gets half the buffer.
            1.0 means no compaction — full buffer with masking only.

    Returns:
        ``NeighborListChangesResult(added, removed)``.
    """
    N, k = lh.size, rh.data.size
    p_idx = rh.indices.indices_in(lh.keys)

    # Build a single query with new particles on the left-hand side
    # (original particles + new particles) and both old and new particles
    # on the right-hand side (old positions at changed indices + new positions).
    lh_combined = Table.union((lh, rh.data))
    rh_combined = Table.union((Table.arange(lh[rh.indices], label=ParticleId), rh.data))
    combined_remap = Index(
        lh_combined.keys, jnp.concatenate([p_idx, jnp.arange(k) + N])
    )

    # single neighborlist call
    all_edges = neighborlist(lh_combined, rh_combined, systems, cutoffs, combined_remap)

    # split into removed / added
    raw = all_edges.indices.indices  # (n_edges, 2)
    c0, c1 = raw[:, 0], raw[:, 1]
    # Removed mask checks for edges that exist in the original set (both indices < N).
    removed_mask = (c0 < N) & (c1 < N)

    # is_stale mask checks that both edges need to be in the original set
    # or one needs to be in the original set and the other needs to be in the new set.
    is_stale = isin(c0, p_idx, N + k) & (c0 < N) | isin(c1, p_idx, N + k) & (c1 < N)
    # Added mask checks for edges that involve at least one new particle.
    added_mask = (c0 < N + k) & (c1 < N + k) & ((c0 >= N) | (c1 >= N)) & ~is_stale

    # remap appended indices N+m -> p_idx[m]
    remapped = jnp.where(raw >= N, p_idx[raw - N], raw)

    # compact each output
    n_total = raw.shape[0]
    shifts = all_edges.shifts

    def _mask_only(mask: Array, indices: Array, shifts: Array) -> Edges[Literal[2]]:
        idx = where_broadcast_last(mask, indices, N)
        sh = where_broadcast_last(mask, shifts, 0)
        return Edges(Index(lh.keys, idx), sh)

    def _compact(mask: Array, indices: Array, label: str) -> Edges[Literal[2]]:
        count = mask.sum()
        runtime_assert(
            count <= capacity,
            f"neighborlist_changes: {label} edges ({{count}}) exceed "
            f"capacity ({{capacity}})",
            fmt_args={"count": count, "capacity": jnp.array(capacity)},
        )
        sel: Array = jnp.where(mask, size=capacity, fill_value=n_total - 1)[0]
        valid = mask.at[sel].get(mode="fill", fill_value=False)
        return _mask_only(valid, indices[sel], shifts[sel])

    if compaction >= 1.0:
        return NeighborListChangesResult(
            _mask_only(added_mask, remapped, shifts),
            _mask_only(removed_mask, remapped, shifts),
        )

    capacity = int(n_total * compaction)
    return NeighborListChangesResult(
        _compact(added_mask, remapped, "added"),
        _compact(removed_mask, remapped, "removed"),
    )
