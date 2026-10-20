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

from typing import Literal, NamedTuple

import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.data import Index, Table
from kups.core.data.wrappers import WithIndices
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import (
    NeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import jit
from kups.core.utils.ops import first_true_indices, where_broadcast_last


class NeighborListChangesResult(NamedTuple):
    added: Edges[Literal[2]]
    removed: Edges[Literal[2]]


@jit(static_argnames=("compaction",))
def neighborlist_changes(
    neighborlist: NeighborList[Literal[2]],
    lh: Table[ParticleId, NeighborListPoints],
    rh: WithIndices[ParticleId, Table[ParticleId, NeighborListPoints]],
    systems: Table[SystemId, NeighborListSystems],
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
        compaction: Fraction of total edges allocated per output (0–1).
            0.5 means each of added/removed gets half the buffer.
            1.0 means no compaction — full buffer with masking only.

    Returns:
        ``NeighborListChangesResult(added, removed)``.
    """
    # Build a single self-graph query with both old and proposed particles in
    # the key table. ``queried_keys`` enumerates edges incident to the changed
    # old slots and the proposed rows. Concatenating in the shifted key space
    # maps out-of-bounds (no-op) old slots to the combined OOB sentinel rather
    # than ``len(lh)``, so an empty per-system change cannot alias the first
    # appended particle and duplicate its edges.
    keys_combined = Table.union((lh, rh.data))
    queried_keys = Index.concatenate(rh.indices, rh.data.index, shift_keys=True)

    # single neighborlist call
    all_edges = neighborlist(keys_combined, systems, queried_keys=queried_keys)

    # Each combined row stands for an ``lh`` particle: itself, or the one it replaces.
    targets = Index.concatenate(lh.index, rh.indices.update_labels(lh.keys))
    remapped = Table(keys_combined.keys, targets)[all_edges.indices]
    old = all_edges.indices.update_labels(lh.keys, allow_missing=True)
    # Removed edges exist in the original set (both endpoints are old rows).
    removed_mask = old.valid_mask.all(-1)
    # Stale edges touch a replaced old row; added edges touch a proposed row.
    is_stale = old.isin(rh.indices).any(-1)
    added_mask = all_edges.indices.valid_mask.all(-1) & ~removed_mask & ~is_stale

    # compact each output
    n_total = len(all_edges)
    shifts = all_edges.shifts

    def _mask_only(
        mask: Array, indices: Index[ParticleId], shifts: Array
    ) -> Edges[Literal[2]]:
        return Edges(
            indices.apply_mask(mask[:, None]), where_broadcast_last(mask, shifts, 0)
        )

    def _compact(mask: Array, label: str) -> Edges[Literal[2]]:
        dtype = remapped.raw_indices.dtype
        count = mask.sum(dtype=dtype)
        runtime_assert(
            count <= capacity,
            f"neighborlist_changes: {label} edges ({{count}}) exceed "
            f"capacity ({{capacity}})",
            fmt_args={"count": count, "capacity": jnp.array(capacity)},
        )
        sel = first_true_indices(mask, capacity, n_total - 1, dtype)
        valid = mask.at[sel].get(mode="fill", fill_value=False)
        return _mask_only(valid, remapped[sel], shifts[sel])

    if compaction >= 1.0:
        return NeighborListChangesResult(
            _mask_only(added_mask, remapped, shifts),
            _mask_only(removed_mask, remapped, shifts),
        )

    capacity = int(n_total * compaction)
    return NeighborListChangesResult(
        _compact(added_mask, "added"), _compact(removed_mask, "removed")
    )
