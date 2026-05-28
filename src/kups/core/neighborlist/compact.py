# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Compactors for the neighbor list pipeline.

Compactor variants:

- [`ReduceCompactor`][kups.core.neighborlist.compact.ReduceCompactor] —
  compresses survivors via ``jnp.where(keep, size=k)`` with a capacity
  assertion. By default it applies pair rh→lh remapping and mirrors remapped
  pairs; fixed-topology callers disable those pair-only options and reuse the
  same row compaction.
- [`MaskOnlyCompactor`][kups.core.neighborlist.compact.MaskOnlyCompactor] —
  preserves the candidate count, replacing failing entries with OOB indices
  and zero shifts. Used by ``RefineMaskNeighborList``.

The pair rh→lh remap (``remap_rh_to_lh``) lets final pair ``Edges`` indices
live in lh-space regardless of which pair compactor ran. Mirroring (doubling
each edge with its reverse) is specific to the default ``ReduceCompactor``
configuration — it pairs with ``RemapDedupMask`` which removed one direction
of each pair upstream.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity
from kups.core.data import Index
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import CandidateBatch, Compactor, PipelineContext
from kups.core.utils.jax import dataclass, field
from kups.core.utils.ops import where_broadcast_last


def remap_rh_to_lh(rh_idx: Array, ctx: PipelineContext) -> Array:
    """Map rh-space indices to lh-space via ``ctx.rh_index_remap``.

    Returns ``rh_idx`` unchanged when no remap is in effect. Out-of-bounds
    rh positions (e.g., padding) resolve to ``max(ctx.lh.size, ctx.rh.size)``.
    """
    if ctx.rh_index_remap is None:
        return rh_idx
    oob = max(ctx.lh.size, ctx.rh.size)
    return ctx.rh_index_remap.at[rh_idx].get(mode="fill", fill_value=oob)


@dataclass
class ReduceCompactor[D: int](Compactor[D]):
    """Compact surviving candidates to a size-bounded ``Edges[D]``.

    The default configuration is the existing pair-neighbor-list behavior:
    remap the rhs column into lh-space and mirror each remapped pair. Fixed
    topology reuses the same row compaction with both pair-only options
    disabled, preserving arbitrary-degree edge columns as emitted.
    """

    avg_edges: Capacity[int]
    remap_rh: bool = field(static=True, default=True)
    mirror_on_remap: bool = field(static=True, default=True)

    def __call__(
        self,
        keep: Array,
        batch: CandidateBatch[D],
        ctx: PipelineContext,
    ) -> Edges[D]:
        oob = max(ctx.lh.size, ctx.rh.size)
        max_edges = self.avg_edges.generate_assertion(keep.sum())
        sort_idxs = jnp.where(keep, size=max_edges.size, fill_value=keep.size)[0]
        shifts = batch.edges.shifts.at[sort_idxs].get(
            mode="fill", fill_value=0, indices_are_sorted=True
        )

        if self.remap_rh:
            rh_idx_remapped = remap_rh_to_lh(batch.rh_idx, ctx)
            lh_edge = batch.lh_idx.at[sort_idxs].get(
                mode="fill", fill_value=oob, indices_are_sorted=True
            )
            rh_edge = rh_idx_remapped.at[sort_idxs].get(
                mode="fill", fill_value=oob, indices_are_sorted=True
            )
            indices = jnp.stack([lh_edge, rh_edge], axis=-1)

            if self.mirror_on_remap and ctx.rh_index_remap is not None:
                shifts = jnp.concatenate([shifts, -shifts], axis=0)
                indices = jnp.concatenate([indices, indices[:, ::-1]], axis=0)
        else:
            indices = batch.edges.indices.indices.at[sort_idxs].get(
                mode="fill", fill_value=oob, indices_are_sorted=True
            )

        return Edges(Index(batch.edges.indices.keys, indices), shifts)


@dataclass
class MaskOnlyCompactor[D: int](Compactor[D]):
    """In-place compaction: failing entries become OOB indices and zero shifts.

    No size change; preserves the candidate count from the selector. Applies
    the shared rh→lh remap so the output indices live in lh-space — matching
    ``ReduceCompactor``'s contract.
    """

    def __call__(
        self,
        keep: Array,
        batch: CandidateBatch[D],
        ctx: PipelineContext,
    ) -> Edges[D]:
        oob = ctx.lh.size
        rh_idx_remapped = remap_rh_to_lh(batch.rh_idx, ctx)
        indices_in = jnp.stack([batch.lh_idx, rh_idx_remapped], axis=-1)
        indices = where_broadcast_last(keep, indices_in, oob)
        shifts = where_broadcast_last(keep, batch.edges.shifts, 0)
        return Edges(Index(batch.edges.indices.keys, indices), shifts)
