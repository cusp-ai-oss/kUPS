# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Compactors for the neighbor list pipeline.

Compactor variants:

- [`ReduceCompactor`][kups.core.neighborlist.compact.ReduceCompactor] —
  compresses surviving candidate rows via ``jnp.where(keep, size=k)`` with a
  capacity assertion, preserving the emitted edge degree.
- [`MaskOnlyCompactor`][kups.core.neighborlist.compact.MaskOnlyCompactor] —
  preserves the candidate count, replacing failing entries with OOB indices
  and zero shifts. Used by ``RefineMaskNeighborList``.

Graph-level output shaping, such as restoring undirected symmetry after
``queried_keys`` deduplication, lives in pipeline postprocessors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import CandidateBatch, Compactor, PipelineContext
from kups.core.utils.jax import dataclass
from kups.core.utils.ops import where_broadcast_last


@dataclass
class ReduceCompactor[D: int](Compactor[D]):
    """Compact surviving candidates to a size-bounded ``Edges[D]``.

    Compacts whole candidate rows, so pair neighbor lists and fixed higher-degree
    topology share the same implementation.
    """

    avg_edges: Capacity[int]

    def __call__(
        self,
        keep: Array,
        batch: CandidateBatch[D],
        ctx: PipelineContext,
    ) -> Edges[D]:
        return self.compact_batch(keep, batch, ctx).edges

    def compact_batch(
        self,
        keep: Array,
        batch: CandidateBatch[D],
        ctx: PipelineContext,
    ) -> CandidateBatch[D]:
        """Compact edges together with the image flags needed by pair kernels."""
        max_edges = self.avg_edges.generate_assertion(keep.sum())
        sort_idxs = jnp.where(keep, size=max_edges.size, fill_value=keep.size)[0]

        def gather(values: Array) -> Array:
            if keep.size == 0:
                # JAX cannot gather a padded output from an empty source array.
                return jnp.zeros((max_edges.size, *values.shape[1:]), values.dtype)
            return values.at[sort_idxs].get(
                mode="fill", fill_value=0, indices_are_sorted=True
            )

        compacted = jax.tree.map(gather, batch)
        return CandidateBatch(
            Edges(
                compacted.edges.indices.apply_mask(sort_idxs[:, None] < keep.size),
                compacted.edges.shifts,
            ),
            compacted.is_minimum_image,
        )


@dataclass
class MaskOnlyCompactor[D: int](Compactor[D]):
    """In-place compaction: failing entries become OOB indices and zero shifts.

    No size change; preserves the candidate count from the selector. Pair
    candidates are already in their output index space.
    """

    def __call__(
        self,
        keep: Array,
        batch: CandidateBatch[D],
        ctx: PipelineContext,
    ) -> Edges[D]:
        indices = batch.edges.indices.apply_mask(keep[:, None])
        shifts = where_broadcast_last(keep, batch.edges.shifts, 0)
        return Edges(indices, shifts)
