# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Distance-agnostic pair list connecting all particles in an inclusion segment.

[`all_connected_neighborlist`][kups.core.neighborlist.all_connected.all_connected_neighborlist]
ignores the cutoff and emits every pair sharing the same inclusion segment that
has differing exclusion segment IDs. Used by Ewald summation to enumerate the
real-space exclusion list.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.data import Index, Table, subselect
from kups.core.neighborlist.common import (
    Candidates,
    candidates_to_batch,
    edge_rhs_table,
    lift_query_candidates,
    query_table,
)
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import ExclusionMask, ForIndicesDedupMask
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.postprocess import MirrorPairEdges
from kups.core.neighborlist.types import (
    CandidateBatch,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


@dataclass
class InclusionGroupSelector:
    """Pairs every particle with every other in the same inclusion segment.

    Ignores the cutoff entirely. Shifts are int-typed minimum-image
    fractional rounds — matches today's ``all_connected_neighborlist``
    (which is Ewald-only and assumed fully periodic).
    """

    capacity: Capacity[int]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        query = query_table(ctx)
        ngraphs = ctx.lh.data.inclusion.num_labels
        selection_result = subselect(
            ctx.lh.data.inclusion.indices,
            query.data.inclusion.indices,
            output_buffer_size=self.capacity,
            num_segments=ngraphs,
        )
        candidates = Candidates(
            lhs=Index(ctx.lh.keys, selection_result.scatter_idxs),
            rhs=Index(query.keys, selection_result.gather_idxs),
        )
        candidates = lift_query_candidates(candidates, ctx)
        rhs = edge_rhs_table(ctx)
        deltas = (
            ctx.lh.data.positions[candidates.lhs.indices]
            - rhs.data.positions[candidates.rhs.indices]
        )
        shifts = jnp.round(deltas).astype(int)
        return candidates_to_batch(
            candidates,
            shifts,
            jnp.ones((candidates.lhs.size,), dtype=bool),
        )


def all_connected_neighborlist(
    lh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    *,
    rh: Table[ParticleId, NeighborListPoints] | None = None,
    for_indices: Index[ParticleId] | None = None,
) -> Edges[Literal[2]]:
    """Neighbor list connecting all pairs sharing the same inclusion segment, ignoring distance.

    Connects every particle pair that belongs to the same inclusion segment and has
    differing exclusion segment IDs. The cell is used only to compute
    minimum-image shifts.

    Requires ``max_count`` to be set on the inclusion ``Index``.
    """
    max_count = lh.data.inclusion.max_count
    assert max_count is not None, "inclusion.max_count must be set"
    query_size = (
        for_indices.size
        if for_indices is not None
        else (rh.size if rh is not None else lh.size)
    )
    capacity = FixedCapacity(max_count).multiply(min(lh.size, query_size))

    pipeline = Pipeline[Literal[2]](
        selector=InclusionGroupSelector(capacity=capacity),
        masks=(ExclusionMask(), ForIndicesDedupMask()),
        compactor=ReduceCompactor(avg_edges=capacity),
        postprocessors=(MirrorPairEdges(),),
    )
    return pipeline(lh, systems, rh=rh, for_indices=for_indices)
