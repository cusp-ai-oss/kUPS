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
from jax import Array

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.data import Index, Table, subselect
from kups.core.neighborlist.common import Candidates, candidates_to_batch
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import ExclusionMask, RemapDedupMask
from kups.core.neighborlist.pipeline import Pipeline
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
        ngraphs = ctx.lh.data.inclusion.num_labels
        selection_result = subselect(
            ctx.lh.data.inclusion.indices,
            ctx.rh.data.inclusion.indices,
            output_buffer_size=self.capacity,
            num_segments=ngraphs,
        )
        candidates = Candidates(
            lhs=Index(ctx.lh.keys, selection_result.scatter_idxs),
            rhs=Index(ctx.rh.keys, selection_result.gather_idxs),
        )
        deltas = (
            ctx.lh.data.positions[candidates.lhs.indices]
            - ctx.rh.data.positions[candidates.rhs.indices]
        )
        shifts = jnp.round(deltas).astype(int)
        return candidates_to_batch(
            candidates,
            shifts,
            jnp.ones((candidates.lhs.size,), dtype=bool),
        )


def all_connected_neighborlist(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints] | None,
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    rh_index_remap: Index[ParticleId] | None = None,
) -> Edges[Literal[2]]:
    """Neighbor list connecting all pairs sharing the same inclusion segment, ignoring distance.

    Connects every particle pair that belongs to the same inclusion segment and has
    differing exclusion segment IDs. The cutoff is ignored for neighbor selection;
    the cell is used only to compute minimum-image shifts.

    Requires ``max_count`` to be set on the inclusion ``Index``.
    """
    if rh is None:
        rh = lh
        rh_index_remap = Index.arange(len(lh), label=ParticleId)

    max_count = lh.data.inclusion.max_count
    assert max_count is not None, "inclusion.max_count must be set"
    capacity = FixedCapacity(max_count).multiply(min(lh.size, rh.size))

    pipeline = Pipeline[Literal[2]](
        selector=InclusionGroupSelector(capacity=capacity),
        masks=(ExclusionMask(), RemapDedupMask()),
        compactor=ReduceCompactor(avg_edges=capacity),
    )
    return pipeline(lh, rh, systems, rh_index_remap)
