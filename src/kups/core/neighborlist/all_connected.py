# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Distance-agnostic pair list connecting all particles in an inclusion segment.

[`all_connected_neighborlist`][kups.core.neighborlist.all_connected.all_connected_neighborlist]
ignores the cutoff and emits every pair sharing the same inclusion segment that
has differing exclusion segment IDs. Used by Ewald summation to enumerate the
real-space exclusion list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import jax.numpy as jnp

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.data import Index, Table, subselect
from kups.core.neighborlist.common import (
    Candidates,
    candidates_to_batch,
    lift_query_candidates,
)
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import ExclusionMask, QueriedKeysDedupMask
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.postprocess import MirrorPairEdges
from kups.core.neighborlist.types import (
    CandidateBatch,
    NeighborList,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.patch import IdPatch, WithPatch
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
        query = ctx.query_table
        inclusion = ctx.keys.data.inclusion
        selection_result = subselect(
            *Index.match(
                inclusion,
                query.data.inclusion.update_labels(inclusion.keys, allow_missing=True),
            ),
            output_buffer_size=self.capacity,
            num_segments=inclusion.num_labels,
        )
        candidates = Candidates(
            key_idx=Index(ctx.keys.keys, selection_result.scatter_idxs),
            query_idx=Index(query.keys, selection_result.gather_idxs),
        )
        candidates = lift_query_candidates(candidates, ctx)
        deltas = (
            ctx.keys[candidates.key_idx].positions
            - ctx.edge_query_table[candidates.query_idx].positions
        )
        shifts = jnp.round(deltas).astype(int)
        return candidates_to_batch(
            candidates,
            shifts,
            jnp.ones((candidates.key_idx.size,), dtype=bool),
        )


@overload
def all_connected_neighborlist[P: NeighborListPoints](
    keys: Table[ParticleId, P],
    systems: Table[SystemId, NeighborListSystems],
    *,
    queries: Table[ParticleId, P],
) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]: ...


@overload
def all_connected_neighborlist[P: NeighborListPoints](
    keys: Table[ParticleId, P],
    systems: Table[SystemId, NeighborListSystems],
    *,
    queried_keys: Index[ParticleId] | None = None,
) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]: ...


def all_connected_neighborlist[P: NeighborListPoints](
    keys: Table[ParticleId, P],
    systems: Table[SystemId, NeighborListSystems],
    *,
    queries: Table[ParticleId, P] | None = None,
    queried_keys: Index[ParticleId] | None = None,
) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]:
    """Neighbor list connecting all pairs sharing the same inclusion segment, ignoring distance.

    Connects every particle pair that belongs to the same inclusion segment and has
    differing exclusion segment IDs. The cell is used only to compute
    minimum-image shifts.

    Requires ``max_count`` to be set on the inclusion ``Index``.
    """
    max_count = keys.data.inclusion.max_count
    assert max_count is not None, "inclusion.max_count must be set"
    query_size = (
        queried_keys.size
        if queried_keys is not None
        else (queries.size if queries is not None else keys.size)
    )
    capacity = FixedCapacity(max_count).multiply(min(keys.size, query_size))

    pipeline = Pipeline[Literal[2]](
        selector=InclusionGroupSelector(capacity=capacity),
        masks=(ExclusionMask(), QueriedKeysDedupMask()),
        compactor=ReduceCompactor(avg_edges=capacity),
        postprocessors=(MirrorPairEdges(),),
    )
    if queries is not None:
        edges = pipeline(keys, systems, queries=queries)
    else:
        edges = pipeline(keys, systems, queried_keys=queried_keys)
    return WithPatch(edges, IdPatch[Any]())


if TYPE_CHECKING:
    x: NeighborList[Literal[2]] = all_connected_neighborlist
