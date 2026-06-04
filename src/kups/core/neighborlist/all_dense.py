# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dense O(N²) neighbor list considering all pairs across all systems."""

from __future__ import annotations

import logging
from typing import Literal, Protocol, overload

import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity, LensCapacity
from kups.core.data import Index, Table
from kups.core.lens import Lens, lens
from kups.core.neighborlist.common import (
    Candidates,
    lift_query_candidates,
    replicate_for_images,
)
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    QueriedKeysDedupMask,
)
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.postprocess import MirrorPairEdges
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit


class IsAllDenseNeighborListParams(Protocol):
    """Protocol for parameters required by ``AllDenseNearestNeighborList``."""

    @property
    def avg_edges(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...


def _all_subselect(
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> Candidates:
    key_indices, queried_keys = jnp.indices((len(keys), len(queries))).reshape(2, -1)
    return Candidates(
        key_idx=Index(keys.keys, key_indices),
        query_idx=Index(queries.keys, queried_keys),
    )


@dataclass
class AllDenseSelector:
    """Selector that emits every ``(i, j)`` pair across all systems."""

    cutoffs: Table[SystemId, Array]
    max_image_candidates: Capacity[int]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        query = ctx.query_table
        candidates = _all_subselect(ctx.keys, query, ctx.systems)
        candidates = lift_query_candidates(candidates, ctx)
        return replicate_for_images(
            candidates,
            ctx.keys,
            ctx.edge_query_table,
            ctx.systems,
            self.cutoffs,
            self.max_image_candidates,
        )


@dataclass
class AllDenseNearestNeighborList:
    """Dense O(N²) neighbor list considering all pairs across all systems.

    This implementation generates all possible particle pairs without spatial
    optimization. It is only suitable for very small systems or testing.

    **Warning**: This crosses system boundaries! Only use for single-system
    simulations. For multiple systems, use
    [DenseNearestNeighborList][kups.core.neighborlist.DenseNearestNeighborList]
    instead.

    Complexity: O(N²) where N is the total number of particles across all systems.

    Attributes:
        avg_edges: Capacity manager for edge array.
        avg_image_candidates: Capacity manager for image candidate pairs.
        cutoffs: Per-system cutoff distances used by this neighbor list.

    Example:
        ```python
        # Construct from state and a lens to the neighbor list parameters:
        nl = AllDenseNearestNeighborList.new(state, lens(lambda s: s.nl_params), cutoffs)

        # Or, if the state implements IsNeighborListState:
        nl = AllDenseNearestNeighborList.from_state(state, cutoffs)

        edges = nl(particles, systems)
        ```
    """

    avg_edges: Capacity[int]
    avg_image_candidates: Capacity[int]
    cutoffs: Table[SystemId, Array]

    @classmethod
    def new[S](
        cls,
        state: S,
        lens: Lens[S, IsAllDenseNeighborListParams],
        cutoffs: Table[SystemId, Array],
    ) -> AllDenseNearestNeighborList:
        params = lens.get(state)
        return AllDenseNearestNeighborList(
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            avg_image_candidates=LensCapacity(
                params.avg_image_candidates,
                lens.focus(lambda x: x.avg_image_candidates),
            ),
            cutoffs=cutoffs,
        )

    @classmethod
    def from_state(
        cls,
        state: IsNeighborListState[IsAllDenseNeighborListParams],
        cutoffs: Table[SystemId, Array],
    ) -> AllDenseNearestNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params), cutoffs)

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> Edges[Literal[2]]: ...
    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]: ...
    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        if keys.data.inclusion.num_labels >= 2:
            logging.warning(
                "AllDenseNearestNeighborList is intended for single-system simulations. "
                "Performance may be degraded when using multiple systems. "
                "Consider using DenseNearestNeighborList or CellListNeighborList instead."
            )
        query_size = (
            queried_keys.size
            if queried_keys is not None
            else (queries.size if queries is not None else keys.size)
        )
        cutoffs = Table.broadcast_to(self.cutoffs, systems)
        pipeline = Pipeline[Literal[2]](
            selector=AllDenseSelector(
                cutoffs=cutoffs,
                max_image_candidates=self.avg_image_candidates.multiply(query_size),
            ),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                QueriedKeysDedupMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                ExclusionMask(),
            ),
            compactor=ReduceCompactor(avg_edges=self.avg_edges.multiply(query_size)),
            postprocessors=(MirrorPairEdges(),),
        )
        if queries is not None:
            return pipeline(keys, systems, queries=queries)
        return pipeline(keys, systems, queried_keys=queried_keys)
