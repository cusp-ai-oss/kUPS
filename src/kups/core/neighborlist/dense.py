# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dense O(N²/K²) neighbor list respecting system boundaries."""

from __future__ import annotations

from typing import Literal, Protocol, overload

from jax import Array

from kups.core.capacity import Capacity, LensCapacity
from kups.core.data import Index, Table, subselect
from kups.core.lens import Lens, lens
from kups.core.neighborlist.common import (
    Candidates,
    lift_query_candidates,
    replicate_for_images,
)
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.pipeline import build_radius_graph
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


class IsDenseNeighborlistParams(Protocol):
    """Protocol for parameters required by ``DenseNearestNeighborList``."""

    @property
    def avg_candidates(self) -> int: ...
    @property
    def avg_edges(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...


def _dense_subselect(
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    max_num_candidates: Capacity[int],
) -> Candidates:
    # Bucket by system *key*, not by position in each Index's own key tuple: a query
    # table whose system Index only spans a subset of the systems (e.g. blocking spheres
    # present in some systems) would otherwise be paired with the wrong systems.
    selection_result = subselect(
        keys.data.system.indices_in(systems.keys),
        queries.data.system.indices_in(systems.keys),
        output_buffer_size=max_num_candidates,
        num_segments=systems.size,
    )
    return Candidates(
        key_idx=Index(keys.keys, selection_result.scatter_idxs, _cls=keys.cls),
        query_idx=Index(queries.keys, selection_result.gather_idxs, _cls=queries.cls),
    )


@dataclass
class DenseSelector:
    """Selector for the per-system dense ``O(N²/K²)`` algorithm."""

    cutoffs: Table[SystemId, Array]
    max_candidates: Capacity[int]
    max_image_candidates: Capacity[int]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        query = ctx.query_table
        candidates = _dense_subselect(
            ctx.keys, query, ctx.systems, max_num_candidates=self.max_candidates
        )
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
class DenseNearestNeighborList:
    """Dense O(N²) neighbor list respecting system boundaries.

    This implementation generates all particle pairs within each system
    separately, avoiding cross-system interactions. Efficient when the cutoff
    is comparable to the box size (cutoff/box ~ 1).

    Complexity: O(N² / K²) where N is total particles and K is number of systems.

    Attributes:
        avg_candidates: Capacity for candidate pair storage.
        avg_edges: Capacity for final edge array.
        avg_image_candidates: Capacity for image candidate pairs.
        cutoffs: Per-system cutoff distances used by this neighbor list.

    When to use:
        - When cutoff/box_size ~ 1 (cutoff comparable to box dimensions)
        - Small box relative to cutoff (few cells would fit)
        - Non-periodic systems

    Example:
        ```python
        # Example: 15 Å cutoff in 20 Å box → cutoff/box = 0.75
        nl = DenseNearestNeighborList.new(state, lens(lambda s: s.nl_params), cutoffs)

        # Or, if the state implements IsNeighborListState:
        nl = DenseNearestNeighborList.from_state(state, cutoffs)

        edges = nl(particles, systems)
        ```
    """

    avg_candidates: Capacity[int]
    avg_edges: Capacity[int]
    avg_image_candidates: Capacity[int]
    cutoffs: Table[SystemId, Array]

    @classmethod
    def new[S](
        cls,
        state: S,
        lens: Lens[S, IsDenseNeighborlistParams],
        cutoffs: Table[SystemId, Array],
    ) -> DenseNearestNeighborList:
        params = lens.get(state)
        return DenseNearestNeighborList(
            avg_candidates=LensCapacity(
                params.avg_candidates, lens.focus(lambda x: x.avg_candidates)
            ),
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
        state: IsNeighborListState[IsDenseNeighborlistParams],
        cutoffs: Table[SystemId, Array],
    ) -> DenseNearestNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params), cutoffs)

    def selector(
        self, query_size: int, systems: Table[SystemId, NeighborListSystems]
    ) -> DenseSelector:
        """Build the candidate selector shared by graph and pair evaluation."""
        return DenseSelector(
            cutoffs=Table.broadcast_to(self.cutoffs, systems),
            max_candidates=self.avg_candidates.multiply(query_size),
            max_image_candidates=self.avg_image_candidates.multiply(query_size),
        )

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
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        return build_radius_graph(
            self, keys, systems, queries=queries, queried_keys=queried_keys
        )
