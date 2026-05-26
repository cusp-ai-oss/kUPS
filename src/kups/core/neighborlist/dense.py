# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dense O(N²/K²) neighbor list respecting system boundaries."""

from __future__ import annotations

from typing import Literal, Protocol

from jax import Array

from kups.core.capacity import Capacity, LensCapacity
from kups.core.data import Index, Table, subselect
from kups.core.lens import Lens, lens
from kups.core.neighborlist.common import Candidates, replicate_for_images
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    RemapDedupMask,
)
from kups.core.neighborlist.pipeline import Pipeline
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
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    max_num_candidates: Capacity[int],
) -> Candidates:
    selection_result = subselect(
        lh.data.system.indices,
        rh.data.system.indices,
        output_buffer_size=max_num_candidates,
        num_segments=systems.size,
    )
    return Candidates(
        lhs=Index(lh.keys, selection_result.scatter_idxs),
        rhs=Index(rh.keys, selection_result.gather_idxs),
    )


@dataclass
class DenseSelector:
    """Selector for the per-system dense ``O(N²/K²)`` algorithm."""

    cutoffs: Table[SystemId, Array]
    max_candidates: Capacity[int]
    max_image_candidates: Capacity[int]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        candidates = _dense_subselect(
            ctx.lh, ctx.rh, ctx.systems, max_num_candidates=self.max_candidates
        )
        return replicate_for_images(
            candidates,
            ctx.lh,
            ctx.rh,
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

    When to use:
        - When cutoff/box_size ~ 1 (cutoff comparable to box dimensions)
        - Small box relative to cutoff (few cells would fit)
        - Non-periodic systems

    Example:
        ```python
        # Example: 15 Å cutoff in 20 Å box → cutoff/box = 0.75
        nl = DenseNearestNeighborList.new(state, lens(lambda s: s.nl_params))

        # Or, if the state implements IsNeighborListState:
        nl = DenseNearestNeighborList.from_state(state)

        edges = nl(particles, None, systems, cutoffs, None)
        ```
    """

    avg_candidates: Capacity[int]
    avg_edges: Capacity[int]
    avg_image_candidates: Capacity[int]

    @classmethod
    def new[S](
        cls, state: S, lens: Lens[S, IsDenseNeighborlistParams]
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
        )

    @classmethod
    def from_state(
        cls, state: IsNeighborListState[IsDenseNeighborlistParams]
    ) -> DenseNearestNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params))

    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        rh_size = rh.size if rh is not None else lh.size
        cutoffs = Table.broadcast_to(cutoffs, systems)
        pipeline = Pipeline[Literal[2]](
            selector=DenseSelector(
                cutoffs=cutoffs,
                max_candidates=self.avg_candidates.multiply(rh_size),
                max_image_candidates=self.avg_image_candidates.multiply(rh_size),
            ),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                RemapDedupMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                ExclusionMask(),
            ),
            compactor=ReduceCompactor(avg_edges=self.avg_edges.multiply(rh_size)),
        )
        return pipeline(lh, rh, systems, rh_index_remap)
