# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dense O(N²) neighbor list considering all pairs across all systems."""

from __future__ import annotations

import logging
from typing import Literal, Protocol

import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity, LensCapacity
from kups.core.data import Index, Table
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
from kups.core.utils.jax import dataclass, jit


class IsAllDenseNeighborListParams(Protocol):
    """Protocol for parameters required by ``AllDenseNearestNeighborList``."""

    @property
    def avg_edges(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...


def _all_subselect(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> Candidates:
    lh_indices, rh_indices = jnp.indices((len(lh), len(rh))).reshape(2, -1)
    return Candidates(lhs=Index(lh.keys, lh_indices), rhs=Index(rh.keys, rh_indices))


@dataclass
class AllDenseSelector:
    """Selector that emits every ``(i, j)`` pair across all systems."""

    cutoffs: Table[SystemId, Array]
    max_image_candidates: Capacity[int]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        candidates = _all_subselect(ctx.lh, ctx.rh, ctx.systems)
        return replicate_for_images(
            candidates,
            ctx.lh,
            ctx.rh,
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

    Example:
        ```python
        # Construct from state and a lens to the neighbor list parameters:
        nl = AllDenseNearestNeighborList.new(state, lens(lambda s: s.nl_params))

        # Or, if the state implements IsNeighborListState:
        nl = AllDenseNearestNeighborList.from_state(state)

        edges = nl(particles, None, systems, cutoffs, None)
        ```
    """

    avg_edges: Capacity[int]
    avg_image_candidates: Capacity[int]

    @classmethod
    def new[S](
        cls, state: S, lens: Lens[S, IsAllDenseNeighborListParams]
    ) -> AllDenseNearestNeighborList:
        params = lens.get(state)
        return AllDenseNearestNeighborList(
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            avg_image_candidates=LensCapacity(
                params.avg_image_candidates,
                lens.focus(lambda x: x.avg_image_candidates),
            ),
        )

    @classmethod
    def from_state(
        cls, state: IsNeighborListState[IsAllDenseNeighborListParams]
    ) -> AllDenseNearestNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params))

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        if lh.data.inclusion.num_labels >= 2:
            logging.warning(
                "AllDenseNearestNeighborList is intended for single-system simulations. "
                "Performance may be degraded when using multiple systems. "
                "Consider using DenseNearestNeighborList or CellListNeighborList instead."
            )
        rh_size = rh.size if rh is not None else lh.size
        cutoffs = Table.broadcast_to(cutoffs, systems)
        pipeline = Pipeline[Literal[2]](
            selector=AllDenseSelector(
                cutoffs=cutoffs,
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
