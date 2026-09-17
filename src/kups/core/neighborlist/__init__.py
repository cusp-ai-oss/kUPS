# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor lists, periodic edges, and composable search pipelines.

Dense and cell-list searches share masks and compaction. ``CellListCache``
retains cell occupants for neighbor searches and incremental pair evaluation.
"""

from kups.core.neighborlist.adaptive import (
    AdaptiveNeighborList,
    NeighborListCandidate,
    NeighborListCost,
    all_dense_cost,
    cell_list_cost,
    dense_cost,
)
from kups.core.neighborlist.all_connected import (
    InclusionGroupSelector,
    all_connected_neighborlist,
)
from kups.core.neighborlist.all_dense import (
    AllDenseNearestNeighborList,
    AllDenseSelector,
    IsAllDenseNeighborListParams,
)
from kups.core.neighborlist.cell_list import (
    CellListNeighborList,
    CellListSelector,
    IsCellListParams,
)
from kups.core.neighborlist.cell_list_cache import (
    CellListCache,
    CellListCacheParameters,
    CellListCacheUpdatePatch,
    CellRows,
    build_cell_list_cache,
    cell_candidates,
    cell_rows,
)
from kups.core.neighborlist.changes import (
    NeighborListChangesResult,
    neighborlist_changes,
)
from kups.core.neighborlist.compact import MaskOnlyCompactor, ReduceCompactor
from kups.core.neighborlist.dense import (
    DenseNearestNeighborList,
    DenseSelector,
    IsDenseNeighborlistParams,
)
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.fixed import (
    EmptyNeighborList,
    FixedEdgesNeighborList,
)
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    QueriedKeysDedupMask,
    TouchesQueriedKeysMask,
)
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.postprocess import MirrorPairEdges
from kups.core.neighborlist.refine import (
    PrecomputedEdgesSelector,
    RefineCutoffNeighborList,
    RefineMaskNeighborList,
)
from kups.core.neighborlist.types import (
    CandidateBatch,
    CandidateSelector,
    Compactor,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    Mask,
    NeighborList,
    NeighborListFactory,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
    Postprocessor,
    SelectableNeighborList,
)

__all__ = [
    "AdaptiveNeighborList",
    "AllDenseNearestNeighborList",
    "AllDenseSelector",
    "CandidateBatch",
    "CandidateSelector",
    "CellListNeighborList",
    "CellListSelector",
    "CellRows",
    "CellListCache",
    "CellListCacheParameters",
    "CellListCacheUpdatePatch",
    "Compactor",
    "DenseNearestNeighborList",
    "DenseSelector",
    "DistanceCutoffMask",
    "Edges",
    "EmptyNeighborList",
    "ExclusionMask",
    "TouchesQueriedKeysMask",
    "FixedEdgesNeighborList",
    "InBoundsMask",
    "InclusionGroupSelector",
    "InclusionMatchMask",
    "IsAllDenseNeighborListParams",
    "IsCellListParams",
    "IsDenseNeighborlistParams",
    "IsNeighborListState",
    "IsUniversalNeighborlistParams",
    "Mask",
    "MaskOnlyCompactor",
    "MirrorPairEdges",
    "NeighborList",
    "NeighborListCandidate",
    "NeighborListChangesResult",
    "NeighborListCost",
    "NeighborListFactory",
    "NeighborListPoints",
    "NeighborListSystems",
    "Pipeline",
    "PipelineContext",
    "Postprocessor",
    "PrecomputedEdgesSelector",
    "ReduceCompactor",
    "SelectableNeighborList",
    "RefineCutoffNeighborList",
    "RefineMaskNeighborList",
    "QueriedKeysDedupMask",
    "UniversalNeighborlistParameters",
    "all_connected_neighborlist",
    "all_dense_cost",
    "build_cell_list_cache",
    "cell_candidates",
    "cell_list_cost",
    "cell_rows",
    "dense_cost",
    "neighborlist_changes",
]
