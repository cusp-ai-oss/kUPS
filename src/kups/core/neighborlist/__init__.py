# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor list construction and edge representations for molecular systems.

This module provides multiple neighbor list algorithms for finding interacting
pairs of particles within cutoff distances, with different performance and
accuracy trade-offs.

## Core Components

- **[Edges][kups.core.neighborlist.Edges]**: Represents connections between particles with periodic shifts
- **[NeighborList][kups.core.neighborlist.NeighborList]**: Protocol for neighbor search implementations
- **[Pipeline][kups.core.neighborlist.Pipeline]**: Selector → mask sequence → compactor

## Neighbor List Implementations

### Primary Implementations

1. **[CellListNeighborList][kups.core.neighborlist.CellListNeighborList]** (Recommended when cutoff << box size)
    - O(N) complexity using spatial hashing
    - Best when cutoff / box_size < 0.3 (cutoff much smaller than box)
    - Honors the cell's per-axis ``periodic`` mask (bulk and bounded non-periodic)

2. **[DenseNearestNeighborList][kups.core.neighborlist.DenseNearestNeighborList]**
    - O(N²/K) complexity (K = number of systems)
    - Best when cutoff / box_size ~ 1 (cutoff comparable to box)

3. **[AllDenseNearestNeighborList][kups.core.neighborlist.AllDenseNearestNeighborList]**
    - O(N²) complexity across all systems
    - Only for single-system simulations or testing
    - Crosses system boundaries (use with caution!)

### Refinement Implementations

These let one expensive base neighbor list be shared across multiple potentials.

4. **[RefineMaskNeighborList][kups.core.neighborlist.RefineMaskNeighborList]**:
   apply different inclusion/exclusion masks to precomputed edges.
5. **[RefineCutoffNeighborList][kups.core.neighborlist.RefineCutoffNeighborList]**:
   refine precomputed edges with new cutoff distances.

## Pipeline Primitives

Every neighbor list above is a [`Pipeline`][kups.core.neighborlist.Pipeline]
of a [`CandidateSelector`][kups.core.neighborlist.CandidateSelector], a
``tuple`` of [`Mask`][kups.core.neighborlist.Mask] criteria, and a
[`Compactor`][kups.core.neighborlist.Compactor]. Users wanting custom
behavior can compose their own pipeline directly.
"""

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
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    RemapDedupMask,
)
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.pipeline import Pipeline
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
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)

__all__ = [
    "AllDenseNearestNeighborList",
    "AllDenseSelector",
    "CandidateBatch",
    "CandidateSelector",
    "CellListNeighborList",
    "CellListSelector",
    "Compactor",
    "DenseNearestNeighborList",
    "DenseSelector",
    "DistanceCutoffMask",
    "Edges",
    "ExclusionMask",
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
    "NeighborList",
    "NeighborListChangesResult",
    "NeighborListPoints",
    "NeighborListSystems",
    "Pipeline",
    "PipelineContext",
    "PrecomputedEdgesSelector",
    "ReduceCompactor",
    "RefineCutoffNeighborList",
    "RefineMaskNeighborList",
    "RemapDedupMask",
    "UniversalNeighborlistParameters",
    "all_connected_neighborlist",
    "neighborlist_changes",
]
