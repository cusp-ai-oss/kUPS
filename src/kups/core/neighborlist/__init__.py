# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor list construction and edge representations for molecular systems.

This module provides multiple neighbor list algorithms for finding interacting
pairs of particles within cutoff distances, with different performance and
accuracy trade-offs.

## Call Contract

Neighbor lists are called as ``neighborlist(keys, systems, *, queries=None,
queried_keys=None)``. ``keys`` is the self-graph/output table. Use keyword-only
``queries`` only for true bipartite queries. Use keyword-only ``queried_keys``
only for self-graph updates; it is mutually exclusive with ``queries`` and names
affected ``keys`` ids after the caller has already written updated particle data
into ``keys``.

## Core Components

- **[Edges][kups.core.neighborlist.Edges]**: Represents connections between particles with periodic shifts
- **[NeighborList][kups.core.neighborlist.NeighborList]**: Protocol for neighbor search implementations
- **[Pipeline][kups.core.neighborlist.Pipeline]**: Selector → mask sequence → compactor → postprocessors

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

### Cutoff-Free Implementations

These cover non-cutoff cases under the same `NeighborList[D]` protocol.

6. **[EmptyNeighborList][kups.core.neighborlist.EmptyNeighborList]**: emits a
   zero-row ``Edges[D]`` for point-cloud constructions.
7. **[FixedEdgesNeighborList][kups.core.neighborlist.FixedEdgesNeighborList]**:
   stores fixed edge topology for bonded edge sets supplied by the state and
   computes current periodic shifts during calls. Affected self-graph calls use
   ``queried_keys`` and return only rows touched by those affected ``keys`` ids.

## Pipeline Primitives

Every neighbor list above is a [`Pipeline`][kups.core.neighborlist.Pipeline]
of a [`CandidateSelector`][kups.core.neighborlist.CandidateSelector], a
``tuple`` of [`Mask`][kups.core.neighborlist.Mask] criteria, a
[`Compactor`][kups.core.neighborlist.Compactor], and zero or more
[`Postprocessor`][kups.core.neighborlist.Postprocessor] transforms. Users
wanting custom behavior can compose their own pipeline directly.
"""

from kups.core.neighborlist.adaptive import (
    CutoffNeighborListPolicy,
    CutoffNeighborListStrategy,
    IsAdaptiveCutoffNeighborListState,
    adaptive_cutoff_neighborlist_from_state,
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
)

__all__ = [
    "AllDenseNearestNeighborList",
    "AllDenseSelector",
    "CandidateBatch",
    "CandidateSelector",
    "CellListNeighborList",
    "CellListSelector",
    "Compactor",
    "CutoffNeighborListPolicy",
    "CutoffNeighborListStrategy",
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
    "IsAdaptiveCutoffNeighborListState",
    "IsAllDenseNeighborListParams",
    "IsCellListParams",
    "IsDenseNeighborlistParams",
    "IsNeighborListState",
    "IsUniversalNeighborlistParams",
    "Mask",
    "MaskOnlyCompactor",
    "MirrorPairEdges",
    "NeighborList",
    "NeighborListChangesResult",
    "NeighborListFactory",
    "NeighborListPoints",
    "NeighborListSystems",
    "Pipeline",
    "PipelineContext",
    "Postprocessor",
    "PrecomputedEdgesSelector",
    "ReduceCompactor",
    "RefineCutoffNeighborList",
    "RefineMaskNeighborList",
    "QueriedKeysDedupMask",
    "UniversalNeighborlistParameters",
    "adaptive_cutoff_neighborlist_from_state",
    "all_connected_neighborlist",
    "neighborlist_changes",
]
