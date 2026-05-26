# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor list construction and edge representations for molecular systems.

This module provides multiple neighbor list algorithms for finding interacting
pairs of particles within cutoff distances, with different performance and
accuracy trade-offs.

## Core Components

- **[Edges][kups.core.neighborlist.Edges]**: Represents connections between particles with periodic shifts
- **[NearestNeighborList][kups.core.neighborlist.NearestNeighborList]**: Protocol for neighbor search implementations
- **[RefineMaskNeighborList][kups.core.neighborlist.RefineMaskNeighborList]**: Applies inclusion/exclusion masks for selective interactions

## Neighbor List Implementations

### Primary Implementations

1. **[CellListNeighborList][kups.core.neighborlist.CellListNeighborList]** (Recommended when cutoff << box size)
    - O(N) complexity using spatial hashing
    - Best when cutoff / box_size < 0.3 (cutoff much smaller than box)
    - Honors the cell's per-axis ``periodic`` mask (bulk and bounded non-periodic)
    - Efficiency improves as cutoff/box ratio decreases

2. **[DenseNearestNeighborList][kups.core.neighborlist.DenseNearestNeighborList]**
    - O(N²/K) complexity (K = number of systems)
    - Best when cutoff / box_size ~ 1 (cutoff comparable to box)
    - Works with or without periodic boundaries
    - More efficient when few cells would fit in box

3. **[AllDenseNearestNeighborList][kups.core.neighborlist.AllDenseNearestNeighborList]**
    - O(N²) complexity across all systems
    - Only for single-system simulations or testing
    - Crosses system boundaries (use with caution!)

### Refinement Implementations

These allow sharing a single base neighbor list across multiple potentials
with different cutoffs or interaction rules (e.g., Lennard-Jones and Coulomb).

4. **[RefineMaskNeighborList][kups.core.neighborlist.RefineMaskNeighborList]**
    - Applies inclusion/exclusion masks to precomputed edges
    - Use for bonded exclusions or group-specific interactions
    - No distance recalculation
    - Share one neighbor list, apply different masks per potential

5. **[RefineCutoffNeighborList][kups.core.neighborlist.RefineCutoffNeighborList]**
    - Refines precomputed edges with new cutoff distances
    - Use for multi-stage construction or adaptive cutoffs
    - Recalculates distances
    - Share one conservative neighbor list, apply different cutoffs per potential

## Features

All neighbor lists handle:
- Per-axis periodic / non-periodic boundaries via the cell's ``periodic`` mask
  (shift vectors are zero on non-periodic axes)
- Multiple systems in parallel with segmentation
- Automatic capacity management for variable neighbor counts
- Integration with JAX transformations (JIT, vmap, etc.)
"""

from kups.core.neighborlist.all_connected import all_connected_neighborlist
from kups.core.neighborlist.all_dense import (
    AllDenseNearestNeighborList,
    IsAllDenseNeighborListParams,
)
from kups.core.neighborlist.cell_list import (
    CellListNeighborList,
    IsCellListParams,
)
from kups.core.neighborlist.changes import (
    NeighborListChangesResult,
    neighborlist_changes,
)
from kups.core.neighborlist.common import (
    CandidateSelector,
    basic_neighborlist,
)
from kups.core.neighborlist.dense import (
    DenseNearestNeighborList,
    IsDenseNeighborlistParams,
)
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.refine import (
    RefineCutoffNeighborList,
    RefineMaskNeighborList,
)
from kups.core.neighborlist.types import (
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NearestNeighborList,
    NeighborListPoints,
    NeighborListSystems,
)

__all__ = [
    "AllDenseNearestNeighborList",
    "CandidateSelector",
    "CellListNeighborList",
    "DenseNearestNeighborList",
    "Edges",
    "IsAllDenseNeighborListParams",
    "IsCellListParams",
    "IsDenseNeighborlistParams",
    "IsNeighborListState",
    "IsUniversalNeighborlistParams",
    "NearestNeighborList",
    "NeighborListChangesResult",
    "NeighborListPoints",
    "NeighborListSystems",
    "RefineCutoffNeighborList",
    "RefineMaskNeighborList",
    "UniversalNeighborlistParameters",
    "all_connected_neighborlist",
    "basic_neighborlist",
    "neighborlist_changes",
]
