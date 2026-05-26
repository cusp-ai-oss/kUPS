# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Protocols for the neighbor list module.

Defines the core
[`NearestNeighborList`][kups.core.neighborlist.protocol.NearestNeighborList]
call signature, the particle and system trait protocols expected by every
implementation, the parameter protocols for capacity hints, and the
[`IsNeighborListState`][kups.core.neighborlist.protocol.IsNeighborListState]
protocol used by the ``from_state`` constructors.
"""

from __future__ import annotations

from typing import Literal, Protocol

from jax import Array

from kups.core.data import Index, Table
from kups.core.neighborlist.edges import Edges
from kups.core.typing import (
    HasCell,
    HasExclusionIndex,
    HasInclusionIndex,
    HasPositions,
    HasSystemIndex,
    ParticleId,
    SystemId,
)


class NeighborListPoints(
    HasPositions,
    HasSystemIndex,
    HasInclusionIndex,
    HasExclusionIndex,
    Protocol,
): ...


class NeighborListSystems(HasCell, Protocol): ...


class NearestNeighborList(Protocol):
    """Protocol for neighbor list construction algorithms.

    Implementations find pairs of particles within a cutoff distance, handling
    periodic boundary conditions and inclusion/exclusion masks.
    """

    def __call__[P: NeighborListPoints](
        self,
        lh: Table[ParticleId, P],
        rh: Table[ParticleId, P] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        """Find all particle pairs within the cutoff distance.

        Args:
            lh: Left-hand particles to find neighbors for
            rh: Right-hand particles to search within (or None for self-neighbors)
            systems: Indexed system data with cell information
            cutoffs: Indexed cutoff data per system
            rh_index_remap: Optional index mapping rh particles back to lh
                particle IDs for self-interaction exclusion. When ``None``,
                rh is treated as disjoint from lh.

        Returns:
            Edges connecting particle pairs within cutoff
        """
        ...


class IsUniversalNeighborlistParams(Protocol):
    """Protocol for parameters required by any neighbor list implementation.

    A superset of ``IsAllDenseNeighborListParams``, ``IsDenseNeighborlistParams``,
    and ``IsCellListParams``. Satisfying this protocol allows constructing any
    of the three neighbor list types.
    """

    @property
    def avg_edges(self) -> int: ...
    @property
    def avg_candidates(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...
    @property
    def cells(self) -> int: ...


class IsNeighborListState[P](Protocol):
    """Protocol for states that expose neighbor list parameters.

    A state satisfying this protocol can be passed to ``from_state()`` on any
    neighbor list class. The type parameter ``P`` determines which neighbor
    list types the state can construct (e.g., ``IsAllDenseNeighborListParams``,
    ``IsDenseNeighborlistParams``, ``IsCellListParams``, or
    ``IsUniversalNeighborlistParams``).
    """

    @property
    def neighborlist_params(self) -> P: ...
