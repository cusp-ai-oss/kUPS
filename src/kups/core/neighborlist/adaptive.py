# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Adaptive cutoff neighbor list with per-call, cost-based dispatch.

:class:`AdaptiveNeighborList` is itself a ``NeighborList[Literal[2]]``. It holds
a tuple of ``(implementation, cost)`` pairs where each ``cost`` is a
:class:`NeighborListCost` -- a numerical estimate of that implementation's
runtime for a given call. On every ``__call__`` it evaluates each cost from the
call's particle and system counts and dispatches to the cheapest implementation,
so a single object routes differently across calls.

The :meth:`AdaptiveNeighborList.new` / :meth:`AdaptiveNeighborList.from_state`
classmethods seed the tuple with the library implementations
(:class:`DenseNearestNeighborList`, :class:`CellListNeighborList`,
:class:`AllDenseNearestNeighborList`) and their default cost guesses. When the
optional ``nvalchemiops`` package is installed, the GPU-backed
:class:`NvalchemiNaiveNeighborList` and :class:`NvalchemiCellListNeighborList`
are seeded too, with lower costs so they win dispatch in their regimes; they
reject bipartite ``queries``, so such calls still route to a library
implementation. Augmenting the set is just appending another
``(implementation, cost)`` pair to ``implementations``; all implementations are
built from one lens, so they share the ``UniversalNeighborlistParameters``
capacities.

Costs use coarse counts rather than cutoff or box geometry: counts are static at
trace time, so the dispatch is a plain Python branch that compiles only the
selected implementation.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Literal, Protocol, overload

from kups.core.data import Index, Table
from kups.core.lens import Lens, lens
from kups.core.neighborlist.all_dense import AllDenseNearestNeighborList
from kups.core.neighborlist.cell_list import CellListNeighborList
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.nvalchemi import (
    NvalchemiCellListNeighborList,
    NvalchemiNaiveNeighborList,
)
from kups.core.neighborlist.types import (
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field


class NeighborListCost(Protocol):
    """Estimates the relative runtime cost of a neighbor list for one call.

    Lower is cheaper; :class:`AdaptiveNeighborList` dispatches to the minimum.
    Return ``math.inf`` to mark an implementation invalid for the given shape.
    """

    def __call__(self, num_particles: int, num_systems: int) -> float: ...


@dataclass
class NeighborListCandidate:
    """A backing neighbor list paired with its cost estimator.

    Attributes:
        neighborlist: The candidate implementation.
        cost: Estimates the implementation's cost for a call's counts.
        supports_queries: Whether the implementation accepts bipartite
            ``queries``; candidates that don't are skipped for such calls.
    """

    neighborlist: NeighborList[Literal[2]]
    cost: NeighborListCost = field(static=True)
    supports_queries: bool = field(static=True, default=True)


# Average particles per system at which O(N) cell-list overtakes O(N^2/K) dense.
_CELL_LIST_CROSSOVER = 10_000


def dense_cost(num_particles: int, num_systems: int) -> float:
    """Default cost for :class:`DenseNearestNeighborList` (``O(N^2/K)``)."""
    return num_particles**2 / max(num_systems, 1)


def cell_list_cost(num_particles: int, num_systems: int) -> float:
    """Default cost for :class:`CellListNeighborList` (``O(N)`` with a large
    constant, crossing dense at ``_CELL_LIST_CROSSOVER`` particles per system)."""
    return _CELL_LIST_CROSSOVER * num_particles


def all_dense_cost(num_particles: int, num_systems: int) -> float:
    """Default cost for :class:`AllDenseNearestNeighborList`.

    ``O(N^2)`` across all particles, so it ties dense for a single system and is
    invalid (``inf``) for multiple systems, which it would incorrectly merge.
    """
    return 0.9 * num_particles**2 if num_systems <= 1 else math.inf


# Factor by which an installed nvalchemi GPU kernel is preferred over the
# equivalent pure-JAX implementation. Any value in ``(0, 1)`` keeps the
# naive/cell-list crossover at ``_CELL_LIST_CROSSOVER`` while letting nvalchemi win.
_NVALCHEMI_PREFERENCE = 0.5


def nvalchemi_naive_cost(num_particles: int, num_systems: int) -> float:
    """Default cost for :class:`NvalchemiNaiveNeighborList`.

    Mirrors :func:`dense_cost` (``O(N^2/K)``) scaled by ``_NVALCHEMI_PREFERENCE``
    so the GPU kernel wins over dense/all-dense below the cell-list crossover.
    """
    return _NVALCHEMI_PREFERENCE * dense_cost(num_particles, num_systems)


def nvalchemi_cell_list_cost(num_particles: int, num_systems: int) -> float:
    """Default cost for :class:`NvalchemiCellListNeighborList`.

    Mirrors :func:`cell_list_cost` (``O(N)``) scaled by ``_NVALCHEMI_PREFERENCE``
    so the GPU kernel wins over the JAX cell list above the crossover, for any
    system count.
    """
    return _NVALCHEMI_PREFERENCE * cell_list_cost(num_particles, num_systems)


def _nvalchemi_installed() -> bool:
    """Whether the optional ``nvalchemiops`` package is importable."""
    return importlib.util.find_spec("nvalchemiops") is not None


@dataclass
class AdaptiveNeighborList(NeighborList[Literal[2]]):
    """Neighbor list that dispatches to the cheapest backing implementation.

    Holds ``(implementation, cost)`` pairs and, on each call, picks the
    implementation whose :class:`NeighborListCost` is smallest for that call's
    counts. Augment by appending pairs to :attr:`implementations`; the seeded
    implementations share the ``UniversalNeighborlistParameters`` capacities, so
    growing one grows the shared state.

    Attributes:
        implementations: Candidate :class:`NeighborListCandidate` pairs. Ties
            resolve to the earlier entry.

    Example:
        ```python
        nl = AdaptiveNeighborList.from_state(state, cutoff)
        edges = nl(particles, systems)
        ```
    """

    implementations: tuple[NeighborListCandidate, ...]

    @classmethod
    def new[S](
        cls,
        state: S,
        lens: Lens[S, IsUniversalNeighborlistParams],
        cutoff: float,
    ) -> AdaptiveNeighborList:
        """Seed the library implementations with their default cost guesses.

        Args:
            state: Object exposing ``UniversalNeighborlistParameters`` via ``lens``.
            lens: Lens focusing the shared ``IsUniversalNeighborlistParams``.
            cutoff: Cutoff radius [Å] bound onto each implementation.

        Returns:
            An ``AdaptiveNeighborList`` over dense, cell-list, and all-dense,
            plus the two ``nvalchemiops`` implementations when that package is
            installed.
        """
        candidates = [
            NeighborListCandidate(
                DenseNearestNeighborList.new(state, lens, cutoff), dense_cost
            ),
            NeighborListCandidate(
                CellListNeighborList.new(state, lens, cutoff), cell_list_cost
            ),
            NeighborListCandidate(
                AllDenseNearestNeighborList.new(state, lens, cutoff), all_dense_cost
            ),
        ]
        if _nvalchemi_installed():
            candidates += [
                NeighborListCandidate(
                    NvalchemiNaiveNeighborList.new(state, lens, cutoff),
                    nvalchemi_naive_cost,
                    supports_queries=False,
                ),
                NeighborListCandidate(
                    NvalchemiCellListNeighborList.new(state, lens, cutoff),
                    nvalchemi_cell_list_cost,
                    supports_queries=False,
                ),
            ]
        return cls(tuple(candidates))

    @classmethod
    def from_state(
        cls,
        state: IsNeighborListState[IsUniversalNeighborlistParams],
        cutoff: float,
    ) -> AdaptiveNeighborList:
        """Seed from a state exposing ``neighborlist_params``."""
        return cls.new(state, lens(lambda s: s.neighborlist_params), cutoff)

    def _choose(
        self, num_particles: int, num_systems: int, *, bipartite: bool = False
    ) -> NeighborList[Literal[2]]:
        """Return the cheapest implementation for the given counts.

        Args:
            num_particles: Total particle count for the call.
            num_systems: Number of systems for the call.
            bipartite: Whether the call passes ``queries``; restricts the choice
                to candidates that support bipartite queries.

        Returns:
            The cheapest eligible backing implementation.
        """
        candidates = [
            c for c in self.implementations if c.supports_queries or not bipartite
        ]
        return min(
            candidates,
            key=lambda candidate: candidate.cost(num_particles, num_systems),
        ).neighborlist

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
        nl = self._choose(keys.size, systems.size, bipartite=queries is not None)
        if queries is not None:
            return nl(keys, systems, queries=queries)
        return nl(keys, systems, queried_keys=queried_keys)
