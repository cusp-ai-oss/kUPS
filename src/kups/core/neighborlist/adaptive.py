# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Adaptive cutoff neighbor-list construction.

The :func:`adaptive_cutoff_neighborlist_from_state` factory picks between
:class:`DenseNearestNeighborList` and :class:`CellListNeighborList` at
*construction time* using a static, count-based
:class:`CutoffNeighborListPolicy`. The returned object is a normal
``NeighborList[Literal[2]]`` — its ``__call__`` runs exactly one algorithm,
with no runtime branching.

The policy intentionally avoids inspecting cutoff values, cell volumes, or
perpendicular box lengths: those are unreliable construction-time signals
in the current architecture (cells can resize, cutoffs can be bound late).
Coarse counts (total particle capacity, number of systems, average
occupancy) are stable enough to drive a deterministic choice.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from jax import Array

from kups.core.data import Table
from kups.core.neighborlist.cell_list import CellListNeighborList
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.neighborlist.types import (
    IsUniversalNeighborlistParams,
    NeighborList,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field


class CutoffNeighborListStrategy(StrEnum):
    """Concrete strategy for ``adaptive_cutoff_neighborlist_from_state``.

    ``AUTO`` defers to :class:`CutoffNeighborListPolicy`. The other two
    bypass the policy and force the corresponding implementation — useful
    for debugging, benchmarking, and reproducibility.
    """

    AUTO = "auto"
    DENSE = "dense"
    CELL_LIST = "cell_list"


class _Sized(Protocol):
    @property
    def size(self) -> int: ...


class IsAdaptiveCutoffNeighborListState[P](Protocol):
    """State protocol consumed by ``adaptive_cutoff_neighborlist_from_state``.

    Extends :class:`IsNeighborListState` with the static shape information
    (particle and system counts) needed to evaluate the count-based policy
    without inspecting cutoff or cell values.
    """

    @property
    def neighborlist_params(self) -> P: ...
    @property
    def particles(self) -> _Sized: ...
    @property
    def systems(self) -> _Sized: ...


@dataclass
class CutoffNeighborListPolicy:
    """Count-based policy for choosing between dense and cell-list.

    Carries both the strategy gate and its tunable thresholds. With the
    default ``strategy = AUTO``, :meth:`choose` returns cell-list once the
    average particles-per-system reaches
    ``min_avg_particles_per_system_for_cell_list`` and dense otherwise — the
    rough break-even under ordinary molecular densities. ``DENSE`` and
    ``CELL_LIST`` short-circuit the heuristic and force the corresponding
    implementation; useful for debugging, benchmarking, and reproducibility.

    Attributes:
        strategy: ``AUTO`` consults the count threshold; ``DENSE`` and
            ``CELL_LIST`` force the matching concrete implementation.
        min_avg_particles_per_system_for_cell_list: Average particles per
            system at or above which cell-list is chosen under ``AUTO``.
    """

    strategy: CutoffNeighborListStrategy = field(
        static=True, default=CutoffNeighborListStrategy.AUTO
    )
    min_avg_particles_per_system_for_cell_list: int = field(static=True, default=10_000)

    def choose(
        self, num_particles: int, num_systems: int
    ) -> CutoffNeighborListStrategy:
        """Return the chosen concrete strategy (never ``AUTO``)."""
        if self.strategy is not CutoffNeighborListStrategy.AUTO:
            return self.strategy
        if num_systems <= 0:
            return CutoffNeighborListStrategy.DENSE
        avg = num_particles / num_systems
        if avg >= self.min_avg_particles_per_system_for_cell_list:
            return CutoffNeighborListStrategy.CELL_LIST
        return CutoffNeighborListStrategy.DENSE


def adaptive_cutoff_neighborlist_from_state(
    state: IsAdaptiveCutoffNeighborListState[IsUniversalNeighborlistParams],
    cutoffs: Table[SystemId, Array],
    *,
    policy: CutoffNeighborListPolicy = CutoffNeighborListPolicy(),
) -> NeighborList[Literal[2]]:
    """Construct a cutoff-bound neighbor list using ``policy``.

    Returns a concrete :class:`DenseNearestNeighborList` or
    :class:`CellListNeighborList`. The returned object runs exactly one
    algorithm; tracing it does not compile both branches.

    Args:
        state: A state exposing ``neighborlist_params`` plus particle/system
            counts via ``state.particles.size`` and ``state.systems.size``.
        cutoffs: Per-system cutoff table bound onto the returned neighbor list.
        policy: Gate + thresholds. Construct
            ``CutoffNeighborListPolicy(strategy=DENSE)`` (or ``CELL_LIST``)
            to force a specific implementation.

    Returns:
        A :class:`NeighborList[Literal[2]]` that runs the chosen algorithm.
    """
    strategy = policy.choose(state.particles.size, state.systems.size)
    match strategy:
        case CutoffNeighborListStrategy.DENSE:
            return DenseNearestNeighborList.from_state(state, cutoffs)
        case CutoffNeighborListStrategy.CELL_LIST:
            return CellListNeighborList.from_state(state, cutoffs)
        case _:
            raise ValueError(f"unreachable strategy: {strategy}")
