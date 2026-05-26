# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Protocols for the neighbor list module.

Defines the core
[`NearestNeighborList`][kups.core.neighborlist.types.NearestNeighborList]
call signature, the particle and system trait protocols expected by every
implementation, the
[`Mask`][kups.core.neighborlist.types.Mask] /
[`Compactor`][kups.core.neighborlist.types.Compactor] /
[`CandidateSelector`][kups.core.neighborlist.types.CandidateSelector]
protocols that compose into a
[`Pipeline`][kups.core.neighborlist.pipeline.Pipeline], the
[`CandidateBatch`][kups.core.neighborlist.types.CandidateBatch]
NamedTuple carried between phases, and the
[`IsNeighborListState`][kups.core.neighborlist.types.IsNeighborListState]
protocol used by the ``from_state`` constructors.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Protocol

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
from kups.core.utils.jax import dataclass


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


class CandidateBatch[D: int](NamedTuple):
    """Candidate set of degree ``D`` carried through the pipeline.

    Reuses [`Edges[D]`][kups.core.neighborlist.edges.Edges] for the
    `(indices, shifts)` layout (`indices` shape `(n, D)`,
    `shifts` shape `(n, D-1, 3)`); adds the
    ``is_minimum_image`` flag that
    [`ExclusionMask`][kups.core.neighborlist.masks.ExclusionMask] needs to
    keep non-minimum periodic copies of excluded pairs.

    Auto-registered as a JAX PyTree because it is a NamedTuple.

    Attributes:
        edges: Candidate edges (indices + fractional shifts).
        is_minimum_image: ``(n,)`` bool — True where the candidate's shift
            equals the minimum-image shift; False for non-MIC replicated
            copies emitted by selectors that handle PBC image expansion.
    """

    edges: Edges[D]
    is_minimum_image: Array

    @property
    def lh_idx(self) -> Array:
        """Pair-specific: raw lh-side index array of shape ``(n,)``. Only meaningful for ``D == 2``."""
        return self.edges.indices.indices[:, 0]

    @property
    def rh_idx(self) -> Array:
        """Pair-specific: raw rh-side index array of shape ``(n,)``. Only meaningful for ``D == 2``."""
        return self.edges.indices.indices[:, 1]


@dataclass
class PipelineContext:
    """Read-only inputs shared by every mask and the compactor.

    Positions in ``lh`` and ``rh`` are in **fractional** coordinates
    (transformed by [`_prepare`][kups.core.neighborlist.pipeline._prepare]).
    There is no ``out_of_bounds`` field — masks/compactors that need an
    OOB sentinel compute ``max(ctx.lh.size, ctx.rh.size)`` locally.

    Attributes:
        lh: Left-hand particle table in fractional coords.
        rh: Right-hand particle table in fractional coords (== ``lh`` when
            the caller passed ``rh=None``).
        systems: Indexed system data with cell information.
        rh_index_remap: Raw remap array mapping rh-positions to lh-space
            particle IDs, or ``None`` when no remap was supplied. Empty
            remaps are replaced with a one-element OOB-sentinel array by
            ``_prepare`` so downstream lookups never see a zero-length array.
    """

    lh: Table[ParticleId, NeighborListPoints]
    rh: Table[ParticleId, NeighborListPoints]
    systems: Table[SystemId, NeighborListSystems]
    rh_index_remap: Array | None


class CandidateSelector[D: int](Protocol):
    """Produces a ``CandidateBatch[D]`` from the pipeline context.

    Owns all candidate-set construction, including any PBC image
    replication required when ``max(cutoff/perp_axis) > 0.5``.
    """

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[D]: ...


class Mask(Protocol):
    """Returns this criterion's bool array; pipeline conjuncts the results.

    Degree-agnostic at the type level — the pair-only masks shipped here
    annotate ``batch: CandidateBatch`` (any ``D``) and internally assume
    ``D == 2`` via ``batch.lh_idx`` / ``batch.rh_idx``. Higher-degree masks
    would not use those properties.

    Cannot change ``batch.edges``, ``batch.is_minimum_image``, or the
    candidate count. Pure ``(batch, ctx) -> Array``.
    """

    def __call__(self, batch: CandidateBatch, ctx: PipelineContext) -> Array: ...


class Compactor[D: int](Protocol):
    """Produces final ``Edges[D]`` from the accumulated ``keep`` mask."""

    def __call__(
        self, keep: Array, batch: CandidateBatch[D], ctx: PipelineContext
    ) -> Edges[D]: ...


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
