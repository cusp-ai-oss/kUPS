# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Protocols for the neighbor list module.

Defines the core
[`NeighborList`][kups.core.neighborlist.types.NeighborList]
call signature, the particle and system trait protocols expected by every
implementation, the
[`Mask`][kups.core.neighborlist.types.Mask] /
[`Compactor`][kups.core.neighborlist.types.Compactor] /
[`Postprocessor`][kups.core.neighborlist.types.Postprocessor] /
[`CandidateSelector`][kups.core.neighborlist.types.CandidateSelector]
protocols that compose into a
[`Pipeline`][kups.core.neighborlist.pipeline.Pipeline], the
[`CandidateBatch`][kups.core.neighborlist.types.CandidateBatch]
dataclass carried between phases, and the
[`IsNeighborListState`][kups.core.neighborlist.types.IsNeighborListState]
protocol used by the ``from_state`` constructors.
"""

from __future__ import annotations

from typing import Literal, Protocol, overload, runtime_checkable

from jax import Array

from kups.core.capacity import Capacity
from kups.core.cell import AnyPeriodicity
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
from kups.core.utils.jax import dataclass, skip_post_init_if_disabled


@runtime_checkable
class NeighborListPoints(
    HasPositions,
    HasSystemIndex,
    HasInclusionIndex,
    HasExclusionIndex,
    Protocol,
): ...


class NeighborListSystems(HasCell[AnyPeriodicity], Protocol): ...


class NeighborList[D: int](Protocol):
    """Protocol for neighbor list construction algorithms.

    Implementations find groups of particles within a cutoff distance, handling
    periodic boundary conditions and inclusion/exclusion masks. The degree
    parameter tracks the arity of the emitted edge tuples.
    """

    @overload
    def __call__[P: NeighborListPoints](
        self,
        keys: Table[ParticleId, P],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, P],
    ) -> Edges[D]: ...
    @overload
    def __call__[P: NeighborListPoints](
        self,
        keys: Table[ParticleId, P],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[D]: ...
    def __call__[P: NeighborListPoints](
        self,
        keys: Table[ParticleId, P],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, P] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        """Find particle groups for a self-graph or bipartite query.

        ``queries`` and ``queried_keys`` are mutually exclusive; the overloads
        above reject calls that pass both.

        Args:
            keys: Particle table that the returned ``Edges`` index into. With
                neither ``queries`` nor ``queried_keys``, the neighbor list
                builds a self-graph over ``keys``.
            systems: Indexed system data with cell information.
            queries: Optional bipartite query table. Each edge then connects a
                ``keys`` particle to a ``queries`` particle.
            queried_keys: Optional subset of ``keys`` particle ids whose
                incident self-graph edges should be returned. The caller must
                already have written any updated particle data into ``keys``.

        Returns:
            Edges whose columns index ``keys`` for self-graph calls.
        """
        ...


@dataclass
class CandidateBatch[D: int]:
    """Candidate set of degree ``D`` carried through the pipeline.

    Reuses [`Edges[D]`][kups.core.neighborlist.edges.Edges] for the
    `(indices, shifts)` layout (`indices` shape `(n, D)`,
    `shifts` shape `(n, D-1, 3)`); adds the
    ``is_minimum_image`` flag that
    [`ExclusionMask`][kups.core.neighborlist.masks.ExclusionMask] needs to
    keep non-minimum periodic copies of excluded pairs.

    Attributes:
        edges: Candidate edges (indices + fractional shifts), keyed by
            [`PipelineContext.edge_keys`][kups.core.neighborlist.types.PipelineContext.edge_keys].
        is_minimum_image: ``(n,)`` bool — True where the candidate's shift
            equals the minimum-image shift; False for non-MIC replicated
            copies emitted by selectors that handle PBC image expansion.
    """

    edges: Edges[D]
    is_minimum_image: Array

    @property
    def key_idx(self) -> Index[ParticleId]:
        """Pair-specific: key-side index of shape ``(n,)``. Only meaningful for ``D == 2``."""
        return self.edges.indices[:, 0]

    @property
    def query_idx(self) -> Index[ParticleId]:
        """Pair-specific: query-side index of shape ``(n,)``. Only meaningful for ``D == 2``."""
        return self.edges.indices[:, 1]


@dataclass
class PipelineContext:
    """Read-only inputs shared by every mask and the compactor.

    Positions in ``keys`` and optional ``queries`` are in **fractional**
    coordinates (transformed by
    [`_prepare`][kups.core.neighborlist.pipeline._prepare]).

    Edges index ``keys``, or for bipartite queries the union of ``keys`` and
    ``queries``, whose particle ids must then be disjoint.

    Attributes:
        keys: Key particle table in fractional coords.
        queries: Query particle table in fractional coords for true bipartite
            queries. ``None`` for full self-graphs and ``queried_keys`` updates.
        systems: Indexed system data with cell information.
        queried_keys: ``keys`` ids to update/query, or ``None`` for a full
            self-graph or bipartite query.
    """

    keys: Table[ParticleId, NeighborListPoints]
    queries: Table[ParticleId, NeighborListPoints] | None
    systems: Table[SystemId, NeighborListSystems]
    queried_keys: Index[ParticleId] | None

    @skip_post_init_if_disabled
    def __post_init__(self) -> None:
        assert self.queries is None or self.queried_keys is None, (
            "PipelineContext cannot combine queries with queried_keys."
        )
        assert self.queries is None or set(self.keys.keys).isdisjoint(
            self.queries.keys
        ), "Bipartite queries need particle ids disjoint from the keys."

    @property
    def edge_keys(self) -> tuple[ParticleId, ...]:
        """Particle ids addressed by candidate and edge indices."""
        if self.queries is None:
            return self.keys.keys
        return tuple(sorted(self.keys.keys + self.queries.keys))

    @property
    def edge_query_table(self) -> Table[ParticleId, NeighborListPoints]:
        """Table holding the second endpoint of each pair."""
        return self.queries if self.queries is not None else self.keys

    @property
    def query_table(self) -> Table[ParticleId, NeighborListPoints]:
        """Table used to enumerate query candidates.

        ``queries`` is reserved for true bipartite calls. ``queried_keys``
        selects a self-graph update subset from ``keys`` and is lifted back to
        ``keys`` ids before masks and compaction see the batch.
        """
        if self.queries is not None:
            return self.queries
        if self.queried_keys is None:
            return self.keys
        return self.keys.subset(self.queried_keys)

    def endpoints[D: int](
        self, batch: CandidateBatch[D]
    ) -> tuple[NeighborListPoints, NeighborListPoints]:
        """Key and query rows of each candidate pair."""
        query = self.edge_query_table
        return (
            self.keys[batch.key_idx.update_labels(self.keys.keys, allow_missing=True)],
            query[batch.query_idx.update_labels(query.keys, allow_missing=True)],
        )


class CandidateSelector[D: int](Protocol):
    """Produces a ``CandidateBatch[D]`` from the pipeline context.

    Owns all candidate-set construction, including any PBC image
    replication required when ``max(cutoff/perp_axis) > 0.5``.
    """

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[D]: ...


@runtime_checkable
class SelectableNeighborList[D: int](NeighborList[D], Protocol):
    """Neighbor list whose selector can be reused with different pipeline masks.

    This additional capability lets pair evaluators retain image metadata and
    defer group masks that differ between terms. Ordinary ``NeighborList``
    implementations need only provide their existing edge-producing call.
    """

    @property
    def cutoffs(self) -> Table[SystemId, Array]: ...

    @property
    def avg_edges(self) -> Capacity[int]: ...

    def selector(
        self, query_size: int, systems: Table[SystemId, NeighborListSystems]
    ) -> CandidateSelector[D]: ...


class Mask[D: int](Protocol):
    """Returns this criterion's bool array; pipeline conjuncts the results.

    The degree parameter tracks which candidate arity the mask accepts. Pair
    masks implement ``Mask[Literal[2]]`` by annotating their ``batch`` argument
    as ``CandidateBatch[Literal[2]]``; degree-agnostic masks use a generic
    ``__call__`` method.

    Cannot change ``batch.edges``, ``batch.is_minimum_image``, or the
    candidate count. Pure ``(batch, ctx) -> Array``.
    """

    def __call__(self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array: ...


class Compactor[D: int](Protocol):
    """Produces compacted ``Edges[D]`` from the accumulated ``keep`` mask."""

    def __call__(
        self, keep: Array, batch: CandidateBatch[D], ctx: PipelineContext
    ) -> Edges[D]: ...


class Postprocessor[D: int](Protocol):
    """Transforms compacted edges using the pipeline context.

    Postprocessors run sequentially after compaction. They may change the
    number of rows, but must preserve the edge degree ``D``.
    """

    def __call__(self, edges: Edges[D], ctx: PipelineContext) -> Edges[D]: ...


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


class NeighborListFactory[State](Protocol):
    """Constructs a pair :class:`NeighborList` for a given state and cutoffs.

    Used by radius-based potential factories so the construction strategy
    can be swapped without coupling the potential to a concrete
    neighbor-list class. The library default is
    :meth:`kups.core.neighborlist.AdaptiveNeighborList.from_state`,
    which is contravariant-compatible with any state satisfying
    :class:`IsNeighborListState`.

    The ``State`` type parameter is contravariant (it appears only in
    input position in ``__call__``), so a factory written against a
    broader state protocol can be passed where a narrower one is expected.
    """

    def __call__(
        self,
        state: State,
        cutoffs: Table[SystemId, Array],
    ) -> NeighborList[Literal[2]]: ...


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
