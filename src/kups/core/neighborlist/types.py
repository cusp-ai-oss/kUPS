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

from typing import Literal, Protocol, overload

from jax import Array

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
from kups.core.utils.jax import dataclass, field, skip_post_init_if_disabled


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
        edges: Candidate edges (indices + fractional shifts). The first column
            is keyed by ``edges.indices.keys``.
        is_minimum_image: ``(n,)`` bool — True where the candidate's shift
            equals the minimum-image shift; False for non-MIC replicated
            copies emitted by selectors that handle PBC image expansion.
        query_keys: Pair-specific key vocabulary for the second edge column.
            ``None`` means it uses ``edges.indices.keys``.
    """

    edges: Edges[D]
    is_minimum_image: Array
    query_keys: tuple[ParticleId, ...] | None = field(default=None, static=True)

    @property
    def key_idx(self) -> Index[ParticleId]:
        """Pair-specific: key-side index of shape ``(n,)``. Only meaningful for ``D == 2``."""
        return self.edges.indices[:, 0]

    @property
    def query_idx(self) -> Index[ParticleId]:
        """Pair-specific: query-side index of shape ``(n,)``. Only meaningful for ``D == 2``."""
        return Index(
            self.query_keys or self.edges.indices.keys,
            self.edges.indices.indices[:, 1],
        )


@dataclass
class PipelineContext:
    """Read-only inputs shared by every mask and the compactor.

    Positions in ``keys`` and optional ``queries`` are in **fractional**
    coordinates (transformed by
    [`_prepare`][kups.core.neighborlist.pipeline._prepare]). There is no
    ``out_of_bounds`` field — masks/compactors that need an OOB sentinel resolve
    the query table first.

    Attributes:
        keys: Key particle table in fractional coords (the table the returned
            ``Edges`` index into).
        queries: Query particle table in fractional coords for true bipartite
            queries. ``None`` for full self-graphs and ``queried_keys`` updates.
        systems: Indexed system data with cell information.
        queried_keys: Raw ``keys`` positions to update/query, or ``None`` for a
            full self-graph or bipartite query.
    """

    keys: Table[ParticleId, NeighborListPoints]
    queries: Table[ParticleId, NeighborListPoints] | None
    systems: Table[SystemId, NeighborListSystems]
    queried_keys: Array | None

    @skip_post_init_if_disabled
    def __post_init__(self) -> None:
        assert self.queries is None or self.queried_keys is None, (
            "PipelineContext cannot combine queries with queried_keys."
        )

    @property
    def edge_query_table(self) -> Table[ParticleId, NeighborListPoints]:
        """Table addressed by the second edge column."""
        return self.queries if self.queries is not None else self.keys

    @property
    def query_table(self) -> Table[ParticleId, NeighborListPoints]:
        """Table used to enumerate query candidates.

        ``queries`` is reserved for true bipartite calls. ``queried_keys``
        selects a self-graph update subset from ``keys`` and is lifted back to
        ``keys`` index space before masks and compaction see the batch.
        """
        if self.queries is not None:
            return self.queries
        if self.queried_keys is None:
            return self.keys
        return self.keys.subset(Index(self.keys.keys, self.queried_keys))


class CandidateSelector[D: int](Protocol):
    """Produces a ``CandidateBatch[D]`` from the pipeline context.

    Owns all candidate-set construction, including any PBC image
    replication required when ``max(cutoff/perp_axis) > 0.5``.
    """

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[D]: ...


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
    :func:`kups.core.neighborlist.adaptive_cutoff_neighborlist_from_state`,
    which is contravariant-compatible with any state satisfying
    :class:`IsAdaptiveCutoffNeighborListState`.

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
