# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Empty and fixed-edge :class:`NeighborList` implementations.

These cover graph-construction cases without cutoff search: always-empty
edge sets for point clouds and state-provided fixed edge topology for bonded
interactions, optionally filtered to a patch-affected subset. They satisfy the
standard :class:`NeighborList[D]` protocol so a unified graph constructor can
ask any neighbor list for edges, regardless of how those edges were obtained.

The public call contract treats ``keys`` as the self-graph/output table.
``queries`` is keyword-only and reserved for true bipartite neighbor-list
queries; fixed topology is a self-graph implementation and does not use
``queries``. ``queried_keys`` is keyword-only, mutually exclusive with
``queries``, and names affected ``keys`` ids after the caller has already
written updated particle data into ``keys``.
``FixedEdgesNeighborList`` returns only fixed topology rows touched by those
ids. Its implementation is a normal selector -> mask -> compactor pipeline over
the fixed edge rows; shifts are computed from the current particle positions
during selection.
"""

from __future__ import annotations

from typing import overload

import jax.numpy as jnp

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.data import Index, Table
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import TouchesQueriedKeysMask
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.types import (
    CandidateBatch,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, jit


@dataclass
class _FixedEdgesSelector[D: int]:
    """Selector that turns stored fixed topology into current edge candidates."""

    indices: Index[ParticleId]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[D]:
        indices = self.indices.update_labels(ctx.keys.keys).indices
        positions = ctx.keys.data.positions.at[indices].get(mode="fill", fill_value=0)
        deltas = positions[:, :1] - positions[:, 1:]
        shifts = ctx.systems.data.cell.minimum_image_shifts(deltas)
        return CandidateBatch(
            edges=Edges(Index(ctx.keys.keys, indices), shifts),
            is_minimum_image=jnp.ones((len(self.indices),), dtype=bool),
        )


@dataclass
class EmptyNeighborList[D: int]:
    """Neighbor list that emits an :class:`Edges[D]` with zero rows.

    The ``degree`` field is the runtime arity carried by the emitted edges;
    it must match the type parameter ``D``.

    Attributes:
        degree: Edge arity (``Literal[0]`` for point clouds, higher for
            unified graph constructors that need a degree-aware empty NL).
    """

    degree: int = field(static=True, default=0)

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> Edges[D]: ...

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[D]: ...

    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        assert queries is None or queried_keys is None, (
            "Neighbor-list calls cannot combine queries with queried_keys."
        )
        del queries, systems, queried_keys
        shift_inner = self.degree - 1 if self.degree > 1 else 0
        return Edges(
            indices=Index(keys.keys, jnp.zeros((0, self.degree), dtype=int)),
            shifts=jnp.zeros((0, shift_inner, 3), dtype=int),
        )


@dataclass
class FixedEdgesNeighborList[D: int]:
    """Neighbor list for a fixed topology edge set.

    Full self-graph calls return all fixed topology rows with shifts computed
    from the current particle positions. Affected self-graph calls pass
    keyword-only ``queried_keys`` after updated particle data has been written
    into ``keys``; the neighbor list returns only fixed rows touched by those
    affected ``keys`` ids. ``queries`` is reserved for true bipartite
    neighbor-list implementations and is not a fixed-edge update mechanism.

    Attributes:
        indices: Fixed edge topology. Shifts are intentionally not stored;
            they are computed from the call's current particle positions.
        avg_edges: Update-only average affected-edge capacity per affected
            ``keys`` id. Full calls ignore this field and use the stored
            topology length; affected calls default to the full edge-buffer
            size when it is not provided.
    """

    indices: Index[ParticleId]
    avg_edges: Capacity[int] | None = None

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> Edges[D]: ...

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[D]: ...

    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        assert queries is None or queried_keys is None, (
            "Neighbor-list calls cannot combine queries with queried_keys."
        )
        assert queries is None, "FixedEdgesNeighborList only supports self-graph calls."
        if queried_keys is None:
            max_edges = FixedCapacity(len(self.indices))
        else:
            max_edges = (
                self.avg_edges.multiply(queried_keys.size)
                if self.avg_edges is not None
                else FixedCapacity(len(self.indices))
            )

        pipeline = Pipeline[D](
            selector=_FixedEdgesSelector(self.indices),
            masks=(TouchesQueriedKeysMask(),),
            compactor=ReduceCompactor(max_edges),
        )
        return pipeline(keys, systems, queried_keys=queried_keys)
