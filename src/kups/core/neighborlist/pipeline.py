# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pipeline runner: selector → mask sequence → compactor → postprocessors.

A [`Pipeline[D]`][kups.core.neighborlist.pipeline.Pipeline] is the
modularized form of a neighbor list. Concrete public NL classes
(``CellListNeighborList``, ``DenseNearestNeighborList``, etc.) build and run
a ``Pipeline`` internally inside their ``__call__``.
"""

from __future__ import annotations

from typing import overload

import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import (
    CandidateSelector,
    Compactor,
    Mask,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
    Postprocessor,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field


@dataclass
class Pipeline[D: int]:
    """Selector → mask sequence → compactor → postprocessors.

    Attributes:
        selector: Produces a ``CandidateBatch[D]`` (handles PBC replication).
        masks: Tuple of mask criteria over ``CandidateBatch[D]``; results
            are conjuncted via ``&``.
        compactor: Produces compacted ``Edges[D]`` from the accumulated mask.
        postprocessors: Edge transforms applied sequentially after compaction.
    """

    selector: CandidateSelector[D]
    masks: tuple[Mask[D], ...] = field(static=True)
    compactor: Compactor[D]
    postprocessors: tuple[Postprocessor[D], ...] = field(default=(), static=True)

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
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        ctx = _prepare(keys, queries, systems, queried_keys)
        batch = self.selector(ctx)
        keep = jnp.ones((len(batch.edges),), dtype=bool)
        for mask in self.masks:
            keep &= mask(batch, ctx)
        edges = self.compactor(keep, batch, ctx)
        for postprocessor in self.postprocessors:
            edges = postprocessor(edges, ctx)
        return edges


def _prepare(
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints] | None,
    systems: Table[SystemId, NeighborListSystems],
    queried_keys: Index[ParticleId] | None,
) -> PipelineContext:
    """Transform positions to fractional coords and resolve ``queried_keys``."""
    assert queries is None or queried_keys is None, (
        "Neighbor-list calls cannot combine queries with queried_keys. "
        "Use queried_keys for self-graph updates, or queries for bipartite queries."
    )
    frames = systems.map_data(lambda s: s.cell.frame.materialize())
    keys_frame = frames[keys.data.system]
    keys = bind(keys, lambda x: x.data.positions).apply(keys_frame.to_fractional)
    if queries is not None:
        queries_frame = frames[queries.data.system]
        queries = bind(queries, lambda x: x.data.positions).apply(
            queries_frame.to_fractional
        )

    queried_keys_raw: Array | None = (
        queried_keys.indices_in(keys.keys) if queried_keys is not None else None
    )

    return PipelineContext(
        keys=keys,
        queries=queries,
        systems=systems,
        queried_keys=queried_keys_raw,
    )
