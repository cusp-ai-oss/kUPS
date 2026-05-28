# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pipeline runner: selector → mask sequence → compactor → postprocessors.

A [`Pipeline[D]`][kups.core.neighborlist.pipeline.Pipeline] is the
modularized form of a neighbor list. Concrete public NL classes
(``CellListNeighborList``, ``DenseNearestNeighborList``, etc.) build and run
a ``Pipeline`` internally inside their ``__call__``.
"""

from __future__ import annotations

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

    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        rh: Table[ParticleId, NeighborListPoints] | None = None,
        for_indices: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        ctx = _prepare(lh, rh, systems, for_indices)
        batch = self.selector(ctx)
        keep = jnp.ones((len(batch.edges),), dtype=bool)
        for mask in self.masks:
            keep &= mask(batch, ctx)
        edges = self.compactor(keep, batch, ctx)
        for postprocessor in self.postprocessors:
            edges = postprocessor(edges, ctx)
        return edges


def _prepare(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints] | None,
    systems: Table[SystemId, NeighborListSystems],
    for_indices: Index[ParticleId] | None,
) -> PipelineContext:
    """Transform positions to fractional coords and resolve ``for_indices``."""
    assert rh is None or for_indices is None, (
        "Neighbor-list calls cannot combine rh with for_indices. "
        "Use for_indices for self-graph updates, or rh for bipartite queries."
    )
    frames = systems.map_data(lambda s: s.cell.frame.materialize())
    lh_frame = frames[lh.data.system]
    lh = bind(lh, lambda x: x.data.positions).apply(lh_frame.to_fractional)
    if rh is not None:
        rh_frame = frames[rh.data.system]
        rh = bind(rh, lambda x: x.data.positions).apply(rh_frame.to_fractional)

    for_indices_raw: Array | None = (
        for_indices.indices_in(lh.keys) if for_indices is not None else None
    )

    return PipelineContext(
        lh=lh,
        rh=rh,
        systems=systems,
        for_indices=for_indices_raw,
    )
