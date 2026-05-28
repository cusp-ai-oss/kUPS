# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pipeline runner: selector → mask sequence → compactor.

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
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field


@dataclass
class Pipeline[D: int]:
    """Selector → mask sequence → compactor.

    Attributes:
        selector: Produces a ``CandidateBatch[D]`` (handles PBC replication).
        masks: Tuple of mask criteria over ``CandidateBatch[D]``; results
            are conjuncted via ``&``.
        compactor: Produces the final ``Edges[D]`` from the accumulated mask.
    """

    selector: CandidateSelector[D]
    masks: tuple[Mask, ...] = field(static=True)
    compactor: Compactor[D]

    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        ctx = _prepare(lh, rh, systems, rh_index_remap)
        batch = self.selector(ctx)
        keep = jnp.ones((len(batch.edges),), dtype=bool)
        for mask in self.masks:
            keep &= mask(batch, ctx)
        return self.compactor(keep, batch, ctx)


def _prepare(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints] | None,
    systems: Table[SystemId, NeighborListSystems],
    rh_index_remap: Index[ParticleId] | None,
) -> PipelineContext:
    """Transform lh/rh positions to fractional coords, resolve the remap, build ctx."""
    if rh is None:
        rh = lh

    frames = systems.map_data(lambda s: s.cell.frame.materialize())
    lh_frame = frames[lh.data.system]
    lh = bind(lh, lambda x: x.data.positions).apply(lh_frame.to_fractional)
    rh_frame = frames[rh.data.system]
    rh = bind(rh, lambda x: x.data.positions).apply(rh_frame.to_fractional)

    rh_remap_raw: Array | None = (
        rh_index_remap.indices_in(lh.keys) if rh_index_remap is not None else None
    )
    if rh_remap_raw is not None and rh_remap_raw.size == 0:
        oob = max(lh.size, rh.size)
        rh_remap_raw = jnp.full((1,), oob, dtype=int)

    return PipelineContext(
        lh=lh,
        rh=rh,
        systems=systems,
        rh_index_remap=rh_remap_raw,
    )
