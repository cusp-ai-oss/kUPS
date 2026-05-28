# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Empty and fixed-edge :class:`NeighborList` implementations.

These cover the cases that live today inline inside
:class:`PointCloudConstructor` (always-empty edge set) and
:class:`EdgeSetGraphConstructor` (state-provided edge topology, optionally
filtered to a patch-affected subset). They satisfy the standard
:class:`NeighborList[D]` protocol so a unified graph constructor can ask
any neighbor list for edges, regardless of how those edges were obtained.

``FixedEdgesNeighborList`` also supports the patch-shaped neighbor-list call
used by graph constructors: when ``rh`` is supplied it requires
``rh_index_remap`` and returns only fixed topology rows touched by those
remapped particle ids. Its implementation is a normal selector -> mask ->
compactor pipeline over the fixed edge rows; shifts are computed from the
current particle positions during selection.
"""

from __future__ import annotations

import jax.numpy as jnp

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.data import Index, Table
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import EdgeInRhMask
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
        indices = self.indices.update_labels(ctx.lh.keys).indices
        positions = ctx.lh.data.positions.at[indices].get(mode="fill", fill_value=0)
        deltas = positions[:, :1] - positions[:, 1:]
        shifts = ctx.systems.data.cell.minimum_image_shifts(deltas)
        return CandidateBatch(
            edges=Edges(Index(ctx.lh.keys, indices), shifts),
            is_minimum_image=jnp.ones((len(self.indices),), dtype=bool),
        )


@dataclass
class EmptyNeighborList[D: int]:
    """Neighbor list that emits an :class:`Edges[D]` with zero rows.

    Replaces the inline empty-edges construction inside
    :class:`PointCloudConstructor`. The ``degree`` field is the runtime
    arity carried by the emitted edges; it must match the type parameter
    ``D``.

    Attributes:
        degree: Edge arity (``Literal[0]`` for point clouds, higher for
            unified graph constructors that need a degree-aware empty NL).
    """

    degree: int = field(static=True, default=0)

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        del rh, systems, rh_index_remap
        shift_inner = self.degree - 1 if self.degree > 1 else 0
        return Edges(
            indices=Index(lh.keys, jnp.zeros((0, self.degree), dtype=int)),
            shifts=jnp.zeros((0, shift_inner, 3), dtype=int),
        )


@dataclass
class FixedEdgesNeighborList[D: int]:
    """Neighbor list for a fixed topology edge set.

    Replaces the full-graph path of :class:`EdgeSetGraphConstructor` and can
    also serve its patch path. Full calls return all fixed topology rows with
    shifts computed from the current particle positions. Patch calls pass
    ``rh`` and ``rh_index_remap``; the remap identifies changed particle ids,
    and the neighbor list returns only fixed rows touched by those particles.

    Attributes:
        indices: Fixed edge topology. Shifts are intentionally not stored;
            they are computed from the call's current particle positions.
        avg_edges: Average affected-edge capacity per patched right-hand row.
            If not provided, defaults to the full edge-buffer size.
    """

    indices: Index[ParticleId]
    avg_edges: Capacity[int] | None = None

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[D]:
        if rh is None:
            max_edges = FixedCapacity(len(self.indices))
            remap = None
        else:
            assert rh_index_remap is not None, (
                "FixedEdgesNeighborList requires rh_index_remap when rh is provided."
            )
            max_edges = (
                self.avg_edges.multiply(rh.size)
                if self.avg_edges is not None
                else FixedCapacity(len(self.indices))
            )
            remap = rh_index_remap

        pipeline = Pipeline[D](
            selector=_FixedEdgesSelector(self.indices),
            masks=(EdgeInRhMask(),),
            compactor=ReduceCompactor(max_edges, remap_rh=False, mirror_on_remap=False),
        )
        return pipeline(lh, rh, systems, remap)
