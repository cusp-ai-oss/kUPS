# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Mask classes for the neighbor list pipeline.

Each [`Mask`][kups.core.neighborlist.types.Mask] is a pure function of
``(batch, ctx)`` returning a fresh boolean array for its own criterion; the
[`Pipeline`][kups.core.neighborlist.pipeline.Pipeline] conjuncts all returned
masks via ``&``. Masks cannot change ``batch.edges`` or
``batch.is_minimum_image``. Pair-only masks express that restriction in
their ``CandidateBatch[Literal[2]]`` type annotation.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array

from kups.core import units
from kups.core.cell import MaterializedFrame
from kups.core.data import Index, Table
from kups.core.neighborlist.common import real_distance_sq
from kups.core.neighborlist.types import CandidateBatch, PipelineContext
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass


@dataclass
class InBoundsMask:
    """Drops candidates whose key/query indices fall outside the valid inclusion-segment range.

    Implements the per-side ``inclusion.valid_mask`` check used to
    guard scatter/gather lookups when the candidate buffer is padded.
    """

    def __call__[D: int](self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        if ctx.queries is None:
            return ctx.keys[batch.edges.indices].inclusion.valid_mask.all(axis=-1)
        key, query = ctx.endpoints(batch)
        return key.inclusion.valid_mask & query.inclusion.valid_mask


@dataclass
class InclusionMatchMask:
    """Drops candidates whose key/query inclusion segments differ."""

    def __call__[D: int](self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        if ctx.queries is None:
            edge_incl = ctx.keys[batch.edges.indices].inclusion
            incl, first_incl = Index.match(edge_incl, edge_incl[:, :1])
            return (incl == first_incl).all(axis=-1)

        key, query = ctx.endpoints(batch)
        key_incl, query_incl = Index.match(key.inclusion, query.inclusion)
        return key_incl == query_incl


@dataclass
class QueriedKeysDedupMask:
    """Deduplicate self-graph update candidates.

    Pair selectors emit candidates in ``keys`` space. When ``ctx.queried_keys``
    is set, the query side was restricted to those affected ``keys`` rows.
    We keep edges whose key endpoint is unaffected, plus one orientation for
    edges where both endpoints are affected: ``key > query``, or for self-image
    rows ``(i, i, ±s)`` the lexicographically positive shift.
    ``MirrorPairEdges`` restores the reverse orientation after compaction.

    Returns all-True for full self-graphs and bipartite queries.
    """

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        if ctx.queried_keys is None:
            return jnp.ones((batch.key_idx.size,), dtype=bool)
        key, query = Index.match(batch.key_idx, batch.query_idx)
        shift = batch.edges.shifts[:, 0, :]
        first_nonzero = jnp.take_along_axis(
            shift, jnp.argmax(shift != 0, axis=-1)[:, None], axis=-1
        )[:, 0]
        return (
            ~batch.key_idx.isin(ctx.queried_keys)
            | (key > query)
            | ((key == query) & (first_nonzero > 0))
        )


@dataclass
class TouchesQueriedKeysMask[D: int]:
    """Keep fixed-topology rows touched by ``ctx.queried_keys``.

    When no affected-index subset is active, every row is kept. This lets fixed
    topology use the same pipeline for full and patch-shaped calls.
    """

    def __call__(self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        if ctx.queried_keys is None:
            return jnp.ones((len(batch.edges),), dtype=bool)
        return batch.edges.indices.isin(ctx.queried_keys).any(-1)


@dataclass
class DistanceCutoffMask:
    """Drops candidates whose squared real-space distance exceeds ``cutoff²``."""

    cutoffs: Table[SystemId, units.Length]

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        cutoffs = Table.broadcast_to(self.cutoffs, ctx.systems)
        shifts = batch.edges.shifts[:, 0, :]
        frame_table: Table[SystemId, MaterializedFrame] = ctx.systems.map_data(
            lambda s: s.cell.frame.materialize()
        )
        key, query = ctx.endpoints(batch)
        frames: MaterializedFrame = frame_table[key.system]
        dist_sq = real_distance_sq(key.positions, query.positions, frames, shifts)
        return dist_sq < cutoffs[key.system] ** 2


@dataclass
class ExclusionMask:
    """Drops minimum-image pairs that share an exclusion segment.

    Non-minimum-image periodic copies of excluded pairs survive (allowed when
    ``batch.is_minimum_image`` is False for that copy).
    """

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        key, query = ctx.endpoints(batch)
        key_excl, query_excl = Index.match(key.exclusion, query.exclusion)
        return (key_excl != query_excl) | ~batch.is_minimum_image
