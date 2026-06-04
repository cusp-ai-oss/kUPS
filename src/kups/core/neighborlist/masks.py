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

from kups.core.cell import MaterializedFrame
from kups.core.data import Index, Table
from kups.core.neighborlist.common import real_distance_sq
from kups.core.neighborlist.types import CandidateBatch, PipelineContext
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, isin


@dataclass
class InBoundsMask:
    """Drops candidates whose key/query indices fall outside the valid inclusion-segment range.

    Implements the per-side ``inclusion.indices < num_labels`` check used to
    guard scatter/gather lookups when the candidate buffer is padded.
    """

    def __call__[D: int](self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        ngraphs = ctx.keys.data.inclusion.num_labels
        key_inclusions = ctx.keys.map_data(lambda d: d.inclusion.indices < ngraphs)
        if ctx.queries is None:
            idx = batch.edges.indices.indices_in(key_inclusions.keys)
            edge_in = key_inclusions.data.at[idx].get(mode="fill", fill_value=False)
            return edge_in.all(axis=-1)

        query_inclusions = ctx.queries.map_data(lambda d: d.inclusion.indices < ngraphs)
        key_idx = batch.key_idx.indices_in(key_inclusions.keys)
        query_idx = batch.query_idx.indices_in(query_inclusions.keys)
        key_in = key_inclusions.data.at[key_idx].get(mode="fill", fill_value=False)
        query_in = query_inclusions.data.at[query_idx].get(
            mode="fill", fill_value=False
        )
        return key_in & query_in


@dataclass
class InclusionMatchMask:
    """Drops candidates whose key/query inclusion segments differ."""

    def __call__[D: int](self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        if ctx.queries is None:
            edge_incl = ctx.keys[batch.edges.indices].inclusion.indices
            return (edge_incl == edge_incl[:, :1]).all(axis=-1)

        key_incl, query_incl = Index.match(
            ctx.keys[batch.key_idx].inclusion, ctx.queries[batch.query_idx].inclusion
        )
        return key_incl == query_incl


@dataclass
class QueriedKeysDedupMask:
    """Deduplicate self-graph update candidates.

    Pair selectors emit candidates in ``keys`` space. When ``ctx.queried_keys``
    is set, the query side was restricted to those affected ``keys`` rows.
    We keep edges whose key endpoint is unaffected, plus one orientation for
    edges where both endpoints are affected. ``MirrorPairEdges`` restores the
    reverse orientation after compaction.

    Returns all-True for full self-graphs and bipartite queries.
    """

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        if ctx.queried_keys is None:
            return jnp.ones((batch.key_idx.size,), dtype=bool)
        return ~isin(batch.key_idx.indices, ctx.queried_keys, ctx.keys.size) | (
            batch.key_idx.indices >= batch.query_idx.indices
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
        return isin(batch.edges.indices.indices, ctx.queried_keys, ctx.keys.size).any(
            -1
        )


@dataclass
class DistanceCutoffMask:
    """Drops candidates whose squared real-space distance exceeds ``cutoff²``."""

    cutoffs: Table[SystemId, Array]

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        cutoffs = Table.broadcast_to(self.cutoffs, ctx.systems)
        shifts = batch.edges.shifts[:, 0, :]
        frame_table: Table[SystemId, MaterializedFrame] = ctx.systems.map_data(
            lambda s: s.cell.frame.materialize()
        )
        key_system = ctx.keys[batch.key_idx].system
        frames: MaterializedFrame = frame_table[key_system]
        if ctx.queries is None:
            pair_positions = ctx.keys[batch.edges.indices].positions
            key_positions = pair_positions[:, 0]
            query_positions = pair_positions[:, 1]
        else:
            key_positions = ctx.keys[batch.key_idx].positions
            query_positions = ctx.queries[batch.query_idx].positions
        dist_sq = real_distance_sq(key_positions, query_positions, frames, shifts)
        return dist_sq < cutoffs[key_system] ** 2


@dataclass
class ExclusionMask:
    """Drops minimum-image pairs that share an exclusion segment.

    Non-minimum-image periodic copies of excluded pairs survive (allowed when
    ``batch.is_minimum_image`` is False for that copy).
    """

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        if ctx.queries is None:
            edge_excl = ctx.keys[batch.edges.indices].exclusion.indices
            return (edge_excl[:, 0] != edge_excl[:, 1]) | ~batch.is_minimum_image

        key_excl, query_excl = Index.match(
            ctx.keys[batch.key_idx].exclusion, ctx.queries[batch.query_idx].exclusion
        )
        return (key_excl != query_excl) | ~batch.is_minimum_image
