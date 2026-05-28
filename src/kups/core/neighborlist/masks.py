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
    """Drops candidates whose lh/rh indices fall outside the valid inclusion-segment range.

    Implements the per-side ``inclusion.indices < num_labels`` check used to
    guard scatter/gather lookups when the candidate buffer is padded.
    """

    def __call__[D: int](self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        ngraphs = ctx.lh.data.inclusion.num_labels
        lh_inclusions = ctx.lh.map_data(lambda d: d.inclusion.indices < ngraphs)
        if ctx.rh is None:
            idx = batch.edges.indices.indices_in(lh_inclusions.keys)
            edge_in = lh_inclusions.data.at[idx].get(mode="fill", fill_value=False)
            return edge_in.all(axis=-1)

        rh_inclusions = ctx.rh.map_data(lambda d: d.inclusion.indices < ngraphs)
        lh_idx = batch.lh_idx.indices_in(lh_inclusions.keys)
        rh_idx = batch.rh_idx.indices_in(rh_inclusions.keys)
        lh_in = lh_inclusions.data.at[lh_idx].get(mode="fill", fill_value=False)
        rh_in = rh_inclusions.data.at[rh_idx].get(mode="fill", fill_value=False)
        return lh_in & rh_in


@dataclass
class InclusionMatchMask:
    """Drops candidates whose lh/rh inclusion segments differ."""

    def __call__[D: int](self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        if ctx.rh is None:
            edge_incl = ctx.lh[batch.edges.indices].inclusion.indices
            return (edge_incl == edge_incl[:, :1]).all(axis=-1)

        lh_incl, rh_incl = Index.match(
            ctx.lh[batch.lh_idx].inclusion, ctx.rh[batch.rh_idx].inclusion
        )
        return lh_incl == rh_incl


@dataclass
class ForIndicesDedupMask:
    """Deduplicate self-graph update candidates.

    Pair selectors emit candidates in ``lh`` space. When ``ctx.for_indices``
    is set, the rhs query side was restricted to those affected ``lh`` rows.
    We keep edges whose lhs endpoint is unaffected, plus one orientation for
    edges where both endpoints are affected. ``MirrorPairEdges`` restores the
    reverse orientation after compaction.

    Returns all-True for full self-graphs and bipartite queries.
    """

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        if ctx.for_indices is None:
            return jnp.ones((batch.lh_idx.size,), dtype=bool)
        return ~isin(batch.lh_idx.indices, ctx.for_indices, ctx.lh.size) | (
            batch.lh_idx.indices >= batch.rh_idx.indices
        )


@dataclass
class TouchesForIndicesMask[D: int]:
    """Keep fixed-topology rows touched by ``ctx.for_indices``.

    When no affected-index subset is active, every row is kept. This lets fixed
    topology use the same pipeline for full and patch-shaped calls.
    """

    def __call__(self, batch: CandidateBatch[D], ctx: PipelineContext) -> Array:
        if ctx.for_indices is None:
            return jnp.ones((len(batch.edges),), dtype=bool)
        return isin(batch.edges.indices.indices, ctx.for_indices, ctx.lh.size).any(-1)


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
        lh_system = ctx.lh[batch.lh_idx].system
        frames: MaterializedFrame = frame_table[lh_system]
        if ctx.rh is None:
            pair_positions = ctx.lh[batch.edges.indices].positions
            lhs_positions = pair_positions[:, 0]
            rhs_positions = pair_positions[:, 1]
        else:
            lhs_positions = ctx.lh[batch.lh_idx].positions
            rhs_positions = ctx.rh[batch.rh_idx].positions
        dist_sq = real_distance_sq(lhs_positions, rhs_positions, frames, shifts)
        return dist_sq < cutoffs[lh_system] ** 2


@dataclass
class ExclusionMask:
    """Drops minimum-image pairs that share an exclusion segment.

    Non-minimum-image periodic copies of excluded pairs survive (allowed when
    ``batch.is_minimum_image`` is False for that copy).
    """

    def __call__(
        self, batch: CandidateBatch[Literal[2]], ctx: PipelineContext
    ) -> Array:
        if ctx.rh is None:
            edge_excl = ctx.lh[batch.edges.indices].exclusion.indices
            return (edge_excl[:, 0] != edge_excl[:, 1]) | ~batch.is_minimum_image

        lh_excl, rh_excl = Index.match(
            ctx.lh[batch.lh_idx].exclusion, ctx.rh[batch.rh_idx].exclusion
        )
        return (lh_excl != rh_excl) | ~batch.is_minimum_image
