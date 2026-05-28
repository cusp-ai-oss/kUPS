# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Mask classes for the neighbor list pipeline.

Each [`Mask`][kups.core.neighborlist.types.Mask] is a pure function of
``(batch, ctx)`` returning a fresh boolean array for its own criterion; the
[`Pipeline`][kups.core.neighborlist.pipeline.Pipeline] conjuncts all returned
masks via ``&``. Masks cannot change ``batch.edges`` or
``batch.is_minimum_image``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

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

    def __call__(self, batch: CandidateBatch, ctx: PipelineContext) -> Array:
        ngraphs = ctx.lh.data.inclusion.num_labels
        lh_in = (
            (ctx.lh.data.inclusion.indices < ngraphs)
            .at[batch.lh_idx]
            .get(mode="fill", fill_value=False)
        )
        rh_in = (
            (ctx.rh.data.inclusion.indices < ngraphs)
            .at[batch.rh_idx]
            .get(mode="fill", fill_value=False)
        )
        return lh_in & rh_in


@dataclass
class InclusionMatchMask:
    """Drops candidates whose lh/rh inclusion segments differ."""

    def __call__(self, batch: CandidateBatch, ctx: PipelineContext) -> Array:
        lh_incl = ctx.lh.data.inclusion.indices[batch.lh_idx]
        rh_incl = ctx.rh.data.inclusion.indices[batch.rh_idx]
        return lh_incl == rh_incl


@dataclass
class RemapDedupMask:
    """Deduplicate the rh→lh remapped subset.

    When ``ctx.rh_index_remap`` is set, ``rh`` is a subset of ``lh`` and each
    rh-position maps to an lh-position via ``rh_index_remap``. We then keep
    only one direction per pair: edges where ``lh_idx`` is **not** in the
    remap (i.e., the pair is lh-only) or where ``lh_idx >= remapped_rh``.

    Returns all-True when no remap is in effect.
    """

    def __call__(self, batch: CandidateBatch, ctx: PipelineContext) -> Array:
        if ctx.rh_index_remap is None:
            return jnp.ones((batch.lh_idx.size,), dtype=bool)
        oob = max(ctx.lh.size, ctx.rh.size)
        rh_remapped = ctx.rh_index_remap.at[batch.rh_idx].get(
            mode="fill", fill_value=oob
        )
        return ~isin(batch.lh_idx, ctx.rh_index_remap, ctx.lh.size) | (
            batch.lh_idx >= rh_remapped
        )


@dataclass
class EdgeInRhMask:
    """Keep fixed-topology rows touched by ``ctx.rh_index_remap``.

    When no remap is active, every row is kept. This lets fixed topology use
    the same pipeline for full and patch-shaped calls.
    """

    def __call__(self, batch: CandidateBatch, ctx: PipelineContext) -> Array:
        if ctx.rh_index_remap is None:
            return jnp.ones((len(batch.edges),), dtype=bool)
        return isin(batch.edges.indices.indices, ctx.rh_index_remap, ctx.lh.size).any(
            -1
        )


@dataclass
class DistanceCutoffMask:
    """Drops candidates whose squared real-space distance exceeds ``cutoff²``."""

    cutoffs: Table[SystemId, Array]

    def __call__(self, batch: CandidateBatch, ctx: PipelineContext) -> Array:
        cutoffs = Table.broadcast_to(self.cutoffs, ctx.systems)
        shifts = batch.edges.shifts[:, 0, :]
        dist_sq = real_distance_sq(
            ctx.lh, ctx.rh, ctx.systems, batch.lh_idx, batch.rh_idx, shifts
        )
        cand_sys = ctx.lh.data.system[batch.lh_idx]
        return dist_sq < cutoffs[cand_sys] ** 2


@dataclass
class ExclusionMask:
    """Drops minimum-image pairs that share an exclusion segment.

    Non-minimum-image periodic copies of excluded pairs survive (allowed when
    ``batch.is_minimum_image`` is False for that copy).
    """

    def __call__(self, batch: CandidateBatch, ctx: PipelineContext) -> Array:
        # batch.edges.indices carries a single set of keys; rebuild a rh-keyed
        # Index so Table indexing aligns when ctx.lh and ctx.rh have distinct keys.
        lh_view = ctx.lh[Index(ctx.lh.keys, batch.lh_idx)]
        rh_view = ctx.rh[Index(ctx.rh.keys, batch.rh_idx)]
        lh_excl, rh_excl = Index.match(lh_view.exclusion, rh_view.exclusion)
        return (lh_excl != rh_excl) | ~batch.is_minimum_image
