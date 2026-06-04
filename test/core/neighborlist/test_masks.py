# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the neighbor list mask criteria.

Each mask is a pure ``(batch, ctx) -> bool array`` function. Tests drive masks
with directly-constructed contexts and candidate batches, covering both the
self-graph (``queries=None``) and bipartite (``queries`` set) branches.
"""

import jax.numpy as jnp

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.table import Table
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    QueriedKeysDedupMask,
    TouchesQueriedKeysMask,
)

from ._builders import make_batch, make_lh, make_pipeline_ctx


class TestInBoundsMask:
    def test_drops_oob_indices_on_either_side(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        ctx = make_pipeline_ctx(lh)
        # Edge 0: both in bounds. Edge 1: rh OOB. Edge 2: lh OOB.
        batch = make_batch(lh.keys, jnp.array([0, 1, 100]), jnp.array([1, 100, 2]))
        result = InBoundsMask()(batch, ctx)
        assert result.tolist() == [True, False, False]

    def test_bipartite_checks_both_tables(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        rh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        ctx = make_pipeline_ctx(lh, rh)
        # Edge 1: rh idx 5 is OOB; edge 2: lh idx 9 is OOB.
        batch = make_batch(lh.keys, jnp.array([0, 1, 9]), jnp.array([1, 5, 0]))
        result = InBoundsMask()(batch, ctx)
        assert result.tolist() == [True, False, False]


class TestInclusionMatchMask:
    def test_matches_inclusion_segments(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.array([0, 0, 1, 1]))
        rh = make_lh(jnp.zeros((4, 3)), jnp.array([0, 1, 0, 1]))
        ctx = make_pipeline_ctx(lh, rh)
        # lh.incl[lh_idx] = [0, 0, 1, 1]; rh.incl[rh_idx] = [0, 1, 0, 1].
        batch = make_batch(lh.keys, jnp.array([0, 1, 2, 3]), jnp.array([0, 1, 0, 3]))
        result = InclusionMatchMask()(batch, ctx)
        assert result.tolist() == [True, False, False, True]


class TestQueriedKeysDedupMask:
    def test_no_queried_keys_returns_all_true(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh)
        batch = make_batch(lh.keys, jnp.array([0, 1, 2]), jnp.array([1, 2, 3]))
        result = QueriedKeysDedupMask()(batch, ctx)
        assert result.tolist() == [True, True, True]

    def test_drops_self_pair_with_queried_keys(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh, queried_keys=jnp.array([1, 3]))
        batch = make_batch(lh.keys, jnp.array([0, 1, 3]), jnp.array([1, 3, 1]))
        result = QueriedKeysDedupMask()(batch, ctx)
        assert result.tolist() == [True, False, True]


class TestTouchesQueriedKeysMask:
    def test_no_queried_keys_keeps_every_row(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh)
        batch = make_batch(lh.keys, jnp.array([0, 1, 2]), jnp.array([1, 2, 3]))
        result = TouchesQueriedKeysMask()(batch, ctx)
        assert result.tolist() == [True, True, True]

    def test_keeps_rows_touching_affected_ids(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh, queried_keys=jnp.array([2]))
        # Affected id is 2: keep rows whose either endpoint is 2.
        batch = make_batch(lh.keys, jnp.array([0, 1, 2]), jnp.array([1, 2, 3]))
        result = TouchesQueriedKeysMask()(batch, ctx)
        assert result.tolist() == [False, True, True]


class TestDistanceCutoffMask:
    def test_filters_by_distance_squared(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.8, 0.0, 0.0]])
        lh = make_lh(positions, jnp.zeros(3, dtype=int))
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
        ctx = make_pipeline_ctx(lh, cell=cell)
        batch = make_batch(lh.keys, jnp.array([0, 0, 1]), jnp.array([1, 2, 2]))
        # Distances: 0.3, 0.8, 0.5. cutoff^2 = 0.55^2 = 0.3025 -> keep [T, F, T].
        cutoffs = Table(ctx.systems.keys, jnp.array([0.55]))
        result = DistanceCutoffMask(cutoffs=cutoffs)(batch, ctx)
        assert result.tolist() == [True, False, True]


class TestExclusionMask:
    def test_drops_matching_exclusion_at_min_image(self):
        lh = make_lh(
            jnp.zeros((4, 3)),
            jnp.zeros(4, dtype=int),
            exclusion_ids=jnp.array([0, 1, 0, 2]),
        )
        rh = make_lh(
            jnp.zeros((4, 3)),
            jnp.zeros(4, dtype=int),
            exclusion_ids=jnp.array([0, 1, 2, 3]),
        )
        ctx = make_pipeline_ctx(lh, rh)
        # lh.excl[lh_idx=[0,1,2,3]] = [0, 1, 0, 2].
        # rh.excl[rh_idx=[0,1,0,1]] = [0, 1, 0, 1]. All min-image.
        batch = make_batch(lh.keys, jnp.array([0, 1, 2, 3]), jnp.array([0, 1, 0, 1]))
        result = ExclusionMask()(batch, ctx)
        assert result.tolist() == [False, False, False, True]

    def test_keeps_non_min_image_periodic_copy_of_excluded_pair(self):
        """Periodic copies of an excluded pair survive because
        ``~is_minimum_image`` short-circuits the drop."""
        lh = make_lh(
            jnp.zeros((2, 3)), jnp.zeros(2, dtype=int), exclusion_ids=jnp.array([0, 0])
        )
        ctx = make_pipeline_ctx(lh)
        batch = make_batch(
            lh.keys,
            jnp.array([0, 0]),
            jnp.array([1, 1]),
            is_minimum_image=jnp.array([True, False]),
        )
        result = ExclusionMask()(batch, ctx)
        # MIC copy: same excl -> drop. Non-MIC copy: kept regardless of excl.
        assert result.tolist() == [False, True]
