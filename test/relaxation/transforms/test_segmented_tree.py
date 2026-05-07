# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-segment pytree reductions used by system-aware Optax transforms."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import SystemId
from kups.relaxation.transforms._segmented_tree import (
    tree_clip_per_row,
    tree_scale_per_row,
    tree_segment_max,
    tree_segment_norm,
    tree_vdot,
    tree_where_per_row,
)

from ...clear_cache import clear_cache  # noqa: F401


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


def _system_table(values, num_systems: int) -> Table[SystemId, jax.Array]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Table(keys, jnp.asarray(values), _cls=SystemId)


class TestTreeVdot:
    def test_scalar_per_row_single_segment(self):
        idx = _system_index([0, 0, 0], 1)
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        out = tree_vdot(a, b, idx)
        npt.assert_allclose(out.data, jnp.array([32.0]))

    def test_two_segments_contracts_trailing_axes(self):
        # Per-row vdot: [1, 1, 2]; seg 0 sums rows 0,1 → 2; seg 1 → 2.
        idx = _system_index([0, 0, 1], 2)
        a = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        out = tree_vdot(a, a, idx)
        npt.assert_allclose(out.data, jnp.array([2.0, 2.0]))

    def test_pytree_sums_across_leaves(self):
        # Single Index covers both dict leaves with the same partitioning.
        idx = _system_index([0, 0, 1, 1], 2)
        a = {
            "x": jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]),
            "y": jnp.array([10.0, 20.0, 30.0, 40.0]),
        }
        # x per-row dots [1,1,4,4] → seg sums [2, 8].
        # y per-row dots [100,400,900,1600] → seg sums [500, 2500].
        out = tree_vdot(a, a, idx)
        npt.assert_allclose(out.data, jnp.array([502.0, 2508.0]))

    def test_tuple_index_prefix_with_distinct_partitions(self):
        # Two leaves with different row counts and partitionings, combined per-segment.
        idx_a = _system_index([0, 0, 1], 2)
        idx_b = _system_index([0, 1], 2)
        a = (
            jnp.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0]]),  # per-row [1,1,9]
            jnp.array([[2.0], [4.0]]),  # per-row [4,16]
        )
        out = tree_vdot(a, a, (idx_a, idx_b))
        # leaf 0 seg sums [2, 9]; leaf 1 seg sums [4, 16]; combined [6, 25].
        npt.assert_allclose(out.data, jnp.array([6.0, 25.0]))

    def test_jit_roundtrip(self):
        idx = _system_index([0, 0, 1, 1], 2)
        a = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        fn = jax.jit(lambda x: tree_vdot(x, x, idx).data)
        npt.assert_allclose(fn(a), tree_vdot(a, a, idx).data)


class TestTreeSegmentMax:
    def test_scalar_rows(self):
        idx = _system_index([0, 0, 1, 1], 2)
        a = jnp.array([1.0, 3.0, 2.0, 5.0])
        out = tree_segment_max(a, idx)
        npt.assert_allclose(out.data, jnp.array([3.0, 5.0]))

    def test_reduces_trailing_axes_before_segment_max(self):
        # Per-row max over trailing axis: [5, 3, 4]; seg max: [5, 4].
        idx = _system_index([0, 0, 1], 2)
        a = jnp.array([[1.0, 5.0], [3.0, 2.0], [4.0, 4.0]])
        out = tree_segment_max(a, idx)
        npt.assert_allclose(out.data, jnp.array([5.0, 4.0]))

    def test_max_across_pytree_leaves(self):
        # Per-leaf seg max combined with element-wise jnp.maximum across leaves.
        idx = _system_index([0, 0, 1, 1], 2)
        a = {
            "p": jnp.array([1.0, 2.0, 3.0, 4.0]),  # seg max [2, 4]
            "q": jnp.array([10.0, -1.0, 0.0, 0.5]),  # seg max [10, 0.5]
        }
        out = tree_segment_max(a, idx)
        npt.assert_allclose(out.data, jnp.array([10.0, 4.0]))


class TestTreeSegmentNorm:
    def test_l2_norm_per_segment(self):
        # Seg 0: sqrt(9+16)=5; seg 1: sqrt(36)=6.
        idx = _system_index([0, 0, 1], 2)
        a = jnp.array([[3.0, 0.0], [0.0, 4.0], [6.0, 0.0]])
        out = tree_segment_norm(a, idx)
        npt.assert_allclose(out.data, jnp.array([5.0, 6.0]))

    def test_matches_sqrt_of_vdot_with_self(self):
        idx = _system_index([0, 1, 0, 1], 2)
        a = jnp.array([[1.0, 2.0], [-3.0, 4.0], [0.0, 5.0], [1.0, 0.0]])
        norm = tree_segment_norm(a, idx)
        sq = tree_vdot(a, a, idx)
        npt.assert_allclose(norm.data, jnp.sqrt(sq.data))


class TestTreeScalePerRow:
    def test_rows_scaled_by_segment_factor(self):
        idx = _system_index([0, 0, 1, 1], 2)
        arr = jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        scale = _system_table([10.0, 100.0], 2)
        out = tree_scale_per_row(arr, scale, idx)
        expected = jnp.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [700.0, 800.0, 900.0],
                [1000.0, 1100.0, 1200.0],
            ]
        )
        npt.assert_allclose(out, expected)

    def test_pytree_preserves_structure(self):
        idx = _system_index([0, 1, 1], 2)
        tree = {
            "a": jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            "b": jnp.array([1.0, 2.0, 3.0]),
        }
        scale = _system_table([10.0, 2.0], 2)
        out = tree_scale_per_row(tree, scale, idx)
        assert set(out.keys()) == {"a", "b"}
        npt.assert_allclose(out["a"], jnp.array([[10.0, 10.0], [4.0, 4.0], [6.0, 6.0]]))
        npt.assert_allclose(out["b"], jnp.array([10.0, 4.0, 6.0]))


class TestTreeClipPerRow:
    def test_clips_to_symmetric_bounds(self):
        idx = _system_index([0, 0, 1, 1], 2)
        arr = jnp.array([[2.0, -5.0], [0.5, 3.0], [10.0, -0.5], [-2.0, 0.0]])
        # Seg 0 limit 1.0 → clip; seg 1 limit inf → untouched.
        limit = _system_table([1.0, jnp.inf], 2)
        out = tree_clip_per_row(arr, limit, idx)
        expected = jnp.array([[1.0, -1.0], [0.5, 1.0], [10.0, -0.5], [-2.0, 0.0]])
        npt.assert_allclose(out, expected)

    def test_pytree_clipped_independently(self):
        idx = _system_index([0, 1, 1], 2)
        tree = {
            "a": jnp.array([[5.0, -5.0], [0.5, 2.0], [10.0, -10.0]]),
            "b": jnp.array([100.0, 0.5, 5.0]),
        }
        limit = _system_table([1.0, 3.0], 2)
        out = tree_clip_per_row(tree, limit, idx)
        npt.assert_allclose(out["a"], jnp.array([[1.0, -1.0], [0.5, 2.0], [3.0, -3.0]]))
        npt.assert_allclose(out["b"], jnp.array([1.0, 0.5, 3.0]))


class TestTreeWherePerRow:
    def test_selects_a_or_b_by_segment_mask(self):
        idx = _system_index([0, 0, 1, 1], 2)
        a = jnp.ones((4, 2))
        b = jnp.zeros((4, 2))
        mask = _system_table([True, False], 2)
        out = tree_where_per_row(mask, a, b, idx)
        npt.assert_allclose(out, jnp.array([[1, 1], [1, 1], [0, 0], [0, 0]]))

    def test_pytree_select(self):
        idx = _system_index([0, 1, 0], 2)
        a = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([[1.0], [2.0], [3.0]])}
        b = {
            "x": jnp.array([-1.0, -2.0, -3.0]),
            "y": jnp.array([[-1.0], [-2.0], [-3.0]]),
        }
        # Seg 0 (rows 0, 2) → take a; seg 1 (row 1) → take b.
        mask = _system_table([True, False], 2)
        out = tree_where_per_row(mask, a, b, idx)
        npt.assert_allclose(out["x"], jnp.array([1.0, -2.0, 3.0]))
        npt.assert_allclose(out["y"], jnp.array([[1.0], [-2.0], [3.0]]))


class TestLayoutValidation:
    def test_mismatched_sub_structures_raises(self):
        # Both trees fall under one Index leaf, but `a` has 2 array leaves and
        # `b` has 1 — _layout_and_leaves must reject this with a clear message.
        idx = _system_index([0, 0, 1], 2)
        a = {"x": jnp.zeros((3, 2)), "y": jnp.zeros((3, 2))}
        b = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match="share the same sub-structure"):
            tree_vdot(a, b, idx)
