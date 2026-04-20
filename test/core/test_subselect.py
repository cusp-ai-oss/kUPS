# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for data utility functions."""

import jax.numpy as jnp
import numpy.testing as npt

from kups.core.capacity import LensCapacity
from kups.core.data import subselect
from kups.core.lens import lens
from kups.core.utils.subselect import offsets_from_counts


class TestOffsetsFromCounts:
    """Test cases for offsets_from_counts."""

    def test_basic(self):
        counts = jnp.array([3, 2, 4, 1])
        offsets = offsets_from_counts(counts)
        npt.assert_array_equal(offsets, jnp.array([0, 3, 5, 9]))

    def test_empty(self):
        counts = jnp.array([])
        offsets = offsets_from_counts(counts)
        npt.assert_array_equal(offsets, jnp.array([]))

    def test_single(self):
        counts = jnp.array([5])
        offsets = offsets_from_counts(counts)
        npt.assert_array_equal(offsets, jnp.array([0]))


class TestSubselect:
    """Test cases for subselect."""

    def test_basic(self):
        needle = jnp.array([1, 3])
        haystack = jnp.array([0, 1, 2, 1, 3, 2, 3])
        capacity = LensCapacity(4, lens(lambda x: x))

        result = subselect(
            needle, haystack, output_buffer_size=capacity, num_segments=4
        )
        scatter_idxs, gather_idxs = result

        expected_gather = jnp.array([1, 3, 4, 6])
        expected_scatter = jnp.array([0, 0, 1, 1])
        npt.assert_array_equal(gather_idxs[:4], expected_gather)
        npt.assert_array_equal(scatter_idxs[:4], expected_scatter)

    def test_with_insufficient_buffer(self):
        needle = jnp.array([1])
        haystack = jnp.array([1, 1, 1])
        capacity = LensCapacity(2, lens(lambda x: x))

        result = subselect(
            needle, haystack, output_buffer_size=capacity, num_segments=2
        )
        scatter_idxs, gather_idxs = result

        assert len(scatter_idxs) == 2
        assert len(gather_idxs) == 2

    def test_sorted(self):
        needle = jnp.array([2, 4])
        haystack = jnp.array([1, 2, 2, 3, 4, 4, 5])
        capacity = LensCapacity(4, lens(lambda x: x))

        result = subselect(
            needle,
            haystack,
            output_buffer_size=capacity,
            num_segments=6,
            is_sorted=True,
        )
        scatter_idxs, gather_idxs = result

        assert len(scatter_idxs) == len(gather_idxs)
        expected_gather = jnp.array([1, 2, 4, 5])
        expected_scatter = jnp.array([0, 0, 1, 1])
        npt.assert_array_equal(gather_idxs[:4], expected_gather)
        npt.assert_array_equal(scatter_idxs[:4], expected_scatter)
