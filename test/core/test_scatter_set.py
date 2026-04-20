# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt
from jax import Array

from kups.core.utils.jax import (
    HasScatterArgs,
    ScatterArgs,
    dataclass,
    tree_scatter_set,
)


@dataclass
class Pair:
    """Simple pytree of two arrays."""

    x: Array
    y: Array


@dataclass
class WithDrop:
    """Pytree implementing HasScatterArgs with mode='drop'."""

    x: Array
    y: Array

    @property
    def scatter_args(self) -> ScatterArgs:
        return {"mode": "drop"}


class TestScatterSetPlainArray:
    """Tests for scatter_set on plain JAX arrays."""

    def test_set_single_index(self):
        arr = jnp.array([0, 0, 0])
        val = jnp.array([9])
        result = tree_scatter_set(arr, val, jnp.array([1]), {})
        npt.assert_array_equal(result, [0, 9, 0])

    def test_set_multiple_indices(self):
        arr = jnp.zeros(5, dtype=jnp.int32)
        val = jnp.array([10, 20], dtype=jnp.int32)
        result = tree_scatter_set(arr, val, jnp.array([0, 3]), {})
        npt.assert_array_equal(result, [10, 0, 0, 20, 0])

    def test_2d_array(self):
        arr = jnp.zeros((4, 2), dtype=jnp.int32)
        val = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        result = tree_scatter_set(arr, val, jnp.array([1, 3]), {})
        npt.assert_array_equal(result, [[0, 0], [1, 2], [0, 0], [3, 4]])


class TestScatterSetPytree:
    """Tests for scatter_set on a dataclass pytree of arrays."""

    def test_basic(self):
        item = Pair(jnp.zeros(4), jnp.ones(4))
        val = Pair(jnp.array([5.0, 6.0]), jnp.array([7.0, 8.0]))
        result = tree_scatter_set(item, val, jnp.array([1, 3]), {})
        npt.assert_array_equal(result.x, [0, 5, 0, 6])
        npt.assert_array_equal(result.y, [1, 7, 1, 8])


class TestScatterSetHasScatterArgs:
    """Tests for scatter_set with a HasScatterArgs dataclass."""

    def test_protocol_satisfied(self):
        obj = WithDrop(jnp.zeros(3), jnp.zeros(3))
        assert isinstance(obj, HasScatterArgs)

    def test_oob_index_dropped(self):
        """Out-of-bounds index should be silently dropped with mode='drop'."""
        item = WithDrop(jnp.zeros(3), jnp.ones(3))
        val = WithDrop(jnp.array([9.0, 8.0]), jnp.array([7.0, 6.0]))
        # Index 5 is OOB for size-3 arrays; mode='drop' should ignore it.
        result = tree_scatter_set(item, val, jnp.array([1, 5]), {})
        npt.assert_array_equal(result.x, [0, 9, 0])
        npt.assert_array_equal(result.y, [1, 7, 1])

    def test_caller_args_override(self):
        """Caller-supplied args should override the object's scatter_args."""
        item = WithDrop(jnp.array([1.0, 2.0, 3.0]), jnp.zeros(3))
        val = WithDrop(jnp.array([99.0]), jnp.array([99.0]))
        # Override mode to 'clip' — OOB index 10 should be clipped to last.
        result = tree_scatter_set(item, val, jnp.array([10]), {"mode": "clip"})
        npt.assert_array_equal(result.x, [1, 2, 99])


class TestScatterSetEmptyIndices:
    """Tests for scatter_set with empty index arrays."""

    def test_noop_on_array(self):
        arr = jnp.array([1, 2, 3])
        result = tree_scatter_set(
            arr, jnp.array([]).astype(jnp.int32), jnp.array([], dtype=jnp.int32), {}
        )
        npt.assert_array_equal(result, [1, 2, 3])

    def test_noop_on_pytree(self):
        item = Pair(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        val = Pair(jnp.array([]).astype(jnp.float32), jnp.array([]).astype(jnp.float32))
        result = tree_scatter_set(item, val, jnp.array([], dtype=jnp.int32), {})
        npt.assert_array_equal(result.x, [1, 2])
        npt.assert_array_equal(result.y, [3, 4])
