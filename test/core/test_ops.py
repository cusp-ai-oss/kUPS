# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt

from kups.core.utils.ops import expand_last_dims, pad_axis, where_broadcast_last


class TestExpandLastDims:
    def test_1d_to_3d(self):
        x = jnp.array([1, 2, 3])
        result = expand_last_dims(x, jnp.zeros((2, 3, 4)))
        assert result.shape == (3, 1, 1)

    def test_2d_to_4d_tuple(self):
        x = jnp.array([[1, 2]])
        result = expand_last_dims(x, (3, 1, 2, 5))
        assert result.shape == (1, 2, 1, 1)

    def test_same_ndim_noop(self):
        x = jnp.ones((2, 3))
        result = expand_last_dims(x, (2, 3))
        assert result.shape == (2, 3)


class TestWhereBroadcastLast:
    def test_basic(self):
        cond = jnp.array([True, False])
        x = jnp.array([[1, 2], [3, 4]])
        y = jnp.array([[5, 6], [7, 8]])
        result = where_broadcast_last(cond, x, y)
        npt.assert_array_equal(result, [[1, 2], [7, 8]])

    def test_scalar_condition(self):
        result = where_broadcast_last(
            jnp.array(True), jnp.array([1, 2]), jnp.array([3, 4])
        )
        npt.assert_array_equal(result, [1, 2])


class TestPadAxis:
    def test_pad_first_axis(self):
        x = jnp.array([[1, 2], [3, 4]])
        result = pad_axis(x, (1, 2), axis=0)
        assert result.shape == (5, 2)
        npt.assert_array_equal(result[0], [0, 0])
        npt.assert_array_equal(result[1:3], [[1, 2], [3, 4]])
        npt.assert_array_equal(result[3:], [[0, 0], [0, 0]])

    def test_pad_last_axis(self):
        x = jnp.array([[1, 2], [3, 4]])
        result = pad_axis(x, (0, 3), axis=1)
        assert result.shape == (2, 5)
        npt.assert_array_equal(result[:, :2], [[1, 2], [3, 4]])
        npt.assert_array_equal(result[:, 2:], 0)

    def test_no_padding(self):
        x = jnp.ones((3, 4))
        result = pad_axis(x, (0, 0), axis=0)
        assert result.shape == (3, 4)
        npt.assert_array_equal(result, x)

    def test_1d(self):
        x = jnp.array([1, 2, 3])
        result = pad_axis(x, (2, 1), axis=0)
        npt.assert_array_equal(result, [0, 0, 1, 2, 3, 0])

    def test_preserves_dtype(self):
        x = jnp.array([1.0, 2.0], dtype=jnp.float32)
        result = pad_axis(x, (1, 1), axis=0)
        assert result.dtype == jnp.float32
