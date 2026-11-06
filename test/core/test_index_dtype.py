# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Indices are created in the configured dtype; arithmetic follows it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.config import index_dtype, set_index_dtype
from kups.core.capacity import FixedCapacity
from kups.core.data import Index
from kups.core.utils.segment import bincount
from kups.core.utils.subselect import subselect

DEFAULT_INT = jnp.asarray(0).dtype
# differs from the default integer under x64, which the test env enables
OTHER_INT = jnp.int32


def _outputs() -> list[jax.Array]:
    idx = Index.new(["b", "a", "b", "c"])
    single = Index.zeros(3)
    return [
        idx.raw_indices,
        Index.integer([0, 2, 2], n=3).raw_indices,
        Index.integer(jnp.array([0, 2, 2], jnp.int16), n=3).raw_indices,
        Index.arange(3).raw_indices,
        Index(("a", "b"), jnp.array([1, 0], jnp.int16)).raw_indices,
        idx.with_raw_indices(jnp.array([0, 0, 1, 2], jnp.int16)).raw_indices,
        idx.counts.data,
        idx.select_per_label(jnp.array([0, 1, 0])),
        single.select_per_label(jnp.array([1])),
        idx.where_rectangular(idx, 2),
        idx.indices_in(("a", "aa", "b", "c")),
        Index.combine(idx, idx).raw_indices,
        idx.where_flat(Index.new(["b"])),
        jax.jit(lambda i: i.where_flat(Index.new(["b"]), FixedCapacity(2)))(idx),
        jax.jit(lambda: Index.arange(3).raw_indices)(),
        *subselect(
            jnp.array([0, 2], dtype=idx.raw_indices.dtype),
            idx.raw_indices,
            output_buffer_size=FixedCapacity(4),
            num_segments=3,
        ),
        *subselect(
            jnp.array([0, 2], dtype=idx.raw_indices.dtype),
            jnp.sort(idx.raw_indices),
            output_buffer_size=FixedCapacity(4),
            num_segments=3,
            is_sorted=True,
        ),
    ]


def test_default_is_jax_default_int():
    assert index_dtype() == DEFAULT_INT
    assert all(out.dtype == DEFAULT_INT for out in _outputs())


def test_outputs_follow_index_dtype():
    ref = _outputs()
    set_index_dtype(OTHER_INT)
    try:
        assert index_dtype() == OTHER_INT
        outs = _outputs()
    finally:
        set_index_dtype(None)
    assert index_dtype() == DEFAULT_INT
    for out, expected in zip(outs, ref, strict=True):
        assert out.dtype == OTHER_INT
        np.testing.assert_array_equal(out, expected)


def test_int32_indices_fixture(int32_indices):
    assert Index.arange(2).raw_indices.dtype == jnp.int32


def test_set_index_dtype_rejects_floats():
    with pytest.raises(ValueError, match="integer"):
        set_index_dtype(jnp.float32)
    assert index_dtype() == DEFAULT_INT


def test_bincount_dtype():
    x = jnp.array([0, 2, 2], dtype=OTHER_INT)
    assert bincount(x, length=3).dtype == OTHER_INT
    narrow = bincount(jnp.zeros(200, jnp.int8), length=2)
    assert narrow.dtype == jnp.int32
    np.testing.assert_array_equal(narrow, [200, 0])
