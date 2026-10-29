# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Index arithmetic follows the dtype of the indices it gets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.data import Index, cast_indices
from kups.core.utils.segment import bincount
from kups.core.utils.subselect import subselect

DEFAULT_INT = jnp.asarray(0).dtype
# differs from the default integer under x64, which the test env enables
OTHER_INT = jnp.int32


def _outputs(dtype) -> list[jax.Array]:
    idx = Index.new(["b", "a", "b", "c"], dtype=dtype)
    single = Index.zeros(3, dtype=dtype)
    return [
        idx.raw_indices,
        Index.integer([0, 2, 2], n=3, dtype=dtype).raw_indices,
        Index.arange(3, dtype=dtype).raw_indices,
        idx.counts.data,
        idx.select_per_label(jnp.array([0, 1, 0])),
        single.select_per_label(jnp.array([1])),
        idx.where_rectangular(idx, 2),
        idx.indices_in(("a", "aa", "b", "c")),
        Index.combine(idx, idx).raw_indices,
        idx.where_flat(Index.new(["b"])),
        jax.jit(lambda i: i.where_flat(Index.new(["b"]), FixedCapacity(2)))(idx),
        *subselect(
            jnp.array([0, 2], dtype=dtype),
            idx.raw_indices,
            output_buffer_size=FixedCapacity(4),
            num_segments=3,
        ),
        *subselect(
            jnp.array([0, 2], dtype=dtype),
            jnp.sort(idx.raw_indices),
            output_buffer_size=FixedCapacity(4),
            num_segments=3,
            is_sorted=True,
        ),
    ]


def test_default_is_jax_default_int():
    assert all(out.dtype == DEFAULT_INT for out in _outputs(int))


def test_outputs_follow_index_dtype():
    for out, ref in zip(_outputs(OTHER_INT), _outputs(int), strict=True):
        assert out.dtype == OTHER_INT
        np.testing.assert_array_equal(out, ref)


def test_bincount_dtype():
    x = jnp.array([0, 2, 2], dtype=OTHER_INT)
    assert bincount(x, length=3).dtype == OTHER_INT
    narrow = bincount(jnp.zeros(200, jnp.int8), length=2)
    assert narrow.dtype == jnp.int32
    np.testing.assert_array_equal(narrow, [200, 0])


def test_cast_indices():
    tree = {"a": Index.arange(3), "b": [Index.new(["x", "y"])], "c": jnp.ones(2)}
    cast = cast_indices(tree, OTHER_INT)
    assert cast["a"].raw_indices.dtype == cast["b"][0].raw_indices.dtype == OTHER_INT
    assert cast["a"].keys == tree["a"].keys
    assert cast["c"].dtype == tree["c"].dtype


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64])
def test_integer_keeps_array_dtype(dtype):
    ids = jnp.array([0, 1], dtype=dtype)
    assert Index.integer(ids, n=2).raw_indices.dtype == ids.dtype
