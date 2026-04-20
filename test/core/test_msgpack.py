# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from kups.core.utils.msgpack import deserialize, serialize


def _roundtrip(obj):
    return deserialize(serialize(obj))


class TestSerializeDeserialize:
    def test_numpy_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _roundtrip(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == arr.dtype

    def test_jax_array(self):
        arr = jnp.array([1, 2, 3], dtype=jnp.int32)
        out = _roundtrip(arr)
        np.testing.assert_array_equal(out, np.asarray(arr))
        assert out.dtype == np.int32

    def test_scalar_array(self):
        arr = jnp.float32(3.14)
        out = _roundtrip(arr)
        np.testing.assert_allclose(out, 3.14, rtol=1e-6)
        assert out.shape == ()

    def test_nested_dict(self):
        obj = {"a": jnp.ones(3), "b": {"c": jnp.zeros(2, dtype=jnp.int32)}}
        out = _roundtrip(obj)
        np.testing.assert_array_equal(out["a"], np.ones(3))
        np.testing.assert_array_equal(out["b"]["c"], np.zeros(2, dtype=np.int32))

    def test_list_of_arrays(self):
        obj = [jnp.array([1.0]), jnp.array([2.0])]
        out = _roundtrip(obj)
        assert len(out) == 2
        np.testing.assert_array_equal(out[0], [1.0])
        np.testing.assert_array_equal(out[1], [2.0])

    def test_mixed_leaves(self):
        obj = {"arr": jnp.array([1.0]), "num": 42, "text": "hello"}
        out = _roundtrip(obj)
        np.testing.assert_array_equal(out["arr"], [1.0])
        assert out["num"] == 42
        assert out["text"] == "hello"

    def test_preserves_dtype(self):
        for dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
            arr = np.array([1, 0, 1], dtype=dtype)
            out = _roundtrip(arr)
            assert out.dtype == dtype, f"dtype mismatch for {dtype}"

    def test_preserves_shape(self):
        arr = np.zeros((2, 3, 4))
        out = _roundtrip(arr)
        assert out.shape == (2, 3, 4)

    def test_empty_array(self):
        arr = np.array([], dtype=np.float64)
        out = _roundtrip(arr)
        assert out.shape == (0,)
        assert out.dtype == np.float64

    def test_returns_bytes(self):
        assert isinstance(serialize(jnp.array([1.0])), bytes)
