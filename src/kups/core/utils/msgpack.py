# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import jax
import msgpack
import numpy as np

_ARRAY_TAG = "__np__"


def _is_array_leaf(leaf: Any) -> bool:
    return isinstance(leaf, dict) and _ARRAY_TAG in leaf


def _encode_leaf(leaf: Any) -> Any:
    arr = np.asarray(leaf)
    return {
        _ARRAY_TAG: True,
        "shape": list(arr.shape),
        "dtype": arr.dtype.str,
        "data": arr.tobytes(),
    }


def _decode_leaf(leaf: Any) -> Any:
    if _is_array_leaf(leaf):
        return np.frombuffer(leaf["data"], dtype=np.dtype(leaf["dtype"])).reshape(
            leaf["shape"]
        )
    return leaf


def serialize(obj: Any) -> bytes:
    """Serialize a pytree with jax/numpy arrays to msgpack bytes.

    Array leaves are encoded as ``{"shape": ..., "dtype": ..., "data": ...}``
    dicts. Non-array leaves (ints, floats, strings, etc.) are passed through
    to msgpack as-is.

    Args:
        obj: A pytree whose leaves are jax/numpy arrays or plain Python values.

    Returns:
        The msgpack-encoded bytes.
    """
    encoded = jax.tree.map(_encode_leaf, obj)
    return cast(bytes, msgpack.packb(encoded, use_bin_type=True))


def deserialize(data: bytes) -> Any:
    """Deserialize msgpack bytes back to a pytree with numpy arrays.

    Inverse of :func:`serialize`. Encoded array dicts are restored to numpy
    arrays; all other values are returned as plain Python objects.

    Args:
        data: Bytes produced by :func:`serialize`.

    Returns:
        The reconstructed pytree with numpy array leaves.
    """
    decoded = msgpack.unpackb(data, raw=False)
    return jax.tree.map(_decode_leaf, decoded, is_leaf=_is_array_leaf)
