# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Global kUPS settings, each with a typed getter and setter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

__all__ = ["index_dtype", "set_index_dtype"]

_index_dtype: np.dtype | None = None


def index_dtype() -> np.dtype:
    """Integer dtype that every ``Index`` stores its positions in.

    Returns:
        The dtype set with :func:`set_index_dtype`, or JAX's default integer
        (``int64`` under x64, else ``int32``) if none is set.
    """
    return jax.dtypes.canonicalize_dtype(int if _index_dtype is None else _index_dtype)


def set_index_dtype(dtype: DTypeLike | None) -> None:
    """Set the integer dtype that every ``Index`` stores its positions in.

    Index constructors and ``Index.__post_init__`` create the positions in
    this dtype; index arithmetic follows the dtype of the positions. Set it
    before building a state: indices that already exist keep their dtype.
    JAX's caches are cleared, so functions traced before are traced again.

    Args:
        dtype: An integer dtype, e.g. ``jnp.int32``, or ``None`` for JAX's
            default integer.
    """
    global _index_dtype
    if dtype is not None:
        dtype = np.dtype(dtype)
        if not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"index dtype must be an integer dtype, got {dtype}")
    if dtype != _index_dtype:
        _index_dtype = dtype
        jax.clear_caches()
