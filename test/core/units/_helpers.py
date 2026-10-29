# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the kups.core.units tests (x64 comes from pytest-env)."""

import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from kups.core import units

# jax's own APIs with their signatures hidden from pyrefly: jax's stubs reject
# quantity arguments and type jit's result as Never
arr: Callable[..., jax.Array] = jnp.array
ones: Callable[..., jax.Array] = jnp.ones
eye: Callable[..., jax.Array] = jnp.eye
arange: Callable[..., jax.Array] = jnp.arange
jit: Callable[..., Any] = jax.jit
make_jaxpr: Callable[..., Any] = jax.make_jaxpr
checkpoint: Callable[..., Any] = jax.checkpoint
jax_grad: Callable[..., Any] = jax.grad  # jax's reverse mode, not units.grad
zeros_like: Callable[..., jax.Array] = jnp.zeros_like
ones_like: Callable[..., jax.Array] = jnp.ones_like
empty_like: Callable[..., jax.Array] = jnp.empty_like
full_like: Callable[..., jax.Array] = jnp.full_like
segment_sum: Callable[..., jax.Array] = jax.ops.segment_sum
linalg_norm: Callable[..., jax.Array] = jnp.linalg.norm
lax_cond: Callable[..., Any] = jax.lax.cond
lax_switch: Callable[..., Any] = jax.lax.switch
jnp_vectorize: Callable[..., Any] = jnp.vectorize
eval_shape: Callable[..., Any] = jax.eval_shape
asarray: Callable[..., jax.Array] = jnp.asarray
debug_print: Callable[..., Any] = jax.debug.print
debug_callback: Callable[..., Any] = jax.debug.callback
np_asarray: Callable[..., Any] = np.asarray
np_array: Callable[..., Any] = np.array


# ASE time unit in seconds, derived from ASE's length/mass/energy base
ASE_TIME = 1.0180505710759415e-14


@contextmanager
def x64_mode(enabled: bool) -> Generator[None, None, None]:
    """Temporarily set ``jax_enable_x64``."""
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def val(q: Any) -> float:
    """Return a quantity's value as a Python float, in its own base."""
    return float(q.value)


def ty(x: Any) -> str:
    """Render dtype, shape and label as ``float64[3]{m|SI}``; jax's for a plain array."""
    if not units.is_quantity(x):
        return str(jax.typeof(x))
    shape = ",".join(map(str, x.shape))
    return f"{x.dtype.name}[{shape}]{{{x.bs.symbol(x.dim) or '1'}|{x.bs.name}}}"


def ulps(a: float, b: float) -> int:
    """Distance between a and b in ulps, capped at 9.

    Returns:
        Representable floats between the two; 9 means beyond the cap (or nan/inf).
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return 9
    n, lo, hi = 0, min(a, b), max(a, b)
    while lo < hi and n < 9:
        lo, n = math.nextafter(lo, math.inf), n + 1
    return n
