# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dtype checks and scale factors for unit conversion and rebasing."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax.typing import DTypeLike

from kups.core.units._base import BaseSystem
from kups.core.units._unit import ONE, Dim, DimensionError, exps


def check_dtype(dtype: DTypeLike, dim: Dim, bs: BaseSystem, what: str) -> None:
    """Require a floating dtype for dimensioned magnitudes."""
    if dim == ONE or jnp.issubdtype(dtype, jnp.floating):
        return
    raise DimensionError(
        f"units.{what}: {jnp.dtype(dtype).name} magnitudes cannot carry "
        f"{bs.symbol(dim)!r}; unit conversions require floating magnitudes. "
        f"Use v.astype(float), or dim=units.ONE for a count, index or mask."
    )


def base_scale(bs: BaseSystem, dim: Dim) -> float:
    """Return ``bs``'s unit size for ``dim``, raising if it is 0 or infinite."""
    s = bs.scale(dim)
    if s == 0.0 or not math.isfinite(s):
        raise DimensionError(
            f"kups.core.units: base {bs.name} has no representable unit for "
            f"{exps(dim)} "
            f"(scale {s!r}). Use a base where this dimension is O(1), "
            f"or take magnitudes with q.to(unit)."
        )
    return s


def cast_advice(dtype: DTypeLike) -> str:
    """Suggest how to get floating magnitudes out of a non-floating ``dtype``."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return "take jnp.real/jnp.imag first -- a cast would drop the imaginary part"
    return "cast with q.astype(dtype) first"


def rebase_factor(
    src: BaseSystem, dst: BaseSystem, dim: Dim, dtype: DTypeLike
) -> float:
    """Return the factor converting ``dim`` magnitudes from ``src`` to ``dst``."""
    f = base_scale(src, dim) / base_scale(dst, dim)
    if f == 1.0:
        return f
    if not jnp.issubdtype(dtype, jnp.floating):
        raise DimensionError(
            f"kups.core.units: {jnp.dtype(dtype).name} magnitudes cannot be "
            "re-expressed from "
            f"base {src.name} in base {dst.name}: scaling by {f!r} requires floating "
            f"magnitudes; {cast_advice(dtype)}."
        )
    return f
