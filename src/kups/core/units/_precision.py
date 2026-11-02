# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dtype and scale-factor checks for unit conversion and rebasing.

Every check here reads only dtypes and scale factors, so it runs while tracing and
behaves the same eagerly and under jit.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from kups.core.units._base import ATOMISTIC, SI, BaseSystem
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
    require_normal(
        f,
        dtype,
        f"re-expressing {src.symbol(dim) or '1'!r} from base {src.name} in base "
        f"{dst.name} needs a factor of {f!r}",
        f"so the magnitudes would be destroyed. Instead {widen(dtype)}build the "
        f"quantity in base {dst.name}.",
    )
    return f


def require_normal(f: float, dtype: DTypeLike, what: str, advice: str) -> None:
    """Raise unless ``f`` is a normal number of ``dtype``.

    Args:
      f: the scale factor to check.
      dtype: the floating dtype ``f`` will be applied in.
      what: the operation and factor, e.g. ``"attaching m ... needs a factor of 2.0"``.
      advice: the consequence and remedy appended after the range.

    Raises:
      DimensionError: if ``|f|`` is below the smallest normal or above the max.
    """
    fi = jnp.finfo(dtype)
    lo, hi = float(fi.smallest_normal), float(fi.max)
    if not lo <= abs(f) <= hi:
        raise DimensionError(
            f"kups.core.units: {what}, which is not a normal {jnp.dtype(dtype).name} "
            f"(range {lo:g} to {hi:g}), {advice}"
        )


X64_HINT = "jax.config.update('jax_enable_x64', True)"


def widen(dtype: DTypeLike) -> str:
    """Suggest a wider dtype as a clause ending in ``", or "``; empty for float64."""
    if jnp.dtype(dtype) == jnp.dtype(jnp.float64):
        return ""
    if jnp.dtype(dtype) == jnp.dtype(jnp.result_type(float)):
        return f"enable {X64_HINT}, or "
    return f"hold the magnitudes in a dtype wider than {jnp.dtype(dtype).name}, or "


def other_base(bs: BaseSystem) -> str:
    """Suggest the other preset base, or "" if ``bs`` is not a preset."""
    if bs is ATOMISTIC:
        return ", e.g. with units.base(units.SI)"
    if bs is SI:
        return ", e.g. with units.base(units.ATOMISTIC)"
    return ""


def checked_cast(v: ArrayLike, dtype: DTypeLike, what: str) -> jax.Array:
    """Cast magnitudes, refusing a dtype that x64 being off makes unavailable.

    ``what`` names the caller in the error.
    """
    out = jnp.astype(v, dtype)
    default = isinstance(dtype, type) and dtype in (float, int, complex, bool)
    want = None if default else jnp.dtype(dtype)
    if want is not None and want != jnp.result_type(out):
        raise DimensionError(
            f"{what}: {want.name} magnitudes are unavailable with x64 off; "
            f"jax returns {jnp.result_type(out).name}. Set {X64_HINT} before the "
            f"first jax call."
        )
    return out
