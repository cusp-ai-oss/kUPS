# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dtype and range checks for unit conversion and rebasing.

Dtypes and scale factors can be checked during tracing. Checks that inspect
magnitudes run only on concrete values; host inputs are checked before transfer
because XLA may flush subnormal values to zero.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from kups.core.units._base import ATOMISTIC, SI, BaseSystem
from kups.core.units._unit import ONE, AnyUnit, Dim, DimensionError, exps


def _check_dtype(dtype: Any, dim: Dim, bs: BaseSystem, what: str) -> None:
    """Require a floating dtype for dimensioned magnitudes."""
    if dim == ONE or jnp.issubdtype(dtype, jnp.floating):
        return
    raise DimensionError(
        f"units.{what}: {jnp.dtype(dtype).name} magnitudes cannot carry "
        f"{bs.symbol(dim)!r}; unit conversions require floating magnitudes. "
        f"Use v.astype(float), or dim=units.ONE for a count, index or mask."
    )


def _host(x: Any) -> Any:
    """Return ``x`` as a numpy array, or None unless it is an inexact host value."""
    if isinstance(x, (jax.Array, jax.core.Tracer)):
        return None
    src = np.asarray(x)
    return src if np.issubdtype(src.dtype, np.inexact) else None


def _check_host(src: Any, f: float, dtype: Any, what: str) -> None:
    """Raise if scaling host magnitudes by ``f`` into ``dtype`` would destroy any."""
    fi = jnp.finfo(dtype)
    inv = 1.0 / abs(f) if f else math.inf
    lo = float(fi.smallest_normal) * max(1.0, inv)
    hi = float(fi.max) * min(1.0, inv)
    mag = np.abs(np.asarray(src))
    n = int(np.count_nonzero(np.isfinite(mag) & (mag != 0) & ((mag < lo) | (mag > hi))))
    if not n:
        return
    scaled = "" if f == 1.0 else f", multiplied by {f!r},"
    raise DimensionError(
        f"kups.core.units: {what}{scaled} destroys {n} of {mag.size} finite "
        f"nonzero magnitudes once they are {jnp.dtype(dtype).name}. Instead "
        f"{_widen(dtype)}measure in a unit closer to the magnitudes' own size."
    )


def _base_scale(bs: BaseSystem, dim: Dim) -> float:
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


def _cast_advice(dtype: Any) -> str:
    """Suggest how to get floating magnitudes out of a non-floating ``dtype``."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return "take jnp.real/jnp.imag first -- a cast would drop the imaginary part"
    return "cast with q.astype(dtype) first"


def _rebase_factor(src: BaseSystem, dst: BaseSystem, dim: Dim, dtype: Any) -> float:
    """Return the factor converting ``dim`` magnitudes from ``src`` to ``dst``."""
    f = _base_scale(src, dim) / _base_scale(dst, dim)
    if f == 1.0:
        return f
    if not jnp.issubdtype(dtype, jnp.floating):
        raise DimensionError(
            f"kups.core.units: {jnp.dtype(dtype).name} magnitudes cannot be "
            "re-expressed from "
            f"base {src.name} in base {dst.name}: scaling by {f!r} requires floating "
            f"magnitudes; {_cast_advice(dtype)}."
        )
    _require_normal(
        f,
        dtype,
        f"re-expressing {src.symbol(dim) or '1'!r} from base {src.name} in base "
        f"{dst.name} needs a factor of {f!r}",
        f"so the magnitudes would be destroyed. Build the quantity in base "
        f"{dst.name}, or set jax_enable_x64.",
    )
    return f


def _require_normal(f: float, dtype: Any, what: str, advice: str) -> None:
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


def _eps(dtype: Any) -> float:
    """Machine epsilon of ``dtype``."""
    return float(jnp.finfo(dtype).eps)


_WARNED: set[tuple[str, BaseSystem]] = set()
_LOW_EPS = _eps(jnp.float32)  # float32 and anything coarser
_X64_HINT = "jax.config.update('jax_enable_x64', True)"


def _widen(dtype: Any) -> str:
    """Suggest a wider dtype as a clause ending in ``", or "``; empty for float64."""
    if jnp.dtype(dtype) == jnp.dtype(jnp.float64):
        return ""
    if jnp.dtype(dtype) == jnp.dtype(jnp.result_type(float)):
        return f"enable {_X64_HINT}, or "
    return f"hold the magnitudes in a dtype wider than {jnp.dtype(dtype).name}, or "


def _other_base(bs: BaseSystem) -> str:
    """Suggest the other preset base, or "" if ``bs`` is not a preset."""
    if bs is ATOMISTIC:
        return ", e.g. units.set_base(units.SI)"
    if bs is SI:
        return ", e.g. units.set_base(units.ATOMISTIC)"
    return ""


def _precision_warning(unit: AnyUnit, bs: BaseSystem, f: float, dtype: Any) -> None:
    """Warn once per unit and base when ``unit`` is near the edge of ``dtype``."""
    if _eps(dtype) < _LOW_EPS:
        return  # only float32 and coarser warn
    key = (unit.name, bs)
    if key in _WARNED or f == 0.0 or abs(math.log10(abs(f))) < 8:
        return  # f == 0 is reported by _check_canonical
    _WARNED.add(key)
    name = jnp.dtype(dtype).name
    edge = "overflow" if abs(f) > 1.0 else "underflow"
    # a dimensionless unit is 1 in every base, so no rebase advice
    rebase = (
        ""
        if unit.dim == ONE
        else f", or work in a base where this unit is O(1){_other_base(bs)}"
    )
    digits = max(1, round(-math.log10(_eps(dtype))))
    warnings.warn(
        f"kups.core.units: {unit} is {f:.3g} in base {bs.name}; powers can {edge} in "
        f"{name}. "
        f"Rescaling keeps about {digits} decimal digits. Enable "
        f"{_X64_HINT}{rebase}.",
        RuntimeWarning,
        # point the warning at the caller, however deep the call
        skip_file_prefixes=(os.path.dirname(__file__),),
    )


def _n_flushed(v: jax.Array, out: jax.Array) -> jax.Array:
    """Count nonzero magnitudes of ``v`` that are zero in ``out``."""
    return jnp.count_nonzero(v) - jnp.count_nonzero(out)


def _n_grown(v: jax.Array, out: jax.Array) -> jax.Array:
    """Count finite magnitudes of ``v`` that are infinite in ``out``."""
    return jnp.count_nonzero(jnp.isinf(out)) - jnp.count_nonzero(jnp.isinf(v))


def _n_destroyed(v: jax.Array, out: jax.Array) -> jax.Array:
    """Count magnitudes of ``v`` flushed to zero or grown to infinity in ``out``."""
    return _n_flushed(v, out) + _n_grown(v, out)


_N_FLUSHED = jax.jit(_n_flushed)
_N_OVERFLOWED = jax.jit(_n_grown)
_N_DESTROYED = jax.jit(_n_destroyed)


def _check_canonical(
    v: jax.Array, out: jax.Array, unit: AnyUnit, bs: BaseSystem, f: float
) -> None:
    """Raise if a canonicalization destroyed a magnitude."""
    if isinstance(out, jax.core.Tracer):
        return  # traced magnitudes cannot be inspected
    # a given f can only lose magnitudes in one direction
    shrinks = abs(f) < 1.0
    n = int((_N_FLUSHED if shrinks else _N_OVERFLOWED)(v, out))
    if n <= 0:
        return
    verb, edge = ("flushes", "zero") if shrinks else ("overflows", "infinity")
    raise DimensionError(
        f"kups.core.units: attaching {unit} in base {bs.name} multiplies the "
        "magnitudes by "
        f"{f!r}, which {verb} {n} of {v.size} finite nonzero "
        f"{jnp.dtype(v.dtype).name} magnitudes to {edge}. Enable {_X64_HINT}"
        + (
            ""
            if unit.dim == ONE
            else f", or work in a base where {unit} is O(1){_other_base(bs)}"
        )
        + "."
    )


def _check_rescaled(v: Any, out: Any, what: str, bs: BaseSystem, f: float) -> None:
    """Raise if a rescale destroyed a magnitude."""
    if isinstance(out, jax.core.Tracer) or isinstance(v, jax.core.Tracer):
        return  # traced magnitudes cannot be inspected
    n = int(_N_DESTROYED(v, out))
    if n <= 0:
        return
    raise DimensionError(
        f"kups.core.units: {what} multiplies the magnitudes by {f!r}, which destroys "
        f"{n} of "
        f"{jnp.asarray(v).size} finite nonzero {jnp.result_type(v).name} magnitudes "
        f"in base {bs.name}. Enable {_X64_HINT}, or measure in a unit closer to "
        f"the magnitudes' own size."
    )


def _checked_cast(v: Any, dtype: Any, bs: BaseSystem) -> jax.Array:
    """Cast magnitudes, rejecting unavailable dtypes and lossy eager casts."""
    out = jnp.astype(v, dtype)
    want = None if dtype in (float, int, complex, bool) else jnp.dtype(dtype)
    if want is not None and want != jnp.result_type(out):
        raise DimensionError(
            f"q.astype: {want.name} magnitudes are unavailable with x64 off; "
            f"jax returns {jnp.result_type(out).name}. Set {_X64_HINT} before the "
            f"first jax call."
        )
    if isinstance(out, jax.core.Tracer) or isinstance(v, jax.core.Tracer):
        return out  # traced magnitudes cannot be inspected
    n = int(_N_DESTROYED(v, out))
    if n <= 0:
        return out
    raise DimensionError(
        f"q.astype: casting to {jnp.result_type(out).name} destroys {n} of "
        f"{jnp.asarray(v).size} finite nonzero magnitudes in base {bs.name}. "
        f"Cast in a base where this dimension is O(1){_other_base(bs)}, "
        f"or keep {jnp.result_type(v).name} and set "
        f"jax_enable_x64 before the first jax call."
    )
