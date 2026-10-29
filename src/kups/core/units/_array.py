# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Array functions for quantities.

Each wrapper strips the units, calls the matching jnp function and relabels the
result, so the lowered HLO matches plain-array code. ``apply`` does the same for any
other function, labelling the result with the product of the operands' dimensions
unless ``dim=`` names it; it does not check that the operands agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from kups.core.units._base import active_base
from kups.core.units._core import (
    Quantity,
    _agree,
    _joint,
    _lift,
    _lower,
    _mag,
    _no_index,
    _norm_impl,
    _operand_dtype,
    _reduce_axis,
    _sqrt_impl,
    apply,
    dim_of,
)
from kups.core.units._precision import _check_dtype, _checked_cast
from kups.core.units._unit import ONE, _require_unit

if TYPE_CHECKING:
    from kups.core.units._core import (
        Acceleration,
        Action,
        Charge,
        Compressibility,
        CoulombConstant,
        Current,
        Density,
        DipoleMoment,
        Energy,
        Entropy,
        Force,
        Frequency,
        Length,
        Mass,
        Momentum,
        NumberDensity,
        Permittivity,
        Power,
        Pressure,
        Stiffness,
        Temperature,
        Time,
        Velocity,
        Voltage,
        Volume,
        Wavenumber,
    )
    from kups.core.units._typed import (
        _E_A,
        _E_K,
        _E_KG,
        _E_M,
        _E_S,
    )
    from kups.core.units._unit import Dimensionless, Unit


# named overloads catch odd-exponent dimensions, which type-level ``//`` would floor
@overload
def sqrt(x: Acceleration) -> Any: ...
@overload
def sqrt(x: Action) -> Any: ...
@overload
def sqrt(x: Charge) -> Any: ...
@overload
def sqrt(x: Compressibility) -> Any: ...
@overload
def sqrt(x: CoulombConstant) -> Any: ...
@overload
def sqrt(x: Current) -> Any: ...
@overload
def sqrt(x: Density) -> Any: ...
@overload
def sqrt(x: DipoleMoment) -> Any: ...
@overload
def sqrt(x: Energy) -> Any: ...
@overload
def sqrt(x: Entropy) -> Any: ...
@overload
def sqrt(x: Force) -> Any: ...
@overload
def sqrt(x: Frequency) -> Any: ...
@overload
def sqrt(x: Length) -> Any: ...
@overload
def sqrt(x: Mass) -> Any: ...
@overload
def sqrt(x: Momentum) -> Any: ...
@overload
def sqrt(x: NumberDensity) -> Any: ...
@overload
def sqrt(x: Permittivity) -> Any: ...
@overload
def sqrt(x: Power) -> Any: ...
@overload
def sqrt(x: Pressure) -> Any: ...
@overload
def sqrt(x: Stiffness) -> Any: ...
@overload
def sqrt(x: Temperature) -> Any: ...
@overload
def sqrt(x: Time) -> Any: ...
@overload
def sqrt(x: Velocity) -> Any: ...
@overload
def sqrt(x: Voltage) -> Any: ...
@overload
def sqrt(x: Volume) -> Any: ...
@overload
def sqrt(x: Wavenumber) -> Any: ...
@overload
def sqrt(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
) -> Quantity[_E_M // 2, _E_KG // 2, _E_S // 2, _E_A // 2, _E_K // 2]: ...
@overload
def sqrt(x: ArrayLike) -> jax.Array: ...
@overload
def sqrt(x: Any) -> Any: ...
def sqrt(x: Any) -> Any:
    """Square root; halves each dimension exponent, which may become fractional.

    Args:
      x: a quantity or plain array.

    Returns:
      The root in ``x``'s base; plain when dimensionless.
    """
    return _sqrt_impl(x)


# Reductions keeping the dimension. ``min``, ``max`` and ``sum`` shadow builtins,
# which this module never uses.


@overload
def sum(x: ArrayLike, axis: Any = ..., keepdims: bool = ...) -> jax.Array: ...
@overload
def sum(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    axis: Any = ...,
    keepdims: bool = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def sum(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Sum over an axis, keeping the dimension.

    Args:
      x: a quantity or plain array.
      axis: axis or axes to reduce, or None for all.
      keepdims: whether to keep the reduced axes with length 1.

    Returns:
      The sum in ``x``'s base; plain when dimensionless.
    """
    return apply(jnp.sum, x, axis=axis, keepdims=keepdims)


@overload
def mean(x: ArrayLike, axis: Any = ..., keepdims: bool = ...) -> jax.Array: ...
@overload
def mean(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    axis: Any = ...,
    keepdims: bool = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def mean(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Mean over an axis, keeping the dimension.

    Args:
      x: a quantity or plain array.
      axis: axis or axes to reduce, or None for all.
      keepdims: whether to keep the reduced axes with length 1.

    Returns:
      The mean in ``x``'s base; plain when dimensionless.
    """
    return apply(jnp.mean, x, axis=axis, keepdims=keepdims)


@overload
def min(x: ArrayLike, axis: Any = ..., keepdims: bool = ...) -> jax.Array: ...
@overload
def min(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    axis: Any = ...,
    keepdims: bool = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def min(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Minimum over an axis, keeping the dimension.

    Args:
      x: a quantity or plain array.
      axis: axis or axes to reduce, or None for all.
      keepdims: whether to keep the reduced axes with length 1.

    Returns:
      The minimum in ``x``'s base; plain when dimensionless.

    Raises:
      DimensionError: if ``axis`` is a quantity.
    """
    _reduce_axis("min", axis)
    return apply(jnp.min, x, axis=axis, keepdims=keepdims)


@overload
def max(x: ArrayLike, axis: Any = ..., keepdims: bool = ...) -> jax.Array: ...
@overload
def max(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    axis: Any = ...,
    keepdims: bool = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def max(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Maximum over an axis, keeping the dimension.

    Args:
      x: a quantity or plain array.
      axis: axis or axes to reduce, or None for all.
      keepdims: whether to keep the reduced axes with length 1.

    Returns:
      The maximum in ``x``'s base; plain when dimensionless.

    Raises:
      DimensionError: if ``axis`` is a quantity.
    """
    _reduce_axis("max", axis)
    return apply(jnp.max, x, axis=axis, keepdims=keepdims)


def prod(x: Any, axis: Any = None, dtype: Any = None, keepdims: bool = False) -> Any:
    """Product over an axis; the dimension is raised to the number of factors.

    Args:
      x: a quantity or plain array.
      axis: axis or axes to reduce, or None for all.
      dtype: dtype to accumulate in, or None for ``x``'s.
      keepdims: whether to keep the reduced axes with length 1.

    Returns:
      The product, of dimension ``dim(x)**n`` for ``n`` reduced factors; plain
      when dimensionless.
    """
    if isinstance(x, Quantity):
        return x.prod(axis=axis, dtype=dtype, keepdims=keepdims)
    return jnp.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


@overload
def absolute(x: ArrayLike) -> jax.Array: ...
@overload
def absolute(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def absolute(x: Any) -> Any:
    """Elementwise absolute value; ``abs(q)`` is equivalent.

    Args:
      x: a quantity or plain array.

    Returns:
      The absolute value in ``x``'s base; plain when dimensionless.
    """
    return apply(jnp.abs, x)


# Ops whose operands share one dimension, which the result keeps.


@overload
def minimum(x: ArrayLike, y: ArrayLike) -> jax.Array: ...
@overload
def minimum(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    y: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def minimum(x: Any, y: Any) -> Any:
    """Elementwise smaller of two operands of the same dimension.

    Args:
      x: a quantity or a plain array.
      y: the other operand, of ``x``'s dimension.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if the operands' dimensions differ.
    """
    mags, dim, bs = _agree([x, y], "take the elementwise minimum of")
    return _lift(jnp.minimum(mags[0], mags[1]), dim, bs)


@overload
def maximum(x: ArrayLike, y: ArrayLike) -> jax.Array: ...
@overload
def maximum(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    y: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def maximum(x: Any, y: Any) -> Any:
    """Elementwise larger of two operands of the same dimension.

    Args:
      x: a quantity or a plain array.
      y: the other operand, of ``x``'s dimension.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if the operands' dimensions differ.
    """
    mags, dim, bs = _agree([x, y], "take the elementwise maximum of")
    return _lift(jnp.maximum(mags[0], mags[1]), dim, bs)


def norm(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ord: Any = None,
    axis: Any = None,
    keepdims: bool = False,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
    """Vector or matrix norm, keeping the operand's dimension.

    Args:
      x: a quantity or plain array.
      ord: order of the norm, as ``jnp.linalg.norm`` takes it.
      axis: axis or axes to reduce, or None for the flattened array.
      keepdims: whether to keep the reduced axes with length 1.

    Returns:
      The norm in ``x``'s base; plain when dimensionless.

    Raises:
      DimensionError: if ``ord`` is 0 and ``x`` has a dimension, since that norm
        is a count.
    """
    return _norm_impl(x, ord, axis, keepdims)


@overload
def zeros_like(x: ArrayLike, dtype: Any = None, shape: Any = None) -> jax.Array: ...
@overload
def zeros_like(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    dtype: Any = None,
    shape: Any = None,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def zeros_like(x: Any, dtype: Any = None, shape: Any = None) -> Any:
    """Zeros with ``x``'s dimension and base, and its shape by default.

    Args:
      x: a quantity or a plain array.
      dtype: dtype of the result, or None for ``x``'s.
      shape: shape of the result, or None for ``x``'s.

    Returns:
      Zeros with ``x``'s dimension and base; plain for plain input.
    """
    return apply(jnp.zeros_like, x, dtype, shape)


@overload
def ones_like(x: ArrayLike, dtype: Any = None, shape: Any = None) -> jax.Array: ...
@overload
def ones_like(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    dtype: Any = None,
    shape: Any = None,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def ones_like(x: Any, dtype: Any = None, shape: Any = None) -> Any:
    """Ones with ``x``'s dimension and base, and its shape by default.

    Args:
      x: a quantity or a plain array.
      dtype: dtype of the result, or None for ``x``'s.
      shape: shape of the result, or None for ``x``'s.

    Returns:
      Ones with ``x``'s dimension and base; plain for plain input.
    """
    return apply(jnp.ones_like, x, dtype, shape)


@overload
def full_like(
    x: ArrayLike, fill: ArrayLike, dtype: Any = None, shape: Any = None
) -> jax.Array: ...
@overload
def full_like(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    fill: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
    | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]
    | float,
    dtype: Any = None,
    shape: Any = None,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def full_like(x: Any, fill: Any, dtype: Any = None, shape: Any = None) -> Any:
    """Array like ``x`` filled with ``fill``, keeping ``x``'s dimension.

    Args:
      x: a quantity or a plain array.
      fill: fill value of ``x``'s dimension, a Unit meaning one of it, or a plain
        zero.
      dtype: dtype of the result, or None for ``x``'s.
      shape: shape of the result, or None for ``x``'s.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if ``fill``'s dimension is not ``x``'s, or the result has a
        dimension but a non-floating dtype.
    """
    mags, dim, bs = _agree([x, fill], "fill")
    out = jnp.full_like(mags[0], mags[1], dtype, shape)
    _check_dtype(out.dtype, dim, bs, "full_like")
    return _lift(out, dim, bs)


@overload
def where(cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> jax.Array: ...
@overload
def where(
    cond: ArrayLike,
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float,
    y: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def where(cond: Any, x: Any, y: Any) -> Any:
    """Elementwise choice between two operands of the same dimension.

    Args:
      cond: boolean array; a quantity is refused.
      x: value where ``cond`` is true.
      y: value where ``cond`` is false.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if ``cond`` is a quantity, or ``x`` and ``y`` differ in
        dimension.
    """
    _no_index("where", "the condition", cond)
    mags, dim, bs = _agree([x, y], "select between")
    return _lift(jnp.where(cond, mags[0], mags[1]), dim, bs)


@overload
def clip(
    x: ArrayLike, lo: ArrayLike | None = None, hi: ArrayLike | None = None
) -> jax.Array: ...
@overload
def clip(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    lo: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float | None = None,
    hi: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float | None = None,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def clip(x: Any, lo: Any = None, hi: Any = None) -> Any:
    """Clamp ``x`` between bounds of its dimension.

    Args:
      x: a quantity or plain array.
      lo: lower bound of ``x``'s dimension, or None.
      hi: upper bound of ``x``'s dimension, or None.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if a bound's dimension is not ``x``'s.
    """
    bounds = [b for b in (lo, hi) if b is not None]
    mags, dim, bs = _agree([x, *bounds], "clip")
    rest = iter(mags[1:])
    low = None if lo is None else next(rest)
    high = None if hi is None else next(rest)
    return _lift(jnp.clip(mags[0], low, high), dim, bs)


@overload
def select_n(which: ArrayLike, *cases: ArrayLike) -> jax.Array: ...
@overload
def select_n(
    which: ArrayLike, *cases: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def select_n(which: Any, *cases: Any) -> Any:
    """Elementwise choice between cases of one dimension, by index.

    Args:
      which: integer or boolean index into ``cases``; a quantity is refused.
      *cases: the candidates, all of one dimension.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if ``which`` is a quantity, or two cases' dimensions differ.
    """
    _no_index("select_n", "which", which)
    mags, dim, bs = _agree(list(cases), "select between")
    return _lift(jax.lax.select_n(which, *mags), dim, bs)


@overload
def stack(qs: Sequence[ArrayLike], axis: int = 0) -> jax.Array: ...
@overload
def stack(
    qs: Sequence[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float], axis: int = 0
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def stack(qs: Any, axis: int = 0) -> Any:
    """Stack quantities of the same dimension along a new axis.

    Args:
      qs: the operands, all of one dimension.
      axis: index of the new axis.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if two operands' dimensions differ.
    """
    mags, dim, bs = _agree(list(qs), "stack")
    return _lift(jnp.stack(mags, axis=axis), dim, bs)


@overload
def concatenate(qs: Sequence[ArrayLike], axis: int = 0) -> jax.Array: ...
@overload
def concatenate(
    qs: Sequence[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float], axis: int = 0
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def concatenate(qs: Any, axis: int = 0) -> Any:
    """Join quantities of the same dimension along an existing axis.

    Args:
      qs: the operands, all of one dimension.
      axis: axis to join along.

    Returns:
      The result in the common base; plain when dimensionless.

    Raises:
      DimensionError: if two operands' dimensions differ.
    """
    mags, dim, bs = _agree(list(qs), "concatenate")
    return _lift(jnp.concatenate(mags, axis=axis), dim, bs)


@overload
def linspace(
    start: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    stop: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float,
    num: int = ...,
    endpoint: bool = ...,
    dtype: Any = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = ...,
    endpoint: bool = ...,
    dtype: Any = ...,
) -> jax.Array: ...
@overload
def linspace(
    start: Any, stop: Any, num: int = ..., endpoint: bool = ..., dtype: Any = ...
) -> Any: ...
def linspace(
    start: Any, stop: Any, num: int = 50, endpoint: bool = True, dtype: Any = None
) -> Any:
    """Evenly spaced values between two endpoints, keeping their dimension.

    Args:
      start: first value.
      stop: last value, of ``start``'s dimension.
      num: number of values.
      endpoint: whether ``stop`` is included.
      dtype: dtype of the result, or None for the endpoints'.

    Returns:
      The result in the operands' base; plain when dimensionless.

    Raises:
      DimensionError: if the operands' dimensions differ, or the result has a
        dimension but a non-floating dtype.
    """
    mags, dim, bs = _agree([start, stop], "space")
    dt = jnp.result_type(*mags, float) if dtype is None else jnp.result_type(dtype)
    _check_dtype(dt, dim, bs, "linspace")
    out = jnp.linspace(mags[0], mags[1], num, endpoint, dtype=dtype)
    return _lift(out, dim, bs)


# Constructors from a shape and a unit.


@overload
def zeros(shape: Any, unit: Dimensionless, dtype: Any = ...) -> jax.Array: ...
@overload
def zeros(
    shape: Any, unit: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K], dtype: Any = ...
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def zeros(shape: Any, unit: Any, dtype: Any = None) -> Any:
    """Zeros of the given shape, carrying ``unit``'s dimension.

    Args:
      shape: shape of the result.
      unit: the unit whose dimension the result carries.
      dtype: dtype of the result, or None for jax's default float.

    Returns:
      The zeros in the active base; plain when dimensionless.

    Raises:
      TypeError: if ``unit`` is not a Unit.
      DimensionError: if ``unit`` has a dimension and ``dtype`` is not floating.
    """
    _require_unit(unit, "the unit of the zeros")
    bs = active_base()
    _check_dtype(
        jnp.result_type(float if dtype is None else dtype), unit.dim, bs, "zeros"
    )
    return _lift(jnp.zeros(shape, dtype), unit.dim, bs)


@overload
def ones(shape: Any, unit: Dimensionless, dtype: Any = ...) -> jax.Array: ...
@overload
def ones(
    shape: Any, unit: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K], dtype: Any = ...
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def ones(shape: Any, unit: Any, dtype: Any = None) -> Any:
    """Array of the given shape with every element equal to one ``unit``.

    Args:
      shape: shape of the result.
      unit: the unit each element is one of.
      dtype: dtype of the result, or None for jax's default float.

    Returns:
      The result in the active base; plain when dimensionless.

    Raises:
      TypeError: if ``unit`` is not a Unit.
      DimensionError: if ``unit`` has a dimension and ``dtype`` is not floating, or
        the unit's magnitude does not fit in it.
    """
    _require_unit(unit, "the unit of the ones")
    return _full_impl(shape, unit, dtype, "ones")


@overload
def full(shape: Any, fill: ArrayLike, dtype: Any = ...) -> jax.Array: ...
@overload
def full(
    shape: Any,
    fill: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    dtype: Any = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
def full(shape: Any, fill: Any, dtype: Any = None) -> Any:
    """Array of the given shape filled with ``fill``, keeping its dimension.

    Args:
      shape: shape of the result.
      fill: a quantity, a Unit (meaning one of it), or a plain value.
      dtype: dtype of the result, or None for ``fill``'s.

    Returns:
      The result in ``fill``'s base; plain when dimensionless.

    Raises:
      DimensionError: if ``fill`` has a dimension and ``dtype`` is not floating,
        or a narrowing cast destroys a concrete finite nonzero magnitude.
    """
    return _full_impl(shape, fill, dtype, "full")


def _full_impl(shape: Any, fill: Any, dtype: Any, name: str) -> Any:
    """Check the fill's dtype, convert it to the base, and broadcast it to ``shape``."""
    bs = _joint(fill)
    dt = _operand_dtype((fill,), dtype)
    if dt is None:
        dt = jnp.result_type(_lower(fill))
    dim = dim_of(fill)
    _check_dtype(dt, dim, bs, name)
    # a dimensionless unit's scale is a float even for an integer result
    scale_dtype = dt if jnp.issubdtype(dt, jnp.inexact) else jnp.result_type(float)
    m, _ = _mag(fill, bs, scale_dtype)
    if dim != ONE and dtype is not None:
        m = _checked_cast(m, dtype, bs)
    return _lift(jnp.full(shape, m, dtype), dim, bs)
