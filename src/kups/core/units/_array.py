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

from kups.core.units._core import (
    Quantity,
    _agree,
    _lift,
    _no_index,
    _norm_impl,
    _sqrt_impl,
    apply,
)
from kups.core.units._precision import _check_dtype

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
    from kups.core.units._unit import Unit


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
