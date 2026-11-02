# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit-aware linear algebra: contractions and decompositions.

Contractions carry the product of the operands' dimensions; ``det`` and ``inv``
derive their own. Decompositions label each output separately and return
dimensionless factors as plain arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Frac
from typing import TYPE_CHECKING, Any, Literal, Unpack, overload

import jax
import jax.numpy as jnp
from jax.lax import PrecisionLike
from jax.typing import ArrayLike, DTypeLike
from typing_extensions import TypedDict

from kups.core.units._core import (
    _joint,
    _lift,
    _mag,
    _shape,
    apply,
    dim_of,
)
from kups.core.units._unit import ONE, ddiv, dmul, dpow

if TYPE_CHECKING:
    from kups.core.units._base import BaseSystem
    from kups.core.units._core import AnyQuantity, Quantity
    from kups.core.units._typed import (
        _E_A,
        _E_K,
        _E_KG,
        _E_M,
        _E_S,
        _F_A,
        _F_K,
        _F_KG,
        _F_M,
        _F_S,
        _G_A,
        _G_K,
        _G_KG,
        _G_M,
        _G_S,
    )
    from kups.core.units._unit import Dim


class _DotOptions(TypedDict, total=False, closed=True):
    precision: PrecisionLike
    preferred_element_type: DTypeLike | None


class _EinsumOptions(TypedDict, total=False, closed=True):
    precision: PrecisionLike
    preferred_element_type: DTypeLike | None
    optimize: str | bool | list[tuple[int, ...]]


def _lowered(
    *xs: AnyQuantity | ArrayLike,
) -> tuple[BaseSystem, *tuple[tuple[ArrayLike, Dim], ...]]:
    """Joint base of ``xs`` and each operand's ``(magnitudes, dim)`` in it."""
    bs = _joint(*xs)
    return bs, *(_mag(x, bs) for x in xs)


@overload
def det(x: AnyQuantity) -> AnyQuantity: ...
@overload
def det(x: ArrayLike) -> jax.Array: ...
def det(x: AnyQuantity | ArrayLike) -> AnyQuantity | jax.Array:
    """Determinant of a square matrix.

    Args:
      x: a quantity or plain array of shape ``(..., n, n)``.

    Returns:
      The determinant, of dimension ``dim(x)**n``; plain when dimensionless.
    """
    return apply(jnp.linalg.det, x, dim=dpow(dim_of(x), Frac(_shape(x)[-1])))


@overload
def inv(
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
) -> Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]: ...
@overload
def inv(x: ArrayLike) -> jax.Array: ...
def inv(x: AnyQuantity | ArrayLike) -> AnyQuantity | jax.Array:
    """Inverse of a square matrix.

    Args:
      x: a quantity or plain array of shape ``(..., n, n)``.

    Returns:
      The inverse, of dimension ``1/dim(x)``; plain when dimensionless.
    """
    return apply(jnp.linalg.inv, x, dim=ddiv(ONE, dim_of(x)))


@overload
def cross(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K],
    axis: int = ...,
) -> jax.Array: ...
@overload
def cross(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    axis: int = ...,
) -> Quantity[_E_M + _F_M, _E_KG + _F_KG, _E_S + _F_S, _E_A + _F_A, _E_K + _F_K]: ...
@overload
def cross(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], b: ArrayLike, axis: int = ...
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def cross(
    a: ArrayLike, b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K], axis: int = ...
) -> Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]: ...
@overload
def cross(a: ArrayLike, b: ArrayLike, axis: int = ...) -> jax.Array: ...
def cross(
    a: AnyQuantity | ArrayLike, b: AnyQuantity | ArrayLike, axis: int = -1
) -> AnyQuantity | jax.Array:
    """Cross product, with result dimension ``dim(a) * dim(b)``.

    Args:
      a: a quantity or plain array whose ``axis`` has length 3.
      b: the other operand.
      axis: the axis holding the vector components.

    Returns:
      The cross product in the operands' base; plain when dimensionless.
    """
    bs, (la, da), (lb, db) = _lowered(a, b)
    return _lift(jnp.cross(la, lb, axis=axis), dmul(da, db), bs)


# typed overloads cover up to three operands; more fall back to Any
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> jax.Array: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> jax.Array: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_E_M + _F_M, _E_KG + _F_KG, _E_S + _F_S, _E_A + _F_A, _E_K + _F_K]: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    b: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> jax.Array: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    c: Quantity[
        -_E_M - _F_M,
        -_E_KG - _F_KG,
        -_E_S - _F_S,
        -_E_A - _F_A,
        -_E_K - _F_K,
    ],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> jax.Array: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    c: Quantity[_G_M, _G_KG, _G_S, _G_A, _G_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[
    _E_M + _F_M + _G_M,
    _E_KG + _F_KG + _G_KG,
    _E_S + _F_S + _G_S,
    _E_A + _F_A + _G_A,
    _E_K + _F_K + _G_K,
]: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    c: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_E_M + _F_M, _E_KG + _F_KG, _E_S + _F_S, _E_A + _F_A, _E_K + _F_K]: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: ArrayLike,
    c: Quantity[_G_M, _G_KG, _G_S, _G_A, _G_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_E_M + _G_M, _E_KG + _G_KG, _E_S + _G_S, _E_A + _G_A, _E_K + _G_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    c: Quantity[_G_M, _G_KG, _G_S, _G_A, _G_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_F_M + _G_M, _F_KG + _G_KG, _F_S + _G_S, _F_A + _G_A, _F_K + _G_K]: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: ArrayLike,
    c: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    c: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    b: ArrayLike,
    c: Quantity[_G_M, _G_KG, _G_S, _G_A, _G_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Quantity[_G_M, _G_KG, _G_S, _G_A, _G_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> jax.Array: ...
@overload
def einsum(
    *operands: str | ArrayLike | AnyQuantity | Sequence[int],
    dim: Dim | None = ...,
    **kwargs: Unpack[_EinsumOptions],
) -> Any: ...
def einsum(
    *operands: str | ArrayLike | AnyQuantity | Sequence[int],
    dim: Dim | None = None,
    **kwargs: Unpack[_EinsumOptions],
) -> Any:
    """Einstein summation with unit tracking.

    Args:
      *operands: the subscripts and arrays, in either ``jnp.einsum`` spelling;
        all quantities are lowered into one base.
      dim: result dimension, or None for the product of the operands'.
      **kwargs: keyword arguments for ``jnp.einsum``, e.g. ``precision``.

    Returns:
      The result in the operands' base; plain when dimensionless.
    """
    return apply(jnp.einsum, *operands, dim=dim, **kwargs)


@overload
def qr(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    mode: Literal["reduced", "complete"] = ...,
) -> tuple[jax.Array, Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]]: ...
@overload
def qr(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], mode: Literal["r"]
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def qr(
    a: ArrayLike, mode: Literal["reduced", "complete"] = ...
) -> tuple[jax.Array, jax.Array]: ...
@overload
def qr(a: ArrayLike, mode: Literal["r"]) -> jax.Array: ...
def qr(
    a: AnyQuantity | ArrayLike, mode: Literal["reduced", "complete", "r"] = "reduced"
) -> tuple[jax.Array, AnyQuantity | jax.Array] | AnyQuantity | jax.Array:
    """QR decomposition; ``Q`` is plain and ``R`` keeps ``a``'s dimension.

    Args:
      a: a quantity or plain array of shape ``(..., m, n)``.
      mode: "reduced", "complete" or "r", as ``jnp.linalg.qr`` takes it.

    Returns:
      ``(Q, R)``, or ``R`` alone for ``mode="r"``.
    """
    bs, (la, da) = _lowered(a)
    if mode == "r":
        return _lift(jnp.linalg.qr(la, mode="r"), da, bs)
    q, r = jnp.linalg.qr(la, mode=mode)
    return q, _lift(r, da, bs)


@overload
def eigh(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    UPLO: str | None = ...,
    symmetrize_input: bool = ...,
) -> tuple[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], jax.Array]: ...
@overload
def eigh(
    a: ArrayLike, UPLO: str | None = ..., symmetrize_input: bool = ...
) -> tuple[jax.Array, jax.Array]: ...
def eigh(
    a: AnyQuantity | ArrayLike, UPLO: str | None = None, symmetrize_input: bool = True
) -> tuple[AnyQuantity | jax.Array, jax.Array]:
    """Eigendecomposition of a symmetric matrix.

    Args:
      a: a symmetric (or Hermitian) quantity or plain array.
      UPLO: which triangle to read, "L" or "U".
      symmetrize_input: whether to symmetrize ``a`` first.

    Returns:
      ``(eigenvalues, eigenvectors)``; eigenvalues carry ``a``'s dimension and
      eigenvectors are a plain unit-norm array.
    """
    bs, (la, da) = _lowered(a)
    w, v = jnp.linalg.eigh(la, UPLO=UPLO, symmetrize_input=symmetrize_input)
    return _lift(w, da, bs), v
