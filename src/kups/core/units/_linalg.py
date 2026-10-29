# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit-aware linear algebra: contractions and decompositions.

Contractions carry the product of the operands' dimensions; ``det``, ``inv`` and
``solve`` derive their own. Decompositions label each output separately and return
dimensionless factors as plain arrays.
"""

from __future__ import annotations

from fractions import Fraction as Frac
from typing import TYPE_CHECKING, Any, Literal, overload

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

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
    from kups.core.units._core import Quantity
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


def _lowered(*xs: Any) -> tuple[BaseSystem, *tuple[tuple[Any, Dim], ...]]:
    """Joint base of ``xs`` and each operand's ``(magnitudes, dim)`` in it."""
    bs = _joint(*xs)
    return bs, *(_mag(x, bs) for x in xs)


def det(x: Any) -> Any:
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
@overload
def inv(x: Any) -> Any: ...
def inv(x: Any) -> Any:
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
@overload
def cross(a: Any, b: Any, axis: int = ...) -> Any: ...
def cross(a: Any, b: Any, axis: int = -1) -> Any:
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
    **kwargs: Any,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def einsum(
    subscripts: str, a: ArrayLike, /, *, dim: None = ..., **kwargs: Any
) -> jax.Array: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Any,
) -> jax.Array: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Any,
) -> Quantity[_E_M + _F_M, _E_KG + _F_KG, _E_S + _F_S, _E_A + _F_A, _E_K + _F_K]: ...
@overload
def einsum(
    subscripts: str,
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: ArrayLike,
    /,
    *,
    dim: None = ...,
    **kwargs: Any,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def einsum(
    subscripts: str,
    a: ArrayLike,
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    /,
    *,
    dim: None = ...,
    **kwargs: Any,
) -> Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]: ...
@overload
def einsum(
    subscripts: str, a: ArrayLike, b: ArrayLike, /, *, dim: None = ..., **kwargs: Any
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
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
    **kwargs: Any,
) -> jax.Array: ...
@overload
def einsum(*operands: Any, dim: Dim | None = ..., **kwargs: Any) -> Any: ...
def einsum(*operands: Any, dim: Dim | None = None, **kwargs: Any) -> Any:
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
def vdot(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K],
    **kwargs: Any,
) -> jax.Array: ...
@overload
def vdot(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    **kwargs: Any,
) -> Quantity[_E_M + _F_M, _E_KG + _F_KG, _E_S + _F_S, _E_A + _F_A, _E_K + _F_K]: ...
@overload
def vdot(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], b: ArrayLike, **kwargs: Any
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def vdot(
    a: ArrayLike, b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K], **kwargs: Any
) -> Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]: ...
@overload
def vdot(a: ArrayLike, b: ArrayLike, **kwargs: Any) -> jax.Array: ...
@overload
def vdot(a: Any, b: Any, **kwargs: Any) -> Any: ...
def vdot(a: Any, b: Any, **kwargs: Any) -> Any:
    """Flattened inner product, with result dimension ``dim(a) * dim(b)``.

    Args:
      a: left operand.
      b: right operand.
      **kwargs: keyword arguments for ``jnp.vdot``, e.g. ``precision``.

    Returns:
      The result in the operands' base; plain when dimensionless.
    """
    bs, (la, da), (lb, db) = _lowered(a, b)
    return _lift(jnp.vdot(la, lb, **kwargs), dmul(da, db), bs)


@overload
def solve(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
) -> jax.Array: ...
@overload
def solve(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
) -> Quantity[_F_M - _E_M, _F_KG - _E_KG, _F_S - _E_S, _F_A - _E_A, _F_K - _E_K]: ...
@overload
def solve(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], b: ArrayLike
) -> Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]: ...
@overload
def solve(
    a: ArrayLike, b: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
) -> Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]: ...
@overload
def solve(a: ArrayLike, b: ArrayLike) -> jax.Array: ...
@overload
def solve(a: Any, b: Any) -> Any: ...
def solve(a: Any, b: Any) -> Any:
    """Solve ``a x = b``; the solution has dimension ``dim(b)/dim(a)``.

    Args:
      a: the square matrix, of shape ``(..., n, n)``.
      b: the right-hand side.

    Returns:
      The solution in the operands' base; plain when dimensionless.
    """
    bs, (la, da), (lb, db) = _lowered(a, b)
    return _lift(jnp.linalg.solve(la, lb), ddiv(db, da), bs)


@overload
def qr(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    mode: Literal["reduced", "complete"] = ...,
) -> tuple[jax.Array, Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]]: ...
@overload
def qr(a: Any, mode: str = ...) -> Any: ...
def qr(a: Any, mode: str = "reduced") -> Any:
    """QR decomposition; ``Q`` is plain and ``R`` keeps ``a``'s dimension.

    Args:
      a: a quantity or plain array of shape ``(..., m, n)``.
      mode: "reduced", "complete" or "r", as ``jnp.linalg.qr`` takes it.

    Returns:
      ``(Q, R)``, or ``R`` alone for ``mode="r"``.
    """
    bs, (la, da) = _lowered(a)
    out = jnp.linalg.qr(la, mode=mode)
    if mode == "r":
        return _lift(out, da, bs)
    q, r = out
    return q, _lift(r, da, bs)


@overload
def eigh(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    UPLO: str | None = ...,
    symmetrize_input: bool = ...,
) -> tuple[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], jax.Array]: ...
@overload
def eigh(
    a: Any, UPLO: str | None = ..., symmetrize_input: bool = ...
) -> tuple[Any, jax.Array]: ...
def eigh(
    a: Any, UPLO: str | None = None, symmetrize_input: bool = True
) -> tuple[Any, Any]:
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


@overload
def svd(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    full_matrices: bool = ...,
    compute_uv: Literal[True] = ...,
    hermitian: bool = ...,
) -> tuple[jax.Array, Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], jax.Array]: ...
@overload
def svd(
    a: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    full_matrices: bool = ...,
    compute_uv: Literal[False] = ...,
    hermitian: bool = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def svd(
    a: Any, full_matrices: bool = ..., compute_uv: bool = ..., hermitian: bool = ...
) -> Any: ...
def svd(
    a: Any, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False
) -> Any:
    """Singular value decomposition.

    Args:
      a: a quantity or plain array of shape ``(..., m, n)``.
      full_matrices: whether ``u`` and ``vh`` are square.
      compute_uv: whether to return the vectors as well as the values.
      hermitian: whether ``a`` may be assumed Hermitian.

    Returns:
      ``(u, s, vh)`` with plain unitary factors and ``s`` in ``a``'s dimension,
      or ``s`` alone when ``compute_uv`` is False.
    """
    bs, (la, da) = _lowered(a)
    out = jnp.linalg.svd(
        la, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian
    )
    if not compute_uv:
        return _lift(out, da, bs)
    lu, s, vh = out
    return lu, _lift(s, da, bs), vh
