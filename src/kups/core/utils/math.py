# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Mathematical utilities for numerical computations.

This module provides specialized numerical algorithms including logarithmic
factorial ratios, polynomial root finding, and optimized matrix operations for
3×3 matrices.
"""

from enum import StrEnum

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.utils.jax import jit, vectorize


@jit
def log_factorial_ratio(N: Array, M: Array) -> Array:
    """Compute log(N!/M!) efficiently using the log-gamma function.

    Uses the identity `log(n!) = lgamma(n+1)` to compute the ratio directly.

    Args:
        N: Integer array (numerator factorials).
        M: Integer array (denominator factorials).

    Returns:
        Result containing `log(N!/M!)` for each pair of elements.

    Example:
        ```python
        N = jnp.array([10, 5])
        M = jnp.array([5, 8])
        result = log_factorial_ratio(N, M)
        # Computes [log(10!/5!), log(5!/8!)]
        ```
    """
    return jax.lax.lgamma(N + 1.0) - jax.lax.lgamma(M + 1.0)


@jit
@vectorize(signature="(4)->(3)")
def cubic_roots(coefficients: Array) -> Array:
    """Find all roots of a cubic polynomial using the companion matrix method.

    Solves `ax³ + bx² + cx + d = 0` by computing eigenvalues of the companion
    matrix. Returns all three roots (real or complex).

    Args:
        coefficients: Array of shape `(..., 4)` containing `[a, b, c, d]` for each
            cubic polynomial.

    Returns:
        Array of shape `(..., 3)` containing the three roots. May be complex-valued.

    Example:
        ```python
        # Solve x^3 - 6x^2 + 11x - 6 = 0 (roots: 1, 2, 3)
        coeffs = jnp.array([1.0, -6.0, 11.0, -6.0])
        roots = cubic_roots(coeffs)
        ```

    Note:
        This method is numerically stable and handles multiple polynomials in
        parallel via vectorization.
    """
    a, b, c, d = coefficients
    C = jnp.array([[0, 0, -d / a], [1, 0, -c / a], [0, 1, -b / a]], dtype=jnp.float_)
    solutions = jnp.linalg.eigvals(C)
    return solutions


@jit
@vectorize(signature="(3,3)->(),(3,3)")
def det_and_inverse_3x3(A: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute determinant and inverse of 3×3 matrices via the adjugate method.

    Efficiently computes both the determinant and inverse of 3×3 matrices using
    explicit formulas for the cofactor matrix. More efficient than general
    matrix inversion for small matrices.

    Args:
        A: Array of shape `(..., 3, 3)` containing 3×3 matrices.

    Returns:
        Tuple of (determinant, inverse):
            - determinant: Array of shape `(...)` containing scalar determinants.
            - inverse: Array of shape `(..., 3, 3)` containing inverted matrices.

    Example:
        ```python
        A = jnp.eye(3)
        det, A_inv = det_and_inverse_3x3(A)
        # det = 1.0, A_inv = identity matrix
        ```
    """
    assert A.shape == (3, 3), "Input must be a 3x3 matrix"

    # Determinant
    det = (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )
    # Cofactor matrix
    cofactor = jnp.array(
        [
            [
                (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]),
                -(A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]),
                (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]),
            ],
            [
                -(A[0, 1] * A[2, 2] - A[0, 2] * A[2, 1]),
                (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]),
                -(A[0, 0] * A[2, 1] - A[0, 1] * A[2, 0]),
            ],
            [
                (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]),
                -(A[0, 0] * A[1, 2] - A[0, 2] * A[1, 0]),
                (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]),
            ],
        ]
    )
    # Adjugate is transpose of cofactor
    adjugate = cofactor.T
    # Inverse
    return det, adjugate / det


@jit(static_argnames=("lower",))
def triangular_3x3_det_and_inverse(
    A: jax.Array, *, lower: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Compute determinant and inverse of triangular 3×3 matrices.

    Optimized computation for triangular matrices that exploits their structure.
    Determinant is simply the product of diagonal elements, and inverse is
    computed via triangular solve.

    Args:
        A: Array of shape `(..., 3, 3)` containing triangular matrices.
        lower: Whether matrices are lower (True) or upper (False) triangular.

    Returns:
        Tuple of (determinant, inverse):
            - determinant: Array of shape `(...)` containing diagonal products.
            - inverse: Array of shape `(..., 3, 3)` containing inverted matrices.

    Example:
        ```python
        L = jnp.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
        det, L_inv = triangular_3x3_det_and_inverse(L, lower=True)
        # det = 18 (product of diagonals: 1*3*6)
        ```
    """

    @vectorize(signature="(3,3)->(),(3,3)")
    def inner(A: jax.Array) -> tuple[jax.Array, jax.Array]:
        det = jnp.diag(A).prod()
        inv = jax.scipy.linalg.solve_triangular(A, jnp.eye(3), lower=lower)
        return det, inv

    return inner(A)


class MatmulSide(StrEnum):
    """Enumeration for matrix multiplication side."""

    LEFT = "left"
    RIGHT = "right"


@jit(static_argnames=("lower", "side"))
def triangular_3x3_matmul(
    L: Array, x: Array, *, lower: bool = True, side: MatmulSide | str = MatmulSide.RIGHT
) -> Array:
    """Optimized matrix-vector multiplication for triangular 3×3 matrices.

    Specialized implementation that exploits triangular structure to avoid
    computing with known-zero elements. On CPU, uses unrolled loops for better
    performance than einsum.

    Args:
        L: Array of shape `(..., 3, 3)` containing triangular matrices.
        x: Array of shape `(..., 3)` containing vectors to multiply.
        lower: Whether `L` is lower (True) or upper (False) triangular.
        side: Multiplication side:
            - `"right"`: Computes `xL`
            - `"left"`: Computes `Lx`

    Returns:
        Shape `(..., 3)` containing the result.

    Example:
        ```python
        L = jnp.array([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
        x = jnp.array([1.0, 2.0, 3.0])
        result = triangular_3x3_matmul(L, x, lower=True, side="right")
        # Computes L @ x efficiently
        ```

    Note:
        Automatically selects between einsum (GPU) and unrolled loops (CPU) for
        optimal performance.
    """
    side = MatmulSide(side)
    cuda = jax.devices()[0].device_kind == "cuda"

    @vectorize(signature="(3,3),(3)->(3)")
    def inner(L: Array, x: Array):
        if cuda:
            if side is MatmulSide.RIGHT:
                return jnp.einsum("ji,j->i", L, x)
            else:
                return jnp.einsum("ij,j->i", L, x)
        # The following unrolled implementation is faster on CPU for small matrices
        if side is MatmulSide.RIGHT:
            if lower:
                L_0, L_1, L_2 = L[:, 0], L[1:, 1], L[2:, 2]
                x_0, x_1, x_2 = x, x[1:], x[2:]
            else:
                L_0, L_1, L_2 = L[:1, 0], L[:2, 1], L[:3, 2]
                x_0, x_1, x_2 = x[:1], x[:2], x
        elif side is MatmulSide.LEFT:
            if lower:
                L_0, L_1, L_2 = L[0, :1], L[1, :2], L[2, :3]
                x_0, x_1, x_2 = x[:1], x[:2], x
            else:
                L_0, L_1, L_2 = L[0, :], L[1, 1:], L[2, 2:]
                x_0, x_1, x_2 = x, x[1:], x[2:]
        else:
            raise ValueError(f"Invalid side argument: {side}")
        return jnp.stack([L_0 @ x_0, L_1 @ x_1, L_2 @ x_2])

    return inner(L, x)


def next_higher_power(value: Array, base: Array | float = 2.0) -> Array:
    """Compute the next higher power of a given base for each element.

    For each element in `value`, finds the smallest power of `base` that is
    greater than or equal to that element.

    Args:
        value: Array of shape `(...)` containing input values.
        base: Base for the power calculation (default is 2.0).

    Returns:
        Array of shape `(...)` containing the next higher powers.

    Example:
        ```python
        values = jnp.array([3, 5, 10])
        result = next_higher_power(values, base=2)
        # result = [4, 8, 16]
        ```
    """
    value = jnp.ceil(value).astype(int)
    log_base = jnp.log(base)
    exponents = jnp.ceil(jnp.log(value) / log_base)
    result = jnp.power(base, exponents)
    # Linear growth for base <= 1
    return jnp.where(base <= 1, value, result).astype(int)
