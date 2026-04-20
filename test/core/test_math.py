# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.utils.math import (
    MatmulSide,
    cubic_roots,
    det_and_inverse_3x3,
    log_factorial_ratio,
    next_higher_power,
    triangular_3x3_det_and_inverse,
    triangular_3x3_matmul,
)


class TestLogFactorialRatio:
    def test_log_factorial_ratio(self):
        """Merged: scalar cases, array cases, edge cases."""
        # basic: log(5!/3!) = log(20)
        npt.assert_allclose(
            log_factorial_ratio(jnp.array([5]), jnp.array([3])),
            jnp.log(5 * 4),
            rtol=1e-10,
        )
        # equal
        npt.assert_allclose(
            log_factorial_ratio(jnp.array([4]), jnp.array([4])),
            jnp.array([0.0]),
            rtol=1e-10,
        )
        # swapped
        npt.assert_allclose(
            log_factorial_ratio(jnp.array([3]), jnp.array([5])),
            -jnp.log(5 * 4),
            rtol=1e-10,
        )
        # zero values
        npt.assert_allclose(
            log_factorial_ratio(jnp.array([0]), jnp.array([0])),
            jnp.array([0.0]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            log_factorial_ratio(jnp.array([3]), jnp.array([0])),
            jnp.log(6),
            rtol=1e-10,
        )
        # large
        npt.assert_allclose(
            log_factorial_ratio(jnp.array([100]), jnp.array([95])),
            jnp.sum(jnp.log(jnp.arange(96, 101))),
            rtol=1e-10,
        )
        # vectorized
        N = jnp.array([5, 4, 10])
        M = jnp.array([3, 4, 7])
        expected = jnp.array([jnp.log(5 * 4), 0.0, jnp.log(10 * 9 * 8)])
        npt.assert_allclose(log_factorial_ratio(N, M), expected, rtol=1e-10)
        # broadcasting
        N = jnp.array([[5, 6], [7, 8]])
        M = jnp.array([3, 4])
        expected = jnp.array(
            [
                [jnp.log(5 * 4), jnp.log(6 * 5)],
                [jnp.log(7 * 6 * 5 * 4), jnp.log(8 * 7 * 6 * 5)],
            ]
        )
        npt.assert_allclose(log_factorial_ratio(N, M), expected, rtol=1e-10)
        # data types
        result_int = log_factorial_ratio(
            jnp.array([5], dtype=jnp.int32), jnp.array([3], dtype=jnp.int32)
        )
        result_float = log_factorial_ratio(
            jnp.array([5.0], dtype=jnp.float64), jnp.array([3.0], dtype=jnp.float64)
        )
        npt.assert_allclose(result_int, result_float, rtol=1e-6)
        # symmetry
        N, M = jnp.array([7]), jnp.array([4])
        npt.assert_allclose(
            log_factorial_ratio(N, M), -log_factorial_ratio(M, N), rtol=1e-10
        )


class TestCubicRoots:
    def test_cubic_roots(self):
        """Merged: standard, complex, degenerate, vectorized."""
        # (x-1)(x-2)(x-3)
        roots = cubic_roots(jnp.array([1, -6, 11, -6]))
        npt.assert_allclose(
            jnp.sort(roots.real), jnp.array([1.0, 2.0, 3.0]), rtol=1e-10
        )
        npt.assert_allclose(roots.imag, 0, atol=1e-12)

        # x^3 - 1: one real root at 1
        roots = cubic_roots(jnp.array([1, 0, 0, -1]))
        real_roots = roots[jnp.abs(roots.imag) < 1e-12].real
        npt.assert_allclose(real_roots, jnp.array([1.0]), rtol=1e-10)
        assert len(real_roots) == 1

        # (x-2)^3
        roots = cubic_roots(jnp.array([1, -6, 12, -8]))
        npt.assert_allclose(roots.real, 2.0, rtol=1e-4)

        # non-monic
        roots = cubic_roots(jnp.array([2, -12, 22, -12]))
        npt.assert_allclose(
            jnp.sort(roots.real), jnp.array([1.0, 2.0, 3.0]), rtol=1e-10
        )

    def test_complex_coefficients(self):
        """Cube roots of unity verification."""
        roots = cubic_roots(jnp.array([1, 0, 0, -1]))
        for root in roots:
            npt.assert_allclose(root**3, 1.0 + 0j, rtol=1e-10)

    def test_degenerate_and_vectorized(self):
        """Merged: degenerate + vectorized + verification + stability."""
        # double root
        roots = cubic_roots(jnp.array([1, -5, 7, -3]))
        npt.assert_allclose(jnp.sort(roots.real), jnp.array([1.0, 1.0, 3.0]), rtol=1e-6)

        # zero leading
        assert cubic_roots(jnp.array([0, 1, -4, 4])).shape == (3,)

        # vectorized
        roots_batch = cubic_roots(jnp.array([[1, 0, 0, -1], [1, 0, 0, -8]]))
        assert roots_batch.shape == (2, 3)
        npt.assert_allclose(
            roots_batch[0][jnp.abs(roots_batch[0].imag) < 1e-12].real,
            jnp.array([1.0]),
            rtol=1e-10,
        )
        npt.assert_allclose(
            roots_batch[1][jnp.abs(roots_batch[1].imag) < 1e-12].real,
            jnp.array([2.0]),
            rtol=1e-10,
        )

        # root verification
        coeffs = jnp.array([1, -2, -1, 2])
        roots = cubic_roots(coeffs)
        a, b, c, d = coeffs
        for root in roots:
            npt.assert_allclose(a * root**3 + b * root**2 + c * root + d, 0, atol=1e-12)

        # numerical stability
        roots = cubic_roots(jnp.array([1e-10, -1, 2, -1]))
        assert jnp.all(jnp.isfinite(roots))


class TestDetAndInverse3x3:
    def test_det_and_inverse(self):
        """Merged: correctness, properties, edge_cases."""
        # identity
        det, inv = det_and_inverse_3x3(jnp.eye(3))
        npt.assert_allclose(det, 1.0, rtol=1e-10)
        npt.assert_allclose(inv, jnp.eye(3), rtol=1e-10)

        # diagonal
        A = jnp.diag(jnp.array([2.0, 3.0, 4.0]))
        det, inv = det_and_inverse_3x3(A)
        npt.assert_allclose(det, 24.0, rtol=1e-10)
        npt.assert_allclose(inv, jnp.diag(jnp.array([1 / 2, 1 / 3, 1 / 4])), rtol=1e-10)

        # random
        key = jax.random.PRNGKey(42)
        A = jax.random.normal(key, (3, 3)) + 2 * jnp.eye(3)
        det, inv = det_and_inverse_3x3(A)
        npt.assert_allclose(det, jnp.linalg.det(A), rtol=1e-10)
        npt.assert_allclose(inv, jnp.linalg.inv(A), rtol=1e-10)

        # A * A^-1 = I
        for A in [
            jnp.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float),
            jnp.array([[2, -1, 0], [1, 2, 1], [0, -1, 1]], dtype=float),
        ]:
            _, inv = det_and_inverse_3x3(A)
            npt.assert_allclose(A @ inv, jnp.eye(3), rtol=1e-5, atol=1e-7)

        # det(A^T) = det(A)
        A = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        det, _ = det_and_inverse_3x3(A)
        det_T, _ = det_and_inverse_3x3(A.T)
        npt.assert_allclose(det, det_T, rtol=1e-10)

        # det(cA) = c^3 * det(A)
        A2 = jnp.array([[1, 2, 0], [3, 1, 1], [0, 1, 2]], dtype=float)
        c = 2.5
        det_A, inv_A = det_and_inverse_3x3(A2)
        det_cA, inv_cA = det_and_inverse_3x3(c * A2)
        npt.assert_allclose(det_cA, c**3 * det_A, rtol=1e-10)
        npt.assert_allclose(inv_cA, (1 / c) * inv_A, rtol=1e-6, atol=1e-8)

        # inv @ A == I (reverse product check)
        for A in [
            jnp.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=float),
            jnp.array([[2, -1, 0], [1, 2, 1], [0, -1, 1]], dtype=float),
        ]:
            _, inv = det_and_inverse_3x3(A)
            npt.assert_allclose(inv @ A, jnp.eye(3), rtol=1e-5, atol=1e-7)

        # near-singular numerical stability
        A_near = jnp.eye(3).at[0, 0].set(1 + 1e-10)
        det_near, inv_near = det_and_inverse_3x3(A_near)
        npt.assert_allclose(det_near, 1 + 1e-10, rtol=1e-6)
        npt.assert_allclose(A_near @ inv_near, jnp.eye(3), rtol=1e-5, atol=1e-7)

        # singular
        singular_A = jnp.array([[1, 2, 3], [2, 4, 6], [1, 2, 3]], dtype=float)
        det, inv = det_and_inverse_3x3(singular_A)
        npt.assert_allclose(det, 0.0, atol=1e-10)

        # vectorized
        A_batch = jax.random.normal(jax.random.PRNGKey(123), (5, 3, 3)) + jnp.eye(3)
        det_batch, inv_batch = det_and_inverse_3x3(A_batch)
        assert det_batch.shape == (5,)
        for i in range(5):
            npt.assert_allclose(det_batch[i], jnp.linalg.det(A_batch[i]), rtol=1e-10)

        # data types
        A_f32 = jnp.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]], dtype=jnp.float32)
        det_f32, _ = det_and_inverse_3x3(A_f32)
        det_f64, _ = det_and_inverse_3x3(A_f32.astype(jnp.float64))
        npt.assert_allclose(det_f32, det_f64, rtol=1e-6)
        assert det_f32.dtype == jnp.float32


class TestTriangular3x3:
    def test_det_and_inverse(self):
        """Merged: determinant + inverse for both lower and upper."""
        for lower in [True, False]:
            A = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
            A = jnp.tril(A) if lower else jnp.triu(A)
            det, inv = triangular_3x3_det_and_inverse(A, lower=lower)
            npt.assert_allclose(det, jnp.linalg.det(A), rtol=1e-10)
            npt.assert_allclose(inv, jnp.linalg.inv(A), atol=1e-15)


class TestTriangular3x3Matmul:
    @pytest.fixture(
        params=[
            {"lower": True, "side": MatmulSide.LEFT},
            {"lower": True, "side": MatmulSide.RIGHT},
            {"lower": False, "side": MatmulSide.LEFT},
            {"lower": False, "side": MatmulSide.RIGHT},
        ]
    )
    def matmul_params(self, request):
        return request.param

    def test_triangular3x3_matmul(self, matmul_params):
        A = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 3))
        L = jnp.tril(A) if matmul_params["lower"] else jnp.triu(A)
        rh = L.mT if matmul_params["side"] is MatmulSide.RIGHT else L
        result = triangular_3x3_matmul(L, x, **matmul_params)
        expected = jnp.einsum("ij,bj->bi", rh, x)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_invalid_side_argument(self):
        A = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
        x = jax.random.normal(jax.random.PRNGKey(1), (3,))
        with pytest.raises(ValueError, match="is not a valid MatmulSide"):
            triangular_3x3_matmul(jnp.tril(A), x, lower=True, side="invalid")


class TestNextHigherPower:
    def test_next_higher_power(self):
        """Merged: powers_of_two, exact_power, base_three."""
        npt.assert_array_equal(
            next_higher_power(jnp.array([3, 5, 9, 17]), base=2),
            jnp.array([4, 8, 16, 32]),
        )
        npt.assert_array_equal(
            next_higher_power(jnp.array([4, 8, 16]), base=2),
            jnp.array([4, 8, 16]),
        )
        npt.assert_array_equal(
            next_higher_power(jnp.array([4, 10, 28]), base=3),
            jnp.array([9, 27, 81]),
        )


if __name__ == "__main__":
    pytest.main([__file__])
