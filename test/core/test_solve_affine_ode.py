# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the augmented-expm affine-ODE solver."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from kups.core.utils.math import solve_affine_ode


def _reference_solution(A: np.ndarray, b: np.ndarray, x0: np.ndarray, dt: float):
    """SciPy reference: x(dt) = expm(A·dt)·x0 + φ₁(A·dt)·dt·b via augmented expm."""
    n = A.shape[-1]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    E = scipy.linalg.expm(M * dt)
    return E[:n, :n] @ x0 + E[:n, n]


def test_solve_affine_matches_scipy_general_matrix():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((3, 3))
    b = rng.standard_normal(3)
    x0 = rng.standard_normal(3)
    dt = 0.137
    expected = _reference_solution(A, b, x0, dt)
    got = np.asarray(
        solve_affine_ode(jnp.asarray(A), jnp.asarray(b), jnp.asarray(x0), dt)
    )
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_solve_affine_matches_scipy_lower_triangular():
    """Lower-triangular A (the kUPS-side coupling matrix in CoupledMomentumStep)."""
    rng = np.random.default_rng(0)
    A = np.tril(rng.standard_normal((3, 3)))
    b = rng.standard_normal(3)
    x0 = rng.standard_normal(3)
    dt = 0.05
    expected = _reference_solution(A, b, x0, dt)
    got = np.asarray(
        solve_affine_ode(jnp.asarray(A), jnp.asarray(b), jnp.asarray(x0), dt)
    )
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_solve_affine_handles_singular_A():
    """A with zero diagonal (singular) must not blow up — the augmented expm handles it."""
    A = jnp.zeros((3, 3))
    b = jnp.array([1.0, 2.0, 3.0])
    x0 = jnp.array([0.5, -0.5, 1.0])
    dt = 0.4
    # ẋ = b  ⇒  x(dt) = x0 + dt·b
    expected = x0 + dt * b
    got = solve_affine_ode(A, b, x0, dt)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_solve_affine_zero_b_reduces_to_matrix_exp():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    x0 = rng.standard_normal(3)
    dt = 0.2
    expected = scipy.linalg.expm(A * dt) @ x0
    got = solve_affine_ode(jnp.asarray(A), jnp.zeros(3), jnp.asarray(x0), dt)
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_solve_affine_zero_A_zero_dt_returns_x0():
    A = jnp.eye(3)
    b = jnp.array([1.0, 2.0, 3.0])
    x0 = jnp.array([5.0, -1.0, 0.0])
    got = solve_affine_ode(A, b, x0, 0.0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(x0), atol=1e-12)


def test_solve_affine_vmap_per_particle():
    """vmap over a leading batch (per-particle solver use in CoupledMomentumStep)."""
    rng = np.random.default_rng(7)
    P = 4
    A = jnp.asarray(np.tril(rng.standard_normal((P, 3, 3))))
    b = jnp.asarray(rng.standard_normal((P, 3)))
    x0 = jnp.asarray(rng.standard_normal((P, 3)))
    dt = jnp.full((P,), 0.1)
    got = jax.vmap(solve_affine_ode)(A, b, x0, dt)
    assert got.shape == (P, 3)
    for p in range(P):
        expected = _reference_solution(
            np.asarray(A[p]), np.asarray(b[p]), np.asarray(x0[p]), 0.1
        )
        np.testing.assert_allclose(np.asarray(got[p]), expected, atol=1e-10, rtol=1e-10)


def test_solve_affine_second_order_convergence_against_euler():
    """Sanity: for small dt, augmented-expm is exact; an Euler step would lag."""
    rng = np.random.default_rng(99)
    A = rng.standard_normal((3, 3))
    b = rng.standard_normal(3)
    x0 = rng.standard_normal(3)
    dt = 0.001
    expected = _reference_solution(A, b, x0, dt)
    # Augmented-expm: exact to machine precision
    got = np.asarray(
        solve_affine_ode(jnp.asarray(A), jnp.asarray(b), jnp.asarray(x0), dt)
    )
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)
    # Euler step: O(dt^2) error
    euler = x0 + dt * (A @ x0 + b)
    euler_err = np.linalg.norm(euler - expected)
    assert euler_err > 1e-15, "Euler must be strictly less accurate than augmented-expm"
