# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for L-BFGS optimizer."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest
from optax import ScaleByLBFGSState

from kups.relaxation.optax.lbfgs import scale_by_ase_lbfgs

from ...clear_cache import clear_cache  # noqa: F401


class TestScaleByAseLBFGS:
    """Tests for scale_by_ase_lbfgs transform."""

    def test_init_creates_correct_state(self):
        """init_fn should create ScaleByLBFGSState with correct structure."""
        optimizer = scale_by_ase_lbfgs(memory_size=10, alpha=70.0)
        params = jnp.array([1.0, 2.0, 3.0])
        state = optimizer.init(params)

        assert isinstance(state, ScaleByLBFGSState)
        assert state.count == 0
        assert state.diff_params_memory.shape == (10, 3)
        assert state.diff_updates_memory.shape == (10, 3)
        assert state.weights_memory.shape == (10,)

    def test_init_with_pytree(self):
        """init_fn should work with PyTree parameters."""
        optimizer = scale_by_ase_lbfgs(memory_size=5)
        params = {"a": jnp.zeros((10, 3)), "b": jnp.zeros((1, 3, 3))}
        state = optimizer.init(params)

        assert isinstance(state, ScaleByLBFGSState)
        assert state.diff_params_memory["a"].shape == (5, 10, 3)
        assert state.diff_params_memory["b"].shape == (5, 1, 3, 3)

    def test_invalid_memory_size_raises(self):
        """memory_size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="memory_size must be >= 1"):
            scale_by_ase_lbfgs(memory_size=0)

        with pytest.raises(ValueError, match="memory_size must be >= 1"):
            scale_by_ase_lbfgs(memory_size=-1)

    def test_first_update_uses_initial_hessian(self):
        """First update should use 1/alpha scaling."""
        alpha = 70.0
        optimizer = scale_by_ase_lbfgs(memory_size=10, alpha=alpha)
        params = jnp.array([1.0, 2.0, 3.0])
        state = optimizer.init(params)
        gradient = jnp.array([7.0, 14.0, 21.0])

        updates, _ = jax.jit(optimizer.update)(gradient, state, params)

        # First step: updates = gradient / alpha
        expected = gradient / alpha
        npt.assert_allclose(updates, expected, rtol=1e-5)

    def test_count_increments(self):
        """count should increment with each update."""
        optimizer = scale_by_ase_lbfgs(memory_size=5)
        params = jnp.array([1.0, 2.0])
        state = optimizer.init(params)
        gradient = jnp.array([0.1, 0.2])

        update_fn = jax.jit(optimizer.update)

        assert state.count == 0
        _, state = update_fn(gradient, state, params)
        assert state.count == 1
        _, state = update_fn(gradient, state, params)
        assert state.count == 2

    def test_convergence_on_quadratic(self):
        """L-BFGS should converge on a simple quadratic potential."""
        optimizer = optax.chain(
            scale_by_ase_lbfgs(memory_size=10, alpha=1.0),
            optax.scale(-1.0),
        )

        # Quadratic: E = 0.5 * x^2, gradient = x
        x = jnp.array([5.0, -3.0, 2.0])
        state = optimizer.init(x)
        update_fn = jax.jit(optimizer.update)

        for _ in range(15):
            gradient = x
            updates, state = update_fn(gradient, state, x)
            x = optax.apply_updates(x, updates)

        npt.assert_allclose(x, jnp.zeros(3), atol=1e-4)

    def test_memory_wraps_around(self):
        """Memory should wrap around after memory_size steps."""
        memory_size = 3
        optimizer = optax.chain(
            scale_by_ase_lbfgs(memory_size=memory_size, alpha=1.0),
            optax.scale(-1.0),
        )

        x = jnp.array([5.0])
        state = optimizer.init(x)
        update_fn = jax.jit(optimizer.update)

        # Run more steps than memory_size
        for _ in range(memory_size + 5):
            gradient = x
            updates, state = update_fn(gradient, state, x)
            x = optax.apply_updates(x, updates)

        # Should not crash and count should be correct
        assert state[0].count == memory_size + 5
