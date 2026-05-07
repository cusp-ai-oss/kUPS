# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system ASE-flavor L-BFGS transform."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

from kups.core.data.index import Index
from kups.core.typing import SystemId
from kups.relaxation.transforms.lbfgs import (
    ScaleByAseLbfgs,
    ScaleByAseLbfgsState,
)

from ...clear_cache import clear_cache  # noqa: F401


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


class TestScaleByASELBFGSGlobalFallback:
    """When index_prefix is None, behavior matches the original optax LBFGS."""

    def test_init_creates_correct_state(self):
        opt = ScaleByAseLbfgs(memory_size=10, alpha=70.0)
        params = jnp.array([1.0, 2.0, 3.0])
        state = opt.init(params)
        assert isinstance(state, ScaleByAseLbfgsState)
        assert int(state.count) == 0
        assert state.diff_params_memory.shape == (10, 3)
        assert state.diff_updates_memory.shape == (10, 3)
        # Per-system weights: (n_systems=1, memory_size=10).
        assert state.weights_memory.data.shape == (1, 10)

    def test_init_with_pytree(self):
        opt = ScaleByAseLbfgs(memory_size=5)
        params = {"a": jnp.zeros((10, 3)), "b": jnp.zeros((1, 3, 3))}
        state = opt.init(params)
        assert state.diff_params_memory["a"].shape == (5, 10, 3)
        assert state.diff_params_memory["b"].shape == (5, 1, 3, 3)

    def test_invalid_memory_size_raises(self):
        with pytest.raises(ValueError, match="memory_size must be >= 1"):
            ScaleByAseLbfgs(memory_size=0)
        with pytest.raises(ValueError, match="memory_size must be >= 1"):
            ScaleByAseLbfgs(memory_size=-1)

    def test_first_update_uses_initial_hessian(self):
        """First step: precond updates = gradient / alpha."""
        alpha = 70.0
        opt = ScaleByAseLbfgs(memory_size=10, alpha=alpha)
        params = jnp.array([1.0, 2.0, 3.0])
        state = opt.init(params)
        gradient = jnp.array([7.0, 14.0, 21.0])
        updates, _ = opt.update(gradient, state, params)
        npt.assert_allclose(updates, gradient / alpha, rtol=1e-5)

    def test_count_increments(self):
        opt = ScaleByAseLbfgs(memory_size=5)
        params = jnp.array([1.0, 2.0])
        state = opt.init(params)
        gradient = jnp.array([0.1, 0.2])
        assert int(state.count) == 0
        _, state = opt.update(gradient, state, params)
        assert int(state.count) == 1
        _, state = opt.update(gradient, state, params)
        assert int(state.count) == 2

    def test_convergence_on_quadratic(self):
        opt = ScaleByAseLbfgs(memory_size=10, alpha=1.0)
        x = jnp.array([5.0, -3.0, 2.0])
        state = opt.init(x)
        for _ in range(15):
            upd, state = opt.update(x, state, x)
            x = optax.apply_updates(x, jax.tree.map(lambda u: -u, upd))
        npt.assert_allclose(x, jnp.zeros(3), atol=1e-4)

    def test_memory_wraps_around(self):
        memory_size = 3
        opt = ScaleByAseLbfgs(memory_size=memory_size, alpha=1.0)
        x = jnp.array([5.0])
        state = opt.init(x)
        for _ in range(memory_size + 5):
            upd, state = opt.update(x, state, x)
            x = optax.apply_updates(x, jax.tree.map(lambda u: -u, upd))
        assert int(state.count) == memory_size + 5


class TestScaleByASELBFGSPerSystem:
    def test_batched_matches_separate(self):
        """Headline guarantee: batched run = concatenated independent runs."""

        def run_alone(x0: jnp.ndarray) -> jnp.ndarray:
            opt = ScaleByAseLbfgs(memory_size=5, alpha=10.0)
            state = opt.init(x0)
            x = x0
            for _ in range(6):
                upd, state = opt.update(x, state, x)
                x = optax.apply_updates(x, jax.tree.map(lambda u: -u, upd))
            return jnp.asarray(x)

        x_a = jnp.array([5.0, -3.0])
        x_b = jnp.array([0.5, 0.2])
        sep = jnp.concatenate([run_alone(x_a), run_alone(x_b)])

        opt = ScaleByAseLbfgs(memory_size=5, alpha=10.0)
        batched = jnp.concatenate([x_a, x_b])
        idx = _system_index([0, 0, 1, 1], 2)
        state = opt.init(batched, index_prefix=idx)
        x = batched
        for _ in range(6):
            upd, state = opt.update(x, state, x)
            x = optax.apply_updates(x, jax.tree.map(lambda u: -u, upd))

        npt.assert_allclose(x, sep, atol=1e-6)

    def test_per_system_block_diagonal(self):
        """A huge gradient on system A must not contaminate system B's update."""
        opt = ScaleByAseLbfgs(memory_size=4, alpha=1.0)
        idx = _system_index([0, 0, 1, 1], 2)
        params = jnp.array([1.0, 1.0, 0.1, 0.1])
        state = opt.init(params, index_prefix=idx)

        # System A: large gradient. System B: tiny gradient.
        gradient = jnp.array([1000.0, 1000.0, 0.01, 0.01])
        updates, _ = opt.update(gradient, state, params)

        # First step is preconditioned by 1/alpha = 1.0, so update == gradient.
        npt.assert_allclose(updates[0:2], jnp.array([1000.0, 1000.0]), rtol=1e-5)
        npt.assert_allclose(updates[2:4], jnp.array([0.01, 0.01]), rtol=1e-5)

    def test_weights_memory_per_system_storage(self):
        """ρᵢ weights are stored independently per system."""
        opt = ScaleByAseLbfgs(memory_size=4, alpha=1.0)
        idx = _system_index([0, 0, 1, 1], 2)
        params = jnp.array([1.0, 1.0, 0.5, 0.5])
        state = opt.init(params, index_prefix=idx)

        # Two updates so the second step records a real ρ in slot 0.
        for _ in range(2):
            upd, state = opt.update(params, state, params)
            params = optax.apply_updates(params, jax.tree.map(lambda u: -u, upd))

        # weights shape (n_systems=2, memory_size=4); slot 0 has been written.
        assert state.weights_memory.data.shape == (2, 4)
        # Each system's slot-0 ρ depends only on its own y·s, so the two
        # systems generally end up with different ρ values.
        rho_a = float(state.weights_memory.data[0, 0])
        rho_b = float(state.weights_memory.data[1, 0])
        assert rho_a != rho_b
