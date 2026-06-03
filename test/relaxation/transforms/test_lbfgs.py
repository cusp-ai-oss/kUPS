# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system ASE-flavor L-BFGS transform."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest
from jax import Array

from kups.core.data.index import Index
from kups.core.typing import SystemId
from kups.relaxation.transforms.lbfgs import (
    ScaleByAseLbfgs,
    ScaleByAseLbfgsState,
)


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


def _run_grad(
    opt: ScaleByAseLbfgs[Any],
    x: Array,
    state: ScaleByAseLbfgsState,
    steps: int,
    *,
    hess: Array | float = 1.0,
) -> tuple[Array, ScaleByAseLbfgsState]:
    """Run ``steps`` of ``opt`` on a quadratic (gradient ``∇L = hess * x``).

    Folding the loop into a single ``lax.scan`` compiles the step once instead
    of re-tracing every Python iteration; the trajectory is bit-identical.
    ``hess`` is a per-element diagonal Hessian (default identity).
    """

    def body(
        carry: tuple[Any, ScaleByAseLbfgsState], _: None
    ) -> tuple[tuple[Any, ScaleByAseLbfgsState], None]:
        x, state = carry
        upd, state = opt.update(hess * x, state, x)
        x = optax.apply_updates(x, jax.tree.map(lambda u: -u, upd))
        return (x, state), None

    (x, state), _ = jax.lax.scan(body, (x, state), None, length=steps)
    return jnp.asarray(x), state


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
        x, _ = _run_grad(opt, x, opt.init(x), 15)
        npt.assert_allclose(x, jnp.zeros(3), atol=1e-4)

    def test_memory_wraps_around(self):
        memory_size = 3
        opt = ScaleByAseLbfgs(memory_size=memory_size, alpha=1.0)
        x = jnp.array([5.0])
        _, state = _run_grad(opt, x, opt.init(x), memory_size + 5)
        assert int(state.count) == memory_size + 5


class TestScaleByASELBFGSPerSystem:
    def test_batched_matches_separate(self):
        """Headline guarantee: batched run = concatenated independent runs."""

        def run_alone(x0: jnp.ndarray) -> jnp.ndarray:
            opt = ScaleByAseLbfgs(memory_size=5, alpha=10.0)
            x, _ = _run_grad(opt, x0, opt.init(x0), 6)
            return x

        x_a = jnp.array([5.0, -3.0])
        x_b = jnp.array([0.5, 0.2])
        sep = jnp.concatenate([run_alone(x_a), run_alone(x_b)])

        opt = ScaleByAseLbfgs(memory_size=5, alpha=10.0)
        batched = jnp.concatenate([x_a, x_b])
        idx = _system_index([0, 0, 1, 1], 2)
        x, _ = _run_grad(opt, batched, opt.init(batched, index_prefix=idx), 6)
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


class TestScaleByASELBFGSAdaptiveScale:
    """``adaptive_scale`` scales the initial inverse Hessian per system."""

    def test_first_step_falls_back_to_inv_alpha(self):
        """No curvature pair exists yet, so the first step still uses 1/alpha."""
        alpha = 70.0
        opt = ScaleByAseLbfgs(memory_size=10, alpha=alpha, adaptive_scale=True)
        params = jnp.array([1.0, 2.0, 3.0])
        gradient = jnp.array([7.0, 14.0, 21.0])
        updates, _ = opt.update(gradient, opt.init(params), params)
        npt.assert_allclose(updates, gradient / alpha, rtol=1e-5)

    def test_adaptive_gamma_equals_curvature_ratio(self):
        """Second step scales by γ = (s·y)/(y·y) from the fresh pair.

        Inputs are chosen so the second gradient is orthogonal to both ``s``
        and ``y``; the two-loop recursion then collapses to
        ``precond = γ * updates``, exposing γ directly.
        """
        opt = ScaleByAseLbfgs(memory_size=5, alpha=70.0, adaptive_scale=True)
        params0 = jnp.array([0.0, 0.0, 0.0])
        updates0 = jnp.array([-2.0, 1.0, 0.0])
        _, state = opt.update(updates0, opt.init(params0), params0)
        # s = [1,0,0], y = [2,0,0]  →  γ = (s·y)/(y·y) = 2/4 = 0.5.
        params1 = jnp.array([1.0, 0.0, 0.0])
        updates1 = jnp.array([0.0, 1.0, 0.0])
        precond, _ = opt.update(updates1, state, params1)
        npt.assert_allclose(precond, 0.5 * updates1, rtol=1e-6)

    def test_non_positive_curvature_falls_back_to_inv_alpha(self):
        """A non-positive (s·y) curvature pair reverts γ to 1/alpha."""
        alpha = 4.0
        params0 = jnp.array([0.0, 0.0, 0.0])
        updates0 = jnp.array([2.0, 1.0, 0.0])
        params1 = jnp.array([1.0, 0.0, 0.0])
        updates1 = jnp.array([0.0, 1.0, 0.0])
        # s = [1,0,0], y = [-2,0,0]  →  s·y = -2 ≤ 0, so γ = 1/alpha.

        def run(adaptive: bool) -> Array:
            opt = ScaleByAseLbfgs(memory_size=5, alpha=alpha, adaptive_scale=adaptive)
            _, state = opt.update(updates0, opt.init(params0), params0)
            precond, _ = opt.update(updates1, state, params1)
            return precond

        npt.assert_allclose(run(True), updates1 / alpha, rtol=1e-6)
        npt.assert_allclose(run(True), run(False), rtol=1e-6)

    def test_batched_matches_separate_with_per_system_curvature(self):
        """Per-system γ keeps batched runs bit-identical to separate ones.

        Each system has a different diagonal Hessian, so their adaptive γ
        differ; a per-system γ is required for the block-diagonal guarantee.
        """

        def run_alone(x0: Array, hess: Array) -> Array:
            opt = ScaleByAseLbfgs(memory_size=5, alpha=10.0, adaptive_scale=True)
            x, _ = _run_grad(opt, x0, opt.init(x0), 6, hess=hess)
            return x

        x_a, hess_a = jnp.array([1.0, -2.0]), jnp.array([1.0, 4.0])
        x_b, hess_b = jnp.array([0.5, 3.0]), jnp.array([10.0, 0.5])
        sep = jnp.concatenate([run_alone(x_a, hess_a), run_alone(x_b, hess_b)])

        opt = ScaleByAseLbfgs(memory_size=5, alpha=10.0, adaptive_scale=True)
        x0 = jnp.concatenate([x_a, x_b])
        hess = jnp.concatenate([hess_a, hess_b])
        idx = _system_index([0, 0, 1, 1], 2)
        x, _ = _run_grad(opt, x0, opt.init(x0, index_prefix=idx), 6, hess=hess)
        npt.assert_allclose(x, sep, atol=1e-6)

    def test_convergence_on_quadratic(self):
        """Adaptive scaling still drives a quadratic to its minimum."""
        opt = ScaleByAseLbfgs(memory_size=10, alpha=70.0, adaptive_scale=True)
        x = jnp.array([5.0, -3.0, 2.0])
        x, _ = _run_grad(opt, x, opt.init(x), 15)
        npt.assert_allclose(x, jnp.zeros(3), atol=1e-4)
