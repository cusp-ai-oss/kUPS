# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system Armijo / strong-Wolfe line-search transforms."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.data.index import Index
from kups.core.typing import SystemId
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms.linesearch import (
    LineSearchState,
    ScaleByBacktrackingLinesearch,
    ScaleByZoomLinesearch,
)


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


def _value_and_grad_fn(
    hess: Array, idx: Index[SystemId]
) -> Callable[[Array], tuple[Any, Array]]:
    """Per-element diagonal quadratic ``E = 0.5 Σ hess·x²``: per-system energy + grad."""
    return lambda p: (idx.sum_over(0.5 * hess * p**2), hess * p)


def _update(
    opt: Optimizer[Array, LineSearchState],
    state: LineSearchState,
    x: Array,
    direction: Array,
    vag: Callable[[Array], tuple[Any, Array]],
) -> tuple[Array, LineSearchState]:
    energies, grad = vag(x)
    return opt.update(
        direction, state, x, grad=grad, energies=energies, value_and_grad_fn=vag
    )


def _run(
    opt: Optimizer[Array, LineSearchState],
    x: Array,
    state: LineSearchState,
    steps: int,
    *,
    hess: Array,
    idx: Index[SystemId],
) -> Array:
    """Steepest-descent + line search on the quadratic; compiled once via scan."""
    vag = _value_and_grad_fn(hess, idx)

    def body(carry: tuple[Array, Any], _: None) -> tuple[tuple[Array, Any], None]:
        x, state = carry
        step, state = _update(opt, state, x, -hess * x, vag)
        return (x + step, state), None

    (x, _), _ = jax.lax.scan(body, (x, state), None, length=steps)
    return x


def _single_step(
    opt: Optimizer[Array, LineSearchState],
    x: Array,
    *,
    hess: Array,
    idx: Index[SystemId],
    direction: Array,
) -> Array:
    step, _ = _update(
        opt, opt.init(x, index_prefix=idx), x, direction, _value_and_grad_fn(hess, idx)
    )
    return step


class TestBacktracking:
    def test_unit_step_satisfies_armijo(self):
        """Identity Hessian: the full t=1 step lands on the minimum."""
        opt = ScaleByBacktrackingLinesearch()
        idx = _system_index([0, 0], 1)
        hess = jnp.ones(2)
        x = jnp.array([3.0, 4.0])
        step = _single_step(opt, x, hess=hess, idx=idx, direction=-hess * x)
        npt.assert_allclose(step, -x, rtol=1e-6)

    def test_convergence(self):
        opt = ScaleByBacktrackingLinesearch()
        idx = _system_index([0, 0, 0], 1)
        hess = jnp.ones(3)
        x = jnp.array([5.0, -3.0, 2.0])
        x = _run(opt, x, opt.init(x, index_prefix=idx), 3, hess=hess, idx=idx)
        npt.assert_allclose(x, jnp.zeros(3), atol=1e-5)

    def test_non_descent_direction_does_not_move(self):
        """A non-descent direction (∇L·d ≥ 0) yields a zero step."""
        opt = ScaleByBacktrackingLinesearch()
        idx = _system_index([0, 0], 1)
        hess = jnp.ones(2)
        x = jnp.array([1.0, 2.0])
        step = _single_step(opt, x, hess=hess, idx=idx, direction=hess * x)
        npt.assert_allclose(step, jnp.zeros(2), atol=1e-12)

    def test_batched_matches_separate(self):
        """Headline guarantee: per-system steps decouple across systems."""

        def run_alone(x0: Array, hess: Array) -> Array:
            opt = ScaleByBacktrackingLinesearch()
            idx = _system_index([0] * x0.shape[0], 1)
            return _run(opt, x0, opt.init(x0, index_prefix=idx), 5, hess=hess, idx=idx)

        x_a, hess_a = jnp.array([5.0, -3.0]), jnp.array([1.0, 4.0])
        x_b, hess_b = jnp.array([0.5, 2.0]), jnp.array([10.0, 0.5])
        sep = jnp.concatenate([run_alone(x_a, hess_a), run_alone(x_b, hess_b)])

        opt = ScaleByBacktrackingLinesearch()
        idx = _system_index([0, 0, 1, 1], 2)
        x0 = jnp.concatenate([x_a, x_b])
        hess = jnp.concatenate([hess_a, hess_b])
        x = _run(opt, x0, opt.init(x0, index_prefix=idx), 5, hess=hess, idx=idx)
        npt.assert_allclose(x, sep, atol=1e-6)

    def test_requires_objective_keywords(self):
        """Omitting the per-system objective keywords is rejected."""
        opt = ScaleByBacktrackingLinesearch()
        x = jnp.array([1.0])
        state = opt.init(x)
        with pytest.raises(ValueError, match="value_and_grad_fn"):
            opt.update(-x, state, x, grad=x)


class TestZoom:
    def test_unit_step_on_quadratic(self):
        """Identity Hessian: strong Wolfe accepts the exact step t=1."""
        opt = ScaleByZoomLinesearch()
        idx = _system_index([0, 0], 1)
        hess = jnp.ones(2)
        x = jnp.array([3.0, 4.0])
        step = _single_step(opt, x, hess=hess, idx=idx, direction=-hess * x)
        npt.assert_allclose(step, -x, rtol=1e-6)

    def test_convergence(self):
        opt = ScaleByZoomLinesearch()
        idx = _system_index([0, 0, 0], 1)
        hess = jnp.ones(3)
        x = jnp.array([5.0, -3.0, 2.0])
        x = _run(opt, x, opt.init(x, index_prefix=idx), 3, hess=hess, idx=idx)
        npt.assert_allclose(x, jnp.zeros(3), atol=1e-5)

    def test_non_descent_direction_does_not_move(self):
        opt = ScaleByZoomLinesearch()
        idx = _system_index([0, 0], 1)
        hess = jnp.ones(2)
        x = jnp.array([1.0, 2.0])
        step = _single_step(opt, x, hess=hess, idx=idx, direction=hess * x)
        npt.assert_allclose(step, jnp.zeros(2), atol=1e-12)

    def test_curvature_condition_satisfied(self):
        """The accepted step meets the strong-Wolfe curvature bound per system."""
        opt = ScaleByZoomLinesearch(c2=0.9)
        idx = _system_index([0, 0], 1)
        hess = jnp.array([1.0, 8.0])  # ill-conditioned: t=1 overshoots
        x = jnp.array([1.0, 1.0])
        direction = -hess * x
        step = _single_step(opt, x, hess=hess, idx=idx, direction=direction)
        dphi0 = float(jnp.vdot(hess * x, direction))
        dphi_t = float(jnp.vdot(hess * (x + step), direction))
        assert abs(dphi_t) <= -0.9 * dphi0 + 1e-6

    def test_batched_matches_separate(self):
        def run_alone(x0: Array, hess: Array) -> Array:
            opt = ScaleByZoomLinesearch()
            idx = _system_index([0] * x0.shape[0], 1)
            return _run(opt, x0, opt.init(x0, index_prefix=idx), 5, hess=hess, idx=idx)

        x_a, hess_a = jnp.array([5.0, -3.0]), jnp.array([1.0, 4.0])
        x_b, hess_b = jnp.array([0.5, 2.0]), jnp.array([10.0, 0.5])
        sep = jnp.concatenate([run_alone(x_a, hess_a), run_alone(x_b, hess_b)])

        opt = ScaleByZoomLinesearch()
        idx = _system_index([0, 0, 1, 1], 2)
        x0 = jnp.concatenate([x_a, x_b])
        hess = jnp.concatenate([hess_a, hess_b])
        x = _run(opt, x0, opt.init(x0, index_prefix=idx), 5, hess=hess, idx=idx)
        npt.assert_allclose(x, sep, atol=1e-6)
