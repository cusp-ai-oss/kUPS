# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system FIRE transform."""

import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.core.data.index import Index
from kups.core.typing import SystemId
from kups.relaxation.transforms.fire import ScaleByFire, ScaleByFireState

from ...clear_cache import clear_cache  # noqa: F401


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


class TestScaleByFireGlobalFallback:
    """When index_prefix is None, behavior matches the original optax FIRE."""

    def test_init_zero_velocity(self):
        x = jnp.array([1.0, 2.0, 3.0])
        state = ScaleByFire(dt_start=0.1).init(x)
        assert isinstance(state, ScaleByFireState)
        npt.assert_array_equal(state.velocity, jnp.zeros(3))
        npt.assert_allclose(state.dt.data, jnp.array([0.1]))
        npt.assert_allclose(state.alpha.data, jnp.array([0.1]))
        npt.assert_array_equal(state.n_pos.data, jnp.array([0], dtype=jnp.int32))

    def test_init_with_pytree(self):
        params = {"a": jnp.zeros((4, 3)), "b": jnp.zeros((1, 3, 3))}
        state = ScaleByFire().init(params)
        npt.assert_array_equal(state.velocity["a"], jnp.zeros((4, 3)))
        npt.assert_array_equal(state.velocity["b"], jnp.zeros((1, 3, 3)))

    def test_convergence_on_quadratic(self):
        new = ScaleByFire(dt_start=0.05, dt_max=0.5)
        x = jnp.array([5.0])
        state = new.init(x)
        for _ in range(100):
            upd, state = new.update(x, state, x)
            x = optax.apply_updates(x, upd)
        npt.assert_allclose(x, jnp.zeros(1), atol=1e-2)


class TestScaleByFirePerSystem:
    def test_batched_matches_separate(self):
        """Batched run must equal concatenation of independent per-system runs."""

        def run_alone(x0: jnp.ndarray) -> jnp.ndarray:
            opt = ScaleByFire(dt_start=0.1)
            state = opt.init(x0)
            x = x0
            for _ in range(8):
                upd, state = opt.update(x, state, x)
                x = optax.apply_updates(x, upd)
            return jnp.asarray(x)

        x_a = jnp.array([5.0, -3.0])
        x_b = jnp.array([0.5, 0.2])
        sep = jnp.concatenate([run_alone(x_a), run_alone(x_b)])

        opt = ScaleByFire(dt_start=0.1)
        batched = jnp.concatenate([x_a, x_b])
        idx = _system_index([0, 0, 1, 1], 2)
        state = opt.init(batched, index_prefix=idx)
        x = batched
        for _ in range(8):
            upd, state = opt.update(x, state, x)
            x = optax.apply_updates(x, upd)

        npt.assert_allclose(x, sep, atol=1e-6)

    def test_dt_evolves_per_system(self):
        """One system makes progress (dt grows), the other oscillates (dt shrinks)."""
        opt = ScaleByFire(dt_start=0.1, n_min=2, f_inc=1.5, f_dec=0.5)
        # System 0: positive-power oscillator-friendly start.
        # System 1: gradient flipped each step → power goes negative.
        idx = _system_index([0, 1], 2)
        x = jnp.array([1.0, 1.0])
        state = opt.init(x, index_prefix=idx)

        # Drive system 0 with consistently negative gradient (downhill);
        # drive system 1 with alternating gradient sign.
        for step in range(6):
            grad = jnp.array([-1.0, 1.0 if step % 2 == 0 else -1.0])
            _, state = opt.update(grad, state, x)

        assert float(state.dt.data[0]) > 0.1  # system 0 grew
        assert float(state.dt.data[1]) < 0.1  # system 1 shrank

    def test_negative_power_resets_velocity_per_system(self):
        """When P <= 0 in one system only, only that system's velocity zeros out."""
        idx = _system_index([0, 0, 1, 1], 2)
        params = jnp.array([[0.0], [0.0], [0.0], [0.0]])

        # Hand-construct a state with non-zero velocity for both systems.
        keys = (SystemId(0), SystemId(1))
        from kups.core.data.table import Table

        state = ScaleByFireState(
            velocity=jnp.array([[1.0], [1.0], [1.0], [1.0]]),
            dt=Table(keys, jnp.array([0.1, 0.1])),
            alpha=Table(keys, jnp.array([0.1, 0.1])),
            n_pos=Table(keys, jnp.array([3, 3], dtype=jnp.int32)),
            index_prefix=idx,
        )

        # Gradient pointing AGAINST current velocity for system 0
        # (so v · F < 0 → P < 0 → reset). Aligned for system 1.
        grad = jnp.array([[1.0], [1.0], [-1.0], [-1.0]])
        opt = ScaleByFire(dt_start=0.1)
        _, new_state = opt.update(grad, state, params)

        npt.assert_allclose(new_state.velocity[0:2], jnp.zeros((2, 1)), atol=1e-6)
        assert float(jnp.linalg.norm(new_state.velocity[2:4])) > 0.0
        assert int(new_state.n_pos.data[0]) == 0  # system 0 reset
        assert int(new_state.n_pos.data[1]) == 4  # system 1 incremented
