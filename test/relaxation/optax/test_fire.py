# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for FIRE optimizer."""

import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.relaxation.optax.fire import ScaleByFireState, scale_by_fire

from ...clear_cache import clear_cache  # noqa: F401


class TestScaleByFire:
    """Tests for scale_by_fire transform."""

    @classmethod
    def setup_class(cls):
        cls.optimizer = scale_by_fire(dt_start=0.1)

    def test_init(self):
        """init_fn creates correct state for arrays and pytrees."""
        params = jnp.array([1.0, 2.0, 3.0])
        state = self.optimizer.init(params)
        assert isinstance(state, ScaleByFireState)
        npt.assert_array_equal(state.velocity, jnp.zeros(3))
        npt.assert_allclose(state.dt, 0.1)
        npt.assert_allclose(state.alpha, 0.1)
        assert state.n_pos == 0

        # PyTree params
        optimizer = scale_by_fire()
        params_tree = {"a": jnp.zeros((10, 3)), "b": jnp.zeros((1, 3, 3))}
        state_tree = optimizer.init(params_tree)
        assert isinstance(state_tree, ScaleByFireState)
        npt.assert_array_equal(state_tree.velocity["a"], jnp.zeros((10, 3)))
        npt.assert_array_equal(state_tree.velocity["b"], jnp.zeros((1, 3, 3)))

    def test_positive_power_increases_n_pos(self):
        """When P > 0, n_pos should increase."""
        optimizer = scale_by_fire(dt_start=0.1, n_min=2)
        params = jnp.array([1.0])
        state = optimizer.init(params)
        gradient = jnp.array([-1.0])

        for i in range(3):
            _, state = optimizer.update(gradient, state, params)

        assert state.n_pos > 0

    def test_dt_bounded(self):
        """dt should not exceed dt_max and not go below dt_min."""
        # dt_max
        optimizer = scale_by_fire(dt_start=0.1, dt_max=0.2, n_min=1, f_inc=2.0)
        params = jnp.array([1.0])
        state = optimizer.init(params)
        gradient = jnp.array([-1.0])
        for _ in range(20):
            _, state = optimizer.update(gradient, state, params)
        assert float(state.dt) <= 0.2

        # dt_min
        optimizer = scale_by_fire(dt_start=0.1, dt_min=0.01, f_dec=0.5)
        state = ScaleByFireState(
            velocity=jnp.array([1.0]),
            dt=jnp.array(0.1),
            alpha=jnp.array(0.1),
            n_pos=jnp.array(0, dtype=jnp.int32),
        )
        gradient = jnp.array([1.0])
        for _ in range(10):
            _, state = optimizer.update(gradient, state, params)
            state = ScaleByFireState(
                velocity=jnp.array([1.0]),
                dt=state.dt,
                alpha=state.alpha,
                n_pos=state.n_pos,
            )
        assert float(state.dt) >= 0.01

    def test_dt_increases_after_n_min(self):
        """dt should increase after n_min positive power steps."""
        optimizer = scale_by_fire(dt_start=0.1, n_min=2, f_inc=1.5)
        params = jnp.array([1.0])
        state = optimizer.init(params)
        initial_dt = float(state.dt)
        gradient = jnp.array([-1.0])

        for _ in range(5):
            _, state = optimizer.update(gradient, state, params)

        assert float(state.dt) > initial_dt

    def test_negative_power_resets_velocity(self):
        """When P <= 0, velocity resets to zero and update is zero."""
        state = ScaleByFireState(
            velocity=jnp.array([1.0]),
            dt=jnp.array(0.1),
            alpha=jnp.array(0.1),
            n_pos=jnp.array(3, dtype=jnp.int32),
        )
        params = jnp.array([1.0])
        gradient = jnp.array([1.0])
        updates, new_state = self.optimizer.update(gradient, state, params)

        npt.assert_array_equal(new_state.velocity, jnp.zeros(1))
        assert new_state.n_pos == 0
        npt.assert_array_equal(updates, jnp.zeros(1))

    def test_composable_with_chain(self):
        """FIRE should work with optax.chain."""
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm=1.0),
            scale_by_fire(dt_start=0.1),
        )
        params = jnp.array([1.0, 2.0, 3.0])
        state = optimizer.init(params)
        gradient = jnp.array([0.1, 0.2, 0.3])

        updates, new_state = optimizer.update(gradient, state, params)

        assert updates.shape == (3,)

    def test_positive_power_moves_downhill(self):
        """With positive power, FIRE converges to the minimum over a few steps."""
        optimizer = scale_by_fire(dt_start=0.1, max_step=0.5)

        x = jnp.array([3.0, -2.0])
        state = optimizer.init(x)
        initial_energy = 0.5 * jnp.sum(x**2)

        for _ in range(5):
            gradient = x
            updates, state = optimizer.update(gradient, state, x)
            x = optax.apply_updates(x, updates)

        new_energy = 0.5 * jnp.sum(x**2)
        assert new_energy < initial_energy

    def test_max_step_clips_updates(self):
        """Position updates should be clipped to max_step."""
        optimizer = scale_by_fire(dt_start=1.0, dt_max=10.0, max_step=0.1)
        params = jnp.array([1.0, 0.0, 0.0])

        state = ScaleByFireState(
            velocity=jnp.array([100.0, 0.0, 0.0]),
            dt=jnp.array(1.0),
            alpha=jnp.array(0.1),
            n_pos=jnp.array(10, dtype=jnp.int32),
        )

        gradient = jnp.array([-10.0, 0.0, 0.0])
        updates, _ = optimizer.update(gradient, state, params)

        update_norm = float(jnp.linalg.norm(updates))
        assert update_norm <= 0.1 + 1e-6

    def test_max_step_none_disables_clipping(self):
        """Setting max_step=None should disable clipping."""
        optimizer = scale_by_fire(dt_start=1.0, max_step=None)
        params = jnp.array([1.0])

        state = ScaleByFireState(
            velocity=jnp.array([100.0]),
            dt=jnp.array(1.0),
            alpha=jnp.array(0.1),
            n_pos=jnp.array(10, dtype=jnp.int32),
        )

        gradient = jnp.array([-10.0])
        updates, _ = optimizer.update(gradient, state, params)

        assert float(jnp.abs(updates[0])) > 1.0

    def test_convergence_on_quadratic(self):
        """FIRE should converge on a simple quadratic potential."""
        optimizer = scale_by_fire(dt_start=0.05, dt_max=0.5, max_step=0.5)

        x = jnp.array([5.0])
        state = optimizer.init(x)

        for _ in range(100):
            gradient = x
            updates, state = optimizer.update(gradient, state, x)
            x = optax.apply_updates(x, updates)

        npt.assert_allclose(x, jnp.zeros(1), atol=1e-2)

    def test_alpha_resets_on_negative_power(self):
        """Alpha should reset to alpha_start when P <= 0."""
        alpha_start = 0.1
        optimizer = scale_by_fire(
            dt_start=0.1, alpha_start=alpha_start, n_min=1, f_alpha=0.5
        )
        params = jnp.array([1.0])
        state = optimizer.init(params)

        gradient = jnp.array([-1.0])
        for _ in range(5):
            _, state = optimizer.update(gradient, state, params)
        assert float(state.alpha) < alpha_start

        state = ScaleByFireState(
            velocity=jnp.array([1.0]),
            dt=state.dt,
            alpha=state.alpha,
            n_pos=state.n_pos,
        )
        gradient = jnp.array([1.0])
        _, state = optimizer.update(gradient, state, params)

        npt.assert_allclose(state.alpha, alpha_start)
