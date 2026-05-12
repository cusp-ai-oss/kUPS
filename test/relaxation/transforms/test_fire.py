# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system FIRE transform.

``ScaleByFire`` takes ``updates`` as the force ``F = -∇L`` (the descent
direction), matching the optax composability convention. Tests therefore
pass ``-grad`` (or pre-flipped forces) into :meth:`update`.
"""

import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import SystemId
from kups.relaxation.optimizer import chain
from kups.relaxation.transforms.clip_by_global_norm import ClipByGlobalNorm
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
        # L(x) = 0.5·x²  ⇒  ∇L = x, force F = -x.
        new = ScaleByFire(dt_start=0.05, dt_max=0.5)
        x = jnp.array([5.0])
        state = new.init(x)
        for _ in range(100):
            upd, state = new.update(-x, state, x)
            x = jnp.asarray(optax.apply_updates(x, upd))
        npt.assert_allclose(x, jnp.zeros(1), atol=1e-2)


class TestScaleByFirePerSystem:
    def test_batched_matches_separate(self):
        """Batched run must equal concatenation of independent per-system runs."""

        def run_alone(x0: jnp.ndarray) -> jnp.ndarray:
            opt = ScaleByFire(dt_start=0.1)
            state = opt.init(x0)
            x = x0
            for _ in range(8):
                upd, state = opt.update(-x, state, x)
                x = jnp.asarray(optax.apply_updates(x, upd))
            return x

        x_a = jnp.array([5.0, -3.0])
        x_b = jnp.array([0.5, 0.2])
        sep = jnp.concatenate([run_alone(x_a), run_alone(x_b)])

        opt = ScaleByFire(dt_start=0.1)
        batched = jnp.concatenate([x_a, x_b])
        idx = _system_index([0, 0, 1, 1], 2)
        state = opt.init(batched, index_prefix=idx)
        x = batched
        for _ in range(8):
            upd, state = opt.update(-x, state, x)
            x = jnp.asarray(optax.apply_updates(x, upd))

        npt.assert_allclose(x, sep, atol=1e-6)

    def test_dt_evolves_per_system(self):
        """One system makes progress (dt grows), the other oscillates (dt shrinks)."""
        opt = ScaleByFire(dt_start=0.1, n_min=2, f_inc=1.5, f_dec=0.5)
        idx = _system_index([0, 1], 2)
        x = jnp.array([1.0, 1.0])
        state = opt.init(x, index_prefix=idx)

        # System 0: constant force → P>0 each step → dt grows.
        # System 1: force flips sign each step → P<0 → dt shrinks.
        for step in range(6):
            force = jnp.array([1.0, -1.0 if step % 2 == 0 else 1.0])
            _, state = opt.update(force, state, x)

        assert float(state.dt.data[0]) > 0.1  # system 0 grew
        assert float(state.dt.data[1]) < 0.1  # system 1 shrank

    def test_negative_power_resets_velocity_per_system(self):
        """When P <= 0 in one system only, only that system's velocity zeros out."""
        idx = _system_index([0, 0, 1, 1], 2)
        params = jnp.array([[0.0], [0.0], [0.0], [0.0]])

        # Hand-construct a state with non-zero velocity for both systems.
        keys = (SystemId(0), SystemId(1))
        state = ScaleByFireState(
            velocity=jnp.array([[1.0], [1.0], [1.0], [1.0]]),
            dt=Table(keys, jnp.array([0.1, 0.1])),
            alpha=Table(keys, jnp.array([0.1, 0.1])),
            n_pos=Table(keys, jnp.array([3, 3], dtype=jnp.int32)),
            index_prefix=idx,
        )

        # Force pointing AGAINST current velocity for system 0
        # (so v · F < 0 → P < 0 → reset). Aligned for system 1.
        force = jnp.array([[-1.0], [-1.0], [1.0], [1.0]])
        opt = ScaleByFire(dt_start=0.1)
        _, new_state = opt.update(force, state, params)

        npt.assert_allclose(new_state.velocity[0:2], jnp.zeros((2, 1)), atol=1e-6)
        assert float(jnp.linalg.norm(new_state.velocity[2:4])) > 0.0
        assert int(new_state.n_pos.data[0]) == 0  # system 0 reset
        assert int(new_state.n_pos.data[1]) == 4  # system 1 incremented


class TestScaleByFireComposability:
    """``ScaleByFire`` should compose cleanly with other optax/kups transforms."""

    def test_chain_with_sign_flip_matches_force_input(self):
        """``chain(optax.scale(-1), ScaleByFire())`` accepts ∇L and gives the
        same trajectory as passing ``-∇L`` directly to a bare ``ScaleByFire``."""
        x0 = jnp.array([5.0])

        direct = ScaleByFire(dt_start=0.05, dt_max=0.5)
        composed = chain(optax.scale(-1.0), ScaleByFire(dt_start=0.05, dt_max=0.5))

        x_direct = x0
        s_direct = direct.init(x_direct)
        x_composed = x0
        s_composed = composed.init(x_composed)

        for _ in range(20):
            # Quadratic: ∇L = x, force = -x.
            upd_d, s_direct = direct.update(-x_direct, s_direct, x_direct)
            x_direct = jnp.asarray(optax.apply_updates(x_direct, upd_d))
            upd_c, s_composed = composed.update(x_composed, s_composed, x_composed)
            x_composed = jnp.asarray(optax.apply_updates(x_composed, upd_c))

        npt.assert_allclose(x_composed, x_direct, atol=1e-6)

    def test_chain_with_clip_caps_input_force(self):
        """Prepending ``ClipByGlobalNorm`` clips the force seen by FIRE."""
        idx = _system_index([0, 0], 1)
        x = jnp.array([10.0, 0.0])

        dt = 0.1
        max_norm = 1.0
        opt = chain(ClipByGlobalNorm(max_norm=max_norm), ScaleByFire(dt_start=dt))
        state = opt.init(x, index_prefix=idx)

        # Huge unclipped force: ||F||=100. After clip ||F̂||=max_norm.
        # First step from v=0 yields Δx = dt² · F̂, so |Δx|∞ ≤ dt² · max_norm.
        upd, _ = opt.update(jnp.array([100.0, 0.0]), state, x)
        assert float(jnp.max(jnp.abs(upd))) <= dt * dt * max_norm + 1e-6
