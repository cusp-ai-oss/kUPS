# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system FIRE 2.0 / ABC-FIRE transform.

``ScaleByFire2`` takes ``updates`` as the force ``F = -∇L`` (the descent
direction), matching the optax composability convention. Tests therefore
pass ``-grad`` (or pre-flipped forces) into :meth:`update`.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
from jax import Array

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import SystemId
from kups.relaxation.optimizer import Optimizer, chain
from kups.relaxation.transforms.clip_by_global_norm import ClipByGlobalNorm
from kups.relaxation.transforms.fire2 import ScaleByFire2, ScaleByFire2State
from kups.relaxation.transforms.max_step_size import MaxStepSize


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


def _run_force(
    opt: Optimizer[Array, Any], x: Array, state: Any, steps: int
) -> tuple[Array, Any]:
    """Run ``steps`` of ``opt`` on a quadratic (force ``F = -x``) via one scan.

    Folding the loop into a single ``lax.scan`` compiles the step once instead
    of re-tracing every Python iteration; the trajectory is bit-identical.
    """

    def body(carry: tuple[Array, Any], _: None) -> tuple[tuple[Array, Any], None]:
        x, state = carry
        upd, state = opt.update(-x, state, x)
        return (jnp.asarray(optax.apply_updates(x, upd)), state), None

    (x, state), _ = jax.lax.scan(body, (x, state), None, length=steps)
    return x, state


def _single_sys_state(
    velocity: jnp.ndarray,
    dt: float = 0.1,
    alpha: float = 0.25,
    n_pos: int = 0,
    n_total: int = 0,
) -> ScaleByFire2State:
    """Hand-construct a one-system FIRE 2.0 state for white-box tests."""
    keys = (SystemId(0),)
    return ScaleByFire2State(
        velocity=velocity,
        dt=Table(keys, jnp.asarray([dt], dtype=jnp.float32)),
        alpha=Table(keys, jnp.asarray([alpha], dtype=jnp.float32)),
        n_pos=Table(keys, jnp.asarray([n_pos], dtype=jnp.int32)),
        n_total=jnp.asarray(n_total, dtype=jnp.int32),
        index_prefix=_system_index([0] * velocity.shape[0], 1),
    )


class TestScaleByFire2GlobalFallback:
    """Single-system behavior: ``index_prefix=None`` falls back to one segment."""

    def test_init(self):
        opt = ScaleByFire2(dt_start=0.1, alpha_start=0.25)
        params = jnp.array([1.0, 2.0, 3.0])
        state = opt.init(params)
        assert isinstance(state, ScaleByFire2State)
        npt.assert_array_equal(state.velocity, jnp.zeros(3))
        npt.assert_allclose(state.dt.data, jnp.array([0.1]))
        npt.assert_allclose(state.alpha.data, jnp.array([0.25]))
        assert int(state.n_pos.data[0]) == 0
        assert int(state.n_total) == 0

    def test_init_pytree(self):
        opt = ScaleByFire2()
        params = {"a": jnp.zeros((4, 3)), "b": jnp.zeros((1, 3, 3))}
        state = opt.init(params)
        assert isinstance(state.velocity, dict)
        npt.assert_array_equal(state.velocity["a"], jnp.zeros((4, 3)))
        npt.assert_array_equal(state.velocity["b"], jnp.zeros((1, 3, 3)))

    def test_n_total_increments_each_step(self):
        opt = ScaleByFire2(dt_start=0.1, n_min=2)
        params = jnp.array([1.0])
        state = opt.init(params)
        force = jnp.array([1.0])  # F = -∇L for ∇L = -1.0
        for i in range(4):
            _, state = opt.update(force, state, params)
            assert int(state.n_total) == i + 1

    def test_positive_power_increases_n_pos(self):
        opt = ScaleByFire2(dt_start=0.1, n_min=2, delaystep_start=False)
        params = jnp.array([1.0])
        state = opt.init(params)
        force = jnp.array([1.0])
        for _ in range(4):
            _, state = opt.update(force, state, params)
        assert int(state.n_pos.data[0]) > 0

    def test_dt_increases_after_n_min_positive_steps(self):
        opt = ScaleByFire2(
            dt_start=0.1,
            dt_max=10.0,
            n_min=2,
            f_inc=1.5,
            delaystep_start=False,
            max_step=None,
        )
        params = jnp.array([1.0])
        state = opt.init(params)
        initial_dt = float(state.dt.data[0])
        force = jnp.array([1.0])
        for _ in range(6):
            _, state = opt.update(force, state, params)
        assert float(state.dt.data[0]) > initial_dt

    def test_dt_decreases_on_negative_power(self):
        opt = ScaleByFire2(dt_start=0.1, dt_min=1e-6, f_dec=0.5, n_min=2)
        state = _single_sys_state(
            velocity=jnp.array([1.0]), dt=0.1, n_pos=5, n_total=10
        )
        params = jnp.array([1.0])
        force = jnp.array([-1.0])  # F=-1, v=+1 → P=-1
        _, new_state = opt.update(force, state, params)
        npt.assert_allclose(float(new_state.dt.data[0]), 0.05)

    def test_dt_bounded_by_dt_min(self):
        opt = ScaleByFire2(dt_start=0.02, dt_min=0.01, f_dec=0.5, n_min=1)
        params = jnp.array([1.0])
        state = _single_sys_state(
            velocity=jnp.array([1.0]), dt=0.02, n_pos=0, n_total=10
        )
        force = jnp.array([-1.0])
        for _ in range(10):
            _, state = opt.update(force, state, params)
            # Re-inject opposing velocity to keep P<=0.
            state = _single_sys_state(
                velocity=jnp.array([1.0]),
                dt=float(state.dt.data[0]),
                alpha=float(state.alpha.data[0]),
                n_pos=int(state.n_pos.data[0]),
                n_total=int(state.n_total),
            )
        assert float(state.dt.data[0]) >= 0.01 - 1e-6

    def test_halfstepback_applies_on_negative_power(self):
        opt = ScaleByFire2(
            dt_start=0.1,
            max_step=None,
            halfstepback=True,
            delaystep_start=False,
        )
        v_old = jnp.array([2.0])
        state = _single_sys_state(velocity=v_old, dt=0.1, n_pos=0, n_total=10)
        force = jnp.array([-1.0])  # F=-1, v=+2 → P=-2
        params = jnp.array([0.0])
        updates, new_state = opt.update(force, state, params)
        # On P<=0: v_pre=0, v_int = dtv*F; new_velocity is v_int. Δx = new_dt*v_int + backtrack.
        new_dt = float(new_state.dt.data[0])
        v_int = new_dt * (-1.0)
        backtrack = -0.5 * new_dt * float(v_old[0])
        expected = new_dt * v_int + backtrack
        npt.assert_allclose(float(jnp.asarray(updates)[0]), expected, rtol=1e-6)

    def test_halfstepback_disabled(self):
        v_old = jnp.array([2.0])
        force = jnp.array([-1.0])
        params = jnp.array([0.0])
        u_with, s_with = ScaleByFire2(
            dt_start=0.1, max_step=None, halfstepback=True, delaystep_start=False
        ).update(
            force,
            _single_sys_state(velocity=v_old, dt=0.1, n_pos=0, n_total=10),
            params,
        )
        u_without, _ = ScaleByFire2(
            dt_start=0.1, max_step=None, halfstepback=False, delaystep_start=False
        ).update(
            force,
            _single_sys_state(velocity=v_old, dt=0.1, n_pos=0, n_total=10),
            params,
        )
        diff = float(jnp.asarray(u_without)[0] - jnp.asarray(u_with)[0])
        expected = 0.5 * float(s_with.dt.data[0]) * float(v_old[0])
        npt.assert_allclose(diff, expected, rtol=1e-6)

    def test_delaystep_start_suppresses_shrink(self):
        opt = ScaleByFire2(
            dt_start=0.1,
            alpha_start=0.25,
            f_dec=0.5,
            n_min=5,
            delaystep_start=True,
        )
        params = jnp.array([1.0])
        state = opt.init(params)
        force = jnp.array([1.0])
        _, new_state = opt.update(force, state, params)
        npt.assert_allclose(float(new_state.dt.data[0]), 0.1)
        npt.assert_allclose(float(new_state.alpha.data[0]), 0.25)

    def test_delaystep_start_disabled_shrinks_immediately(self):
        opt = ScaleByFire2(
            dt_start=0.1,
            f_dec=0.5,
            dt_min=1e-6,
            n_min=5,
            delaystep_start=False,
        )
        params = jnp.array([1.0])
        state = opt.init(params)
        force = jnp.array([1.0])
        _, new_state = opt.update(force, state, params)
        npt.assert_allclose(float(new_state.dt.data[0]), 0.05)

    def test_negative_power_resets_velocity_and_alpha(self):
        alpha_start = 0.25
        opt = ScaleByFire2(
            dt_start=0.1,
            alpha_start=alpha_start,
            max_step=None,
            halfstepback=False,
            delaystep_start=False,
        )
        state = _single_sys_state(
            velocity=jnp.array([5.0]),
            dt=0.1,
            alpha=0.01,  # decayed
            n_pos=7,
            n_total=20,
        )
        params = jnp.array([1.0])
        force = jnp.array([-1.0])  # P = v·F = 5·(-1) = -5 < 0
        _, new_state = opt.update(force, state, params)
        assert int(new_state.n_pos.data[0]) == 0
        npt.assert_allclose(float(new_state.alpha.data[0]), alpha_start)
        # v_pre=0, v_int = dtv*F; new_velocity = v_int (no mixing on P<=0).
        npt.assert_allclose(
            float(jnp.asarray(new_state.velocity)[0]),
            float(new_state.dt.data[0]) * (-1.0),
            rtol=1e-6,
        )

    def test_abc_differs_from_non_abc_at_small_n(self):
        v_old = jnp.array([1.0, 0.5])
        force = jnp.array([1.0, 0.5])  # F · v > 0
        params = jnp.array([1.0, 0.0])

        def run(use_abc: bool) -> ScaleByFire2State:
            state = _single_sys_state(
                velocity=v_old, dt=0.1, alpha=0.25, n_pos=0, n_total=10
            )
            # Index covers 2 particles in 1 system.
            state = ScaleByFire2State(
                velocity=state.velocity,
                dt=state.dt,
                alpha=state.alpha,
                n_pos=state.n_pos,
                n_total=state.n_total,
                index_prefix=_system_index([0, 0], 1),
            )
            _, s = ScaleByFire2(
                dt_start=0.1,
                alpha_start=0.25,
                max_step=None,
                delaystep_start=False,
                use_abc=use_abc,
            ).update(force, state, params)
            return s

        s_plain = run(False)
        s_abc = run(True)
        assert int(s_plain.n_pos.data[0]) == 1
        assert int(s_abc.n_pos.data[0]) == 1
        diff = float(
            jnp.linalg.norm(jnp.asarray(s_abc.velocity) - jnp.asarray(s_plain.velocity))
        )
        assert diff > 1e-3

    def test_abc_per_component_clip(self):
        max_step = 0.05
        opt = ScaleByFire2(
            dt_start=0.1,
            alpha_start=0.25,
            max_step=max_step,
            use_abc=True,
            delaystep_start=False,
        )
        state = _single_sys_state(
            velocity=jnp.array([100.0, 0.0]),
            dt=0.1,
            n_pos=5,
            n_total=20,
        )
        state = ScaleByFire2State(
            velocity=state.velocity,
            dt=state.dt,
            alpha=state.alpha,
            n_pos=state.n_pos,
            n_total=state.n_total,
            index_prefix=_system_index([0, 0], 1),
        )
        params = jnp.array([0.0, 0.0])
        force = jnp.array([1.0, 0.0])
        _, new_state = opt.update(force, state, params)
        limit = max_step / float(new_state.dt.data[0])
        assert float(jnp.max(jnp.abs(jnp.asarray(new_state.velocity)))) <= limit + 1e-6

    def test_max_step_none_disables_clipping(self):
        opt = ScaleByFire2(
            dt_start=1.0, dt_max=1.0, max_step=None, delaystep_start=False
        )
        state = _single_sys_state(
            velocity=jnp.array([100.0]), dt=1.0, n_pos=10, n_total=20
        )
        params = jnp.array([0.0])
        force = jnp.array([10.0])
        updates, _ = opt.update(force, state, params)
        assert float(jnp.abs(jnp.asarray(updates)[0])) > 1.0

    def test_max_step_clips_non_abc(self):
        max_step = 0.1
        opt = ScaleByFire2(
            dt_start=1.0,
            dt_max=10.0,
            max_step=max_step,
            use_abc=False,
            delaystep_start=False,
        )
        state = _single_sys_state(
            velocity=jnp.array([100.0, 0.0]), dt=1.0, n_pos=10, n_total=20
        )
        state = ScaleByFire2State(
            velocity=state.velocity,
            dt=state.dt,
            alpha=state.alpha,
            n_pos=state.n_pos,
            n_total=state.n_total,
            index_prefix=_system_index([0, 0], 1),
        )
        params = jnp.array([0.0, 0.0])
        force = jnp.array([1.0, 0.0])
        updates, _ = opt.update(force, state, params)
        assert float(jnp.max(jnp.abs(jnp.asarray(updates)))) <= max_step + 1e-6

    def test_convergence_on_quadratic(self):
        # L(x) = 0.5·x²  ⇒  ∇L = x, force F = -x.
        opt = ScaleByFire2(dt_start=0.05, dt_max=0.5, max_step=0.5)
        x = jnp.array([5.0])
        x, _ = _run_force(opt, x, opt.init(x), 200)
        npt.assert_allclose(x, jnp.zeros(1), atol=1e-2)

    def test_convergence_on_quadratic_abc(self):
        opt = ScaleByFire2(dt_start=0.05, dt_max=0.5, max_step=0.5, use_abc=True)
        x = jnp.array([5.0])
        x, _ = _run_force(opt, x, opt.init(x), 200)
        npt.assert_allclose(x, jnp.zeros(1), atol=1e-2)


class TestScaleByFire2PerSystem:
    def test_batched_matches_separate(self):
        """Batched run must equal concatenation of independent per-system runs."""

        def run_alone(x0: jnp.ndarray) -> jnp.ndarray:
            opt = ScaleByFire2(dt_start=0.1, max_step=None, delaystep_start=False)
            x, _ = _run_force(opt, x0, opt.init(x0), 8)
            return x

        x_a = jnp.array([5.0, -3.0])
        x_b = jnp.array([0.5, 0.2])
        sep = jnp.concatenate([run_alone(x_a), run_alone(x_b)])

        opt = ScaleByFire2(dt_start=0.1, max_step=None, delaystep_start=False)
        batched = jnp.concatenate([x_a, x_b])
        idx = _system_index([0, 0, 1, 1], 2)
        x, _ = _run_force(opt, batched, opt.init(batched, index_prefix=idx), 8)
        npt.assert_allclose(x, sep, atol=1e-6)

    def test_per_system_dmax_clip(self):
        """Each system's per-step ∞-norm of Δx is independently bounded."""
        opt = ScaleByFire2(
            dt_start=1.0,
            dt_max=10.0,
            max_step=0.1,
            use_abc=False,
            delaystep_start=False,
        )
        idx = _system_index([0, 0, 1, 1], 2)
        # Hand-crafted state with large per-system velocities.
        keys = (SystemId(0), SystemId(1))
        state = ScaleByFire2State(
            velocity=jnp.array([100.0, 0.0, 50.0, 0.0]),
            dt=Table(keys, jnp.array([1.0, 1.0])),
            alpha=Table(keys, jnp.array([0.25, 0.25])),
            n_pos=Table(keys, jnp.array([10, 10], dtype=jnp.int32)),
            n_total=jnp.asarray(20, dtype=jnp.int32),
            index_prefix=idx,
        )
        params = jnp.array([0.0, 0.0, 0.0, 0.0])
        # Both systems get positive power.
        force = jnp.array([1.0, 0.0, 1.0, 0.0])
        upd, _ = opt.update(force, state, params)
        # ∞-norm of each system's update ≈ max_step. LAMMPS dmax bounds
        # ``dtv·|v_old|_∞``, not the final Δx after mixing — allow a small
        # O(dtv²·F) overshoot.
        assert float(jnp.max(jnp.abs(upd[0:2]))) <= 0.1 * 1.001
        assert float(jnp.max(jnp.abs(upd[2:4]))) <= 0.1 * 1.001

    def test_n_pos_evolves_per_system(self):
        """Two systems, one consistently downhill, the other oscillating."""
        opt = ScaleByFire2(
            dt_start=0.1,
            n_min=2,
            max_step=None,
            delaystep_start=False,
        )
        idx = _system_index([0, 1], 2)
        x = jnp.array([1.0, 1.0])
        state = opt.init(x, index_prefix=idx)
        # System 0: constant force → P>0 every step.
        # System 1: force flips sign each step → P<0 → n_pos keeps resetting.
        for step in range(6):
            force = jnp.array([-1.0, -1.0 if step % 2 == 0 else 1.0])
            _, state = opt.update(force, state, x)
        # System 0 should have accumulated several positive-power steps.
        # System 1 should have reset n_pos to 0 frequently.
        assert int(state.n_pos.data[0]) > int(state.n_pos.data[1])


class TestScaleByFire2Composability:
    """``ScaleByFire2`` should compose cleanly with other transforms."""

    def test_chain_with_sign_flip_matches_force_input(self):
        """``chain(optax.scale(-1), ScaleByFire2())`` accepts ∇L and gives the
        same trajectory as passing ``-∇L`` directly to a bare ``ScaleByFire2``."""
        x0 = jnp.array([5.0])

        direct = ScaleByFire2(dt_start=0.05, dt_max=0.5, max_step=0.5)
        composed = chain(
            optax.scale(-1.0),
            ScaleByFire2(dt_start=0.05, dt_max=0.5, max_step=0.5),
        )

        def body(
            carry: tuple[Array, Any, Array, Any], _: None
        ) -> tuple[tuple[Array, Any, Array, Any], None]:
            x_d, s_d, x_c, s_c = carry
            upd_d, s_d = direct.update(-x_d, s_d, x_d)
            x_d = jnp.asarray(optax.apply_updates(x_d, upd_d))
            upd_c, s_c = composed.update(x_c, s_c, x_c)
            x_c = jnp.asarray(optax.apply_updates(x_c, upd_c))
            return (x_d, s_d, x_c, s_c), None

        init = (x0, direct.init(x0), x0, composed.init(x0))
        (x_direct, _, x_composed, _), _ = jax.lax.scan(body, init, None, length=20)
        npt.assert_allclose(x_composed, x_direct, atol=1e-6)

    def test_chain_with_external_clip_caps_displacement(self):
        """``MaxStepSize`` appended after FIRE 2.0 caps the per-particle Δx,
        independently of (and on top of) the built-in LAMMPS ``dmax``."""
        idx = _system_index([0, 0], 1)
        x = jnp.zeros((2, 3))  # 2 particles, 3D — natural per-particle shape

        external_cap = 0.01
        opt = chain(
            ScaleByFire2(dt_start=1.0, dt_max=10.0, max_step=1.0),
            MaxStepSize(max_step_size=external_cap),
        )
        state = opt.init(x, index_prefix=idx)

        force = jnp.array([[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        upd, _ = opt.update(force, state, x)
        per_particle = jnp.linalg.norm(upd, axis=-1)
        assert float(jnp.max(per_particle)) <= external_cap + 1e-6

    def test_chain_clip_then_fire_runs(self):
        """``ClipByGlobalNorm → ScaleByFire2`` chain converges on a quadratic."""
        idx = _system_index([0], 1)
        x = jnp.array([5.0])

        opt = chain(
            ClipByGlobalNorm(max_norm=2.0),
            ScaleByFire2(dt_start=0.05, dt_max=0.5, max_step=0.5),
        )
        x, _ = _run_force(opt, x, opt.init(x, index_prefix=idx), 200)
        npt.assert_allclose(x, jnp.zeros(1), atol=1e-2)
