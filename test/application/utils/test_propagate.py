# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for blocked stepping: ``make_cycle_function(LoopPropagator(propagator, K))``.

``LoopPropagator`` fuses ``K`` steps into one device dispatch and returns only the
block's final state, so the per-step and blocked paths share one interface. A blocked
run therefore saves the last frame of each block; ``run_simulation_cycles`` is unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax import Array

from kups.application.utils.propagate import (
    make_cycle_function,
    run_simulation_cycles,
)
from kups.core.propagator import LoopPropagator
from kups.core.utils.jax import dataclass


@dataclass
class _State:
    step: Array
    value: Array


def _stepper(key: Array, s: _State) -> _State:
    del key
    return _State(step=s.step + 1, value=s.value + 1.0)


def _state() -> _State:
    return _State(step=jnp.array([0]), value=jnp.array(0.0))


class _Log:
    """Logger stand-in: records ``state.value`` at each ``log`` call."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def __enter__(self) -> _Log:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def log(self, state: _State, step: int) -> None:
        self.values.append(float(state.value))


def test_block_advances_block_size_steps():
    """One block call fuses block_size steps and returns the block's final state."""
    out = make_cycle_function(LoopPropagator(_stepper, 5))(jax.random.key(0), _state())
    npt.assert_array_equal(out.value.step, jnp.array([5]))
    npt.assert_allclose(out.value.value, 5.0)


def test_blocked_matches_per_step_final_state():
    """Deterministic propagator: 2 blocks of 5 reaches the same state as 10 per-step."""
    key = jax.random.key(0)
    per = run_simulation_cycles(
        key, make_cycle_function(_stepper), _state(), 10, _Log()
    )
    blk = run_simulation_cycles(
        key, make_cycle_function(LoopPropagator(_stepper, 5)), _state(), 2, _Log()
    )
    npt.assert_array_equal(per.step, blk.step)
    npt.assert_allclose(per.value, blk.value)


def test_saves_last_frame_of_each_block():
    """A blocked run logs once per block -- the last frame of each."""
    log = _Log()
    out = run_simulation_cycles(
        jax.random.key(0),
        make_cycle_function(LoopPropagator(_stepper, 5)),
        _state(),
        4,
        log,
    )
    npt.assert_array_equal(out.step, jnp.array([20]))  # 4 blocks x 5
    npt.assert_allclose(log.values, [5.0, 10.0, 15.0, 20.0])  # block-final frames only


def test_convergence_stops_early():
    out = run_simulation_cycles(
        jax.random.key(0),
        make_cycle_function(LoopPropagator(_stepper, 5)),
        _state(),
        10,
        _Log(),
        convergence_fn=lambda s: bool(s.value >= 10.0),
    )
    npt.assert_array_equal(out.step, jnp.array([10]))  # stops after the 2nd block
