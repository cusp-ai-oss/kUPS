# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Test cycle dispatch, sample retention and recovery from capacity failures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array

from kups.application.utils.propagate import (
    make_cycle_function,
    make_sampled_cycle_function,
    run_sampled_cycles,
    run_simulation_cycles,
)
from kups.core.assertion import runtime_assert
from kups.core.lens import bind
from kups.core.propagator import LoopPropagator
from kups.core.utils.jax import dataclass, field


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
        # Scalar samples must remain arrays, including after transfer to the host.
        assert isinstance(state.value, (Array, np.ndarray))
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


@pytest.mark.parametrize("block_size", [1, 4, 8])
def test_sampled_blocks_preserve_random_keys_and_every_frame(block_size: int):
    def stochastic(key: Array, state: _State) -> _State:
        return _State(state.step + 1, state.value + jax.random.uniform(key))

    key = jax.random.key(41)
    expected_log, actual_log = _Log(), _Log()
    expected = run_simulation_cycles(
        key, make_cycle_function(stochastic), _state(), 11, expected_log
    )
    actual = run_sampled_cycles(
        key,
        make_sampled_cycle_function(stochastic, lambda s: s),
        _state(),
        11,
        actual_log,
        block_size,
    )
    npt.assert_array_equal(actual.step, expected.step)
    npt.assert_array_equal(actual.value, expected.value)
    npt.assert_array_equal(actual_log.values, expected_log.values)


@dataclass
class _CapacityState:
    value: Array
    capacity: int = field(static=True)


def test_sampled_block_repair_rolls_back_successful_prefix_and_retries():
    """A failure after two steps must not duplicate their samples or random draws."""

    def repair(state: _CapacityState, _: Array) -> _CapacityState:
        return bind(state, lambda s: s.capacity).set(20)

    def step(key: Array, state: _CapacityState) -> _CapacityState:
        del key
        value = state.value + 1
        runtime_assert(
            value <= state.capacity, "capacity exceeded", fix_fn=repair, fix_args=value
        )
        return bind(state, lambda s: s.value).set(value)

    log = _Log()
    actual = run_sampled_cycles(
        jax.random.key(42),
        make_sampled_cycle_function(step, lambda s: _State(s.value, s.value)),
        _CapacityState(jnp.array(0), 2),
        11,
        log,
        8,
    )
    assert int(actual.value) == 11
    assert actual.capacity == 20
    npt.assert_array_equal(log.values, jnp.arange(1, 12))


def test_sampled_cycles_do_not_dispatch_when_empty():
    log = _Log()
    state = _state()
    result = run_sampled_cycles(
        jax.random.key(43),
        make_sampled_cycle_function(_stepper, lambda s: s),
        state,
        0,
        log,
        8,
    )
    assert result is state
    assert log.values == []
