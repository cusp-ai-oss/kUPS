# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the blocked path of ``run_simulation_cycles`` (block_size > 1).

Blocking is built by :func:`make_block_function` (the loop inserted before compilation)
and driven by ``run_simulation_cycles`` with ``block_size`` set; the per-step frames are
replayed through a minimal ``log_block`` stand-in. The HDF5 side is covered by
test_storage.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.application.utils.propagate import (
    make_block_function,
    make_cycle_function,
    run_simulation_cycles,
)
from kups.core.assertion import runtime_assert
from kups.core.utils.jax import dataclass

_VALUE = lambda s: s.value  # noqa: E731 -- single-leaf view used across the tests


@dataclass
class _State:
    step: Array
    value: Array


def _stepper(key: Array, s: _State) -> _State:
    del key
    return _State(step=s.step + 1, value=s.value + 1.0)


def _state() -> _State:
    return _State(step=jnp.array([0]), value=jnp.array(0.0))


class _Recorder:
    """Minimal blocked-logger stand-in: records each block's frames, no HDF5."""

    def __init__(self) -> None:
        self.frames: list[Array] = []

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def log_block(self, frames: list[Array], start: int) -> None:
        # scan_with emits stacked view *leaves*; the views here have a single leaf.
        (leaf,) = frames
        self.frames.append(leaf)


class _StepLog:
    """Per-step Logger stand-in: records state.value each step."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def __enter__(self) -> _StepLog:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def log(self, state: _State, step: int) -> None:
        self.values.append(float(state.value))


def test_logs_every_timestep_in_order():
    """Blocking is transparent: every timestep's frame is captured, in order."""
    rec = _Recorder()
    cycle_fn = make_block_function(_stepper, 3, _VALUE)
    out = run_simulation_cycles(
        jax.random.key(0), cycle_fn, _state(), 9, rec, block_size=3
    )
    npt.assert_array_equal(out.step, jnp.array([9]))  # 3 + 3 + 3
    npt.assert_array_equal(jnp.concatenate(rec.frames), jnp.arange(1.0, 10.0))


def test_checks_assertions():
    """A runtime assertion that fails inside a block surfaces at the block boundary."""

    def asserting_step(key: Array, s: _State) -> _State:
        del key
        value = s.value + 1.0
        runtime_assert(value < 5.0, message="value too large")
        return _State(step=s.step + 1, value=value)

    cycle_fn = make_block_function(asserting_step, 5, _VALUE)
    with pytest.raises(AssertionError):
        run_simulation_cycles(
            jax.random.key(0), cycle_fn, _state(), 10, _Recorder(), block_size=5
        )


def test_convergence_stops_early():
    cycle_fn = make_block_function(_stepper, 10, _VALUE)
    out = run_simulation_cycles(
        jax.random.key(0),
        cycle_fn,
        _state(),
        100,
        _Recorder(),
        block_size=10,
        convergence_fn=lambda s: bool(s.value >= 20.0),
    )
    npt.assert_array_equal(out.step, jnp.array([20]))  # stops after the 2nd block


def test_blocked_matches_per_step_deterministic():
    """Deterministic propagator: the blocked path (block_size>1) produces the same final
    state and the same per-timestep trajectory as the per-step path (block_size=1)."""
    key = jax.random.key(0)
    per_log = _StepLog()
    per = run_simulation_cycles(
        key, make_cycle_function(_stepper), _state(), 9, per_log
    )
    rec = _Recorder()
    blk = run_simulation_cycles(
        key, make_block_function(_stepper, 3, _VALUE), _state(), 9, rec, block_size=3
    )
    npt.assert_array_equal(per.value, blk.value)
    npt.assert_array_equal(jnp.array(per_log.values), jnp.concatenate(rec.frames))


def test_block_equals_num_steps():
    """block_size == num_cycles: exactly one full block, no remainder."""
    rec = _Recorder()
    cycle_fn = make_block_function(_stepper, 5, _VALUE)
    out = run_simulation_cycles(
        jax.random.key(0), cycle_fn, _state(), 5, rec, block_size=5
    )
    npt.assert_array_equal(out.step, jnp.array([5]))
    assert len(rec.frames) == 1
    npt.assert_array_equal(jnp.concatenate(rec.frames), jnp.arange(1.0, 6.0))


def test_requires_num_cycles_multiple_of_block_size():
    """num_cycles not divisible by block_size is rejected (caller must align)."""
    cycle_fn = make_block_function(_stepper, 4, _VALUE)
    with pytest.raises(ValueError, match="multiple of block_size"):
        run_simulation_cycles(
            jax.random.key(0), cycle_fn, _state(), 10, _Recorder(), block_size=4
        )
