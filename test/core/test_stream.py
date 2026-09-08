# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax import Array

from kups.application.utils.propagate import make_cycle_function
from kups.core.assertion import runtime_assert
from kups.core.data import Table
from kups.core.lens import bind
from kups.core.propagator import (
    LoopPropagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.stream import RefillPropagator
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass


@dataclass
class State:
    values: Array
    requested: Table[SystemId, Array]
    capacity: Array


def test_refill_and_capacity_repair_preserve_committed_state() -> None:
    calls: list[Array] = []
    traces: list[int] = []

    def refill(s: State, mask: Table[SystemId, Array]) -> State:
        calls.append(s.values.copy())
        return (
            bind(s)
            .focus(lambda x: (x.values, x.requested))
            .set(
                (jnp.where(mask.data, 10, s.values), mask.set_data(jnp.zeros(2, bool)))
            )
        )

    def grow(s: State, required: Array) -> State:
        return bind(s).focus(lambda x: x.capacity).set(required)

    def advance(key: Array, s: State) -> State:
        del key
        traces.append(1)
        runtime_assert(s.capacity >= 1, "capacity", fix_fn=grow, fix_args=jnp.array(1))
        return bind(s).focus(lambda x: x.values).apply(lambda values: values + 1)

    gate: RefillPropagator[State] = RefillPropagator(lambda s: s.requested, refill)
    cycle = make_cycle_function(
        SequentialPropagator((gate, ResetOnErrorPropagator(advance)))
    )
    state = State(
        jnp.array([5, 7]),
        Table.arange(jnp.array([True, False]), label=SystemId),
        jnp.array(0),
    )
    state = propagate_and_fix(cycle, jax.random.key(0), state)
    assert len(calls) == 1
    npt.assert_array_equal(calls[0], [5, 7])
    npt.assert_array_equal(state.values, [11, 8])
    assert len(traces) == 1

    state = propagate_and_fix(cycle, jax.random.key(1), state)
    npt.assert_array_equal(state.values, [12, 9])
    assert len(calls) == 1


def test_refill_is_serviced_once_per_host_cycle() -> None:
    calls: list[int] = []

    def refill(s: State, mask: Table[SystemId, Array]) -> State:
        calls.append(int(s.values[0]))
        return (
            bind(s).focus(lambda x: x.requested).set(mask.set_data(jnp.zeros(1, bool)))
        )

    def advance(key: Array, s: State) -> State:
        del key
        return (
            bind(s)
            .focus(lambda x: (x.values, x.requested))
            .set((s.values + 1, s.requested.set_data(jnp.ones(1, bool))))
        )

    gate: RefillPropagator[State] = RefillPropagator(lambda s: s.requested, refill)
    cycle = make_cycle_function(
        SequentialPropagator((gate, ResetOnErrorPropagator(advance)))
    )
    state = State(
        jnp.zeros(1, int), Table.arange(jnp.ones(1, bool), label=SystemId), jnp.array(1)
    )
    for i in range(20):
        state = propagate_and_fix(cycle, jax.random.key(i), state, max_tries=2)
    assert calls == list(range(20))
    npt.assert_array_equal(state.values, [20])


def test_blocked_refill_preserves_progress_across_capacity_repair() -> None:
    calls: list[int] = []

    def refill(s: State, mask: Table[SystemId, Array]) -> State:
        calls.append(int(s.values[0]))
        return (
            bind(s)
            .focus(lambda x: (x.values, x.requested))
            .set((jnp.zeros_like(s.values), mask.set_data(jnp.zeros(1, bool))))
        )

    def grow(s: State, required: Array) -> State:
        return bind(s).focus(lambda x: x.capacity).set(required)

    def advance(key: Array, s: State) -> State:
        del key
        runtime_assert(
            (s.values[0] < 3) | (s.capacity >= 1),
            "capacity",
            fix_fn=grow,
            fix_args=jnp.array(1),
        )
        values = jnp.minimum(s.values + 1, 7)
        return (
            bind(s)
            .focus(lambda x: (x.values, x.requested))
            .set((values, s.requested.set_data(values == 7)))
        )

    gate: RefillPropagator[State] = RefillPropagator(lambda s: s.requested, refill)
    loop = LoopPropagator(
        ResetOnErrorPropagator(advance),
        lambda s: jnp.where(s.requested.data.any(), 0, 4),
    )
    cycle = make_cycle_function(SequentialPropagator((gate, loop)))
    state = State(
        jnp.zeros(1, int), Table.arange(jnp.ones(1, bool), label=SystemId), jnp.array(0)
    )
    state = propagate_and_fix(cycle, jax.random.key(0), state)
    # Three successful steps survive the capacity failure, then four more run.
    npt.assert_array_equal(state.values, [7])
    assert calls == [0]
    state = propagate_and_fix(cycle, jax.random.key(1), state)
    npt.assert_array_equal(state.values, [4])
    assert calls == [0, 7]
    state = propagate_and_fix(cycle, jax.random.key(2), state)
    npt.assert_array_equal(state.values, [7])
    assert calls == [0, 7]
