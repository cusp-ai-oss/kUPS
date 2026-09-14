# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import assert_type, override

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.application.utils.propagate import make_cycle_function, run_simulation_cycles
from kups.core.assertion import runtime_assert
from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.logging import NullLogger
from kups.core.propagator import (
    LoopPropagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.stream import RefillPropagator, reserve_slots, slot_owners
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit


@pytest.mark.parametrize("n_slots,capacity", [(1, 1), (2, 3)])
def test_reservations_and_owners(n_slots: int, capacity: int) -> None:
    slots = reserve_slots(n_slots, capacity)
    assert_type(slots, Table[SystemId, Index[ParticleId]])
    assert slots.keys == tuple(SystemId(i) for i in range(n_slots))
    assert slots.data.keys == tuple(ParticleId(i) for i in range(n_slots * capacity))
    assert slots.data.max_count == 1
    npt.assert_array_equal(
        slots.data.indices, jnp.arange(n_slots * capacity).reshape(n_slots, capacity)
    )

    owners = slot_owners(slots)
    assert_type(owners, Table[ParticleId, Index[SystemId]])
    assert owners.keys == slots.data.keys
    assert owners.data.keys == slots.keys
    assert owners.data.max_count == capacity
    npt.assert_array_equal(
        owners.data.indices, jnp.repeat(jnp.arange(n_slots), capacity)
    )


@pytest.mark.parametrize("n_slots,capacity", [(0, 8), (2, 0), (-1, 8), (2, -1)])
def test_reservations_require_positive_dimensions(n_slots: int, capacity: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        reserve_slots(n_slots, capacity)


@pytest.mark.parametrize("compiled", [False, True])
def test_slot_owners_preserve_noncontiguous_keys_and_unreserved_rows(
    compiled: bool,
) -> None:
    slots = Table(
        (SystemId(11), SystemId(29)),
        Index(
            tuple(ParticleId(100 + 2 * i) for i in range(6)),
            jnp.array([[4, 0, 6], [3, 1, 6]]),
            max_count=1,
            _cls=ParticleId,
        ),
    )
    owners = (jit(slot_owners) if compiled else slot_owners)(slots)
    assert owners.keys == slots.data.keys
    assert owners.data.keys == slots.keys
    assert owners.data.max_count == 3
    npt.assert_array_equal(owners.data.indices, [0, 1, 2, 1, 0, 2])
    npt.assert_array_equal(
        owners.data.valid_mask, [True, True, False, True, True, False]
    )
    npt.assert_array_equal(owners[slots.data].indices, [[0, 0, 2], [1, 1, 2]])


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


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_refill_logs_committed_cycles_not_repairs(
    caplog: pytest.LogCaptureFixture,
    level: int,
) -> None:
    caplog.set_level(level)
    calls: list[int] = []
    logged: list[tuple[int, int]] = []

    class Recorder(NullLogger[State]):
        @override
        def log(self, state: State, step: int) -> None:
            logged.append((step, int(state.values[0])))

    def refill(s: State, mask: Table[SystemId, Array]) -> State:
        calls.append(int(s.values[0]))
        return (
            bind(s).focus(lambda x: x.requested).set(mask.set_data(jnp.zeros(1, bool)))
        )

    def advance(key: Array, s: State) -> State:
        del key
        return (
            bind(s)
            .focus(lambda x: (x.values, x.requested.data))
            .set((s.values + 1, jnp.ones(1, bool)))
        )

    cycle = make_cycle_function(
        SequentialPropagator(
            (
                RefillPropagator(lambda s: s.requested, refill),
                ResetOnErrorPropagator(advance),
            )
        )
    )
    initial = State(
        jnp.zeros(1, int), Table.arange(jnp.ones(1, bool), label=SystemId), jnp.array(1)
    )
    run_simulation_cycles(jax.random.key(0), cycle, initial, 3, Recorder())
    assert calls == [0, 1, 2]
    assert logged == [(0, 1), (1, 2), (2, 3)]
    repairs = [r for r in caplog.records if "Applying assertion fix:" in r.getMessage()]
    assert len(repairs) == (3 if level == logging.DEBUG else 0)
    assert all(
        r.levelno == logging.DEBUG and "Assertion created at:" in r.getMessage()
        for r in repairs
    )


def test_fatal_assertion_prevents_refill_side_effects() -> None:
    calls: list[int] = []

    def refill(s: State, mask: Table[SystemId, Array]) -> State:
        calls.append(1)
        return s

    def fail(key: Array, s: State) -> State:
        del key
        runtime_assert(jnp.array(False), "unrecoverable")
        return s

    cycle = make_cycle_function(
        SequentialPropagator(
            (
                RefillPropagator(lambda s: s.requested, refill),
                ResetOnErrorPropagator(fail),
            )
        )
    )
    initial = State(
        jnp.zeros(1), Table.arange(jnp.ones(1, bool), label=SystemId), jnp.array(1)
    )
    with pytest.raises(AssertionError, match="unrecoverable"):
        propagate_and_fix(cycle, jax.random.key(0), initial)
    assert calls == []


def test_raising_refill_aborts_without_retrying_callback() -> None:
    calls: list[int] = []

    def refill(s: State, mask: Table[SystemId, Array]) -> State:
        calls.append(1)
        raise ValueError("replacement rejected")

    cycle = make_cycle_function(RefillPropagator(lambda s: s.requested, refill))
    initial = State(
        jnp.zeros(1), Table.arange(jnp.ones(1, bool), label=SystemId), jnp.array(1)
    )
    with pytest.raises(ValueError, match="replacement rejected"):
        propagate_and_fix(cycle, jax.random.key(0), initial)
    assert calls == [1]
