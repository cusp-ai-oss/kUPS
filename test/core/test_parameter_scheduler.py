# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for ParameterScheduler functionality."""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.lens import SimpleLens
from kups.core.parameter_scheduler import (
    AcceptanceHistory,
    Correlation,
    ParameterSchedulerState,
    acceptance_target_schedule,
)
from kups.core.schedule import PropertyScheduler
from kups.core.utils.jax import dataclass, field


def get_scheduler_params(state: "SimpleTestState") -> ParameterSchedulerState:
    """Get scheduler parameters from test state."""
    return state.scheduler_params


@dataclass
class SimpleTestState:
    """Simple test state for parameter scheduler testing."""

    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.0, 2.0]),
            multiplicity=jnp.array([2.0, 2.0]),
            target=jnp.array([0.5, 0.5]),
            tolerance=jnp.array([0.1, 0.1]),
            correlation=Correlation.NEGATIVE,
            bounds=(None, None),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 5)),
                index=jnp.array([0, 0]),
            ),
        )
    )
    other_data: float = field(default=42.0)


@dataclass
class BoundedUpperState:
    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.0, 2.0]),
            multiplicity=jnp.array([3.0, 3.0]),
            target=jnp.array([0.5, 0.5]),
            tolerance=jnp.array([0.05, 0.05]),
            correlation=Correlation.NEGATIVE,
            bounds=(None, jnp.array([2.5, 2.5])),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 4)),
                index=jnp.array([0, 0]),
            ),
        )
    )


@dataclass
class BoundedLowerState:
    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.0, 2.0]),
            multiplicity=jnp.array([4.0, 4.0]),
            target=jnp.array([0.6, 0.6]),
            tolerance=jnp.array([0.05, 0.05]),
            correlation=Correlation.NEGATIVE,
            bounds=(jnp.array([0.8, 1.5]), None),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 5)),
                index=jnp.array([0, 0]),
            ),
        )
    )


@dataclass
class BoundedBothState:
    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.0, 5.0]),
            multiplicity=jnp.array([2.0, 2.0]),
            target=jnp.array([0.5, 0.5]),
            tolerance=jnp.array([0.05, 0.05]),
            correlation=Correlation.NEGATIVE,
            bounds=(jnp.array([0.9, 4.0]), jnp.array([1.5, 6.0])),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 3)),
                index=jnp.array([0, 0]),
            ),
        )
    )


@dataclass
class SingleSystemState:
    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.5]),
            multiplicity=jnp.array([1.5]),
            target=jnp.array([0.6]),
            tolerance=jnp.array([0.1]),
            correlation=Correlation.NEGATIVE,
            bounds=(None, None),
            history=AcceptanceHistory(
                values=jnp.zeros((1, 3)),
                index=jnp.array([0]),
            ),
        )
    )


@dataclass
class NegativeCorrelationState:
    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.0, 2.0]),
            multiplicity=jnp.array([3.0, 3.0]),
            target=jnp.array([0.5, 0.5]),
            tolerance=jnp.array([0.1, 0.1]),
            correlation=Correlation.NEGATIVE,
            bounds=(None, None),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 5)),
                index=jnp.array([0, 0]),
            ),
        )
    )


@dataclass
class PositiveCorrelationState:
    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.0, 2.0]),
            multiplicity=jnp.array([3.0, 3.0]),
            target=jnp.array([0.5, 0.5]),
            tolerance=jnp.array([0.1, 0.1]),
            correlation=Correlation.POSITIVE,
            bounds=(None, None),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 5)),
                index=jnp.array([0, 0]),
            ),
        )
    )


@dataclass
class BatchSchedulerState:
    scheduler_params: ParameterSchedulerState = field(
        default_factory=lambda: ParameterSchedulerState(
            value=jnp.array([1.0, 2.0]),
            multiplicity=jnp.array([2.0, 2.0]),
            target=jnp.array([0.5, 0.5]),
            tolerance=jnp.array([0.1, 0.1]),
            correlation=Correlation.NEGATIVE,
            bounds=(None, None),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 4)), index=jnp.array([0, 0])
            ),
        )
    )


_default_lens = SimpleLens(lambda s: s.scheduler_params)
_default_scheduler = PropertyScheduler(
    lens=_default_lens, schedule=acceptance_target_schedule
)


class TestParameterSchedulerBasics:
    def test_creation_history_and_call(self):
        """Merged: history creation, scheduler creation, basic call."""
        # History creation
        values = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        index = jnp.array([0, 1])
        history = AcceptanceHistory(values=values, index=index)
        npt.assert_array_equal(history.values, values)
        npt.assert_array_equal(history.index, index)

        # Scheduler creation and call
        lens = SimpleLens(get_scheduler_params)
        scheduler = PropertyScheduler(lens=lens, schedule=acceptance_target_schedule)
        assert scheduler.lens == lens
        assert scheduler.schedule is acceptance_target_schedule

        test_state = SimpleTestState()
        acceptance_values = jnp.array([0.3, 0.7])
        result_state = scheduler(test_state, acceptance_values)
        assert hasattr(result_state, "scheduler_params")
        assert result_state.other_data == 42.0

        new_history = result_state.scheduler_params.history
        expected_values = test_state.scheduler_params.history.values.at[
            jnp.arange(2), jnp.array([0, 0])
        ].set(acceptance_values)
        npt.assert_array_equal(new_history.values, expected_values)
        npt.assert_array_equal(new_history.index, jnp.array([1, 1]))


class TestParameterAdjustment:
    def test_correlation_and_cycle(self):
        """Merged: negative, positive, full_history_cycle."""
        scheduler = _default_scheduler
        acceptance_values = jnp.array([0.3, 0.7])
        initial_values = jnp.array([1.0, 2.0])

        # negative: no change before cycle
        state = SimpleTestState()
        result = scheduler(state, acceptance_values)
        npt.assert_array_equal(result.scheduler_params.value, initial_values)

        # positive: no change before cycle
        pos_state = PositiveCorrelationState()
        result = scheduler(pos_state, acceptance_values)
        npt.assert_array_equal(result.scheduler_params.value, initial_values)

        # full history cycle
        state = SimpleTestState()
        for _ in range(5):
            state = scheduler(state, acceptance_values)
        npt.assert_array_equal(state.scheduler_params.history.index, jnp.array([0, 0]))
        npt.assert_array_almost_equal(
            state.scheduler_params.value, jnp.array([0.5, 4.0])
        )


class TestHistoryManagement:
    def test_cycling_and_storage(self):
        """Merged: index_cycling + value_storage."""
        scheduler = _default_scheduler

        # index cycling
        state = SimpleTestState()
        indices = [state.scheduler_params.history.index]
        for _ in range(7):
            state = scheduler(state, jnp.array([0.4, 0.6]))
            indices.append(state.scheduler_params.history.index)
        expected_indices = [
            jnp.array([0, 0]),
            jnp.array([1, 1]),
            jnp.array([2, 2]),
            jnp.array([3, 3]),
            jnp.array([4, 4]),
            jnp.array([0, 0]),
            jnp.array([1, 1]),
            jnp.array([2, 2]),
        ]
        for i, expected in enumerate(expected_indices):
            npt.assert_array_equal(indices[i], expected)

        # value storage
        state = SimpleTestState()
        test_values = [
            jnp.array([0.1, 0.2]),
            jnp.array([0.3, 0.4]),
            jnp.array([0.5, 0.6]),
            jnp.array([0.7, 0.8]),
            jnp.array([0.9, 1.0]),
        ]
        for v in test_values:
            state = scheduler(state, v)
        expected_history = jnp.array(
            [
                [0.1, 0.3, 0.5, 0.7, 0.9],
                [0.2, 0.4, 0.6, 0.8, 1.0],
            ]
        )
        npt.assert_array_equal(state.scheduler_params.history.values, expected_history)


class TestEdgeCases:
    def test_edge_cases_and_bounds(self):
        """Merged: target_equal, zero, perfect, single_system, upper/lower/both bounds."""
        scheduler = _default_scheduler
        initial_values = jnp.array([1.0, 2.0])

        # target == acceptance => no change
        state = SimpleTestState()
        for _ in range(5):
            state = scheduler(state, jnp.array([0.5, 0.5]))
        npt.assert_array_almost_equal(state.scheduler_params.value, initial_values)

        # zero acceptance => decrease
        state = SimpleTestState()
        for _ in range(5):
            state = scheduler(state, jnp.array([0.0, 0.0]))
        npt.assert_array_almost_equal(
            state.scheduler_params.value, initial_values * 0.5
        )

        # perfect acceptance => increase
        state = SimpleTestState()
        for _ in range(5):
            state = scheduler(state, jnp.array([1.0, 1.0]))
        npt.assert_array_almost_equal(
            state.scheduler_params.value, initial_values * 2.0
        )

        # single system
        state = SingleSystemState()
        for _ in range(3):
            state = scheduler(state, jnp.array([0.4]))
        npt.assert_array_almost_equal(state.scheduler_params.value, jnp.array([1.0]))

        # upper bound
        state = BoundedUpperState()
        for _ in range(4):
            state = scheduler(state, jnp.array([0.9, 0.9]))
        npt.assert_array_almost_equal(
            state.scheduler_params.value, jnp.array([2.5, 2.5])
        )

        # lower bound
        state = BoundedLowerState()
        for _ in range(5):
            state = scheduler(state, jnp.array([0.0, 0.0]))
        npt.assert_array_almost_equal(
            state.scheduler_params.value, jnp.array([0.8, 1.5])
        )

        # both bounds
        state = BoundedBothState()
        for _ in range(3):
            state = scheduler(state, jnp.array([0.9, 0.1]))
        npt.assert_array_almost_equal(
            state.scheduler_params.value, jnp.array([1.5, 4.0])
        )


class TestCorrelationBehavior:
    def test_multiplicity_application(self):
        multiplicity = 3.0
        initial_values = jnp.array([1.0, 2.0])
        acceptance_values = jnp.array([0.3, 0.3])
        scheduler = _default_scheduler

        neg_state = NegativeCorrelationState()
        pos_state = PositiveCorrelationState()

        for _ in range(5):
            neg_state = scheduler(neg_state, acceptance_values)
        for _ in range(5):
            pos_state = scheduler(pos_state, acceptance_values)

        npt.assert_array_almost_equal(
            neg_state.scheduler_params.value, initial_values * (1.0 / multiplicity)
        )
        npt.assert_array_almost_equal(
            pos_state.scheduler_params.value, initial_values * multiplicity
        )


class TestJAXCompatibility:
    def test_jit_and_vmap(self):
        """Merged: JIT compilation + vmap compatibility."""
        scheduler = _default_scheduler

        # JIT
        @jax.jit
        def jitted_call(state, vals):
            return scheduler(state, vals)

        test_state = SimpleTestState()
        acceptance_values = jnp.array([0.4, 0.6])
        result_state = jitted_call(test_state, acceptance_values)
        expected_state = scheduler(test_state, acceptance_values)
        npt.assert_array_equal(
            result_state.scheduler_params.value, expected_state.scheduler_params.value
        )
        npt.assert_array_equal(
            result_state.scheduler_params.history.values,
            expected_state.scheduler_params.history.values,
        )

        # vmap
        batch_size = 3
        states = [
            BatchSchedulerState(
                scheduler_params=ParameterSchedulerState(
                    value=jnp.array([1.0, 2.0]) * (i + 1),
                    multiplicity=jnp.array([2.0, 2.0]),
                    target=jnp.array([0.5, 0.5]),
                    tolerance=jnp.array([0.1, 0.1]),
                    correlation=Correlation.NEGATIVE,
                    bounds=(None, None),
                    history=AcceptanceHistory(
                        values=jnp.zeros((2, 4)), index=jnp.array([0, 0])
                    ),
                )
            )
            for i in range(batch_size)
        ]
        stacked = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *states)
        batch_acc = jnp.array([[0.3, 0.7]] * batch_size)
        result = jax.vmap(scheduler, in_axes=(0, 0))(stacked, batch_acc)
        assert result.scheduler_params.value.shape == (batch_size, 2)
        assert result.scheduler_params.history.values.shape == (batch_size, 2, 4)
