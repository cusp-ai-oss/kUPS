# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for property scheduling functionality."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.lens import SimpleLens
from kups.core.schedule import (
    ComposedSchedule,
    ConstantSchedule,
    CosineAnnealingSchedule,
    ExponentialSchedule,
    IncrementSchedule,
    LinearSchedule,
    PropertyScheduler,
    StepFunctionSchedule,
)
from kups.core.utils.jax import dataclass, field


@dataclass
class SimpleState:
    """Simple test state for scheduler testing."""

    temperature: jax.Array = field(default_factory=lambda: jnp.array(300.0))
    step: jax.Array = field(default_factory=lambda: jnp.array(0))


def get_temperature(state: SimpleState) -> jax.Array:
    return state.temperature


@pytest.fixture
def simple_state():
    return SimpleState()


class TestSchedules:
    """Tests for all schedule types."""

    def test_constant_schedule(self):
        """ConstantSchedule returns fixed value regardless of input."""
        schedule = ConstantSchedule(value=jnp.array(500.0))
        for step in [0, 100, 1000]:
            result = schedule(jnp.array(step), jnp.array(300.0))
            npt.assert_array_almost_equal(result, jnp.array(500.0))

    def test_increment_schedule(self):
        """IncrementSchedule increments current value by fixed amount."""
        schedule = IncrementSchedule(increment=jnp.array(1))
        npt.assert_array_almost_equal(
            schedule(jnp.array(0), jnp.array(0)), jnp.array(1)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(0), jnp.array(10)), jnp.array(11)
        )
        schedule2 = IncrementSchedule(increment=jnp.array(5))
        npt.assert_array_almost_equal(
            schedule2(jnp.array(0), jnp.array(10)), jnp.array(15)
        )

    def test_linear_schedule(self):
        """LinearSchedule interpolates between start and end."""
        schedule = LinearSchedule(
            start=jnp.array(300.0),
            end=jnp.array(500.0),
            total_steps=jnp.array(100),
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(0), jnp.array(0.0)), jnp.array(300.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(50), jnp.array(0.0)), jnp.array(400.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(100), jnp.array(0.0)), jnp.array(500.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(200), jnp.array(0.0)), jnp.array(500.0)
        )

    def test_exponential_schedule(self):
        """ExponentialSchedule applies multiplicative decay/growth with bounds."""
        decay = ExponentialSchedule(rate=jnp.array(0.9), bounds=(None, None))
        npt.assert_array_almost_equal(
            decay(jnp.array(0), jnp.array(100.0)), jnp.array(90.0)
        )

        bounded = ExponentialSchedule(
            rate=jnp.array(0.5), bounds=(jnp.array(60.0), None)
        )
        npt.assert_array_almost_equal(
            bounded(jnp.array(0), jnp.array(100.0)), jnp.array(60.0)
        )

    def test_step_function_schedule(self):
        """StepFunctionSchedule returns values at discrete thresholds."""
        schedule = StepFunctionSchedule(
            steps=jnp.array([0, 100, 200]),
            values=(jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)),
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(50), jnp.array(0.0)), jnp.array(1.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(100), jnp.array(0.0)), jnp.array(2.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(250), jnp.array(0.0)), jnp.array(3.0)
        )

    def test_cosine_annealing_schedule(self):
        """CosineAnnealingSchedule smoothly anneals from max to min."""
        schedule = CosineAnnealingSchedule(
            min_value=jnp.array(300.0),
            max_value=jnp.array(500.0),
            total_steps=jnp.array(1000),
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(0), jnp.array(0.0)), jnp.array(500.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(500), jnp.array(0.0)), jnp.array(400.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(1000), jnp.array(0.0)), jnp.array(300.0)
        )

    def test_composed_schedule(self):
        """ComposedSchedule chains two schedules at transition point."""
        schedule = ComposedSchedule(
            first=LinearSchedule(
                start=jnp.array(0.0),
                end=jnp.array(100.0),
                total_steps=jnp.array(100),
            ),
            second=ConstantSchedule(value=jnp.array(200.0)),
            transition_input=jnp.array(100),
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(50), jnp.array(0.0)), jnp.array(50.0)
        )
        npt.assert_array_almost_equal(
            schedule(jnp.array(100), jnp.array(0.0)), jnp.array(200.0)
        )


class TestPropertyScheduler:
    """Tests for PropertyScheduler."""

    def test_property_scheduler_updates_state(self, simple_state):
        """PropertyScheduler applies schedule and updates state via lens."""
        scheduler = PropertyScheduler(
            lens=SimpleLens(get_temperature),
            schedule=LinearSchedule(
                start=jnp.array(300.0),
                end=jnp.array(500.0),
                total_steps=jnp.array(100),
            ),
        )
        result = scheduler(simple_state, jnp.array(50))
        npt.assert_array_almost_equal(result.temperature, jnp.array(400.0))


class TestJAXCompatibility:
    """Tests for JAX JIT and vmap compatibility."""

    def test_jax_transforms(self):
        """All schedules should be JIT-compatible, LinearSchedule also vmap-compatible."""
        schedules = [
            ConstantSchedule(value=jnp.array(500.0)),
            IncrementSchedule(increment=jnp.array(1)),
            LinearSchedule(
                start=jnp.array(300.0),
                end=jnp.array(500.0),
                total_steps=jnp.array(100),
            ),
            ExponentialSchedule(rate=jnp.array(0.9), bounds=(None, None)),
            CosineAnnealingSchedule(
                min_value=jnp.array(300.0),
                max_value=jnp.array(500.0),
                total_steps=jnp.array(100),
            ),
        ]

        for schedule in schedules:

            @jax.jit
            def apply(step, current, s=schedule):
                return s(step, current)

            result = apply(jnp.array(50), jnp.array(100.0))
            assert result is not None

        # Also test vmap for LinearSchedule
        linear = schedules[2]
        steps = jnp.array([0, 50, 100])
        results = jax.vmap(linear, in_axes=(0, None))(steps, jnp.array(0.0))
        expected = jnp.array([300.0, 400.0, 500.0])
        npt.assert_array_almost_equal(results, expected)


class TestPyTreeSupport:
    """Tests for PyTree value support."""

    def test_linear_schedule_pytree(self):
        """LinearSchedule works with PyTree values."""
        schedule = LinearSchedule(
            start={"temp": jnp.array(300.0), "pressure": jnp.array(1.0)},
            end={"temp": jnp.array(500.0), "pressure": jnp.array(2.0)},
            total_steps=jnp.array(100),
        )
        current = {"temp": jnp.array(0.0), "pressure": jnp.array(0.0)}
        result = schedule(jnp.array(50), current)
        npt.assert_array_almost_equal(result["temp"], jnp.array(400.0))
        npt.assert_array_almost_equal(result["pressure"], jnp.array(1.5))
