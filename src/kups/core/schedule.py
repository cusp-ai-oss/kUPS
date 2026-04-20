# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Property scheduling for time-dependent simulation parameters.

This module provides a framework for scheduling property changes during simulation,
such as temperature annealing, pressure ramps, and time step adaptation.

Key components:

- **[Schedule][kups.core.schedule.Schedule]**: Protocol for property schedules
- **[Scheduler][kups.core.schedule.Scheduler]**: Protocol for state schedulers
- **[PropertyScheduler][kups.core.schedule.PropertyScheduler]**: Applies schedules to state properties
- **[ConstantSchedule][kups.core.schedule.ConstantSchedule]**: Returns a fixed value
- **[IncrementSchedule][kups.core.schedule.IncrementSchedule]**: Increments current value by fixed amount
- **[LinearSchedule][kups.core.schedule.LinearSchedule]**: Linear interpolation between values
- **[ExponentialSchedule][kups.core.schedule.ExponentialSchedule]**: Exponential decay/growth
- **[StepFunctionSchedule][kups.core.schedule.StepFunctionSchedule]**: Discrete step changes
- **[CosineAnnealingSchedule][kups.core.schedule.CosineAnnealingSchedule]**: Smooth cosine-based annealing

The `Input` and `Value` types are generic, enabling both simple step-based scheduling
(where `Value = Array`) and complex acceptance-based scheduling (where `Value = ParameterSchedulerState`).
"""

from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.lens import Lens
from kups.core.utils.jax import dataclass, field


class Schedule[Input, Value](Protocol):
    """Protocol for property schedules.

    A schedule defines how a property value changes based on some input.
    Both `Input` and `Value` are generic, allowing flexible scheduling strategies.

    Type Parameters:
        Input: Type of scheduling input (step number, acceptance rate, time, etc.)
        Value: Type of value being scheduled (Array, ParameterSchedulerState, etc.)

    Example:
        ```python
        # Simple step-based schedule
        class MySchedule:
            def __call__(self, step: Array, current: Array) -> Array:
                return current * 0.999  # 0.1% decay per step

        # Complex state-based schedule
        class AcceptanceSchedule:
            def __call__(self, acceptance: Array, state: SchedulerState) -> SchedulerState:
                # Update state based on acceptance rate
                ...
        ```
    """

    def __call__(self, input: Input, current: Value, /) -> Value:
        """Compute the scheduled value.

        Args:
            input: Scheduling input (e.g., step number, acceptance rate)
            current: Current value of the property

        Returns:
            New value for the property
        """
        ...


class Scheduler[State, Input](Protocol):
    """Protocol for schedulers that update state based on input.

    This protocol defines the interface for any scheduler that can be used
    with [MCMCPropagator][kups.core.propagator.MCMCPropagator] or other
    components that need to adjust parameters based on some input signal.

    Type Parameters:
        State: Simulation state type
        Input: Type of input signal (e.g., acceptance flags, step number)

    Example:
        ```python
        class MyScheduler:
            def __call__(self, state: State, input: Array) -> State:
                # Update state based on input
                return updated_state
        ```
    """

    def __call__(self, state: State, input: Input) -> State:
        """Update state based on input.

        Args:
            state: Current simulation state
            input: Input signal (e.g., acceptance flags)

        Returns:
            Updated state
        """
        ...


@dataclass
class ConstantSchedule[Value](Schedule[Array, Value]):
    """Schedule that returns a constant value.

    Ignores both input and current value, always returning the specified constant.

    Supports PyTree values.

    Type Parameters:
        Value: Type of value being scheduled (Array or PyTree of Arrays)

    Attributes:
        value: The constant value to return (can be any PyTree)

    Example:
        ```python
        schedule = ConstantSchedule(value=jnp.array(300.0))
        # Always returns 300.0 regardless of step or current value
        new_temp = schedule(step, current_temp)  # Returns 300.0
        ```
    """

    value: Value

    def __call__(self, input: Array, current: Value) -> Value:
        del input, current  # Unused
        return self.value


@dataclass
class IncrementSchedule[Value](Schedule[Any, Value]):
    """Schedule that increments the current value by a fixed amount.

    Ignores the input and returns `current + increment`.
    Supports PyTree values - all leaves are incremented by the same amount.

    Type Parameters:
        Value: Type of value being scheduled (Array or PyTree of Arrays)

    Attributes:
        increment: The amount to add to the current value

    Example:
        ```python
        schedule = IncrementSchedule(increment=jnp.array(1))
        new_step = schedule(input, current_step)  # Returns current_step + 1
        ```
    """

    increment: Value

    def __call__(self, input: Any, current: Value) -> Value:
        del input  # Unused
        return jax.tree.map(lambda c, i: c + i, current, self.increment)


@dataclass
class LinearSchedule[Value](Schedule[Array, Value]):
    """Linear interpolation between start and end values.

    Computes: `start + (end - start) * clamp(input / total_steps, 0, 1)`

    The value transitions linearly from `start` to `end` over `total_steps`,
    then remains at `end` for all subsequent steps.

    Supports PyTree values - all leaves are interpolated uniformly.

    Type Parameters:
        Value: Type of value being scheduled (Array or PyTree of Arrays)

    Attributes:
        start: Initial value at step 0 (PyTree structure must match end)
        end: Final value at total_steps (PyTree structure must match start)
        total_steps: Number of steps over which to interpolate

    Example:
        ```python
        # Temperature ramp from 300K to 500K over 10000 steps
        schedule = LinearSchedule(
            start=jnp.array(300.0),
            end=jnp.array(500.0),
            total_steps=jnp.array(10000)
        )
        temp = schedule(jnp.array(5000), current)  # Returns 400.0

        # Can also interpolate PyTrees
        schedule = LinearSchedule(
            start={"temp": jnp.array(300.0), "pressure": jnp.array(1.0)},
            end={"temp": jnp.array(500.0), "pressure": jnp.array(2.0)},
            total_steps=jnp.array(10000)
        )
        ```
    """

    start: Value
    end: Value
    total_steps: Array

    def __call__(self, input: Array, current: Value) -> Value:
        del current  # Unused - linear schedule ignores current value
        t = jnp.clip(input / self.total_steps, 0.0, 1.0)
        return jax.tree.map(lambda s, e: s + (e - s) * t, self.start, self.end)


@dataclass
class ExponentialSchedule[Value](Schedule[Array, Value]):
    """Exponential decay or growth of a property.

    Computes: `clamp(current * rate, min_value, max_value)`

    Each call multiplies the current value by the rate, enabling exponential
    decay (rate < 1) or growth (rate > 1).

    Supports PyTree values - all leaves are scaled by the same rate.

    Type Parameters:
        Value: Type of value being scheduled (Array or PyTree of Arrays)

    Attributes:
        rate: Multiplicative factor per step (< 1 for decay, > 1 for growth)
        bounds: Optional (min, max) bounds to clamp the result (PyTree or None)

    Example:
        ```python
        # Exponential cooling with 0.1% decay per step, minimum 100K
        schedule = ExponentialSchedule(
            rate=jnp.array(0.999),
            bounds=(jnp.array(100.0), None)
        )
        new_temp = schedule(step, current_temp)  # Decays toward 100K
        ```
    """

    rate: Array
    bounds: tuple[Value | None, Value | None] = (None, None)

    def __call__(self, input: Array, current: Value) -> Value:
        del input  # Unused - exponential schedule only uses current
        new = jax.tree.map(lambda c: c * self.rate, current)
        min_val, max_val = self.bounds
        if min_val is not None:
            new = jax.tree.map(jnp.maximum, new, min_val)
        if max_val is not None:
            new = jax.tree.map(jnp.minimum, new, max_val)
        return new


@dataclass
class StepFunctionSchedule[Value](Schedule[Array, Value]):
    """Step function schedule with discrete value changes.

    Returns the value corresponding to the largest step threshold not exceeding
    the current input.

    Supports PyTree values - the `values` attribute should be a list/tuple of
    PyTrees with matching structure.

    Type Parameters:
        Value: Type of value being scheduled (Array or PyTree of Arrays)

    Attributes:
        steps: Array of step thresholds (must be sorted ascending)
        values: Tuple of values corresponding to each threshold (PyTree structure)

    Example:
        ```python
        # Change temperature at specific steps
        schedule = StepFunctionSchedule(
            steps=jnp.array([0, 1000, 5000, 10000]),
            values=(
                jnp.array(300.0),
                jnp.array(350.0),
                jnp.array(400.0),
                jnp.array(300.0),
            )
        )
        # step 0-999: 300.0
        # step 1000-4999: 350.0
        # step 5000-9999: 400.0
        # step 10000+: 300.0

        # Can also use PyTree values
        schedule = StepFunctionSchedule(
            steps=jnp.array([0, 1000]),
            values=(
                {"temp": jnp.array(300.0), "pressure": jnp.array(1.0)},
                {"temp": jnp.array(400.0), "pressure": jnp.array(2.0)},
            )
        )
        ```
    """

    steps: Array
    values: tuple[Value, ...]

    def __call__(self, input: Array, current: Value) -> Value:
        del current  # Unused
        idx = jnp.searchsorted(self.steps, input, side="right") - 1
        idx = jnp.clip(idx, 0, len(self.values) - 1)
        # Use jax.lax.switch to select from values based on index
        return jax.lax.switch(idx, [lambda v=v: v for v in self.values])


@dataclass
class CosineAnnealingSchedule[Value](Schedule[Array, Value]):
    """Cosine annealing schedule for smooth value transitions.

    Computes: `min_value + 0.5 * (max_value - min_value) * (1 + cos(pi * input / total_steps))`

    Provides smooth transitions that slow down at the extremes, starting at
    `max_value` and annealing to `min_value` over `total_steps`.

    Supports PyTree values - all leaves are annealed uniformly.

    Type Parameters:
        Value: Type of value being scheduled (Array or PyTree of Arrays)

    Attributes:
        min_value: Minimum (target) value (PyTree structure must match max_value)
        max_value: Maximum (starting) value (PyTree structure must match min_value)
        total_steps: Period of one cosine cycle

    Example:
        ```python
        # Cosine annealing from 500K to 300K over 10000 steps
        schedule = CosineAnnealingSchedule(
            min_value=jnp.array(300.0),
            max_value=jnp.array(500.0),
            total_steps=jnp.array(10000)
        )
        temp = schedule(jnp.array(5000), current)  # Returns 400.0 (midpoint)
        ```
    """

    min_value: Value
    max_value: Value
    total_steps: Array

    def __call__(self, input: Array, current: Value) -> Value:
        del current  # Unused - cosine schedule ignores current value
        t = jnp.clip(input / self.total_steps, 0.0, 1.0)
        cos_factor = 0.5 * (1 + jnp.cos(jnp.pi * t))
        return jax.tree.map(
            lambda min_v, max_v: min_v + (max_v - min_v) * cos_factor,
            self.min_value,
            self.max_value,
        )


@dataclass
class ComposedSchedule[Value](Schedule[Array, Value]):
    """Compose two schedules sequentially.

    Uses the first schedule for inputs below `transition_input`, then switches
    to the second schedule (with input offset by `transition_input`).

    Supports PyTree values - both schedules must return the same PyTree structure.

    Type Parameters:
        Value: Type of value being scheduled (Array or PyTree of Arrays)

    Attributes:
        first: Schedule to use before transition
        second: Schedule to use after transition
        transition_input: Input value at which to switch schedules

    Example:
        ```python
        # Linear warmup followed by exponential decay
        schedule = ComposedSchedule(
            first=LinearSchedule(
                start=jnp.array(100.0),
                end=jnp.array(500.0),
                total_steps=jnp.array(1000)
            ),
            second=ExponentialSchedule(
                rate=jnp.array(0.999),
                bounds=(jnp.array(100.0), None)
            ),
            transition_input=jnp.array(1000)
        )
        ```
    """

    first: Schedule[Array, Value] = field(static=True)
    second: Schedule[Array, Value] = field(static=True)
    transition_input: Array

    def __call__(self, input: Array, current: Value) -> Value:
        first_result = self.first(input, current)
        second_result = self.second(input - self.transition_input, current)
        return jax.tree.map(
            lambda f, s: jnp.where(input < self.transition_input, f, s),
            first_result,
            second_result,
        )


@dataclass
class PropertyScheduler[State, Input, Value](Scheduler[State, Input]):
    """Applies a Schedule to update a property in state.

    This is a generic scheduler that works with any Schedule implementation.
    It reads the current value via a lens, applies the schedule, and writes
    the result back.

    Type Parameters:
        State: Simulation state type
        Input: Schedule input type
        Value: Type of value being scheduled

    Attributes:
        lens: Lens to access and update the scheduled property
        schedule: Schedule function that computes new values

    Example:
        ```python
        # Step-based temperature scheduling
        temp_scheduler = PropertyScheduler(
            lens=lens(lambda s: s.temperature),
            schedule=LinearSchedule(
                start=jnp.array(500.0),
                end=jnp.array(300.0),
                total_steps=jnp.array(10000)
            )
        )

        # Apply with explicit input
        state = temp_scheduler(state, step)

        # Acceptance-based scheduling (for MCMC)
        step_size_scheduler = PropertyScheduler(
            lens=lens(lambda s: s.scheduler_state),
            schedule=acceptance_target_schedule
        )

        # Called in MCMC loop
        state = step_size_scheduler(state, acceptance)
        ```
    """

    lens: Lens[State, Value] = field(static=True)
    schedule: Schedule[Input, Value] = field(static=True)

    def __call__(self, state: State, input: Input) -> State:
        """Apply the schedule to update the state.

        Args:
            state: Current simulation state
            input: Scheduling input (step, acceptance, etc.)

        Returns:
            Updated state with scheduled property modified
        """
        current = self.lens.get(state)
        new = self.schedule(input, current)
        return self.lens.set(state, new)
