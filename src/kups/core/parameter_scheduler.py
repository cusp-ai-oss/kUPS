# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Adaptive parameter scheduling based on acceptance rates.

This module implements automatic parameter adjustment for Monte Carlo simulations,
dynamically tuning parameters (e.g., step sizes, temperatures) to maintain target
acceptance rates.

Key components:

- **[ParameterSchedulerState][kups.core.parameter_scheduler.ParameterSchedulerState]**: State containing the scheduled parameter value, configuration, and history.
- **[acceptance_target_schedule][kups.core.parameter_scheduler.acceptance_target_schedule]**: Schedule function that adjusts values based on acceptance rates.

Example:
    ```python
    from kups.core.schedule import PropertyScheduler
    from kups.core.parameter_scheduler import acceptance_target_schedule

    scheduler = PropertyScheduler(
        lens=lens(lambda s: s.scheduler_state),
        schedule=acceptance_target_schedule,
    )
    state = scheduler(state, acceptance)
    ```
"""

from enum import Enum
from typing import Any, overload

import jax.numpy as jnp
from jax import Array

from kups.core import units
from kups.core.units._typed import _E_A, _E_K, _E_KG, _E_M, _E_S
from kups.core.utils.jax import dataclass, field


class Correlation(Enum):
    """Defines how parameters correlate with acceptance rates.

    Attributes:
        POSITIVE: Parameter increases lead to higher acceptance (e.g., temperature)
        NEGATIVE: Parameter increases lead to lower acceptance (e.g., step size)
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class AcceptanceHistory:
    """Circular buffer for tracking acceptance rates over time.

    Attributes:
        values: Array of shape `(n_systems, history_length)` storing acceptance flags
        index: Current write position in the circular buffer for each system
    """

    values: Array
    index: Array


@dataclass
class ParameterSchedulerState[V]:
    """State for parameter scheduling algorithm.

    Contains both the parameter value being scheduled and the configuration
    for the scheduling algorithm.

    Attributes:
        value: The parameter value being scheduled (e.g., step size, temperature)
        multiplicity: Factor by which to multiply/divide parameter on adjustment
        target: Target acceptance rate to achieve (typically 0.2-0.5 for MC)
        tolerance: Acceptance within +/-tolerance of target won't trigger adjustment
        correlation: How parameter correlates with acceptance rate
        bounds: Optional (min, max) bounds to clip parameter values
        history: Circular buffer tracking recent acceptance rates

    Example:
        ```python
        state = ParameterSchedulerState(
            value=jnp.array([0.1, 0.2]) * u.angstrom,  # Step sizes for 2 systems
            multiplicity=jnp.array([1.5, 1.5]),
            target=jnp.array([0.4, 0.4]),
            tolerance=jnp.array([0.05, 0.05]),
            correlation=Correlation.NEGATIVE,
            bounds=(jnp.array([0.01, 0.01]) * u.angstrom, jnp.array([1.0, 1.0]) * u.angstrom),
            history=AcceptanceHistory(
                values=jnp.zeros((2, 10)),
                index=jnp.array([0, 0]),
            ),
        )
        ```
    """

    value: V
    multiplicity: Array
    target: Array
    tolerance: Array
    correlation: Correlation = field(static=True)
    bounds: tuple[V | None, V | None]
    history: AcceptanceHistory

    @overload
    @classmethod
    def create(
        cls,
        n_systems: int = 1,
        initial_value: float = 0.1,
        multiplicity: float = 1.1,
        target: float = 0.5,
        tolerance: float = 0.05,
        correlation: Correlation = Correlation.NEGATIVE,
        lower_bound: float = 0.0,
        upper_bound: float | None = None,
        history_length: int = 100,
    ) -> "ParameterSchedulerState[Array]": ...
    @overload
    @classmethod
    def create(
        cls,
        n_systems: int = 1,
        *,
        initial_value: units.Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
        multiplicity: float = 1.1,
        target: float = 0.5,
        tolerance: float = 0.05,
        correlation: Correlation = Correlation.NEGATIVE,
        lower_bound: units.Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float = 0.0,
        upper_bound: units.Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | None = None,
        history_length: int = 100,
    ) -> """ParameterSchedulerState[
        units.Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
    ]""": ...
    @classmethod
    def create(
        cls,
        n_systems: int = 1,
        initial_value: units.Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float = 0.1,
        multiplicity: float = 1.1,
        target: float = 0.5,
        tolerance: float = 0.05,
        correlation: Correlation = Correlation.NEGATIVE,
        lower_bound: units.Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float = 0.0,
        upper_bound: units.Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | float
        | None = None,
        history_length: int = 100,
    ) -> "ParameterSchedulerState[Any]":
        """Create a ParameterSchedulerState with sensible defaults for ~50 % acceptance.

        Args:
            n_systems: Number of independent simulation systems.
            initial_value: Starting parameter value (e.g. step size).
            multiplicity: Multiplicative adjustment factor applied each scheduling
                cycle; must be > 1. Smaller values (e.g. 1.1) give finer
                adaptation; larger values (e.g. 2.0) give faster adaptation.
            target: Target acceptance rate in ``[0, 1]``. Defaults to ``0.5``
                as a general-purpose starting point; use ``0.234`` for
                high-dimensional moves (optimal Metropolis-Hastings).
            tolerance: Half-width of the dead-band around ``target``. No
                adjustment is made while the observed acceptance lies within
                ``[target - tolerance, target + tolerance]``.
            correlation: Whether the parameter correlates positively or
                negatively with acceptance. Step sizes use
                [Correlation.NEGATIVE][kups.core.parameter_scheduler.Correlation]
                (larger step → lower acceptance).
            lower_bound: Minimum allowed parameter value (inclusive), in
                ``initial_value``'s dimension; a plain ``0.0`` takes its unit. Set
                to ``0.0`` for step sizes; use ``-inf`` (times the unit for a
                quantity) for unconstrained minima.
            upper_bound: Maximum allowed parameter value (inclusive), or
                ``None`` for no upper constraint.
            history_length: Number of recent MC moves recorded in the circular
                acceptance-history buffer. The scheduler updates the parameter
                once per complete buffer cycle.

        Returns:
            Initialised [ParameterSchedulerState][kups.core.parameter_scheduler.ParameterSchedulerState].
        """
        value: Any = units.full((n_systems,), initial_value)
        upper = (
            units.full((n_systems,), upper_bound) if upper_bound is not None else None
        )
        return cls(
            value=value,
            multiplicity=jnp.full((n_systems,), multiplicity),
            target=jnp.full((n_systems,), target),
            tolerance=jnp.full((n_systems,), tolerance),
            correlation=correlation,
            bounds=(units.full_like(value, lower_bound), upper),
            history=AcceptanceHistory(
                values=jnp.zeros((n_systems, history_length)),
                index=jnp.zeros((n_systems,), dtype=int),
            ),
        )


def acceptance_target_schedule[V](
    input: Array, current: ParameterSchedulerState[V]
) -> ParameterSchedulerState[V]:
    """Adjust parameter values to achieve target acceptance rate.

    Implements multiplicative adjustment based on acceptance history:
    tracks acceptance over a rolling window, compares average to target,
    and adjusts multiplicatively based on correlation direction.

    Args:
        input: Boolean/float array of shape `(n_systems,)` indicating acceptance.
        current: Current scheduler state containing value and history.

    Returns:
        Updated scheduler state with adjusted value and updated history.
    """
    history = current.history
    history_size = history.values.shape[1]
    idx = history.index

    # Elementwise, so selects over the result update the history in place.
    slot = jnp.arange(history_size) == idx[:, None]
    value = input.astype(history.values.dtype)[:, None]
    new_history_values = jnp.where(slot, value, history.values)
    new_indices = (idx + 1) % history_size
    avg_acceptance = jnp.mean(new_history_values, axis=-1)

    if current.correlation is Correlation.POSITIVE:
        multiplicity = current.multiplicity
    else:
        multiplicity = 1 / current.multiplicity

    mask = avg_acceptance < current.target
    state: ParameterSchedulerState[Any] = current
    new_value = units.where(
        mask, state.value * multiplicity, state.value / multiplicity
    )
    new_value = units.clip(new_value, *state.bounds)

    accept_mask = jnp.abs(avg_acceptance - current.target) > current.tolerance
    new_value = units.where(accept_mask, new_value, state.value)
    result_value = units.where(new_indices == 0, new_value, state.value)

    return ParameterSchedulerState(
        value=result_value,
        multiplicity=current.multiplicity,
        target=current.target,
        tolerance=current.tolerance,
        correlation=current.correlation,
        bounds=current.bounds,
        history=AcceptanceHistory(new_history_values, new_indices),
    )
