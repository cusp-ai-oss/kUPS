# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Exponential Moving Average (EMA) with numerical stability.

This module implements an exponential moving average that maintains numerical
stability using Kahan summation to reduce floating-point errors during
accumulation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.utils.jax import dataclass, jit, kahan_summation


@dataclass
class EMA[PyTree]:
    """Exponential moving average for arbitrary PyTree structures.

    Computes a weighted moving average where recent values have exponentially
    higher weight. Uses Kahan summation for numerical stability to prevent
    accumulation of floating-point errors.

    The EMA update follows:

    $$
    \\text{data}_t = \\alpha \\cdot \\text{data}_{t-1} + \\text{data}_\\text{new}
    $$

    $$
    \\text{weight}_t = \\alpha \\cdot \\text{weight}_{t-1} + 1
    $$

    The final average is: `data_t / weight_t`

    Attributes:
        data: Accumulated weighted sum as a PyTree.
        weight: Total accumulated weight as a scalar.
        alpha: Decay factor in range (0, 1). Higher values give more weight to history.
        _compensate: Kahan summation error compensation for data (internal).
        _weight_compensate: Kahan summation error compensation for weight (internal).

    Example:
        ```python
        # Initialize with first observation
        ema = EMA.init(jnp.array([1.0, 2.0]), alpha=0.9)

        # Update with new observations
        ema = ema.update(jnp.array([3.0, 4.0]))
        ema = ema.update(jnp.array([5.0, 6.0]))

        # Get the moving average
        average = ema.get()
        ```
    """

    data: PyTree
    weight: Array
    alpha: float
    _compensate: PyTree
    _weight_compensate: Array

    @staticmethod
    def init[T](data: T, alpha: float) -> EMA[T]:
        """Initialize an EMA tracker with zero state.

        Args:
            data: Template PyTree structure to track. Used only for shape/structure.
            alpha: Decay factor in range (0, 1). Values closer to 1 give more weight
                to historical data.

        Returns:
            Initialized EMA with zero-valued data and weight.
        """
        zeros = jax.tree.map(jnp.zeros_like, data)
        weight = jnp.zeros((), dtype=jnp.float64)
        return EMA(zeros, weight, alpha, zeros, weight)

    def update(self, new_data: PyTree) -> EMA[PyTree]:
        """Update the EMA with a new observation.

        Args:
            new_data: New data point to incorporate into the moving average.

        Returns:
            Updated EMA state.
        """
        return _ema_update(self, new_data)

    def get(self) -> PyTree:
        """Compute the current moving average.

        Returns:
            The weighted average of all observed data.
        """
        return _ema_get(self)


@jit
def _ema_update[PyTree](ema: EMA[PyTree], new_data: PyTree) -> EMA[PyTree]:
    decayed = jax.tree.map(lambda x: ema.alpha * x, ema.data)
    data, compensate = kahan_summation(decayed, new_data, compensate=ema._compensate)
    weight = ema.weight * ema.alpha
    weight, weight_compensate = kahan_summation(
        weight, jnp.ones((), dtype=jnp.float64), compensate=ema._weight_compensate
    )
    return EMA(data, weight, ema.alpha, compensate, weight_compensate)


@jit
def _ema_get[PyTree](ema: EMA[PyTree]) -> PyTree:
    return jax.tree.map(lambda x: x / ema.weight, ema.data)
