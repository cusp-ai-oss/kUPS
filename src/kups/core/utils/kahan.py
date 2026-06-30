# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Compensated summation as a composable accumulator.

This module wraps Kahan's compensated summation in a small value type. Adding
to a ``KahanSummand`` returns a new summand carrying both the running sum and
the rounding error lost so far, reducing floating-point drift when many values
are accumulated.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kups.core.utils.jax import dataclass, kahan_summation


@dataclass
class KahanSummand[T]:
    """Numerically stable accumulator for repeated addition.

    Holds a running sum together with a compensation term that captures the
    low-order bits dropped by rounding at each step. The ``+`` operator (and
    ``+=``) accumulates a value and returns a new ``KahanSummand``, so it can be
    folded over an iterable or passed to ``sum`` like a plain number.

    Values may be arbitrary PyTrees, mirroring
    [kahan_summation][kups.core.utils.jax.kahan_summation].

    Attributes:
        value: Current running sum as a PyTree.
        compensate: Accumulated rounding error; the best estimate of the true
            sum is ``value - compensate``.

    Example:
        ```python
        accumulator = KahanSummand.init(jnp.zeros(3))
        for x in observations:
            accumulator += x
        result = accumulator.total
        ```
    """

    value: T
    compensate: T

    @classmethod
    def init(cls, value: T) -> KahanSummand[T]:
        """Create a summand seeded with `value` and zero compensation.

        Args:
            value: Initial running sum as a PyTree.

        Returns:
            A summand starting from `value`.
        """
        return cls(value, jax.tree.map(jnp.zeros_like, value))

    @property
    def total(self) -> T:
        """Best single-value estimate of the sum, ``value - compensate``."""
        return jax.tree.map(jnp.subtract, self.value, self.compensate)

    def difference(self, other: KahanSummand[T]) -> T:
        """Difference ``self - other`` of two accumulators.

        Subtracts the running sums and the compensations separately before
        combining them. For accumulators that differ by a small increment this
        recovers the increment to full precision: the running sums cancel
        exactly, and the low-order bits each of them dropped are carried by the
        compensations. Folding the compensations in first (as ``+`` does) would
        instead annihilate them against the much larger running sums.

        Args:
            other: Accumulator to subtract.

        Returns:
            The difference as a PyTree, without compensation.
        """
        value = jax.tree.map(jnp.subtract, self.value, other.value)
        compensate = jax.tree.map(jnp.subtract, self.compensate, other.compensate)
        return jax.tree.map(jnp.subtract, value, compensate)

    def __add__(self, other: KahanSummand[T] | T) -> KahanSummand[T]:
        """Accumulate a value or another summand.

        Args:
            other: A PyTree to add, or another ``KahanSummand`` whose value and
                compensation are both folded in.

        Returns:
            A new summand with the updated running sum and compensation.
        """
        addend: T
        compensate: T
        if isinstance(other, KahanSummand):
            addend = other.value
            compensate = jax.tree.map(jnp.add, self.compensate, other.compensate)
        else:
            addend = other
            compensate = self.compensate
        value, error = kahan_summation(self.value, addend, compensate=compensate)
        return KahanSummand(value, error)

    def __radd__(self, other: KahanSummand[T] | T) -> KahanSummand[T]:
        """Accumulate from the left so `sum` and `other + summand` work.

        Args:
            other: A value to add from the left, e.g. the ``0`` start value of
                ``sum``.

        Returns:
            A new summand with the updated running sum and compensation.
        """
        return self.__add__(other)
