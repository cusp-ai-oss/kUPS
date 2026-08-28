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
from jax import Array

from kups.core.utils.jax import (
    dataclass,
    drop_nonfinite_compensation,
    kahan_summation,
)


@dataclass
class KahanSummand[T]:
    """Numerically stable accumulator for repeated addition.

    Holds a running sum together with a compensation term that captures the
    low-order bits dropped by rounding at each step. The ``+`` operator (and
    ``+=``) accumulates a value and returns a new ``KahanSummand``, so it can be
    folded over an iterable or passed to ``sum`` like a plain number.

    Values may be arbitrary PyTrees, mirroring
    [kahan_summation][kups.core.utils.jax.kahan_summation].

    An infinite value carries a zero compensation, so an infinite total stays
    infinite rather than becoming NaN.

    Attributes:
        value: Current running sum as a PyTree.
        compensate: Accumulated rounding error; the best estimate of the true
            sum is ``value - compensate``; zero where `value` is not finite.

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

        Adding a plain value applies the compensation to the addend, which keeps
        the running sum close to the true total when many small values are
        accumulated. Adding another summand instead adds the two compensations to
        each other and folds in the exact rounding error of the value addition:
        the addend is then of comparable magnitude, and subtracting a compensation
        from it would round the compensation away entirely.

        Args:
            other: A PyTree to add, or another ``KahanSummand`` whose value and
                compensation are both folded in.

        Returns:
            A new summand with the updated running sum and compensation.
        """
        if not isinstance(other, KahanSummand):
            value, error = kahan_summation(
                self.value, other, compensate=self.compensate
            )
            return KahanSummand(value, error)

        def excess(total: Array, a: Array, b: Array) -> Array:
            """Amount by which `fl(a + b)` exceeds `a + b`, exactly.

            Knuth's 2Sum, which needs no ordering of the operands. The cheaper
            ``(total - a) - b`` is only exact for ``|a| >= |b|``, which a sum over
            potential terms cannot guarantee.
            """
            b_part = total - a
            a_part = total - b_part
            return (a_part - a) + (b_part - b)

        value = jax.tree.map(jnp.add, self.value, other.value)
        compensate = jax.tree.map(
            lambda c, o, e: c + o + e,
            self.compensate,
            other.compensate,
            jax.tree.map(excess, value, self.value, other.value),
        )
        return KahanSummand(value, drop_nonfinite_compensation(compensate))

    def __radd__(self, other: KahanSummand[T] | T) -> KahanSummand[T]:
        """Accumulate from the left so `sum` and `other + summand` work.

        Args:
            other: A value to add from the left, e.g. the ``0`` start value of
                ``sum``.

        Returns:
            A new summand with the updated running sum and compensation.
        """
        return self.__add__(other)

    def __mul__(self, other: float) -> KahanSummand[T]:
        """Scale the running sum and its compensation by `other`.

        Args:
            other: Scalar factor.

        Returns:
            A summand representing `other` times the accumulated sum.
        """
        return jax.tree.map(lambda x: other * x, self)

    def __rmul__(self, other: float) -> KahanSummand[T]:
        """Scale from the left, see [__mul__][kups.core.utils.kahan.KahanSummand.__mul__]."""
        return self.__mul__(other)
