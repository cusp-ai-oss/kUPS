# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.utils.kahan import KahanSummand


@pytest.fixture(autouse=True)
def _disable_x64():
    """Force float32 for this module without leaking the global flag.

    Compensated summation only earns its keep once rounding is coarse enough to
    lose an addend. The suite runs in float64, where demonstrating that needs
    extreme magnitudes; float32 exercises the same code path at ordinary ones.
    """
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


class TestKahanSummand:
    def test_init(self):
        """Seeds value and zeroes compensation for scalars and PyTrees."""
        s = KahanSummand.init(jnp.array([1.0, 2.0]))
        npt.assert_array_equal(s.value, [1.0, 2.0])
        npt.assert_array_equal(s.compensate, [0.0, 0.0])

        tree = KahanSummand.init({"a": jnp.array(1.0), "b": jnp.zeros(2)})
        npt.assert_array_equal(tree.compensate["a"], 0.0)
        npt.assert_array_equal(tree.compensate["b"], jnp.zeros(2))

    def test_add_accumulates(self):
        """`+` and `+=` accumulate values into the running sum."""
        s = KahanSummand.init(jnp.array(0.0)) + jnp.array(3.0) + jnp.array(4.0)
        npt.assert_allclose(s.value, 7.0)

        s = KahanSummand.init(jnp.array([1.0, 2.0]))
        s += jnp.array([3.0, 4.0])
        npt.assert_allclose(s.value, [4.0, 6.0])

    def test_add_two_summands(self):
        """Adding two summands folds in both value and compensation."""
        a = KahanSummand.init(jnp.array(1.0)) + jnp.array(0.5)
        b = KahanSummand.init(jnp.array(2.0)) + jnp.array(0.25)
        npt.assert_allclose((a + b).value, 3.75)

    def test_sum_builtin(self):
        """`sum` works via __radd__ on its 0 start value."""
        summands = [KahanSummand.init(jnp.array(float(i))) for i in range(4)]
        total = sum(summands)
        assert isinstance(total, KahanSummand)
        npt.assert_allclose(total.value, 6.0)

    def test_numerical_stability(self):
        """Compensation recovers precision a naive sum loses."""
        # `small` is half an ulp of `big`, so every naive addition rounds away.
        # Sixteen of them make exactly 1.0, which the compensation carries.
        big = jnp.array(2.0**20)
        small = jnp.array(2.0**-4)
        assert big.dtype == jnp.float32
        s = KahanSummand.init(big)
        naive = big
        for _ in range(16):
            s += small
            naive = naive + small
        npt.assert_array_equal(s.total - big, 1.0)
        npt.assert_array_equal(naive - big, 0.0)

    def test_total(self):
        """`total` reports the compensated sum for scalars and PyTrees."""
        s = KahanSummand(jnp.array(4.0), jnp.array(0.25))
        npt.assert_allclose(s.total, 3.75)

        tree = KahanSummand({"a": jnp.array([2.0, 3.0])}, {"a": jnp.array([0.5, 1.0])})
        npt.assert_allclose(tree.total["a"], [1.5, 2.0])

    def test_difference_recovers_small_increment(self):
        """`difference` resolves an increment the running sums cannot."""
        big = jnp.array(1e5)
        delta = jnp.array(1e-2)
        old = KahanSummand.init(big)
        for _ in range(50):
            old += delta
        new = old + delta

        npt.assert_allclose(new.difference(old), delta, rtol=1e-5)
        # float32 cannot represent the increment on top of `big`: the running
        # sums differ by a multiple of ulp(1e5) = 2**-7 instead.
        assert abs(float(new.value - old.value) - float(delta)) > 1e-3

    def test_difference_pytree(self):
        """`difference` maps over PyTrees."""
        a = KahanSummand.init({"x": jnp.array([1.0, 2.0])}) + {
            "x": jnp.array([1.0, 1.0])
        }
        b = KahanSummand.init({"x": jnp.array([1.0, 2.0])})
        npt.assert_allclose(a.difference(b)["x"], [1.0, 1.0])

    def test_jit_and_immutability(self):
        """Works under jit and leaves the original untouched."""
        s0 = KahanSummand.init(jnp.array([1.0, 2.0]))
        added = jax.jit(lambda s, x: s + x)(s0, jnp.array([3.0, 4.0]))
        npt.assert_allclose(added.value, [4.0, 6.0])
        npt.assert_array_equal(s0.value, [1.0, 2.0])
