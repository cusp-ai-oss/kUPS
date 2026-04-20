# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.utils.ema import EMA


class TestEMA:
    def test_init(self):
        """Merged: scalar, array, complex_pytree init."""
        # scalar
        ema = EMA.init(5.0, 0.9)
        assert ema.data == 0.0
        assert ema.alpha == 0.9
        assert ema.weight == 0.0
        assert ema._compensate == 0.0
        assert ema._weight_compensate == 0.0

        # array
        data = jnp.array([1.0, 2.0, 3.0])
        ema = EMA.init(data, 0.8)
        npt.assert_allclose(ema.data, jnp.zeros_like(data))
        assert ema.alpha == 0.8
        npt.assert_allclose(ema._compensate, jnp.zeros_like(data))

        # complex pytree
        data = {
            "scalar": 1.0,
            "array": jnp.array([1.0, 2.0]),
            "nested": {"inner": jnp.array([[1, 2], [3, 4]])},
        }
        ema = EMA.init(data, 0.95)
        assert ema.data["scalar"] == 0.0
        npt.assert_allclose(ema.data["array"], jnp.zeros(2))
        npt.assert_allclose(ema.data["nested"]["inner"], jnp.zeros((2, 2)))

    def test_scalar_updates(self):
        """Merged: multiple_updates + weight_accumulation + exact_multi_step."""
        alpha = 0.9
        values = [10.0, 20.0, 30.0]
        ema = EMA.init(0.0, alpha)

        # first update
        ema = ema.update(values[0])
        npt.assert_allclose(ema.get(), values[0], rtol=1e-10)
        npt.assert_allclose(ema.weight, 1.0, rtol=1e-10)

        # second update
        ema = ema.update(values[1])
        expected2 = (alpha * values[0] + values[1]) / (alpha + 1)
        npt.assert_allclose(ema.get(), expected2, rtol=1e-10)
        npt.assert_allclose(ema.weight, alpha + 1.0, rtol=1e-10)

        # third update
        ema = ema.update(values[2])
        expected_data = alpha * (alpha * values[0] + values[1]) + values[2]
        expected_weight = alpha * (alpha + 1) + 1
        npt.assert_allclose(ema.get(), expected_data / expected_weight, rtol=1e-10)
        npt.assert_allclose(ema.weight, expected_weight, rtol=1e-10)

        # exact multi-step with alpha=0.5
        ema = EMA.init(0.0, 0.5)
        ema = ema.update(2.0)
        npt.assert_allclose(ema.get(), 2.0, rtol=1e-15)
        ema = ema.update(4.0)
        npt.assert_allclose(ema.get(), 5.0 / 1.5, rtol=1e-15)
        ema = ema.update(6.0)
        npt.assert_allclose(ema.get(), 8.5 / 1.75, rtol=1e-15)

    def test_convergence(self):
        """Merged: convergence_properties + alpha_effect."""
        # convergence to constant
        ema = EMA.init(0.0, 0.9)
        for _ in range(100):
            ema = ema.update(15.0)
        npt.assert_allclose(ema.get(), 15.0, rtol=1e-10)

        # alpha effect
        values = [10.0, 20.0]
        ema_high = EMA.init(0.0, 0.95)
        ema_low = EMA.init(0.0, 0.1)
        for v in values:
            ema_high = ema_high.update(v)
            ema_low = ema_low.update(v)
        npt.assert_allclose(ema_high.get(), (0.95 * 10.0 + 20.0) / 1.95, rtol=1e-10)
        npt.assert_allclose(ema_low.get(), (0.1 * 10.0 + 20.0) / 1.1, rtol=1e-10)

    def test_array_and_pytree(self):
        """Merged: array_update + complex_pytree_update."""
        alpha = 0.8
        # array
        ema = EMA.init(jnp.zeros(3), alpha)
        v1, v2 = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
        ema = ema.update(v1)
        npt.assert_allclose(ema.get(), v1, rtol=1e-10)
        ema = ema.update(v2)
        npt.assert_allclose(ema.get(), (alpha * v1 + v2) / (alpha + 1), rtol=1e-10)

        # pytree
        alpha = 0.9
        data = {
            "scalar": 0.0,
            "array": jnp.zeros(2),
            "nested": {"values": jnp.zeros(3)},
        }
        new1 = {
            "scalar": 10.0,
            "array": jnp.array([1.0, 2.0]),
            "nested": {"values": jnp.ones(3)},
        }
        new2 = {
            "scalar": 20.0,
            "array": jnp.array([3.0, 4.0]),
            "nested": {"values": jnp.full(3, 2.0)},
        }

        ema = EMA.init(data, alpha)
        ema = ema.update(new1)
        npt.assert_allclose(ema.get()["scalar"], 10.0, rtol=1e-10)
        ema = ema.update(new2)
        npt.assert_allclose(
            ema.get()["scalar"], (alpha * 10.0 + 20.0) / (alpha + 1), rtol=1e-10
        )
        npt.assert_allclose(
            ema.get()["array"],
            (alpha * jnp.array([1.0, 2.0]) + jnp.array([3.0, 4.0])) / (alpha + 1),
            rtol=1e-10,
        )

    def test_numerical_stability(self):
        """Merged: small_alpha + large_values."""
        # small alpha -> result ~ last value
        ema = EMA.init(0.0, 1e-10)
        for i in range(10):
            ema = ema.update(float(i))
        npt.assert_allclose(ema.get(), 9.0, rtol=1e-8)

        # large values
        alpha = 0.9
        ema = EMA.init(0.0, alpha)
        ema = ema.update(1e10)
        ema = ema.update(2e10)
        npt.assert_allclose(ema.get(), (alpha * 1e10 + 2e10) / (alpha + 1), rtol=1e-10)

    def test_edge_cases(self):
        """Merged: zero_alpha, one_alpha, empty_tree, immutability."""
        # alpha=0 -> always last value
        ema = EMA.init(0.0, 0.0)
        for v in [5.0, 10.0, 15.0]:
            ema = ema.update(v)
            npt.assert_allclose(ema.get(), v, rtol=1e-10)

        # alpha=1 -> cumulative average
        ema = EMA.init(0.0, 1.0)
        ema = ema.update(2.0)
        npt.assert_allclose(ema.get(), 2.0, rtol=1e-10)
        ema = ema.update(4.0)
        npt.assert_allclose(ema.get(), 3.0, rtol=1e-10)
        ema = ema.update(6.0)
        npt.assert_allclose(ema.get(), 4.0, rtol=1e-10)

        # empty tree
        ema = EMA.init({}, 0.9)
        result = ema.update({}).get()
        assert isinstance(result, dict) and len(result) == 0

    def test_jax_transformations_and_gradient(self):
        """Merged: jax_transformations + gradient_through_ema."""
        # JIT
        data = jnp.array([0.0, 0.0])
        alpha = 0.9

        def update_and_get(ema_state, new_value):
            return ema_state.update(new_value).get()

        ema = EMA.init(data, alpha)
        new_value = jnp.array([1.0, 2.0])
        npt.assert_allclose(
            jax.jit(update_and_get)(ema, new_value),
            update_and_get(ema, new_value),
            rtol=1e-15,
        )

        # gradient
        def ema_loss(initial_value, updates):
            ema = EMA.init(0.0, 0.9)
            ema = ema.update(initial_value)
            for update in updates:
                ema = ema.update(update)
            return ema.get() ** 2

        grads = jax.grad(ema_loss)(1.0, jnp.array([2.0, 3.0]))
        expected_grad = (
            2
            * ema_loss(1.0, jnp.array([2.0, 3.0])) ** 0.5
            * (0.9**2)
            / (0.9**2 + 0.9 + 1)
        )
        npt.assert_allclose(grads, expected_grad, rtol=1e-8)

    def test_immutability(self):
        """Test that EMA updates don't modify the original."""
        data = jnp.array([1.0, 2.0])
        ema_original = EMA.init(data, 0.8)
        original_data = ema_original.data.copy()
        original_weight = ema_original.weight

        ema_updated = ema_original.update(jnp.array([3.0, 4.0]))
        npt.assert_allclose(ema_original.data, original_data)
        assert ema_original.weight == original_weight
        npt.assert_allclose(ema_updated.data, jnp.array([3.0, 4.0]), rtol=1e-15)
        npt.assert_allclose(ema_updated.weight, 1.0, rtol=1e-15)
