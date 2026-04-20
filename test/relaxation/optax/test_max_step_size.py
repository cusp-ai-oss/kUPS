# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for max_step_size transform."""

import jax.numpy as jnp
import numpy.testing as npt

from kups.relaxation.optax.max_step_size import max_step_size

from ...clear_cache import clear_cache  # noqa: F401


class TestMaxStepSize:
    def test_no_scaling_when_below_limit(self):
        """Updates smaller than max_step_size should not be scaled."""
        transform = max_step_size(1.0)
        state = transform.init(None)
        updates = jnp.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
        new_updates, _ = transform.update(updates, state)
        npt.assert_allclose(new_updates, updates)

    def test_scaling_when_above_limit(self):
        """Updates larger than max_step_size should be scaled down."""
        transform = max_step_size(1.0)
        state = transform.init(None)
        updates = jnp.array([[2.0, 0.0, 0.0]])  # norm = 2.0
        new_updates, _ = transform.update(updates, state)
        # Should scale by 1.0/2.0 = 0.5
        expected = jnp.array([[1.0, 0.0, 0.0]])
        npt.assert_allclose(new_updates, expected, atol=1e-6)

    def test_scaling_with_multiple_particles(self):
        """Max step size should be determined by the largest particle update."""
        transform = max_step_size(1.0)
        state = transform.init(None)
        updates = jnp.array(
            [
                [0.5, 0.0, 0.0],  # norm = 0.5
                [4.0, 0.0, 0.0],  # norm = 4.0 (largest)
            ]
        )
        new_updates, _ = transform.update(updates, state)
        # Scale factor = 1.0/4.0 = 0.25
        expected = jnp.array(
            [
                [0.125, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        npt.assert_allclose(new_updates, expected, atol=1e-6)

    def test_pytree_input(self):
        """Should work with pytree inputs."""
        transform = max_step_size(1.0)
        state = transform.init(None)
        updates = {
            "a": jnp.array([[0.5, 0.0, 0.0]]),  # norm = 0.5
            "b": jnp.array([[2.0, 0.0, 0.0]]),  # norm = 2.0 (largest)
        }
        new_updates, _ = transform.update(updates, state)
        # Scale factor = 1.0/2.0 = 0.5
        npt.assert_allclose(new_updates["a"], jnp.array([[0.25, 0.0, 0.0]]), atol=1e-6)
        npt.assert_allclose(new_updates["b"], jnp.array([[1.0, 0.0, 0.0]]), atol=1e-6)

    def test_preserves_direction(self):
        """Scaling should preserve the direction of updates."""
        transform = max_step_size(1.0)
        state = transform.init(None)
        updates = jnp.array([[3.0, 4.0, 0.0]])  # norm = 5.0
        new_updates, _ = transform.update(updates, state)
        # Scale factor = 1.0/5.0 = 0.2
        expected = jnp.array([[0.6, 0.8, 0.0]])
        npt.assert_allclose(new_updates, expected, atol=1e-6)

    def test_custom_max_step_size(self):
        """Should respect the configured max_step_size value."""
        transform = max_step_size(0.5)
        state = transform.init(None)
        updates = jnp.array([[2.0, 0.0, 0.0]])  # norm = 2.0
        new_updates, _ = transform.update(updates, state)
        # Scale factor = 0.5/2.0 = 0.25
        expected = jnp.array([[0.5, 0.0, 0.0]])
        npt.assert_allclose(new_updates, expected, atol=1e-6)

    def test_exact_boundary(self):
        """Updates exactly at max_step_size should not be scaled."""
        transform = max_step_size(1.0)
        state = transform.init(None)
        updates = jnp.array([[1.0, 0.0, 0.0]])  # norm = 1.0
        new_updates, _ = transform.update(updates, state)
        npt.assert_allclose(new_updates, updates, atol=1e-6)
