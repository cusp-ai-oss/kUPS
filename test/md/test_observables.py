# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MD observable utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.data import Index
from kups.core.typing import SystemId
from kups.md.observables import remove_center_of_mass_momentum


class TestRemoveCenterOfMassMomentum:
    def test_removes_total_momentum_per_system(self):
        momenta = jnp.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [1.0, 1.0, 0.0],
                [3.0, 5.0, 2.0],
            ]
        )
        masses = jnp.array([1.0, 3.0, 2.0, 2.0])
        system = Index.integer(jnp.array([0, 0, 1, 1]), n=2, label=SystemId)

        result = remove_center_of_mass_momentum(momenta, masses, system)

        expected = jnp.array(
            [
                [1.5, -0.75, 0.0],
                [-1.5, 0.75, 0.0],
                [-1.0, -2.0, -1.0],
                [1.0, 2.0, 1.0],
            ]
        )
        npt.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(system.sum_over(result).data, 0.0, atol=1e-12)

    def test_is_jittable(self):
        momenta = jnp.array([[3.0, -1.0, 2.0], [1.0, 5.0, -2.0]])
        masses = jnp.array([2.0, 1.0])
        system = Index.integer(jnp.array([0, 0]), n=1, label=SystemId)

        result = jax.jit(remove_center_of_mass_momentum)(momenta, masses, system)

        npt.assert_allclose(system.sum_over(result).data, 0.0, atol=1e-12)
