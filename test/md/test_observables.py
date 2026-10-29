# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MD observable utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kups.core.data import Index
from kups.core.typing import SystemId
from kups.core.units import catalog as u
from kups.core.units import testing
from kups.md.observables import remove_center_of_mass_momentum, system_kinetic_energy
from test.units import AMU_A_PER_FS, EV

ENERGY = AMU_A_PER_FS**2 / u.amu
"""The kinetic-energy unit of momenta in ``AMU_A_PER_FS`` and masses in amu."""


class TestRemoveCenterOfMassMomentum:
    def test_removes_total_momentum_per_system(self):
        momenta = (
            jnp.array(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [3.0, 5.0, 2.0],
                ]
            )
            * AMU_A_PER_FS
        )
        masses = jnp.array([1.0, 3.0, 2.0, 2.0]) * u.amu
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
        testing.assert_allclose(result, expected, AMU_A_PER_FS, rtol=1e-12, atol=1e-12)
        testing.assert_allclose(
            system.sum_over(result).data, 0.0, AMU_A_PER_FS, atol=1e-12
        )

    def test_is_jittable(self):
        momenta = jnp.array([[3.0, -1.0, 2.0], [1.0, 5.0, -2.0]]) * AMU_A_PER_FS
        masses = jnp.array([2.0, 1.0]) * u.amu
        system = Index.integer(jnp.array([0, 0]), n=1, label=SystemId)

        result = jax.jit(remove_center_of_mass_momentum)(momenta, masses, system)

        testing.assert_allclose(
            system.sum_over(result).data, 0.0, AMU_A_PER_FS, atol=1e-12
        )


class TestSystemKineticEnergy:
    def test_internal_excludes_center_of_mass_motion(self):
        """Pure center-of-mass drift contributes to total KE but not internal KE."""
        masses = jnp.array([1.0, 1.0]) * u.amu
        system = Index.integer(jnp.array([0, 0]), n=1, label=SystemId)
        drift = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * AMU_A_PER_FS

        total = system_kinetic_energy(drift, masses, system)
        internal = system_kinetic_energy(drift, masses, system, remove_com=True)

        testing.assert_allclose(total, [1.0], ENERGY, rtol=1e-12)
        testing.assert_allclose(internal, jnp.array([0.0]), ENERGY, atol=1e-12)

    def test_internal_equals_total_with_zero_total_momentum(self):
        """With no net momentum, internal KE equals total KE, summed per system."""
        masses = jnp.array([1.0, 1.0, 2.0, 2.0]) * u.amu
        system = Index.integer(jnp.array([0, 0, 1, 1]), n=2, label=SystemId)
        momenta = (
            jnp.array(
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
            )
            * AMU_A_PER_FS
        )

        total = system_kinetic_energy(momenta, masses, system)
        internal = system_kinetic_energy(momenta, masses, system, remove_com=True)

        testing.assert_allclose(internal, total, EV, rtol=1e-12)
        testing.assert_allclose(total, [1.0, 2.0], ENERGY, rtol=1e-12)
