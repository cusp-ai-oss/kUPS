# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.core.constants import BAR, PASCAL
from kups.mcmc.fugacity import peng_robinson_log_fugacity


class TestPengRobinsonFugacity:
    def setup_method(self):
        # CO2-like reference parameters used across tests
        self.Pc = 7.84e6 * PASCAL
        self.Tc = 303.75
        self.omega = 0.22394

    @staticmethod
    def _pr_coefficients(
        pressure, critical_pressure, temperature, critical_temperature, acentric_factor
    ):
        Pr = pressure / critical_pressure
        Tr = temperature / critical_temperature
        m = 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor**2
        sqrt_alpha = 1 + m * (1 - np.sqrt(Tr))
        A = 0.45724 * Pr / (Tr**2) * (sqrt_alpha**2)
        B = 0.0778 * Pr / Tr
        return A, B

    @staticmethod
    def _pr_cubic_roots(A, B):
        """Solve PR cubic for Z; return real roots."""
        # Z^3 - (1 - B) Z^2 + (A - 3 B^2 - 2 B) Z - (A B - B**2 - B**3) = 0
        coeffs = [
            1.0,
            -(1.0 - B),
            (A - 3.0 * B**2 - 2.0 * B),
            -(A * B - B**2 - B**3),
        ]
        roots = np.roots(coeffs)
        real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-12)].real
        return real_roots

    @staticmethod
    def _pr_log_phi(Z, A, B):
        sq2 = np.sqrt(2.0)
        return (
            Z
            - 1.0
            - np.log(Z - B)
            - A
            / (2.0 * sq2 * B)
            * np.log((Z + (1.0 + sq2) * B) / (Z + (1.0 - sq2) * B))
        )

    def test_low_pressure_ideal_limit(self):
        """Z ~ 1 and ln f ~ ln p at very low pressure."""
        p = 1e-2 * BAR  # 0.01 bar

        res = peng_robinson_log_fugacity(
            pressure=jnp.array(p),
            temperature=jnp.array(300.0),
            critical_pressure=jnp.array([self.Pc]),
            critical_temperature=jnp.array([self.Tc]),
            acentric_factor=jnp.array([self.omega]),
        )

        npt.assert_allclose(res.compressibility, jnp.array(1.0), rtol=0, atol=1e-4)
        npt.assert_allclose(res.log_fugacity, jnp.log(jnp.array(p)), rtol=0, atol=1e-4)

    def test_against_numpy_roots_at_moderate_pressure(self):
        """Cross-check Z and ln f against independent numpy.roots solution."""
        T = 300.0
        p = 50.0 * BAR

        out = peng_robinson_log_fugacity(
            pressure=jnp.array(p),
            temperature=jnp.array(T),
            critical_pressure=jnp.array([self.Pc]),
            critical_temperature=jnp.array([self.Tc]),
            acentric_factor=jnp.array([self.omega]),
        )

        A, B = self._pr_coefficients(
            float(p), float(self.Pc), float(T), float(self.Tc), float(self.omega)
        )
        roots = self._pr_cubic_roots(A, B)
        assert roots.size >= 1
        Z = roots.max()
        ln_phi = self._pr_log_phi(Z, A, B)
        ln_f_np = ln_phi + np.log(p)

        npt.assert_allclose(out.compressibility, jnp.array(Z), rtol=1e-10, atol=1e-10)
        npt.assert_allclose(
            out.log_fugacity, jnp.array(ln_f_np), rtol=1e-10, atol=1e-10
        )


class TestPengRobinsonFugacityMixtures:
    def setup_method(self):
        # Use CO2-like parameters for both components to simplify checks
        self.Pc = 7.84e6 * PASCAL
        self.Tc = 303.75
        self.omega = 0.22394

    def test_binary_low_pressure_ideal_limit(self):
        """At very low pressure, ln f_i ≈ ln(y_i p) and Z ≈ 1 for all components."""
        p = 1e-2 * BAR
        T = 300.0

        # Binary mixture with identical species properties and non-trivial composition
        Pc = jnp.array([self.Pc, self.Pc])
        Tc = jnp.array([self.Tc, self.Tc])
        omega = jnp.array([self.omega, self.omega])
        y = jnp.array([0.25, 0.75])
        kij = jnp.zeros((2, 2))

        res = peng_robinson_log_fugacity(
            pressure=jnp.array(p),
            temperature=jnp.array(T),
            critical_pressure=Pc,
            critical_temperature=Tc,
            acentric_factor=omega,
            composition=y,
            interaction=kij,
        )

        # Z should be ~1 for both components in this limit
        npt.assert_allclose(res.compressibility, jnp.ones(2), rtol=0, atol=1e-4)
        # ln f_i should approach ln(y_i p)
        expected = jnp.log(y * p)
        npt.assert_allclose(res.log_fugacity, expected, rtol=0, atol=2e-4)

    def test_binary_reduces_to_pure_when_first_component_dominates(self):
        """When y -> [1, 0], component-1 fugacity reduces to pure-component value."""
        p = 50.0 * BAR
        T = 300.0

        Pc = jnp.array([self.Pc, self.Pc])
        Tc = jnp.array([self.Tc, self.Tc])
        omega = jnp.array([self.omega, self.omega])
        kij = jnp.zeros((2, 2))

        # Mixture nearly pure in component 1
        y_mix = jnp.array([1.0, 0.0])

        mix = peng_robinson_log_fugacity(
            pressure=jnp.array(p),
            temperature=jnp.array(T),
            critical_pressure=Pc,
            critical_temperature=Tc,
            acentric_factor=omega,
            composition=y_mix,
            interaction=kij,
        )

        # Pure component calculation with a single species
        pure = peng_robinson_log_fugacity(
            pressure=jnp.array(p),
            temperature=jnp.array(T),
            critical_pressure=jnp.array([self.Pc]),
            critical_temperature=jnp.array([self.Tc]),
            acentric_factor=jnp.array([self.omega]),
        )

        # Component 1 should match the pure-component results
        npt.assert_allclose(
            mix.log_fugacity[0], pure.log_fugacity[0], rtol=1e-10, atol=1e-10
        )
        npt.assert_allclose(
            mix.compressibility[0], pure.compressibility[0], rtol=1e-10, atol=1e-10
        )
