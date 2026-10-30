# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Morse bond potential implementation."""

import jax
import jax.numpy as jnp
import pytest

from kups.core import units
from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.core.units import catalog as u
from kups.potential.classical.harmonic import (
    HarmonicBondParameters,
    harmonic_bond_energy,
)
from kups.potential.classical.morse import (
    MorseBondParameters,
    morse_bond_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph
from test.units import EV, A, assert_close

from .conftest import make_particles, make_systems

_LABELS = ("A", "B")

_jit_energy = jax.jit(morse_bond_energy)


def _positions_for_distance(r: units.Length) -> units.Length:
    """Generate positions for two atoms separated by distance r along x-axis."""
    return jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * r


class TestMorseBondEnergy:
    """Test the morse_bond_energy function."""

    @classmethod
    def setup_class(cls):
        cls.species = ["A", "B"]
        cls.system_ids = [0, 0]
        cls.cells_lv = jnp.eye(3)[None] * 10.0
        cls.r0 = jnp.array([[1.5, 1.2], [1.2, 1.8]]) * A
        cls.D = jnp.array([[2.0, 1.5], [1.5, 2.5]]) * EV
        cls.alpha = jnp.array([[2.0, 1.8], [1.8, 2.2]]) / A
        cls.params = MorseBondParameters(
            labels=_LABELS, r0=cls.r0, D=cls.D, alpha=cls.alpha
        )

    def _make_graph(self, positions):
        particles = make_particles(positions, self.species, self.system_ids)
        systems = make_systems(self.cells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1]])),
            shifts=jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        return HyperGraph(particles, systems, edges)

    def _energy(self, positions, params):
        graph = self._make_graph(positions)
        return _jit_energy(
            GraphPotentialInput(graph=graph, parameters=params)
        ).data.data[0]

    def _gradient(self, positions, params):
        magnitudes = positions.to(A)

        def energy(p: jax.Array) -> jax.Array:
            """The energy of Angstrom magnitudes, in eV."""
            return self._energy(p * A, params).to(EV)

        return jax.jit(jax.grad(energy))(magnitudes)

    def test_energy_values(self):
        """Merged: equilibrium, nonequilibrium, dissociation, asymmetry, formula."""
        r0 = self.r0[0, 1]

        # Equilibrium: energy=0
        assert_close(
            self._energy(_positions_for_distance(r0), self.params), 0.0, atol=1e-10
        )

        # Nonequilibrium: energy > 0
        assert (
            self._energy(_positions_for_distance(r0 + 0.3 * A), self.params).to(EV) > 0
        )

        # Dissociation limit
        D_01 = self.D[0, 1]
        assert_close(
            self._energy(_positions_for_distance(100.0 * A), self.params),
            D_01,
            rtol=1e-6,
        )

        # Compression vs extension asymmetry
        disp = 0.3 * A
        E_compressed = self._energy(_positions_for_distance(r0 - disp), self.params)
        E_extended = self._energy(_positions_for_distance(r0 + disp), self.params)
        assert E_compressed.to(EV) > E_extended.to(EV)

        # Energy formula
        r = 1.0 * A
        D, alpha = self.D[0, 1], self.alpha[0, 1]
        expected = D * (1 - jnp.exp(-alpha * (r - r0))) ** 2
        assert_close(
            self._energy(_positions_for_distance(r), self.params), expected, rtol=1e-6
        )

    def test_gradient_values(self):
        """Merged: equilibrium gradient=0, nonequilibrium gradient nonzero."""
        r0 = self.r0[0, 1]
        assert jnp.allclose(
            self._gradient(_positions_for_distance(r0), self.params), 0.0, atol=1e-6
        )
        assert (
            jnp.linalg.norm(
                self._gradient(_positions_for_distance(r0 + 0.3 * A), self.params)
            )
            > 1e-6
        )

    def test_compare_with_harmonic_near_equilibrium(self):
        r0 = self.r0[0, 1]
        positions = _positions_for_distance(r0 + 0.01 * A)
        graph = self._make_graph(positions)

        morse_E = self._energy(positions, self.params)
        harmonic_params = HarmonicBondParameters(
            labels=_LABELS, x0=self.r0, k=self.D * self.alpha**2
        )
        harmonic_E = harmonic_bond_energy(
            GraphPotentialInput(graph=graph, parameters=harmonic_params)
        ).data.data[0]
        assert_close(morse_E, harmonic_E, rtol=0.05)

    def test_from_harmonic(self):
        r0 = jnp.array([[1.0, 1.2], [1.2, 1.5]]) * A
        k = jnp.array([[100.0, 80.0], [80.0, 120.0]]) * EV / A**2
        D = jnp.array([[2.0, 1.5], [1.5, 2.5]]) * EV
        params = MorseBondParameters.from_harmonic(_LABELS, r0, k, D)
        assert_close(params.alpha, units.sqrt(k / D))
        assert_close(params.r0, r0)
        assert_close(params.D, D)

    def test_from_uff(self):
        # UFF H_ and O_3: Eq. 6 gives k_OH = 1120 kcal/mol/Å^2 (as RDKit), so
        # alpha = sqrt(k / 2D) for D = 100 kcal/mol.
        params = MorseBondParameters.from_uff(
            ("H_", "O_3"),
            jnp.array([0.354, 0.658]) * A,
            jnp.array([4.528, 8.741]),
            jnp.array([0.712, 2.300]) * u.e,
            jnp.full((2, 2), 100.0) * (u.kcal / u.mol),
        )
        assert_close(params.alpha[0, 1], 2.3664 / A, rtol=1e-3)

    def test_assertion_wrong_order(self):
        positions = _positions_for_distance(1.0 * A)
        particles = make_particles(positions, self.species, self.system_ids)
        systems = make_systems(self.cells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 0]])),
            shifts=jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        )
        graph = HyperGraph(particles, systems, edges)
        with pytest.raises(AssertionError, match="pairwise interactions"):
            morse_bond_energy(GraphPotentialInput(graph=graph, parameters=self.params))
