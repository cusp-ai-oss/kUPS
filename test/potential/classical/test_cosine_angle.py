# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UFF-style cosine angle potential."""

import jax
import jax.numpy as jnp
import pytest

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.cosine_angle import (
    CosineAngleParameters,
    _compute_cosine_coefficients,
    cosine_angle_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from .conftest import make_particles, make_systems

_LABELS = ("A", "B")

_jit_energy = jax.jit(cosine_angle_energy)


def _positions_for_angle(theta) -> jax.Array:
    """Generate positions where angle at atom 0 equals theta."""
    return jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [jnp.cos(theta), jnp.sin(theta), 0.0]]
    )


class TestCosineAngleEnergy:
    """Test the cosine_angle_energy function."""

    @classmethod
    def setup_class(cls):
        cls.unitcells_lv = jnp.eye(3)[None] * 10.0
        cls.species = ["A", "A", "A"]
        cls.system_ids = [0, 0, 0]
        cls.k = jnp.ones((2, 2, 2))

    def _make_graph(self, positions):
        particles = make_particles(positions, self.species, self.system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2]])),
            shifts=jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        )
        return HyperGraph(particles, systems, edges)

    def _energy(self, positions, params):
        graph = self._make_graph(positions)
        return _jit_energy(
            GraphPotentialInput(graph=graph, parameters=params)
        ).data.data[0]

    def _gradient(self, positions, params):
        return jax.jit(jax.grad(lambda p: self._energy(p, params)))(positions)

    def test_energy_values(self):
        """Merged: equilibrium, nonequilibrium, and zero force constant tests."""
        # Equilibrium: energy=0 at theta0
        for theta0_deg in [90, 120, 180]:
            theta0 = jnp.radians(theta0_deg)
            params = CosineAngleParameters(
                labels=_LABELS, theta0=jnp.ones((2, 2, 2)) * theta0, k=self.k
            )
            assert jnp.isclose(
                self._energy(_positions_for_angle(theta0), params), 0.0, atol=1e-6
            )

        # Nonequilibrium: energy > 0
        for theta0_deg, test_deg in [(90, 60), (120, 90), (180, 90)]:
            theta0 = jnp.radians(theta0_deg)
            params = CosineAngleParameters(
                labels=_LABELS, theta0=jnp.ones((2, 2, 2)) * theta0, k=self.k
            )
            assert self._energy(_positions_for_angle(jnp.radians(test_deg)), params) > 0

        # Zero force constant
        params = CosineAngleParameters(
            labels=_LABELS,
            theta0=jnp.ones((2, 2, 2)) * jnp.pi / 2,
            k=jnp.zeros((2, 2, 2)),
        )
        assert jnp.isclose(
            self._energy(_positions_for_angle(0.7), params), 0.0, atol=1e-10
        )

    def test_gradient_values(self):
        """Merged: equilibrium gradient=0, nonequilibrium gradient nonzero."""
        # Equilibrium: gradient=0
        for theta0_deg in [90, 120, 180]:
            theta0 = jnp.radians(theta0_deg)
            params = CosineAngleParameters(
                labels=_LABELS, theta0=jnp.ones((2, 2, 2)) * theta0, k=self.k
            )
            assert jnp.allclose(
                self._gradient(_positions_for_angle(theta0), params), 0.0, atol=1e-6
            )

        # Nonequilibrium: gradient nonzero
        for theta0_deg, test_deg in [(90, 60), (120, 90), (180, 90)]:
            theta0 = jnp.radians(theta0_deg)
            params = CosineAngleParameters(
                labels=_LABELS, theta0=jnp.ones((2, 2, 2)) * theta0, k=self.k
            )
            grad = self._gradient(_positions_for_angle(jnp.radians(test_deg)), params)
            assert jnp.linalg.norm(grad) > 1e-6

    def test_assertion_wrong_order(self):
        particles = make_particles(
            _positions_for_angle(jnp.pi / 2), self.species, self.system_ids
        )
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1]])),
            shifts=jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        graph = HyperGraph(particles, systems, edges)
        params = CosineAngleParameters(
            labels=_LABELS, theta0=jnp.ones((2, 2, 2)) * jnp.pi / 2, k=self.k
        )
        with pytest.raises(AssertionError, match="triplet interactions"):
            cosine_angle_energy(GraphPotentialInput(graph=graph, parameters=params))

    def test_multiple_angles(self):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        species = ["A", "A", "A", "A"]
        system_ids = [0, 0, 0, 0]
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2], [3, 0, 2]])),
            shifts=jnp.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        )
        graph = HyperGraph(particles, systems, edges)
        params = CosineAngleParameters(
            labels=_LABELS, theta0=jnp.ones((2, 2, 2)) * jnp.pi / 2, k=self.k
        )
        result = cosine_angle_energy(
            GraphPotentialInput(graph=graph, parameters=params)
        )
        assert result.data.data.shape == (1,)
        assert result.data.data[0] > 0


class TestCosineCoefficients:
    """Test the coefficient computation."""

    @pytest.mark.parametrize(
        "theta0_deg,expected",
        [
            (90, (0.25, 0.0, 0.25)),
            (120, (0.5, 2 / 3, 1 / 3)),
        ],
    )
    def test_coefficients(self, theta0_deg, expected):
        c0, c1, c2 = _compute_cosine_coefficients(jnp.radians(theta0_deg))
        assert jnp.isclose(c0, expected[0], rtol=1e-5)
        assert jnp.isclose(c1, expected[1], atol=1e-10, rtol=1e-5)
        assert jnp.isclose(c2, expected[2], rtol=1e-5)
