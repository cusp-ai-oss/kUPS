# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the UFF-style inversion potential implementation."""

import jax
import jax.numpy as jnp
import pytest

from kups.core.data.index import Index
from kups.core.neighborlist import Edges
from kups.potential.classical.inversion import InversionParameters, inversion_energy
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from .conftest import make_particles, make_systems

_LABELS = ("A", "B")

_jit_energy = jax.jit(inversion_energy)


def _make_positions(h: float) -> jax.Array:
    """Generate positions with center at height h above the neighbor plane."""
    return jnp.array(
        [
            [0.0, 0.0, h],
            [1.0, 0.0, 0.0],
            [-0.5, 0.866025, 0.0],
            [-0.5, -0.866025, 0.0],
        ]
    )


def _compute_omega(h: float) -> jax.Array:
    """Compute the out-of-plane angle omega for positions with given h."""
    r_ji = jnp.array([1.0, 0.0, -h])
    r_jk = jnp.array([-0.5, 0.866025, -h])
    r_jl = jnp.array([-0.5, -0.866025, -h])
    n = jnp.cross(r_jk, r_jl)
    n_unit = n / jnp.linalg.norm(n)
    sin_omega = jnp.dot(r_ji, n_unit) / jnp.linalg.norm(r_ji)
    return jnp.arcsin(jnp.clip(sin_omega, -1.0, 1.0))


class TestInversionEnergy:
    """Test the inversion_energy function."""

    @classmethod
    def setup_class(cls):
        cls.unitcells_lv = jnp.eye(3)[None] * 10.0
        cls.species = ["A", "A", "A", "A"]
        cls.system_ids = [0, 0, 0, 0]
        shape = (2, 2, 2, 2)
        cls.omega0 = jnp.zeros(shape)
        cls.k = jnp.ones(shape) * 10.0
        cls.params = InversionParameters(labels=_LABELS, omega0=cls.omega0, k=cls.k)

    def _make_graph(self, positions):
        particles = make_particles(positions, self.species, self.system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2, 3]])),
            shifts=jnp.zeros((1, 3, 3)),
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
        """Merged: equilibrium, nonequilibrium, and formula tests."""
        # Equilibrium sp2: energy=0
        positions = _make_positions(0.0)
        assert jnp.isclose(self._energy(positions, self.params), 0.0, atol=1e-6)

        # Nonequilibrium: energy > 0
        for h in [0.2, 0.5, 1.0]:
            assert self._energy(_make_positions(h), self.params) > 0

        # SP2 energy formula
        h = 0.2
        omega = _compute_omega(h)
        expected = 10.0 * (1.0 - jnp.cos(omega))
        assert jnp.isclose(
            self._energy(_make_positions(h), self.params), expected, rtol=1e-5
        )

    def test_gradient_values(self):
        """Merged: equilibrium gradient=0, nonequilibrium gradient nonzero."""
        assert jnp.allclose(
            self._gradient(_make_positions(0.0), self.params), 0.0, atol=1e-6
        )
        for h in [0.2, 0.5, 1.0]:
            assert (
                jnp.linalg.norm(self._gradient(_make_positions(h), self.params)) > 1e-6
            )

    def test_nonsp2_barrier_at_planar(self):
        omega0_rad = jnp.radians(30.0)
        shape = (2, 2, 2, 2)
        barrier = 22.0
        params = InversionParameters(
            labels=_LABELS,
            omega0=jnp.ones(shape) * omega0_rad,
            k=jnp.ones(shape) * barrier,
        )
        positions = _make_positions(0.0)
        energy = self._energy(positions, params)
        assert jnp.isclose(energy, barrier, rtol=1e-5)

    def test_nonsp2_scaled_energy_formula(self):
        omega0_rad = jnp.radians(30.0)
        shape = (2, 2, 2, 2)
        barrier = 10.0
        params = InversionParameters(
            labels=_LABELS,
            omega0=jnp.ones(shape) * omega0_rad,
            k=jnp.ones(shape) * barrier,
        )

        h = 0.3
        positions = _make_positions(h)
        omega = _compute_omega(h)

        c2 = 1.0 / (4.0 * jnp.sin(omega0_rad) ** 2)
        c1 = -4.0 * c2 * jnp.cos(omega0_rad)
        c0 = c2 * (2.0 * jnp.cos(omega0_rad) ** 2 + 1.0)
        barrier_factor = c0 + c1 + c2
        k_scaled = barrier / barrier_factor
        expected = k_scaled * (c0 + c1 * jnp.cos(omega) + c2 * jnp.cos(2 * omega))

        assert jnp.isclose(self._energy(positions, params), expected, rtol=1e-4)

    def test_zero_force_constant(self):
        params = InversionParameters(
            labels=_LABELS, omega0=self.omega0, k=jnp.zeros_like(self.k)
        )
        positions = _make_positions(0.5)
        assert jnp.isclose(self._energy(positions, params), 0.0, atol=1e-10)

    def test_assertion_wrong_order(self):
        positions = _make_positions(0.0)[:3]
        species = ["A", "A", "A"]
        system_ids = [0, 0, 0]
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2]])),
            shifts=jnp.zeros((1, 2, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        with pytest.raises(AssertionError, match="4-body interactions"):
            inversion_energy(GraphPotentialInput(graph=graph, parameters=self.params))

    def test_different_species(self):
        species = ["A", "B", "B", "A"]
        system_ids = [0, 0, 0, 0]
        positions = _make_positions(0.0)
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2, 3]])),
            shifts=jnp.zeros((1, 3, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        result = inversion_energy(
            GraphPotentialInput(graph=graph, parameters=self.params)
        )
        assert jnp.isclose(result.data.data[0], 0.0, atol=1e-6)

    def test_multiple_inversions(self):
        positions = _make_positions(0.2)
        particles = make_particles(positions, self.species, self.system_ids)
        systems = make_systems(self.unitcells_lv)
        edges = Edges(
            indices=Index(
                particles.keys,
                jnp.array([[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]),
            ),
            shifts=jnp.zeros((3, 3, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        result = inversion_energy(
            GraphPotentialInput(graph=graph, parameters=self.params)
        )
        assert result.data.data[0] > 0.0
        assert jnp.isfinite(result.data.data[0])

    def test_multiple_graphs(self):
        positions = jnp.concatenate([_make_positions(0.0), _make_positions(0.5)])
        species = ["A"] * 8
        system_ids = [0, 0, 0, 0, 1, 1, 1, 1]
        particles = make_particles(positions, species, system_ids)
        systems = make_systems(jnp.eye(3)[None].repeat(2, axis=0) * 10.0)
        edges = Edges(
            indices=Index(particles.keys, jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])),
            shifts=jnp.zeros((2, 3, 3)),
        )
        graph = HyperGraph(particles, systems, edges)
        result = inversion_energy(
            GraphPotentialInput(graph=graph, parameters=self.params)
        )
        assert result.data.data.shape == (2,)
        assert jnp.isclose(result.data.data[0], 0.0, atol=1e-6)
        assert result.data.data[1] > 0.0
