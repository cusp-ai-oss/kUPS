# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the harmonic potential implementation."""

import jax
import jax.numpy as jnp
import pytest

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist import Edges
from kups.core.patch import WithPatch
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.potential.classical.harmonic import (
    HarmonicAngleParameters,
    HarmonicBondParameters,
    harmonic_angle_energy,
    harmonic_bond_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph


@dataclass
class _PointData:
    positions: jax.Array
    labels: Index[str]
    system: Index[SystemId]


@dataclass
class _SystemData:
    unitcell: UnitCell


def _make_particles(
    positions: jax.Array, species: list[str], system_ids: list[int]
) -> Table[ParticleId, _PointData]:
    labels = Index.new(species)
    system = Index.new(system_ids)
    return Table.arange(_PointData(positions, labels, system), label=ParticleId)


def _make_systems(lattice_vectors: jax.Array) -> Table[SystemId, _SystemData]:
    unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
    return Table.arange(_SystemData(unitcell), label=SystemId)


def _make_graph(particles, systems, edge_indices, edge_shifts):
    edges = Edges(
        indices=Index(particles.keys, edge_indices),
        shifts=edge_shifts,
    )
    return HyperGraph(particles, systems, edges)


_LABELS = ("A", "B")

_jit_harmonic_bond_energy = jax.jit(harmonic_bond_energy)
_jit_harmonic_angle_energy = jax.jit(harmonic_angle_energy)


class TestHarmonicBondEnergy:
    """Test the harmonic_bond_energy function."""

    @classmethod
    def setup_class(cls):
        cls.positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cls.species = ["A", "B"]
        cls.system_ids = [0, 0]
        cls.unitcells_lv = jnp.eye(3)[None] * 10.0

        cls.particles = _make_particles(cls.positions, cls.species, cls.system_ids)
        cls.systems = _make_systems(cls.unitcells_lv)

        edge_indices = jnp.array([[0, 1]])
        edge_shifts = jnp.array([[[0.0, 0.0, 0.0]]])
        cls.edges = Edges(
            indices=Index(cls.particles.keys, edge_indices), shifts=edge_shifts
        )
        cls.graph = HyperGraph(cls.particles, cls.systems, cls.edges)

        cls.x0 = jnp.array([[1.5, 1.2], [1.2, 1.8]])
        cls.k = jnp.array([[100.0, 80.0], [80.0, 120.0]])
        cls.parameters = HarmonicBondParameters(labels=_LABELS, x0=cls.x0, k=cls.k)

    def test_bond_energy_scenarios(self):
        """Merged: basic + equilibrium + zero_force_constant (same 2-particle A-B shape)."""
        energy_fn = _jit_harmonic_bond_energy

        # Basic
        inp = GraphPotentialInput(graph=self.graph, parameters=self.parameters)
        result = energy_fn(inp)
        assert isinstance(result, WithPatch)
        assert result.data.data.shape == (1,)
        r = 1.0
        x0_ij = self.x0[0, 1]
        k_ij = self.k[0, 1]
        expected_energy = (r - x0_ij) ** 2 * k_ij
        assert jnp.isclose(result.data.data[0], expected_energy, rtol=1e-6)

        # Equilibrium distance -> zero energy (same shape: 2 particles, A-B)
        eq_dist = self.x0[0, 1]
        positions_eq = jnp.array([[0.0, 0.0, 0.0], [eq_dist, 0.0, 0.0]])
        particles_eq = _make_particles(positions_eq, self.species, self.system_ids)
        graph_eq = _make_graph(
            particles_eq,
            self.systems,
            jnp.array([[0, 1]]),
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_eq = energy_fn(
            GraphPotentialInput(graph=graph_eq, parameters=self.parameters)
        )
        assert jnp.isclose(result_eq.data.data[0], 0.0, atol=1e-10)

        # Zero force constant (same shape graph)
        k_zero = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        params_zero = HarmonicBondParameters(labels=_LABELS, x0=self.x0, k=k_zero)
        result_zero = energy_fn(
            GraphPotentialInput(graph=self.graph, parameters=params_zero)
        )
        assert jnp.isclose(result_zero.data.data[0], 0.0, atol=1e-10)

        # Same species (A-A) -- different Index keys, use raw function
        positions_aa = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        particles_aa = _make_particles(positions_aa, ["A", "A"], [0, 0])
        graph_aa = _make_graph(
            particles_aa,
            self.systems,
            jnp.array([[0, 1]]),
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_aa = harmonic_bond_energy(
            GraphPotentialInput(graph=graph_aa, parameters=self.parameters)
        )
        expected_aa = (1.5 - self.x0[0, 0]) ** 2 * self.k[0, 0]
        assert jnp.isclose(result_aa.data.data[0], expected_aa, rtol=1e-6)

        # Multiple bonds in same graph (different shape: 3 particles, 2 edges)
        positions_m = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        particles_m = _make_particles(positions_m, ["A", "B", "A"], [0, 0, 0])
        graph_m = _make_graph(
            particles_m,
            self.systems,
            jnp.array([[0, 1], [1, 2]]),
            jnp.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]),
        )
        result_m = harmonic_bond_energy(
            GraphPotentialInput(graph=graph_m, parameters=self.parameters)
        )
        energy1 = (1.0 - self.x0[0, 1]) ** 2 * self.k[0, 1]
        energy2 = (1.5 - self.x0[1, 0]) ** 2 * self.k[1, 0]
        assert jnp.isclose(result_m.data.data[0], energy1 + energy2, rtol=1e-6)

    def test_energy_calculation_multiple_graphs(self):
        """Test with multiple graphs (batched)."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        particles = _make_particles(positions, ["A", "B", "A", "B"], [0, 0, 1, 1])
        systems = _make_systems(jnp.eye(3)[None].repeat(2, axis=0) * 10.0)
        graph = _make_graph(
            particles,
            systems,
            jnp.array([[0, 1], [2, 3]]),
            jnp.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]),
        )
        result = harmonic_bond_energy(
            GraphPotentialInput(graph=graph, parameters=self.parameters)
        )
        assert result.data.data.shape == (2,)
        x0_ij = self.x0[0, 1]
        k_ij = self.k[0, 1]
        assert jnp.isclose(result.data.data[0], (1.0 - x0_ij) ** 2 * k_ij, rtol=1e-6)
        assert jnp.isclose(result.data.data[1], (2.0 - x0_ij) ** 2 * k_ij, rtol=1e-6)

    def test_gradient_compatibility(self):
        """Merged: position + parameter gradients."""

        # Position gradients
        def energy_fn_pos(positions):
            particles = _make_particles(positions, self.species, self.system_ids)
            graph = _make_graph(
                particles,
                self.systems,
                jnp.array([[0, 1]]),
                jnp.array([[[0.0, 0.0, 0.0]]]),
            )
            return harmonic_bond_energy(
                GraphPotentialInput(graph=graph, parameters=self.parameters)
            ).data.data.sum()

        grads_pos = jax.grad(energy_fn_pos)(self.positions)
        assert grads_pos.shape == self.positions.shape
        assert jnp.all(jnp.isfinite(grads_pos))

        # Parameter gradients
        def energy_fn_k(k):
            params = HarmonicBondParameters(labels=_LABELS, x0=self.x0, k=k)
            return harmonic_bond_energy(
                GraphPotentialInput(graph=self.graph, parameters=params)
            ).data.data.sum()

        grads_k = jax.grad(energy_fn_k)(self.k)
        assert grads_k.shape == self.k.shape
        assert jnp.all(jnp.isfinite(grads_k))

    def test_assertion_wrong_order(self):
        edge_indices = jnp.array([[0, 1, 0]])
        edge_shifts = jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        edges = Edges(
            indices=Index(self.particles.keys, edge_indices), shifts=edge_shifts
        )
        graph = HyperGraph(self.particles, self.systems, edges)
        inp = GraphPotentialInput(graph=graph, parameters=self.parameters)
        with pytest.raises(AssertionError, match="pairwise interactions"):
            harmonic_bond_energy(inp)


class TestHarmonicAngleEnergy:
    """Test the harmonic_angle_energy function."""

    @classmethod
    def setup_class(cls):
        cls.positions = jnp.array(
            [
                [1.0, 0.0, 0.0],  # Atom 0 (center)
                [0.0, 0.0, 0.0],  # Atom 1
                [1.0, 1.0, 0.0],  # Atom 2
            ]
        )
        cls.species = ["A", "B", "A"]
        cls.system_ids = [0, 0, 0]
        cls.unitcells_lv = jnp.eye(3)[None] * 10.0

        cls.particles = _make_particles(cls.positions, cls.species, cls.system_ids)
        cls.systems = _make_systems(cls.unitcells_lv)

        edge_indices = jnp.array([[0, 1, 2]])
        edge_shifts = jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        cls.edges = Edges(
            indices=Index(cls.particles.keys, edge_indices), shifts=edge_shifts
        )
        cls.graph = HyperGraph(cls.particles, cls.systems, cls.edges)

        cls.theta0 = jnp.array(
            [[[120.0, 90.0], [90.0, 120.0]], [[90.0, 90.0], [120.0, 90.0]]]
        )
        cls.k = jnp.array([[[10.0, 8.0], [12.0, 10.0]], [[8.0, 12.0], [10.0, 8.0]]])
        cls.parameters = HarmonicAngleParameters(
            labels=_LABELS, theta0=cls.theta0, k=cls.k
        )

    def test_angle_energy_scenarios(self):
        """Merged: basic + equilibrium + zero_force_constant."""
        energy_fn = _jit_harmonic_angle_energy

        # Basic energy
        inp = GraphPotentialInput(graph=self.graph, parameters=self.parameters)
        result = energy_fn(inp)
        assert isinstance(result, WithPatch)
        assert result.data.data.shape == (1,)
        theta0_ijk = self.theta0[0, 1, 0]
        k_ijk = self.k[0, 1, 0]
        expected_energy = (90.0 - theta0_ijk) ** 2 * k_ijk
        assert jnp.isclose(result.data.data[0], expected_energy, rtol=1e-5)

        # At equilibrium -> zero energy
        assert jnp.isclose(result.data.data[0], 0.0, atol=1e-6)

        # Zero force constant
        k_zero = jnp.zeros_like(self.k)
        params_zero = HarmonicAngleParameters(
            labels=_LABELS, theta0=self.theta0, k=k_zero
        )
        result_zero = energy_fn(
            GraphPotentialInput(graph=self.graph, parameters=params_zero)
        )
        assert jnp.isclose(result_zero.data.data[0], 0.0, atol=1e-10)

    def test_linear_configuration(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        particles = _make_particles(positions, self.species, self.system_ids)
        graph = _make_graph(
            particles,
            self.systems,
            jnp.array([[0, 1, 2]]),
            jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        )
        result = _jit_harmonic_angle_energy(
            GraphPotentialInput(graph=graph, parameters=self.parameters)
        )
        theta0_ijk = self.theta0[0, 1, 0]
        k_ijk = self.k[0, 1, 0]
        expected_energy = (180.0 - theta0_ijk) ** 2 * k_ijk
        assert jnp.isclose(result.data.data[0], expected_energy, rtol=1e-5)

    def test_energy_calculation_multiple_graphs(self):
        positions = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.5, 0.866, 0.0],
            ]
        )
        species = ["A", "B", "A", "A", "B", "A"]
        system_ids = [0, 0, 0, 1, 1, 1]
        particles = _make_particles(positions, species, system_ids)
        systems = _make_systems(jnp.eye(3)[None].repeat(2, axis=0) * 10.0)
        graph = _make_graph(
            particles,
            systems,
            jnp.array([[0, 1, 2], [3, 4, 5]]),
            jnp.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ),
        )
        result = harmonic_angle_energy(
            GraphPotentialInput(graph=graph, parameters=self.parameters)
        )
        assert result.data.data.shape == (2,)
        theta0_ijk = self.theta0[0, 1, 0]
        k_ijk = self.k[0, 1, 0]
        assert jnp.isclose(
            result.data.data[0], (90.0 - theta0_ijk) ** 2 * k_ijk, rtol=1e-5
        )
        assert jnp.isclose(
            result.data.data[1], (120.0 - theta0_ijk) ** 2 * k_ijk, rtol=1e-2
        )

    def test_gradient_compatibility(self):
        """Merged: position + parameter gradients."""

        def energy_fn_pos(positions):
            particles = _make_particles(positions, self.species, self.system_ids)
            graph = _make_graph(
                particles,
                self.systems,
                jnp.array([[0, 1, 2]]),
                jnp.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            )
            return harmonic_angle_energy(
                GraphPotentialInput(graph=graph, parameters=self.parameters)
            ).data.data.sum()

        grads_pos = jax.grad(energy_fn_pos)(self.positions)
        assert grads_pos.shape == self.positions.shape
        assert jnp.all(jnp.isfinite(grads_pos))

        def energy_fn_k(k):
            params = HarmonicAngleParameters(labels=_LABELS, theta0=self.theta0, k=k)
            return harmonic_angle_energy(
                GraphPotentialInput(graph=self.graph, parameters=params)
            ).data.data.sum()

        grads_k = jax.grad(energy_fn_k)(self.k)
        assert grads_k.shape == self.k.shape
        assert jnp.all(jnp.isfinite(grads_k))

    def test_assertion_wrong_order(self):
        edge_indices = jnp.array([[0, 1]])
        edge_shifts = jnp.array([[[0.0, 0.0, 0.0]]])
        edges = Edges(
            indices=Index(self.particles.keys, edge_indices), shifts=edge_shifts
        )
        graph = HyperGraph(self.particles, self.systems, edges)
        inp = GraphPotentialInput(graph=graph, parameters=self.parameters)
        with pytest.raises(AssertionError, match="triplet interactions"):
            harmonic_angle_energy(inp)
