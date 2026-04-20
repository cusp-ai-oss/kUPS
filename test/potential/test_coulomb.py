# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist import Edges
from kups.core.patch import WithPatch
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.potential.classical.coulomb import (
    coulomb_vacuum_energy,
)
from kups.potential.classical.ewald import TO_STANDARD_UNITS
from kups.potential.common.graph import GraphPotentialInput, HyperGraph


@dataclass
class PointCloudParticles:
    """Simple point cloud particles with positions and charges."""

    positions: jax.Array
    charges: jax.Array
    system: Index[SystemId]


@dataclass
class _SystemData:
    unitcell: UnitCell
    cutoff: jax.Array


def _make_particles(
    positions: jax.Array,
    charges: jax.Array,
    system_ids: jax.Array,
) -> Table[ParticleId, PointCloudParticles]:
    system = Index.new([SystemId(i) for i in system_ids.tolist()])
    return Table.arange(
        PointCloudParticles(positions, charges, system), label=ParticleId
    )


def _make_systems(
    lattice_vectors: jax.Array,
) -> Table[SystemId, _SystemData]:
    unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
    cutoff = jnp.full((lattice_vectors.shape[0],), 100.0)
    return Table.arange(_SystemData(unitcell, cutoff), label=SystemId)


def _make_graph(
    positions: jax.Array,
    charges: jax.Array,
    system_ids: jax.Array,
    lattice_vectors: jax.Array,
    edge_indices: jax.Array,
    edge_shifts: jax.Array,
) -> HyperGraph:
    particles = _make_particles(positions, charges, system_ids)
    systems = _make_systems(lattice_vectors)
    edges = Edges(
        indices=Index(particles.keys, edge_indices),
        shifts=edge_shifts,
    )
    return HyperGraph(particles, systems, edges)


_jit_coulomb_vacuum_energy = jax.jit(coulomb_vacuum_energy)


class TestCoulombVacuumEnergy:
    """Test the coulomb_vacuum_energy function."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cls.charges = jnp.array([1.0, -1.0])
        cls.lattice_vectors = jnp.eye(3)[None] * 10.0
        cls.system_ids = jnp.array([0, 0])
        cls.indices = jnp.array([[0, 1]])
        cls.shifts = jnp.array([[[0.0, 0.0, 0.0]]])
        cls.graph = _make_graph(
            cls.positions,
            cls.charges,
            cls.system_ids,
            cls.lattice_vectors,
            cls.indices,
            cls.shifts,
        )

    def test_coulomb_vacuum_energy_values(self):
        """Merged value tests: basic, like charges, distance, zero, large, magnitudes, fractional, coulomb law."""
        energy_fn = _jit_coulomb_vacuum_energy

        # Basic opposite charges at r=1
        inp = GraphPotentialInput(parameters=None, graph=self.graph)
        result = energy_fn(inp)
        expected = (1.0 * -1.0 / 1.0) * TO_STANDARD_UNITS / 2
        assert isinstance(result, WithPatch)
        assert result.data.data.shape == (1,)
        assert jnp.isclose(result.data.data[0], expected, rtol=1e-10)

        # Like charges (repulsive)
        graph_like = _make_graph(
            self.positions,
            jnp.array([1.0, 1.0]),
            self.system_ids,
            self.lattice_vectors,
            self.indices,
            self.shifts,
        )
        result_like = energy_fn(GraphPotentialInput(parameters=None, graph=graph_like))
        expected_like = (1.0 * 1.0 / 1.0) * TO_STANDARD_UNITS / 2
        assert result_like.data.data[0] > 0.0
        assert jnp.isclose(result_like.data.data[0], expected_like, rtol=1e-10)

        # Distance dependence (1/r): r=2 should give half the energy magnitude
        graph_far = _make_graph(
            jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            self.charges,
            self.system_ids,
            self.lattice_vectors,
            self.indices,
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_far = energy_fn(GraphPotentialInput(parameters=None, graph=graph_far))
        assert jnp.isclose(
            result.data.data[0] / result_far.data.data[0], 2.0, rtol=1e-10
        )

        # Zero charges
        graph_zero = _make_graph(
            self.positions,
            jnp.array([0.0, 0.0]),
            self.system_ids,
            self.lattice_vectors,
            self.indices,
            self.shifts,
        )
        result_zero = energy_fn(GraphPotentialInput(parameters=None, graph=graph_zero))
        assert jnp.isclose(result_zero.data.data[0], 0.0, atol=1e-12)

        # Large distances
        graph_large = _make_graph(
            jnp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            self.charges,
            self.system_ids,
            jnp.eye(3)[None] * 200.0,
            self.indices,
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_large = energy_fn(
            GraphPotentialInput(parameters=None, graph=graph_large)
        )
        expected_large = (1.0 * -1.0 / 100.0) * TO_STANDARD_UNITS / 2
        assert jnp.isclose(result_large.data.data[0], expected_large, rtol=1e-10)

        # Different charge magnitudes
        graph_mag = _make_graph(
            self.positions,
            jnp.array([2.0, -3.0]),
            self.system_ids,
            self.lattice_vectors,
            self.indices,
            self.shifts,
        )
        result_mag = energy_fn(GraphPotentialInput(parameters=None, graph=graph_mag))
        expected_mag = (2.0 * -3.0 / 1.0) * TO_STANDARD_UNITS / 2
        assert jnp.isclose(result_mag.data.data[0], expected_mag, rtol=1e-10)

        # Fractional charges
        graph_frac = _make_graph(
            self.positions,
            jnp.array([0.5, -0.3]),
            self.system_ids,
            self.lattice_vectors,
            self.indices,
            self.shifts,
        )
        result_frac = energy_fn(GraphPotentialInput(parameters=None, graph=graph_frac))
        expected_frac = (0.5 * -0.3 / 1.0) * TO_STANDARD_UNITS / 2
        assert jnp.isclose(result_frac.data.data[0], expected_frac, rtol=1e-10)

        # Coulomb law validation (q=1.5, -2.0, r=3.0)
        graph_cl = _make_graph(
            jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            jnp.array([1.5, -2.0]),
            self.system_ids,
            self.lattice_vectors,
            jnp.array([[0, 1]]),
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_cl = energy_fn(GraphPotentialInput(parameters=None, graph=graph_cl))
        expected_cl = (1.5 * -2.0 / 3.0) * TO_STANDARD_UNITS / 2
        assert jnp.isclose(result_cl.data.data[0], expected_cl, rtol=1e-12)

    def test_multiple_interactions(self):
        """Test system with multiple charge interactions."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        charges = jnp.array([1.0, -1.0, 1.0])
        system_ids = jnp.array([0, 0, 0])
        indices = jnp.array([[0, 1], [0, 2], [1, 2]])
        shifts = jnp.zeros((3, 1, 3))

        graph = _make_graph(
            positions,
            charges,
            system_ids,
            jnp.eye(3)[None] * 10.0,
            indices,
            shifts,
        )
        result = coulomb_vacuum_energy(
            GraphPotentialInput(parameters=None, graph=graph)
        )
        expected_energy = -1.0 / jnp.sqrt(2.0) * TO_STANDARD_UNITS / 2
        assert jnp.isclose(result.data.data[0], expected_energy, rtol=1e-10)

    def test_batch_processing(self):
        """Test energy calculation with multiple graphs in batch."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0])
        system_ids = jnp.array([0, 0, 1, 1])
        indices = jnp.array([[0, 1], [2, 3]])
        shifts = jnp.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])

        graph = _make_graph(
            positions,
            charges,
            system_ids,
            jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
            indices,
            shifts,
        )
        result = coulomb_vacuum_energy(
            GraphPotentialInput(parameters=None, graph=graph)
        )
        expected_energies = jnp.array([-1.0, -0.5]) * TO_STANDARD_UNITS / 2
        assert result.data.data.shape == (2,)
        assert jnp.allclose(result.data.data, expected_energies, rtol=1e-10)

    def test_energy_symmetry(self):
        """Test that energy is symmetric w.r.t. particle order."""
        result1 = _jit_coulomb_vacuum_energy(
            GraphPotentialInput(parameters=None, graph=self.graph)
        )

        graph_swapped = _make_graph(
            jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            jnp.array([-1.0, 1.0]),
            self.system_ids,
            self.lattice_vectors,
            jnp.array([[0, 1]]),
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result2 = _jit_coulomb_vacuum_energy(
            GraphPotentialInput(parameters=None, graph=graph_swapped)
        )
        assert jnp.isclose(result1.data.data[0], result2.data.data[0], rtol=1e-12)

    def test_gradients(self):
        """Merged gradient tests: charge gradients + position gradients."""

        # Charge gradients
        def energy_fn_charges(charges):
            graph = _make_graph(
                self.positions,
                charges,
                self.system_ids,
                self.lattice_vectors,
                self.indices,
                self.shifts,
            )
            return coulomb_vacuum_energy(
                GraphPotentialInput(parameters=None, graph=graph)
            ).data.data[0]

        gradients_q = jax.grad(energy_fn_charges)(self.charges)
        assert gradients_q.shape == self.charges.shape
        assert jnp.all(jnp.isfinite(gradients_q))
        assert jnp.isclose(gradients_q[0] / gradients_q[1], -1.0, rtol=1e-10)

        # Position gradients
        def energy_fn_pos(positions):
            graph = _make_graph(
                positions,
                self.charges,
                self.system_ids,
                self.lattice_vectors,
                self.indices,
                jnp.array([[[0.0, 0.0, 0.0]]]),
            )
            return coulomb_vacuum_energy(
                GraphPotentialInput(parameters=None, graph=graph)
            ).data.data[0]

        gradients_p = jax.grad(energy_fn_pos)(self.positions)
        assert gradients_p.shape == self.positions.shape
        assert jnp.all(jnp.isfinite(gradients_p))


class TestCoulombPhysicalValidation:
    """Test physical correctness of Coulomb calculations."""

    def test_superposition_principle(self):
        """Test that energies obey superposition principle."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0])
        system_ids = jnp.array([0, 0, 0, 0])
        indices = jnp.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        shifts = jnp.zeros((6, 1, 3))

        graph = _make_graph(
            positions,
            charges,
            system_ids,
            jnp.eye(3)[None] * 10.0,
            indices,
            shifts,
        )
        result = coulomb_vacuum_energy(
            GraphPotentialInput(parameters=None, graph=graph)
        )
        sqrt2 = jnp.sqrt(2.0)
        expected_energy = (-4.0 + 2.0 / sqrt2) * TO_STANDARD_UNITS / 2
        assert jnp.isclose(result.data.data[0], expected_energy, rtol=1e-10)

    def test_energy_conservation_scaling(self):
        """Test that energy scales correctly when all charges are scaled."""
        positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        system_ids = jnp.array([0, 0])
        indices = jnp.array([[0, 1]])
        shifts = jnp.array([[[2.0, 0.0, 0.0]]])
        lv = jnp.eye(3)[None] * 10.0

        graph1 = _make_graph(
            positions, jnp.array([1.0, -1.0]), system_ids, lv, indices, shifts
        )
        graph2 = _make_graph(
            positions, jnp.array([2.0, -2.0]), system_ids, lv, indices, shifts
        )

        result1 = _jit_coulomb_vacuum_energy(
            GraphPotentialInput(parameters=None, graph=graph1)
        )
        result2 = _jit_coulomb_vacuum_energy(
            GraphPotentialInput(parameters=None, graph=graph2)
        )

        assert jnp.isclose(result2.data.data[0] / result1.data.data[0], 4.0, rtol=1e-10)
