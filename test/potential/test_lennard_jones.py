# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import view
from kups.core.neighborlist import Edges
from kups.core.patch import WithPatch
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    LennardJonesParameters,
    PairTailCorrectedLennardJonesParameters,
    global_lennard_jones_tail_correction_energy,
    global_lennard_jones_tail_correction_pressure,
    lennard_jones_energy,
    make_global_lennard_jones_tail_correction_pressure,
    pair_tail_corrected_lennard_jones_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from ..clear_cache import clear_cache  # noqa: F401


@dataclass
class _LJPointData:
    positions: jax.Array
    labels: Index[str]
    system: Index[SystemId]


@dataclass
class _SystemData:
    unitcell: UnitCell
    cutoff: jax.Array


def _make_particles(
    positions: jax.Array,
    species: list[str],
    system_ids: jax.Array,
) -> Table[ParticleId, _LJPointData]:
    labels = Index.new(species)
    system = Index.new(system_ids)
    return Table.arange(_LJPointData(positions, labels, system), label=ParticleId)


def _make_systems(
    lattice_vectors: jax.Array, cutoff: jax.Array
) -> Table[SystemId, _SystemData]:
    unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
    return Table.arange(_SystemData(unitcell, cutoff), label=SystemId)


def _make_graph(
    positions: jax.Array,
    species: list[str],
    system_ids: jax.Array,
    lattice_vectors: jax.Array,
    cutoff: jax.Array,
    edge_indices: jax.Array,
    edge_shifts: jax.Array,
) -> HyperGraph:
    particles = _make_particles(positions, species, system_ids)
    systems = _make_systems(lattice_vectors, cutoff)
    edges = Edges(
        indices=Index(particles.keys, edge_indices),
        shifts=edge_shifts,
    )
    return HyperGraph(particles, systems, edges)


def _make_point_cloud_graph(
    positions: jax.Array,
    species: list[str],
    system_ids: jax.Array,
    lattice_vectors: jax.Array,
    cutoff: jax.Array,
) -> HyperGraph:
    """Make a HyperGraph with zero edges (for global corrections)."""
    particles = _make_particles(positions, species, system_ids)
    systems = _make_systems(lattice_vectors, cutoff)
    edges = Edges(
        indices=Index(particles.keys, jnp.zeros((0, 0), dtype=int)),
        shifts=jnp.zeros((0, 0, 3), dtype=int),
    )
    return HyperGraph(particles, systems, edges)


_LABELS = ("A", "B")


def _cutoff(*vals: float) -> Table[SystemId, jax.Array]:
    return Table(
        tuple(SystemId(i) for i in range(len(vals))),
        jnp.array(list(vals)),
    )


_jit_lj_energy = jax.jit(lennard_jones_energy)
_jit_pair_tail_energy = jax.jit(pair_tail_corrected_lennard_jones_energy)
_jit_global_tail_energy = jax.jit(global_lennard_jones_tail_correction_energy)
_jit_global_tail_pressure = jax.jit(global_lennard_jones_tail_correction_pressure)


class TestLennardJonesParameters:
    @classmethod
    def setup_class(cls):
        cls.sigma = jnp.array([2.8, 3.05])
        cls.epsilon = jnp.array([27.0, 79.0])

    def test_initialization(self):
        params = LennardJonesParameters.from_lorentz_berthelot_mixing(
            labels=_LABELS, sigma=self.sigma, epsilon=self.epsilon, cutoff=_cutoff(12.0)
        )
        npt.assert_allclose(params.sigma, jnp.array([[2.8, 2.925], [2.925, 3.05]]))
        npt.assert_allclose(
            params.epsilon,
            jnp.array([[27.0, 46.18441296], [46.18441296, 79.0]]),
        )

    def test_single_species(self):
        params = LennardJonesParameters.from_lorentz_berthelot_mixing(
            labels=("X",),
            sigma=jnp.array([2.8]),
            epsilon=jnp.array([27.0]),
            cutoff=_cutoff(12.0),
        )
        npt.assert_allclose(params.sigma, jnp.array([[2.8]]))
        npt.assert_allclose(params.epsilon, jnp.array([[27.0]]))

    def test_2d_input_error(self):
        try:
            LennardJonesParameters.from_lorentz_berthelot_mixing(
                labels=_LABELS,
                sigma=jnp.array([[2.8, 3.05]]),
                epsilon=jnp.array([27.0]),
                cutoff=_cutoff(12.0),
            )
            assert False, "Should have raised"
        except AssertionError:
            pass


class TestLennardJonesEnergy:
    @classmethod
    def setup_class(cls):
        cls.sigma = jnp.array([[1.0, 1.2], [1.2, 1.0]])
        cls.epsilon = jnp.array([[1.0, 0.8], [0.8, 1.0]])
        cls.parameters = LennardJonesParameters(
            labels=_LABELS,
            sigma=cls.sigma,
            epsilon=cls.epsilon,
            cutoff=_cutoff(12.0),
        )
        cls.graph = _make_graph(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            species=["A", "B"],
            system_ids=jnp.array([0, 0]),
            lattice_vectors=jnp.eye(3)[None] * 10.0,
            cutoff=jnp.array([12.0]),
            edge_indices=jnp.array([[0, 1]]),
            edge_shifts=jnp.array([[[0.0, 0.0, 0.0]]]),
        )

    def test_lj_energy_values(self):
        """Merged: basic + same_species + beyond_cutoff + multiple_pairs + zero_epsilon + very_close + symmetry."""
        energy_fn = _jit_lj_energy
        s, e = self.sigma, self.epsilon

        # Basic A-B at r=1
        result = energy_fn(GraphPotentialInput(self.parameters, self.graph))
        assert isinstance(result, WithPatch)
        assert result.data.data.shape == (1,)
        expected = 4 * e[0, 1] * ((s[0, 1] / 1.0) ** 12 - (s[0, 1] / 1.0) ** 6)
        assert jnp.isclose(result.data.data[0] * 2, expected, rtol=1e-6)

        # Same species A-A at r=1.5
        graph_aa = _make_graph(
            jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            ["A", "A"],
            jnp.array([0, 0]),
            jnp.eye(3)[None] * 10.0,
            jnp.array([12.0]),
            jnp.array([[0, 1]]),
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_aa = energy_fn(GraphPotentialInput(self.parameters, graph_aa))
        expected_aa = 4 * e[0, 0] * ((s[0, 0] / 1.5) ** 12 - (s[0, 0] / 1.5) ** 6)
        assert jnp.isclose(result_aa.data.data[0] * 2, expected_aa, rtol=1e-6)

        # Beyond cutoff -> ~0
        graph_far = _make_graph(
            jnp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            ["A", "B"],
            jnp.array([0, 0]),
            jnp.eye(3)[None] * 200.0,
            jnp.array([12.0]),
            jnp.array([[0, 1]]),
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_far = energy_fn(GraphPotentialInput(self.parameters, graph_far))
        assert jnp.abs(result_far.data.data[0]) < 1e-10

        # Multiple pairs (3 particles)
        graph_mp = _make_graph(
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ["A", "B", "A"],
            jnp.array([0, 0, 0]),
            jnp.eye(3)[None] * 10.0,
            jnp.array([12.0]),
            jnp.array([[0, 1], [0, 2], [1, 2]]),
            jnp.zeros((3, 1, 3)),
        )
        result_mp = energy_fn(GraphPotentialInput(self.parameters, graph_mp))
        e01 = 4 * e[0, 1] * ((s[0, 1] / 1.0) ** 12 - (s[0, 1] / 1.0) ** 6)
        e02 = 4 * e[0, 0] * ((s[0, 0] / 1.0) ** 12 - (s[0, 0] / 1.0) ** 6)
        r12 = jnp.sqrt(2.0)
        e12 = 4 * e[1, 0] * ((s[1, 0] / r12) ** 12 - (s[1, 0] / r12) ** 6)
        assert jnp.isclose(result_mp.data.data[0] * 2, e01 + e02 + e12, rtol=1e-6)

        # Zero epsilon -> zero energy
        params_zero = LennardJonesParameters(
            labels=_LABELS,
            sigma=s,
            epsilon=jnp.zeros((2, 2)),
            cutoff=_cutoff(12.0),
        )
        result_zero = energy_fn(GraphPotentialInput(params_zero, self.graph))
        assert jnp.isclose(result_zero.data.data[0], 0.0, atol=1e-10)

        # Very close particles -> large positive energy
        graph_close = _make_graph(
            jnp.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
            ["A", "B"],
            jnp.array([0, 0]),
            jnp.eye(3)[None] * 10.0,
            jnp.array([12.0]),
            jnp.array([[0, 1]]),
            jnp.array([[[0.0, 0.0, 0.0]]]),
        )
        result_close = energy_fn(GraphPotentialInput(self.parameters, graph_close))
        assert result_close.data.data[0] > 1e6
        assert jnp.isfinite(result_close.data.data[0])

        # Symmetry (reverse edge direction)
        r1 = result.data.data[0]
        graph_rev = _make_graph(
            self.graph.particles.data.positions,
            ["A", "B"],
            jnp.array([0, 0]),
            jnp.eye(3)[None] * 10.0,
            jnp.array([12.0]),
            jnp.array([[1, 0]]),
            self.graph.edges.shifts,
        )
        r2 = energy_fn(GraphPotentialInput(self.parameters, graph_rev)).data.data[0]
        assert jnp.isclose(r1, r2, rtol=1e-10)

    def test_multiple_graphs(self):
        graph = _make_graph(
            positions=jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            species=["A", "B", "A", "B"],
            system_ids=jnp.array([0, 0, 1, 1]),
            lattice_vectors=jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
            cutoff=jnp.array([12.0, 12.0]),
            edge_indices=jnp.array([[0, 1], [2, 3]]),
            edge_shifts=jnp.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]]),
        )
        result = lennard_jones_energy(GraphPotentialInput(self.parameters, graph))
        assert result.data.data.shape == (2,)
        s, e = self.sigma[0, 1], self.epsilon[0, 1]
        e0 = 4 * e * ((s / 1.0) ** 12 - (s / 1.0) ** 6)
        e1 = 4 * e * ((s / 2.0) ** 12 - (s / 2.0) ** 6)
        assert jnp.isclose(result.data.data[0] * 2, e0, rtol=1e-6)
        assert jnp.isclose(result.data.data[1] * 2, e1, rtol=1e-6)

    def test_energy_is_differentiable(self):
        """Test that jax.grad works through lennard_jones_energy."""

        def energy_fn(sigma, epsilon):
            params = LennardJonesParameters(
                labels=_LABELS,
                sigma=sigma,
                epsilon=epsilon,
                cutoff=_cutoff(12.0),
            )
            return lennard_jones_energy(
                GraphPotentialInput(params, self.graph)
            ).data.data.sum()

        grad_sigma, grad_epsilon = jax.grad(energy_fn, argnums=(0, 1))(
            self.sigma, self.epsilon
        )
        assert grad_sigma.shape == self.sigma.shape
        assert grad_epsilon.shape == self.epsilon.shape
        assert jnp.all(jnp.isfinite(grad_sigma))
        assert jnp.all(jnp.isfinite(grad_epsilon))
        assert jnp.any(jnp.abs(grad_sigma) > 1e-10)
        assert jnp.any(jnp.abs(grad_epsilon) > 1e-10)


class TestPairTailCorrectedLennardJonesEnergy:
    @classmethod
    def setup_class(cls):
        cls.sigma = jnp.array([[1.0, 1.2], [1.2, 1.0]])
        cls.epsilon = jnp.array([[1.0, 0.8], [0.8, 1.0]])
        cls.lj_params = LennardJonesParameters(
            labels=_LABELS,
            sigma=cls.sigma,
            epsilon=cls.epsilon,
            cutoff=_cutoff(12.0),
        )
        cls.graph = _make_graph(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            species=["A", "B"],
            system_ids=jnp.array([0, 0]),
            lattice_vectors=jnp.eye(3)[None] * 10.0,
            cutoff=jnp.array([12.0]),
            edge_indices=jnp.array([[0, 1]]),
            edge_shifts=jnp.array([[[0.0, 0.0, 0.0]]]),
        )

    def test_tail_correction_regimes(self):
        """Merged: non_corrected + boundary + corrected + outside regimes."""
        energy_fn = _jit_pair_tail_energy
        lj_fn = _jit_lj_energy
        target = lj_fn(GraphPotentialInput(self.lj_params, self.graph)).data.data

        # Non-corrected regime (truncation_radius > r)
        params_nc = PairTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=self.sigma,
            epsilon=self.epsilon,
            cutoff=_cutoff(2.0),
            truncation_radius=_cutoff(1.5),
        )
        corrected_nc = energy_fn(GraphPotentialInput(params_nc, self.graph)).data.data
        npt.assert_allclose(corrected_nc, target)

        # Boundary regime (cutoff just above r)
        params_b = PairTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=self.sigma,
            epsilon=self.epsilon,
            cutoff=_cutoff(1.2),
            truncation_radius=_cutoff(1.0),
        )
        corrected_b = energy_fn(GraphPotentialInput(params_b, self.graph)).data.data
        npt.assert_allclose(corrected_b, target)

        # Corrected regime (truncation_radius < r < cutoff)
        params_c = PairTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=self.sigma,
            epsilon=self.epsilon,
            cutoff=_cutoff(1.2),
            truncation_radius=_cutoff(0.8),
        )
        corrected_c = energy_fn(GraphPotentialInput(params_c, self.graph)).data.data
        npt.assert_array_less(corrected_c, target)
        npt.assert_array_less(0.0, corrected_c)

        # Outside regime (r > cutoff)
        params_o = PairTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=self.sigma,
            epsilon=self.epsilon,
            cutoff=_cutoff(1.0),
            truncation_radius=_cutoff(0.8),
        )
        result_o = energy_fn(GraphPotentialInput(params_o, self.graph)).data.data
        npt.assert_allclose(result_o, 0.0)


class TestGlobalTailCorrectedLennardJonesEnergy:
    @classmethod
    def setup_class(cls):
        cls.sigma = jnp.array([[1.0, 1.2], [1.2, 1.0]])
        cls.epsilon = jnp.array([[1.0, 0.8], [0.8, 1.0]])
        cls.unitcells = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
        cls.parameters = GlobalTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=cls.sigma,
            epsilon=cls.epsilon,
            tail_corrected=jnp.ones((2, 2), dtype=jnp.bool_),
            cutoff=_cutoff(12.0),
        )
        cls.graph = _make_point_cloud_graph(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            species=["A", "B"],
            system_ids=jnp.array([0, 0]),
            lattice_vectors=jnp.eye(3)[None] * 10.0,
            cutoff=jnp.array([12.0]),
        )

    def test_global_tail_correction(self):
        result = _jit_global_tail_energy(
            GraphPotentialInput(self.parameters, self.graph)
        )
        N = jnp.array([1, 1])
        V = self.unitcells.volume[0]
        density = N[:, None] * N[None, :] / V
        cutoff = self.parameters.cutoff.data[0]
        term1 = (self.sigma / cutoff) ** 3
        target = (
            8
            / 3
            * jnp.pi
            * density
            * self.epsilon
            * self.sigma**3
            * (term1**3 / 3 - term1)
        ).sum()
        assert result.data.data.shape == (1,)
        npt.assert_allclose(result.data.data, target, rtol=1e-6)


class TestGlobalTailCorrectionPressure:
    @classmethod
    def setup_class(cls):
        cls.parameters = GlobalTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=jnp.array([[1.0, 1.2], [1.2, 1.0]]),
            epsilon=jnp.array([[1.0, 0.8], [0.8, 1.0]]),
            tail_corrected=jnp.ones((2, 2), dtype=jnp.bool_),
            cutoff=_cutoff(12.0),
        )
        cls.graph = _make_point_cloud_graph(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            species=["A", "B"],
            system_ids=jnp.array([0, 0]),
            lattice_vectors=jnp.eye(3)[None] * 10.0,
            cutoff=jnp.array([12.0]),
        )

    def test_pressure_scenarios(self):
        """Merged: basic + disabled_interactions."""
        # Basic
        result = _jit_global_tail_pressure(
            GraphPotentialInput(self.parameters, self.graph)
        )
        assert isinstance(result, WithPatch)
        assert result.data.data.shape == (1,)
        assert jnp.isfinite(result.data.data[0])

        # Disabled interactions
        params_disabled = GlobalTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=self.parameters.sigma,
            epsilon=self.parameters.epsilon,
            tail_corrected=jnp.array([[True, False], [False, True]]),
            cutoff=_cutoff(12.0),
        )
        result_d = _jit_global_tail_pressure(
            GraphPotentialInput(params_disabled, self.graph)
        )
        assert result_d.data.data.shape == (1,)
        assert jnp.isfinite(result_d.data.data[0])


class TestPotentialFactoryFunctions:
    def test_make_global_pressure(self):
        particles = _make_particles(
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            ["A", "B"],
            jnp.array([0, 0]),
        )
        systems = _make_systems(jnp.eye(3)[None] * 10.0, jnp.array([12.0]))
        params = GlobalTailCorrectedLennardJonesParameters(
            labels=_LABELS,
            sigma=jnp.array([[1.0, 1.2], [1.2, 1.0]]),
            epsilon=jnp.array([[1.0, 0.8], [0.8, 1.0]]),
            tail_corrected=jnp.ones((2, 2), dtype=jnp.bool_),
            cutoff=_cutoff(12.0),
        )
        state = {"particles": particles, "systems": systems, "parameters": params}
        pressure_fn = make_global_lennard_jones_tail_correction_pressure(
            particles_view=view(lambda x: x["particles"]),
            systems_view=view(lambda x: x["systems"]),
            parameter_view=view(lambda x: x["parameters"]),
        )
        result = pressure_fn(jax.random.key(42), state)
        assert isinstance(result, Table)
        assert result.data.shape == (1,)
        assert jnp.isfinite(result.data[0])
