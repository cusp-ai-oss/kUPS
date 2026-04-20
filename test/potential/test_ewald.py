# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import lens
from kups.core.neighborlist import AllDenseNearestNeighborList, Edges
from kups.core.result import as_result_function
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell, make_supercell
from kups.core.utils.jax import dataclass
from kups.potential.classical.coulomb import coulomb_vacuum_energy
from kups.potential.classical.ewald import (
    TO_STANDARD_UNITS,
    EwaldLongRangeInput,
    EwaldParameters,
    estimate_ewald_parameters,
    ewald_long_range_energy,
    ewald_self_interaction_energy,
    ewald_short_range_energy,
    kvecs_from_kmax,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph, PointCloud

from ..clear_cache import clear_cache  # noqa: F401


@dataclass
class PointCloudParticles:
    """Simple point cloud particles with positions, charges, and system index."""

    positions: Array
    charges: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class SystemData:
    """System data with unit cell and cutoff."""

    unitcell: UnitCell
    cutoff: Array


# The Madelung constant for a simple cubic lattice
# with alternating Na and Cl ions
# https://en.wikipedia.org/wiki/Madelung_constant
MADELUNG_CONSTANT = -1.7475646


def _make_particle_data(
    positions: Array,
    charges: Array,
    n_systems: int = 1,
    *,
    system_ids: Array | None = None,
    inclusion_ids: Array | None = None,
    exclusion_ids: Array | None = None,
    inclusion_max_count: int | None = None,
    exclusion_max_count: int | None = None,
) -> PointCloudParticles:
    """Helper to build PointCloudParticles with Index fields."""
    n = len(positions)
    if system_ids is None:
        system_ids = jnp.zeros(n, dtype=int)
    if inclusion_ids is None:
        inclusion_ids = system_ids
    if exclusion_ids is None:
        exclusion_ids = jnp.arange(n, dtype=int)

    system_keys = tuple(SystemId(i) for i in range(n_systems))
    incl_keys = tuple(InclusionId(i) for i in range(int(inclusion_ids.max()) + 1))
    excl_keys = tuple(ExclusionId(i) for i in range(int(exclusion_ids.max()) + 1))

    return PointCloudParticles(
        positions=positions,
        charges=charges,
        system=Index(system_keys, system_ids, inclusion_max_count, _cls=SystemId),
        inclusion=Index(
            incl_keys, inclusion_ids, inclusion_max_count, _cls=InclusionId
        ),
        exclusion=Index(
            excl_keys, exclusion_ids, exclusion_max_count, _cls=ExclusionId
        ),
    )


def _make_systems(unitcell: UnitCell, cutoff: Array) -> Table[SystemId, SystemData]:
    """Build a Table[SystemId, SystemData] from a batched UnitCell and cutoff."""
    n_sys = unitcell.volume.shape[0]
    keys = tuple(SystemId(i) for i in range(n_sys))
    return Table(keys, SystemData(unitcell=unitcell, cutoff=cutoff))


def _build_neighborlist(particles, systems, n_particles):
    """Build neighbor list with sufficient initial capacity.

    Args:
        particles: Particle table.
        systems: System table.
        n_particles: Number of particles (used as avg edges per particle).
            Total capacity = n_particles * n_particles (dense O(N^2)).
    """
    cutoffs = systems.map_data(lambda d: d.cutoff)

    @jax.jit
    @as_result_function
    def nn_search(neighborlist: AllDenseNearestNeighborList):
        return neighborlist(particles, None, systems, cutoffs)

    # avg_edges is per-particle; total = avg_edges * n_particles.
    # Using n_particles gives total = n_particles^2, sufficient for dense lists.
    statics = AllDenseNearestNeighborList(
        avg_edges=FixedCapacity(n_particles),
        avg_image_candidates=FixedCapacity(n_particles),
    )
    while (edge_result := nn_search(statics)).failed_assertions:
        statics = edge_result.fix_or_raise(statics)
    edge_result.raise_assertion()
    return edge_result.value


class TestEwald:
    """Tests for Ewald summation and exclusion correction.

    Grouped into a single class to share JIT caches across tests.
    """

    def test_exclusion_correction_connects_bonded_pairs(self):
        """Exclusion correction: negative vacuum Coulomb over exactly the bonded pairs.

        Setup: 4 particles in one system; (0,1) bonded at distance 1, (2,3) bonded
        at distance 1. Charges alternate +1/-1 so q_i*q_j = -1 for each bonded pair.
        """
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0])

        pdata = _make_particle_data(
            positions,
            charges,
            n_systems=1,
            inclusion_ids=jnp.array([0, 0, 1, 1]),
            inclusion_max_count=2,
        )
        particles = Table.arange(pdata, label=ParticleId)

        unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 100.0)
        systems = _make_systems(unitcell, jnp.array([50.0]))

        # Manually build bonded-pair edges: (0,1), (1,0), (2,3), (3,2)
        edge_idx = jnp.array([[0, 1], [1, 0], [2, 3], [3, 2]])
        edge_shifts = jnp.zeros((4, 1, 3), dtype=float)
        edges = Edges(
            indices=Index(particles.keys, edge_idx),
            shifts=edge_shifts,
        )

        graph = HyperGraph(particles=particles, systems=systems, edges=edges)
        result = coulomb_vacuum_energy(GraphPotentialInput(None, graph))

        npt.assert_allclose(result.data.data[0], -2.0 * TO_STANDARD_UNITS, rtol=1e-5)

    def test_exclusion_correction_pbc_cross_boundary_bond(self):
        """Exclusion correction uses minimum-image distance for a bond across the boundary."""
        positions = jnp.array([[1.0, 0.0, 0.0], [19.0, 0.0, 0.0]])
        charges = jnp.array([1.0, -1.0])

        pdata = _make_particle_data(
            positions,
            charges,
            n_systems=1,
            inclusion_ids=jnp.array([0, 0]),
            inclusion_max_count=2,
        )
        particles = Table.arange(pdata, label=ParticleId)

        unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 20.0)
        systems = _make_systems(unitcell, jnp.array([50.0]))

        # Manually build edges for the cross-boundary bond: (0,1), (1,0)
        # Particle 0 at x=1, particle 1 at x=19, box=20.
        # diff_vec = pos[j] - pos[i] + shift * lattice_vectors
        # Edge (0,1): 19 - 1 + shift*20 = 18 + shift*20; shift=-1 gives -2, distance=2
        # Edge (1,0): 1 - 19 + shift*20 = -18 + shift*20; shift=+1 gives +2, distance=2
        edge_idx = jnp.array([[0, 1], [1, 0]])
        edge_shifts = jnp.array([[[-1, 0, 0]], [[1, 0, 0]]])
        edges = Edges(
            indices=Index(particles.keys, edge_idx),
            shifts=edge_shifts,
        )

        graph = HyperGraph(particles=particles, systems=systems, edges=edges)
        result = coulomb_vacuum_energy(GraphPotentialInput(None, graph))

        npt.assert_allclose(result.data.data[0], -0.5 * TO_STANDARD_UNITS, rtol=1e-5)

    def test_ewald_potential_excludes_bonded_pairs(self):
        """Ewald(all pairs) - vacuum Coulomb(excluded pairs) gives exclusion-corrected energy.

        Verifies the exclusion correction identity by computing each component:
            E_with_exclusions = E_ewald_atomic - E_vacuum_coulomb_excluded
        """
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 0.0, 0.0], [11.5, 0.0, 0.0]]
        )
        charges = jnp.array([1.0, -1.0, 1.0, -1.0])
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3, dtype=float) * 20.0)

        estimates = estimate_ewald_parameters(charges, cell, epsilon_total=1e-4)
        params = EwaldParameters(
            alpha=Table((SystemId(0),), jnp.array([estimates.alpha])),
            cutoff=Table((SystemId(0),), jnp.array([estimates.real_cutoff])),
            reciprocal_lattice_shifts=Table(
                (SystemId(0),), kvecs_from_kmax(cell, estimates.k_max)[None]
            ),
        )

        # 1. Atomic Ewald: no bonded exclusions
        pdata = _make_particle_data(positions, charges, n_systems=1)
        all_particles = Table.arange(pdata, label=ParticleId)
        systems = _make_systems(cell[None], params.cutoff.data)

        edges = _build_neighborlist(all_particles, systems, len(charges))

        graph = HyperGraph(particles=all_particles, systems=systems, edges=edges)
        sr_inp = GraphPotentialInput(params, graph)
        e_atomic = (
            ewald_short_range_energy(sr_inp).data.data[0]
            + ewald_long_range_energy(
                EwaldLongRangeInput(PointCloud(all_particles, systems), params, None)
            ).data.data[0]
            + ewald_self_interaction_energy(sr_inp).data.data[0]
        )

        # 2. Vacuum Coulomb of the excluded (bonded) pairs via manual edges
        # Bonds: (0,1) and (2,3), all within the same image (no PBC shift needed)
        excl_edge_idx = jnp.array([[0, 1], [1, 0], [2, 3], [3, 2]])
        excl_edge_shifts = jnp.zeros((4, 1, 3), dtype=float)
        excl_edges = Edges(
            indices=Index(all_particles.keys, excl_edge_idx),
            shifts=excl_edge_shifts,
        )
        excl_graph = HyperGraph(
            particles=all_particles, systems=systems, edges=excl_edges
        )
        e_excl = coulomb_vacuum_energy(GraphPotentialInput(None, excl_graph)).data.data[
            0
        ]

        # 3. Verify the identity: E_with_exclusions = E_atomic - E_excluded
        e_with_excl = e_atomic - e_excl

        # Cross-check: analytic vacuum Coulomb for the two bonds
        e_excl_analytic = (
            charges[0] * charges[1] / 1.5 + charges[2] * charges[3] / 1.5
        ) * TO_STANDARD_UNITS
        npt.assert_allclose(e_excl, e_excl_analytic, rtol=1e-5)

        # Verify the corrected energy is consistent
        npt.assert_allclose(e_with_excl, e_atomic - e_excl_analytic, rtol=1e-4)

    def test_ewald_summation(self):
        eps = 5e-5
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Na
                [1.0, 1.0, 0.0],  # Na
                [1.0, 0.0, 1.0],  # Na
                [0.0, 1.0, 1.0],  # Na
                [1.0, 1.0, 1.0],  # Cl
                [1.0, 0.0, 0.0],  # Cl
                [0.0, 1.0, 0.0],  # Cl
                [0.0, 0.0, 1.0],  # Cl
            ],
            dtype=float,
        )
        charges = jnp.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        cell = TriclinicUnitCell.from_matrix(jnp.eye(3, dtype=float) * 2)

        with npt.assert_raises(ValueError):
            estimate_ewald_parameters(charges, cell, epsilon_total=eps)

        # The unitcell needs to be at least as large as the cutoff
        REPEATS = 5
        cell, (positions, charges) = make_supercell(
            cell, REPEATS, (positions, charges), lens(lambda x: x[0])
        )
        estimates = estimate_ewald_parameters(charges, cell, epsilon_total=eps)
        assert estimates.error_real < eps, (
            f"Real space target accuracy cannot be reached. {estimates.error_real}"
        )
        params = EwaldParameters(
            alpha=Table((SystemId(0),), jnp.asarray([estimates.alpha])),
            cutoff=Table((SystemId(0),), jnp.asarray([estimates.real_cutoff])),
            reciprocal_lattice_shifts=Table(
                (SystemId(0),), kvecs_from_kmax(cell, estimates.k_max)[None]
            ),
        )

        pdata = _make_particle_data(positions, charges, n_systems=1)
        particles = Table.arange(pdata, label=ParticleId)
        systems = _make_systems(cell[None], params.cutoff.data)

        edges = _build_neighborlist(particles, systems, len(charges))

        graph = HyperGraph(
            particles=particles,
            systems=systems,
            edges=edges,
        )
        sr_input = GraphPotentialInput(params, graph)
        sr_result = ewald_short_range_energy(sr_input)

        lr_input = EwaldLongRangeInput(PointCloud(particles, systems), params, None)
        lr_result = ewald_long_range_energy(lr_input)
        self_result = ewald_self_interaction_energy(sr_input)

        total_energy = sr_result.data.data + lr_result.data.data + self_result.data.data
        npt.assert_allclose(
            MADELUNG_CONSTANT * (REPEATS**3) * 4,
            total_energy[0] / TO_STANDARD_UNITS,
            rtol=eps,
        )


class TestEwaldParametersMake:
    """Tests for EwaldParameters.make with Table inputs."""

    def test_single_system(self):
        """Single NaCl system produces valid parameters."""
        L = 10.0
        positions = jnp.array([[0.0, 0.0, 0.0], [L / 2, L / 2, L / 2]])
        charges = jnp.array([1.0, -1.0])
        particles = Table.arange(
            _make_particle_data(positions, charges), label=ParticleId
        )
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * L)
        systems = Table.arange(
            SystemData(unitcell=uc, cutoff=jnp.array([4.0])),
            label=SystemId,
        )
        params = EwaldParameters.make(particles, systems, real_cutoff=4.0)
        assert params.alpha.data.shape == (1,)
        assert float(params.alpha.data[0]) > 0
        assert len(params.cutoff.keys) == 1
        assert float(params.cutoff.data[0]) > 0
        assert params.reciprocal_lattice_shifts.data.shape[0] == 1
        assert params.reciprocal_lattice_shifts.data.shape[2] == 3

    def test_two_systems(self):
        """Two systems with different sizes produce correct shapes."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.5, 2.5, 2.5],  # system 0
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],  # system 1
            ]
        )
        charges = jnp.array([1.0, -1.0, 2.0, -2.0])
        sys_ids = jnp.array([0, 0, 1, 1])
        particles = Table.arange(
            _make_particle_data(positions, charges, n_systems=2, system_ids=sys_ids),
            label=ParticleId,
        )
        systems = Table(
            (SystemId(0), SystemId(1)),
            SystemData(
                unitcell=TriclinicUnitCell.from_matrix(
                    jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 20.0])
                ),
                cutoff=jnp.array([4.0, 4.0]),
            ),
        )
        params = EwaldParameters.make(particles, systems, real_cutoff=4.0)
        assert params.alpha.data.shape == (2,)
        assert len(params.cutoff.keys) == 2
        assert params.reciprocal_lattice_shifts.data.shape[0] == 2
        # k-vectors zero-padded to max count
        assert params.reciprocal_lattice_shifts.data.ndim == 3

    def test_custom_cutoff(self):
        """Explicit real_cutoff is respected."""
        positions = jnp.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]])
        charges = jnp.array([1.0, -1.0])
        particles = Table.arange(
            _make_particle_data(positions, charges), label=ParticleId
        )
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 5.0)
        systems = Table.arange(
            SystemData(unitcell=uc, cutoff=jnp.array([2.0])),
            label=SystemId,
        )
        params = EwaldParameters.make(particles, systems, real_cutoff=2.0)
        npt.assert_allclose(params.cutoff.data[0], 2.0)
