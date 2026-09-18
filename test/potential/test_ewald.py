# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array, random
from scipy.special import erfc

from kups.core.capacity import FixedCapacity
from kups.core.cell import (
    Cell,
    Periodic3D,
    PeriodicCell,
    TriclinicFrame,
    make_supercell,
)
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.data.wrappers import WithIndices
from kups.core.lens import bind, lens
from kups.core.neighborlist import AllDenseNearestNeighborList, Edges
from kups.core.result import as_result_function
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.core.utils.kahan import KahanSummand
from kups.potential.classical.coulomb import coulomb_vacuum_energy
from kups.potential.classical.ewald import (
    TO_STANDARD_UNITS,
    EwaldCache,
    EwaldLongRangeInput,
    EwaldParameters,
    estimate_ewald_parameters,
    ewald_long_range_energy,
    ewald_net_charge_energy,
    ewald_self_interaction_energy,
    ewald_short_range_energy,
    ewald_short_range_pair_kernel,
    prefactor,
    structure_factor,
)
from kups.potential.classical.ewald.reciprocal import _structure_factor_full
from kups.potential.common.graph import GraphPotentialInput, HyperGraph, PointCloud


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
    """System data with cell and cutoff."""

    cell: Cell[Periodic3D]
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


def _make_systems(cell: Cell[Periodic3D], cutoff: Array) -> Table[SystemId, SystemData]:
    """Build a Table[SystemId, SystemData] from a batched Cell and cutoff."""
    n_sys = cell.volume.shape[0]
    keys = tuple(SystemId(i) for i in range(n_sys))
    return Table(keys, SystemData(cell=cell, cutoff=cutoff))


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
        return neighborlist(particles, systems)

    # avg_edges is per-particle; total = avg_edges * n_particles.
    # Using n_particles gives total = n_particles^2, sufficient for dense lists.
    statics = AllDenseNearestNeighborList(
        avg_edges=FixedCapacity(n_particles),
        avg_image_candidates=FixedCapacity(n_particles),
        cutoffs=cutoffs,
    )
    while (edge_result := nn_search(statics)).failed_assertions:
        statics = edge_result.fix_or_raise(statics)
    edge_result.raise_assertion()
    return edge_result.value


def _parameters(cell, alpha, cutoff, k_max):
    systems = _make_systems(cell[None], jnp.asarray([cutoff]))
    return EwaldParameters.from_cutoffs(
        systems,
        systems.set_data(jnp.asarray([alpha], dtype=cell.vectors.dtype)),
        systems.set_data(jnp.asarray([cutoff], dtype=cell.vectors.dtype)),
        systems.set_data(jnp.asarray([k_max], dtype=cell.vectors.dtype)),
        compact=True,
    )


class TestEwald:
    """Tests for Ewald summation and exclusion correction.

    Grouped into a single class to share JIT caches across tests.
    """

    @pytest.mark.parametrize("singleton_alpha", [False, True])
    def test_short_range_kernel_values_and_radial_derivative(
        self, singleton_alpha: bool
    ) -> None:
        parameters = TestReciprocalReduction()._input().parameters
        alpha = jnp.array([0.2]) if singleton_alpha else jnp.array([0.2, 0.3, 0.5, 0.4])
        parameters = bind(parameters, lambda p: p.alpha).set(
            Table.arange(alpha, label=SystemId)
        )
        system = Index.integer(jnp.array([[0], [2]]), n=4, label=SystemId)
        distance = jnp.array([[0.02, 0.9, 2.3, 6.1], [0.1, 1.1, 3.3, 5.5]])
        left = jnp.array([[0.4], [-0.6]])
        right = jnp.array([-0.2, 0.7, 0.0, 0.3])

        def energy(r2: Array) -> Array:
            return ewald_short_range_pair_kernel(
                parameters, left, right, jnp.zeros((*r2.shape, 3)), r2, system
            )

        a = np.asarray(alpha[0] if singleton_alpha else alpha[jnp.array([0, 2]), None])
        r = np.asarray(distance)
        qq = TO_STANDARD_UNITS * np.asarray(left * right)
        screened = erfc(a * r)
        expected = qq * screened / r
        derivative = (
            -qq
            * (screened / r**2 + 2 * a / np.sqrt(np.pi) * np.exp(-((a * r) ** 2)) / r)
            / (2 * r)
        )
        npt.assert_allclose(
            jax.jit(energy)(distance**2), expected, rtol=2e-11, atol=1e-13
        )
        actual_derivative = jax.jit(jax.grad(lambda r2: energy(r2).sum()))(distance**2)
        npt.assert_allclose(actual_derivative, derivative, rtol=2e-11, atol=1e-13)

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

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 100.0))
        systems = _make_systems(cell, jnp.array([50.0]))

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

        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 20.0))
        systems = _make_systems(cell, jnp.array([50.0]))

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
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3, dtype=float) * 20.0))

        # The exclusion-correction identity is independent of Ewald precision
        # (e_excl is checked analytically), so a coarse k-space suffices here.
        estimates = estimate_ewald_parameters(charges, cell, epsilon_total=1e-3)
        params = _parameters(
            cell, estimates.alpha, estimates.real_cutoff, estimates.k_max
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
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3, dtype=float) * 2))

        with npt.assert_raises(ValueError):
            estimate_ewald_parameters(charges, cell, epsilon_total=eps)

        # The cell needs to be at least as large as the cutoff
        REPEATS = 5
        cell, (positions, charges) = make_supercell(
            cell, REPEATS, (positions, charges), lens(lambda x: x[0])
        )
        estimates = estimate_ewald_parameters(charges, cell, epsilon_total=eps)
        assert estimates.error_real < eps, (
            f"Real space target accuracy cannot be reached. {estimates.error_real}"
        )
        params = _parameters(
            cell, estimates.alpha, estimates.real_cutoff, estimates.k_max
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

    def test_net_charge_energy_closed_form(self):
        """E_net matches -(pi / (2 V alpha^2)) * Q^2 in standard units."""
        L = 10.0
        positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])
        charges = jnp.array([1.0, 1.0, 0.5])  # Q = 2.5
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3, dtype=float) * L))
        alpha = 0.4
        params = _parameters(cell, alpha, 5.0, 0.0)
        pdata = _make_particle_data(positions, charges, n_systems=1)
        particles = Table.arange(pdata, label=ParticleId)
        systems = _make_systems(cell[None], jnp.array([5.0]))
        inp = EwaldLongRangeInput(PointCloud(particles, systems), params, None)

        e_net = ewald_net_charge_energy(inp).data[0]
        Q, V = 2.5, L**3
        expected = -jnp.pi / (2 * V * alpha**2) * Q**2 * TO_STANDARD_UNITS
        npt.assert_allclose(e_net, expected, rtol=1e-6)

    def test_zero_kvector_excluded(self):
        """The k=0 reciprocal mode is excluded (prefactor 0); net charge replaces it."""
        L = 10.0
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3, dtype=float) * L))
        params = _parameters(cell, 0.4, 5.0, 1.0)
        pdata = _make_particle_data(jnp.zeros((1, 3)), jnp.array([1.0]), n_systems=1)
        particles = Table.arange(pdata, label=ParticleId)
        systems = _make_systems(cell[None], jnp.array([5.0]))
        inp = EwaldLongRangeInput(PointCloud(particles, systems), params, None)

        pref = prefactor(inp)
        zero = np.all(np.asarray(inp.kvecs[0]) == 0, axis=-1)
        assert zero.any() and not zero.all()
        npt.assert_array_equal(pref[0, zero], 0.0)
        assert bool(jnp.all(pref[0, ~zero] > 0))

    def test_net_charge_total_energy_alpha_stable(self):
        """For Q != 0 the total energy is finite and ~independent of alpha.

        The neutralizing-background correction restores alpha-independence of the
        total Ewald energy; doubling alpha leaves the (converged) total unchanged.
        """
        L = 12.0
        positions = jnp.array([[0.0, 0.0, 0.0], [6.0, 6.0, 6.0]])
        charges = jnp.array([1.0, 1.0])  # Q = 2, non-neutral
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3, dtype=float) * L))

        rc = 5.0
        pdata = _make_particle_data(positions, charges, n_systems=1)
        particles = Table.arange(pdata, label=ParticleId)
        systems = _make_systems(cell[None], jnp.array([rc]))
        edges = _build_neighborlist(particles, systems, len(charges))
        graph = HyperGraph(particles=particles, systems=systems, edges=edges)

        def total_energy(alpha: float) -> Array:
            params = _parameters(cell, alpha, rc, 6.0)
            sr_inp = GraphPotentialInput(params, graph)
            lr_inp = EwaldLongRangeInput(PointCloud(particles, systems), params, None)
            return (
                ewald_short_range_energy(sr_inp).data.data[0]
                + ewald_long_range_energy(lr_inp).data.data[0]
                + ewald_self_interaction_energy(sr_inp).data.data[0]
            )

        e_lo = total_energy(0.5)
        e_hi = total_energy(1.0)  # alpha doubled
        assert bool(jnp.isfinite(e_lo)) and bool(jnp.isfinite(e_hi))
        npt.assert_allclose(e_lo, e_hi, atol=5e-3, rtol=1e-3)


class TestEwaldParametersMake:
    """Tests for EwaldParameters.make with Table inputs."""

    @pytest.mark.parametrize("backend,expected", [("cpu", 1024), ("gpu", 8192)])
    def test_backend_tile_defaults_preserve_explicit_limits(
        self, monkeypatch, backend, expected
    ):
        monkeypatch.setattr(jax, "default_backend", lambda: backend)
        particles = Table.arange(
            _make_particle_data(
                jnp.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
                jnp.array([1.0, -1.0]),
            ),
            label=ParticleId,
        )
        systems = Table.arange(
            SystemData(
                cell=PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10)),
                cutoff=jnp.array([4.0]),
            ),
            label=SystemId,
        )
        params = EwaldParameters.make(particles, systems, real_cutoff=4.0)
        assert params.reciprocal_particle_chunk_size == expected
        direct = EwaldParameters(
            params.alpha, params.cutoff, params.reciprocal_lattice_shifts, params.k_max
        )
        assert direct.reciprocal_particle_chunk_size == expected
        explicit = EwaldParameters.from_cutoffs(
            systems,
            params.alpha,
            params.cutoff,
            params.k_max,
            reciprocal_particle_chunk_size=17,
        )
        assert explicit.reciprocal_particle_chunk_size == 17

    def test_single_system(self):
        """Single NaCl system produces valid parameters."""
        L = 10.0
        positions = jnp.array([[0.0, 0.0, 0.0], [L / 2, L / 2, L / 2]])
        charges = jnp.array([1.0, -1.0])
        particles = Table.arange(
            _make_particle_data(positions, charges), label=ParticleId
        )
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * L))
        systems = Table.arange(
            SystemData(cell=cell, cutoff=jnp.array([4.0])),
            label=SystemId,
        )
        params = EwaldParameters.make(particles, systems, real_cutoff=4.0)
        assert params.alpha.data.shape == (1,)
        assert float(params.alpha.data[0]) > 0
        assert len(params.cutoff.keys) == 1
        assert float(params.cutoff.data[0]) > 0
        assert params.k_max.data.shape == (1,)
        assert params.reciprocal_lattice_shifts.data.shape[1] > 0
        assert params.reciprocal_lattice_shifts.data.shape[::2] == (1, 3)

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
                cell=PeriodicCell(
                    TriclinicFrame.from_matrix(
                        jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 20.0])
                    )
                ),
                cutoff=jnp.array([4.0, 4.0]),
            ),
        )
        params = EwaldParameters.make(particles, systems, real_cutoff=4.0)
        assert params.alpha.data.shape == (2,)
        assert len(params.cutoff.keys) == 2
        assert params.k_max.data.shape == (2,)
        inp = EwaldLongRangeInput(PointCloud(particles, systems), params)
        assert inp.kvecs.shape == params.reciprocal_lattice_shifts.data.shape

    def test_custom_cutoff(self):
        """Explicit real_cutoff is respected."""
        positions = jnp.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]])
        charges = jnp.array([1.0, -1.0])
        particles = Table.arange(
            _make_particle_data(positions, charges), label=ParticleId
        )
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 5.0))
        systems = Table.arange(
            SystemData(cell=cell, cutoff=jnp.array([2.0])),
            label=SystemId,
        )
        params = EwaldParameters.make(particles, systems, real_cutoff=2.0)
        npt.assert_allclose(params.cutoff.data[0], 2.0)


class TestStructureFactorCache:
    """Tests for the compensated structure factor cache used by incremental MC."""

    REPEATS = 4
    L = 8.0
    N_STEPS = 1000
    # (2, 2, 2) is the charge-ordering wavevector of the lattice below, where the
    # structure factor peaks at |S| = N and accumulation error is largest.
    # Single precision keeps the accumulation error well above the float64
    # reference; the test suite otherwise runs with x64 enabled.
    DTYPE = jnp.float32

    def _setup(
        self,
    ) -> tuple[
        Table[ParticleId, PointCloudParticles],
        Callable[..., EwaldLongRangeInput[Any]],
    ]:
        """Build a charge-ordered cubic lattice and a builder for its Ewald input.

        Returns:
            Tuple of the particle table and a function mapping positions (plus an
            optional cache and previous-particle data) to an Ewald input.
        """
        grid = jnp.stack(
            jnp.meshgrid(*(3 * [jnp.arange(self.REPEATS)]), indexing="ij"), axis=-1
        ).reshape(-1, 3)
        positions = (grid * (self.L / self.REPEATS)).astype(self.DTYPE)
        charges = ((-1.0) ** grid.sum(-1)).astype(self.DTYPE)
        cell = PeriodicCell(
            TriclinicFrame.from_matrix(jnp.eye(3, dtype=self.DTYPE) * self.L)
        )
        params = _parameters(cell, 0.35, 3.0, 3.0)
        systems = _make_systems(cell[None], params.cutoff.data)
        # Indices are built once outside any trace; only positions vary.
        particles = Table.arange(
            _make_particle_data(positions, charges), label=ParticleId
        )

        def make_input(
            pos: Array,
            cache: EwaldCache[Any, Any] | None = None,
            changes: WithIndices[ParticleId, PointCloudParticles] | None = None,
        ) -> EwaldLongRangeInput[Any]:
            patched = bind(particles).focus(lambda p: p.data.positions).set(pos)
            return EwaldLongRangeInput(
                PointCloud(patched, systems), params, cache, None, changes
            )

        return particles, make_input

    def _cache(self, summand: KahanSummand[Array]) -> EwaldCache[Any, Any]:
        """Wrap a structure factor accumulator in an otherwise-zero cache."""
        zeros = EwaldCache.make(1, summand.value.shape[1])
        return EwaldCache(
            summand,
            zeros.short_range,
            zeros.long_range,
            zeros.self_interaction,
            zeros.exclusion,
        )

    def _walk(self, compensated: bool) -> tuple[Array, Array]:
        """Run a chain of single-particle moves, returning final S(k) and positions.

        Args:
            compensated: Keep the Kahan compensation between updates; if ``False``
                it is dropped each step, leaving the plain running sum.

        Returns:
            Tuple of the S(k) the energy would consume and the final positions.
        """
        particles, make_input = self._setup()
        n = len(particles)
        moved = random.randint(random.key(2), (self.N_STEPS,), 0, n)
        deltas = random.normal(random.key(1), (self.N_STEPS, 3), self.DTYPE) * 1e-4

        @jax.jit
        def step(
            carry: tuple[KahanSummand[Array], Array], move: tuple[Array, Array]
        ) -> tuple[tuple[KahanSummand[Array], Array], None]:
            sk, pos = carry
            idx, delta = move
            previous = WithIndices(
                Index.integer(idx[None], n=n, label=ParticleId),
                jax.tree.map(lambda x: x[idx][None], particles.data),
            )
            previous = (
                bind(previous).focus(lambda p: p.data.positions).set(pos[idx][None])
            )
            if not compensated:
                sk = KahanSummand.init(sk.value)
            new_pos = pos.at[idx].add(delta)
            sk, _ = structure_factor(make_input(new_pos, self._cache(sk), previous))
            return (sk, new_pos), None

        positions = particles.data.positions
        sk0, _ = structure_factor(make_input(positions))
        (sk, pos), _ = jax.lax.scan(step, (sk0, positions), (moved, deltas))
        # Without compensation only the running sum survives, as before this change.
        return (sk.total if compensated else sk.value), pos

    def _reference(self, positions: Array) -> np.ndarray:
        """Structure factor of `positions` in float64, shape ``(n_kvecs, 2)``."""
        particles, make_input = self._setup()
        kvecs = np.asarray(make_input(positions).kvecs[0], dtype=np.float64)
        phase = np.asarray(positions, dtype=np.float64) @ kvecs.T
        q = np.asarray(particles.data.charges, dtype=np.float64)[:, None]
        return np.stack(
            [(q * np.cos(phase)).sum(0), (q * np.sin(phase)).sum(0)], axis=-1
        )

    def test_compensation_reduces_drift(self):
        """Carrying the compensation shrinks drift over a chain of moves."""
        reference = self._reference(self._walk(compensated=True)[1])
        errors = {
            compensated: float(
                np.abs(np.asarray(self._walk(compensated)[0][0]) - reference).max()
            )
            for compensated in (True, False)
        }
        assert errors[True] < errors[False] / 4, errors


class TestReciprocalReduction:
    """Uneven batches, bounded tiles, and the full-energy derivative of a cache."""

    @pytest.mark.parametrize("grid", [False, True])
    @pytest.mark.parametrize(
        "previous_systems", ["original", "relabelled", "negative_inactive"]
    )
    def test_incremental_probe_with_subset_vocabulary(
        self,
        grid: bool,
        previous_systems: Literal["original", "relabelled", "negative_inactive"],
    ):
        proposal = self._proposal(self._input(grid=grid))
        previous = proposal.changes_from_prev
        assert previous is not None
        keys = proposal.point_cloud.particles.keys
        subset = Index(
            tuple(keys[i] for i in (0, 2, 3, 5, 18)),
            jnp.array([0, 1, 2, 3, 4, 5, 5]),
        )
        old = previous.data
        if previous_systems != "original":
            # Preserve inactive rows while changing the system vocabulary.
            system = old.system.apply_mask(old.system.indices >= 0).update_labels(
                (SystemId(-1), *old.system.keys)
            )
            if previous_systems == "negative_inactive":
                system = bind(system, lambda idx: idx.indices).set(
                    jnp.where(old.system.indices >= 0, system.indices, -1)
                )
            old = bind(old, lambda p: p.system).set(system)
        proposal = bind(proposal, lambda p: p.changes_from_prev).set(
            WithIndices(subset, old)
        )
        actual, _ = structure_factor(proposal)
        expected, _ = structure_factor(
            bind(proposal, lambda p: (p.cache, p.cache_lens, p.changes_from_prev)).set(
                (None, None, None)
            )
        )
        npt.assert_allclose(actual.total, expected.total, rtol=1e-12, atol=1e-12)

    def _input(self, grid=False, dtype=jnp.float64, chunks=(7, 11), *, batch=4):
        n = 19  # System 3 (when present) is empty; the last two slots are inactive.
        positions = random.uniform(random.key(81), (n, 3), dtype) * 7
        charges = (
            random.normal(random.key(82), (n,), dtype).at[jnp.array([3, 4])].set(0)
        )
        ids = jnp.array([2, 0, 1, 0, 2, 1, 2, 1, 0, 2, 0, 1, 0, 2, 1, 0, 2, 4, -1])
        if batch == 1:
            ids = jnp.where(ids < 0, -1, jnp.where(ids < 4, 0, batch))
        positions = positions.at[-2:].set(jnp.nan)
        charges = charges.at[-2:].set(jnp.nan)
        particles = Table.arange(
            _make_particle_data(positions, charges, batch, system_ids=ids),
            label=ParticleId,
        )
        matrix = jnp.array(
            [[5.0, 0.0, 0.0], [1.0, 10.0, 0.0], [-0.5, 1.5, 20.0]], dtype
        )
        cells = PeriodicCell(
            TriclinicFrame.from_matrix(
                matrix[None] * jnp.linspace(1.0, 1.3, batch, dtype=dtype)[:, None, None]
            )
        )
        systems = _make_systems(cells, jnp.full(batch, 3.0, dtype))
        params = EwaldParameters.from_cutoffs(
            systems,
            systems.set_data(jnp.full(batch, 0.35, dtype)),
            systems.set_data(jnp.full(batch, 3.0, dtype)),
            systems.set_data(jnp.asarray([0.85, 0.8, 0.75, 0.8][:batch], dtype)),
            compact=not grid,
            reciprocal_particle_chunk_size=chunks[0],
            reciprocal_k_chunk_size=chunks[1],
        )
        return EwaldLongRangeInput(PointCloud(particles, systems), params)

    def _reference(self, inp):
        p = inp.point_cloud.particles.data
        kv = np.asarray(inp.kvecs, dtype=np.float64)
        result = np.zeros((*kv.shape[:2], 2))
        for s in range(len(kv)):
            mask = np.asarray(p.system.indices) == s
            phase = np.asarray(p.positions, dtype=np.float64)[mask] @ kv[s].T
            q = np.asarray(p.charges, dtype=np.float64)[mask, None]
            result[s] = np.stack(
                ((q * np.cos(phase)).sum(0), (q * np.sin(phase)).sum(0)), axis=-1
            )
        return result

    def _proposal(self, inp):
        p = inp.point_cloud.particles
        idx = Index.integer(
            jnp.array([0, 2, 3, 5, 18, 19, -1]), n=len(p), label=ParticleId
        )
        # The two invalid probe rows deliberately carry real old data: neither
        # may subtract a particle or wrap around to the last particle slot.
        old = jax.tree.map(lambda a: a[jnp.array([0, 2, 3, 5, 18, 0, 0])], p.data)
        new = bind(p.data, lambda x: (x.positions, x.charges, x.system)).set(
            (
                p.data.positions.at[jnp.array([0, 2, 3, 5])].add(0.05).at[18].set(1.0),
                p.data.charges.at[3].set(0.7).at[18].set(-0.6),
                bind(p.data.system, lambda x: x.indices).set(
                    p.data.system.indices.at[0].set(1).at[2].set(4).at[18].set(2)
                ),
            )
        )
        sk, _ = structure_factor(inp)
        cache = bind(
            EwaldCache.make(len(inp.point_cloud.systems), sk.value.shape[1]),
            lambda x: x.structure_factor,
        ).set(sk)
        return bind(
            inp,
            lambda x: (
                x.point_cloud,
                x.cache,
                x.cache_lens,
                x.changes_from_prev,
            ),
        ).set(
            (
                bind(inp.point_cloud, lambda x: x.particles).set(p.set_data(new)),
                cache,
                lens(lambda c: c),
                WithIndices(idx, old),
            )
        )

    @pytest.mark.parametrize("grid", [False, True])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("chunks", [(2, 11), (1024, 128), (8, 128)])
    @pytest.mark.parametrize("batch", [1, 4])
    def test_full_and_incremental(self, grid, dtype, chunks, batch):
        inp = self._input(grid, dtype, chunks, batch=batch)
        actual, _ = jax.jit(structure_factor)(inp)
        # Grid/direct phase evaluation rounds differently in float32.
        tol = 5e-6 if dtype == jnp.float32 else 2e-13
        npt.assert_allclose(actual.total, self._reference(inp), atol=tol, rtol=tol)

        proposal = self._proposal(inp)
        updated, patch = jax.jit(structure_factor)(proposal)
        npt.assert_allclose(
            updated.total, self._reference(proposal), atol=tol, rtol=tol
        )
        accept = inp.point_cloud.systems.set_data(
            jnp.array([True, False, True, False][:batch])
        )
        accepted = jax.jit(lambda c: patch(c, accept))(proposal.cache)
        for name in ("value", "compensate"):
            npt.assert_array_equal(
                getattr(accepted.structure_factor, name),
                jnp.where(
                    accept.data[:, None, None],
                    getattr(updated, name),
                    getattr(proposal.cache.structure_factor, name),
                ),
            )

    @pytest.mark.parametrize("grid", [False, True])
    @pytest.mark.parametrize("batch", [1, 4])
    @pytest.mark.parametrize("chunks", [(2, 11), (7, 11), (1024, 128)])
    def test_cached_derivatives_match_full_energy(self, grid, batch, chunks):
        proposal = self._proposal(self._input(grid, chunks=chunks, batch=batch))
        p = proposal.point_cloud.particles.data
        # Keep differentiation inputs finite, including inactive particle slots.
        pos, q = jnp.nan_to_num(p.positions), jnp.nan_to_num(p.charges)
        cells = proposal.point_cloud.systems.data.cell

        def energy(pos, q, scale, incremental):
            particles = proposal.point_cloud.particles.set_data(
                bind(p, lambda x: (x.positions, x.charges)).set((pos, q))
            )
            systems = proposal.point_cloud.systems.map_data(
                lambda s: bind(s, lambda x: x.cell).set(
                    PeriodicCell(TriclinicFrame.from_matrix(cells.vectors * scale))
                )
            )
            inp = bind(proposal, lambda x: x.point_cloud).set(
                PointCloud(particles, systems)
            )
            if not incremental:
                inp = bind(inp, lambda x: x.changes_from_prev).set(None)
            return ewald_long_range_energy(inp).data.data.sum()

        # Cell derivatives are taken at the cached cell, as in the virial path;
        # an actual cell move must rebuild the structure factors.
        args = (pos, q, jnp.array(1.0))
        cached = jax.jit(jax.value_and_grad(lambda *xs: energy(*xs, True), (0, 1, 2)))
        full = jax.jit(jax.value_and_grad(lambda *xs: energy(*xs, False), (0, 1, 2)))
        full_result = full(*args)
        for actual, expected in zip(
            jax.tree.leaves(cached(*args)), jax.tree.leaves(full_result)
        ):
            npt.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)

        # Full and cached evaluation share their JVP. Check that rule against
        # finite differences of values, independently for positions, charges
        # (including zero charges), and cell scale.
        value = jax.jit(lambda *xs: energy(*xs, False))
        step = 2e-5
        for i, gradient in enumerate(full_result[1]):
            direction = jnp.cos(jnp.arange(args[i].size)).reshape(args[i].shape)
            plus, minus = list(args), list(args)
            plus[i] = args[i] + step * direction
            minus[i] = args[i] - step * direction
            numerical = (value(*plus) - value(*minus)) / (2 * step)
            npt.assert_allclose(
                jnp.sum(gradient * direction), numerical, atol=2e-7, rtol=2e-7
            )

        # Mixed charge/position derivatives exercise the custom JVP twice,
        # including a zero charge on an unchanged particle.
        second_derivatives = []
        for incremental in (False, True):
            grad_pq = jax.grad(lambda x, y: energy(x, y, args[2], incremental), (0, 1))
            _, tangent = jax.jvp(
                grad_pq, (pos, q), (jnp.ones_like(pos) * 0.03, jnp.ones_like(q) * 0.02)
            )
            second_derivatives.append(tangent)
        for actual, expected in zip(*second_derivatives):
            npt.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)

    @pytest.mark.parametrize("empty", ["particles", "kvecs", "changes"])
    def test_empty_inputs(self, empty):
        inp = self._input()
        if empty == "particles":
            inp = bind(inp, lambda x: x.point_cloud).set(
                bind(inp.point_cloud, lambda x: x.particles).set(
                    Table(
                        (),
                        jax.tree.map(lambda a: a[:0], inp.point_cloud.particles.data),
                        _cls=ParticleId,
                    )
                )
            )
        elif empty == "kvecs":
            inp = bind(
                inp, lambda x: x.parameters.reciprocal_lattice_shifts.data
            ).apply(lambda shifts: shifts[:, :0])
        else:
            inp = self._proposal(inp)
            inp = bind(inp, lambda x: x.changes_from_prev).set(
                WithIndices(
                    Index.integer(jnp.zeros(0, dtype=int), n=19, label=ParticleId),
                    jax.tree.map(lambda a: a[:0], inp.changes_from_prev.data),
                )
            )
        sk, _ = jax.jit(structure_factor)(inp)
        expected = (
            inp.cache.structure_factor.total
            if empty == "changes"
            else self._reference(inp)
        )
        npt.assert_allclose(sk.total, expected, atol=2e-13)

    def test_invalid_chunk_size(self):
        inp = self._input()
        p = inp.point_cloud.particles.data
        for sizes in ({"particle_chunk_size": 0}, {"k_chunk_size": -1}):
            with pytest.raises(ValueError, match="chunk sizes must be positive"):
                _structure_factor_full(
                    p.positions, p.charges, inp.kvecs, batch_mask=p.system, **sizes
                )


class TestReciprocalShifts:
    def test_grid_encloses_retained_vectors_with_tight_axis_bounds(self):
        inp = TestReciprocalReduction()._input(grid=True)
        compact = EwaldParameters.from_cutoffs(
            inp.point_cloud.systems,
            inp.parameters.alpha,
            inp.parameters.cutoff,
            inp.parameters.k_max,
            compact=True,
        )
        bounds = inp.parameters.reciprocal_shift_bound
        assert len(set(bounds)) > 1  # Anisotropic, skew cells need distinct bounds.
        shifts = np.asarray(compact.reciprocal_lattice_shifts.data)
        npt.assert_array_equal(bounds, np.max(np.abs(shifts), axis=(0, 1)))
        actual_grid = np.asarray(inp.parameters.reciprocal_lattice_shifts.data)
        assert actual_grid.shape[1] == (bounds[0] + 1) * (2 * bounds[1] + 1) * (
            2 * bounds[2] + 1
        )

    @pytest.mark.parametrize("grid", [False, True])
    def test_vectors_and_energy_against_independent_sphere(self, grid):
        inp = TestReciprocalReduction()._input(grid)
        bound = inp.parameters.reciprocal_shift_bound
        assert isinstance(bound, tuple) and len(bound) == 3
        if not grid:
            assert bound == (0, 0, 0)
        actual = np.asarray(jax.jit(lambda x: x.kvecs)(inp))
        cells = np.asarray(inp.point_cloud.systems.data.cell.vectors)
        shifts = np.stack(
            np.meshgrid(
                np.arange(9), np.arange(-8, 9), np.arange(-8, 9), indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 3)
        reference_energy = []
        p = inp.point_cloud.particles.data
        for s, cell in enumerate(cells):
            vectors = shifts @ (2 * np.pi * np.linalg.inv(cell).T)
            squared = (vectors**2).sum(-1)
            inside = squared <= float(inp.parameters.k_max.data[s]) ** 2
            expected = vectors[inside]
            selected = actual[s]
            if grid:
                selected = selected[
                    (selected**2).sum(-1) <= float(inp.parameters.k_max.data[s]) ** 2
                ]
            # Compaction pads with k=0, whose weight is zero.
            npt.assert_allclose(
                selected[np.any(selected != 0, axis=-1)],
                expected[np.any(expected != 0, axis=-1)],
                atol=2e-15,
            )
            active = inside & (squared > 0)
            q = np.asarray(p.charges)[np.asarray(p.system.indices) == s]
            pos = np.asarray(p.positions)[np.asarray(p.system.indices) == s]
            sf = (q[:, None] * np.exp(1j * (pos @ vectors[active].T))).sum(0)
            k2 = squared[active]
            weights = (2 - (shifts[active, 0] == 0)) * 2 * np.pi / np.linalg.det(cell)
            weights *= np.exp(-k2 / (4 * float(inp.parameters.alpha.data[s]) ** 2)) / k2
            reference_energy.append(
                np.sum(weights * np.abs(sf) ** 2) * TO_STANDARD_UNITS
            )
        energy = jax.jit(ewald_long_range_energy)(inp).data.data
        npt.assert_allclose(
            energy - ewald_net_charge_energy(inp).data,
            reference_energy,
            atol=2e-13,
            rtol=2e-13,
        )

    @pytest.mark.parametrize("compact", [False, True])
    def test_zero_cutoff_has_only_background_energy(self, compact):
        inp = TestReciprocalReduction()._input()
        parameters = EwaldParameters.from_cutoffs(
            inp.point_cloud.systems,
            inp.parameters.alpha,
            inp.parameters.cutoff,
            inp.parameters.k_max.map_data(jnp.zeros_like),
            compact=compact,
        )
        assert parameters.reciprocal_shift_bound == (0, 0, 0)
        inp = bind(inp, lambda x: x.parameters).set(parameters)
        npt.assert_array_equal(jax.jit(prefactor)(inp), 0)
        npt.assert_allclose(
            jax.jit(ewald_long_range_energy)(inp).data.data,
            ewald_net_charge_energy(inp).data,
            atol=2e-13,
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_cutoff_boundary_and_padding(self, dtype):
        inp = TestReciprocalReduction()._input(dtype=dtype)
        cells = PeriodicCell(
            TriclinicFrame.from_matrix(
                jnp.broadcast_to(jnp.eye(3, dtype=dtype) * 10, (4, 3, 3))
            )
        )
        inp = bind(inp, lambda x: x.point_cloud.systems.data.cell).set(cells)
        systems = inp.point_cloud.systems
        params = EwaldParameters.from_cutoffs(
            systems,
            inp.parameters.alpha,
            inp.parameters.cutoff,
            systems.set_data(jnp.full(4, 2 * jnp.pi / 10, dtype)),
            compact=True,
        )
        inp = bind(inp, lambda x: x.parameters).set(params)
        result = jax.jit(as_result_function(lambda x: x.kvecs))(inp)
        result.raise_assertion()
        assert np.isfinite(np.asarray(result.value)).all()
        n_kvecs = params.reciprocal_lattice_shifts.data.shape[1]
        padded = bind(inp, lambda x: x.parameters.reciprocal_lattice_shifts.data).apply(
            lambda shifts: jnp.pad(shifts, ((0, 0), (0, 5), (0, 0)))
        )
        padded_vectors = jax.jit(lambda x: x.kvecs)(padded)
        npt.assert_allclose(padded_vectors[:, :n_kvecs], result.value)
        npt.assert_array_equal(padded_vectors[:, n_kvecs:], 0)

    @pytest.mark.parametrize("compact", [False, True])
    def test_k_max_is_retained_and_masks_stored_shifts(self, compact):
        inp = TestReciprocalReduction()._input(grid=not compact)
        assert inp.parameters.k_max is not None
        shifts = inp.parameters.reciprocal_lattice_shifts.data
        assert jnp.issubdtype(shifts.dtype, jnp.integer)
        inp = bind(inp, lambda x: x.parameters.k_max.data).apply(
            lambda cutoff: cutoff / 2
        )
        weights = jax.jit(prefactor)(inp)
        outside = (
            jnp.sum(inp.kvecs**2, axis=-1) > inp.parameters.k_max.data[:, None] ** 2
        )
        assert bool(outside.any())
        npt.assert_array_equal(weights[outside], 0)
        npt.assert_array_equal(inp.parameters.reciprocal_lattice_shifts.data, shifts)
