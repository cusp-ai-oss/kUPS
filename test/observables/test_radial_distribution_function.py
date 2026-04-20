# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for radial distribution function computation."""

from collections import namedtuple
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy.testing as npt
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import view
from kups.core.neighborlist import AllDenseNearestNeighborList, Edges
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.observables.radial_distribution_function import (
    RadialDistributionFunction,
    radial_distribution_function,
)

from ..clear_cache import clear_cache  # noqa: F401

_SystemData = namedtuple("_SystemData", ["unitcell", "cutoff"])


def _make_particles(
    positions_data: Array, system_ids: Array, num_systems: int
) -> Table[ParticleId, Any]:
    """Helper to create Table particles for tests."""
    _Particles = namedtuple("_Particles", ["positions", "system"])
    system = Index(
        tuple(map(SystemId, range(num_systems))),
        jnp.asarray(system_ids, dtype=int),
    )
    return Table.arange(
        _Particles(positions_data, system),
        label=ParticleId,
    )


def _make_systems(unitcell: UnitCell, rmax: float) -> Table[SystemId, Any]:
    """Helper to create Table systems for tests."""
    n_sys = unitcell.lattice_vectors.shape[0]
    return Table.arange(
        _SystemData(unitcell, jnp.full(n_sys, rmax)),
        label=SystemId,
    )


class RDFTestState(NamedTuple):
    """Test state for RDF calculations."""

    positions: Table[ParticleId, Any]
    systems: Table[SystemId, Any]
    rmax: float
    bins: int


@dataclass
class MockNeighborList:
    """Mock neighbor list returning two reciprocal pairs."""

    def __call__(self, *args, **kwargs) -> Edges[Literal[2]]:
        indices = Index(
            (ParticleId(0), ParticleId(1)),
            jnp.array([[0, 1], [1, 0]]),
        )
        shifts = jnp.zeros((2, 1, 3))
        return Edges(indices, shifts)


@dataclass
class EmptyNeighborList:
    """Mock neighbor list returning no pairs."""

    def __call__(self, *args, **kwargs) -> Edges[Literal[2]]:
        indices = Index((ParticleId(0),), jnp.empty((0, 2), dtype=int))
        shifts = jnp.empty((0, 1, 3))
        return Edges(indices, shifts)


def _box10() -> UnitCell:
    """10x10x10 cubic unit cell."""
    return TriclinicUnitCell.from_matrix((jnp.eye(3) * 10.0)[None])


class TestRadialDistributionFunction:
    """Test radial distribution function computation."""

    def test_two_particles_peak_and_class_interface(self):
        """Two particles 1.0 apart: RDF peak near r=1.0, works with class interface."""
        positions = _make_particles(
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            jnp.array([0, 0]),
            1,
        )
        systems = _make_systems(_box10(), 5.0)
        rmax, bins = 5.0, 50
        dr = rmax / bins

        # Test functional interface
        result = radial_distribution_function(
            positions, systems, rmax, bins, MockNeighborList()
        )
        assert result.shape == (1, bins)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)
        assert result[0, 0] == 0.0

        r_centers = jnp.linspace(dr / 2, rmax - dr / 2, bins)
        bin_at_1 = int(jnp.argmin(jnp.abs(r_centers - 1.0)))
        max_bin = int(jnp.argmax(result[0]))
        assert abs(max_bin - bin_at_1) <= 2

        # Test class interface
        state = RDFTestState(positions=positions, systems=systems, rmax=5.0, bins=50)
        rdf_calc = RadialDistributionFunction(
            positions=view(lambda s: s.positions),  # type: ignore
            systems=view(lambda s: s.systems),  # type: ignore
            rmax=view(lambda s: s.rmax),  # type: ignore
            bins=view(lambda s: s.bins),  # type: ignore
            neighborlist=MockNeighborList(),
        )
        class_result = rdf_calc(jax.random.PRNGKey(42), state)
        assert class_result.shape == (1, 50)
        assert jnp.all(jnp.isfinite(class_result))

        # Test different rmax/bins combinations work
        for test_rmax in [1.0, 10.0]:
            test_systems = _make_systems(_box10(), test_rmax)
            r = radial_distribution_function(
                positions, test_systems, test_rmax, 50, MockNeighborList()
            )
            assert r.shape == (1, 50)

    def test_no_neighbors_gives_zero_rdf(self):
        """Single particle or no pairs produces all-zero RDF."""
        positions = _make_particles(jnp.array([[0.0, 0.0, 0.0]]), jnp.array([0]), 1)
        systems = _make_systems(_box10(), 5.0)

        result = radial_distribution_function(
            positions, systems, 5.0, 50, EmptyNeighborList()
        )
        assert result.shape == (1, 50)
        npt.assert_allclose(result, 0.0, atol=1e-10)

    def test_ideal_gas_normalization(self):
        """Dilute system: RDF values are non-negative and bounded."""
        positions = _make_particles(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [3.0, 3.0, 0.0],
                ]
            ),
            jnp.array([0, 0, 0, 0]),
            1,
        )
        unitcell = TriclinicUnitCell.from_matrix((jnp.eye(3) * 20.0)[None])
        systems = _make_systems(unitcell, 8.0)

        result = radial_distribution_function(
            positions,
            systems,
            8.0,
            40,
            AllDenseNearestNeighborList(
                avg_edges=FixedCapacity(4), avg_image_candidates=FixedCapacity(4)
            ),
        )

        large_r_bins = result[0, -10:]
        assert jnp.all(large_r_bins >= 0.0)
        assert jnp.all(large_r_bins < 10.0)

    def test_multiple_systems(self):
        """Two systems with different pair distances produce peaks at correct r."""
        positions = _make_particles(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            jnp.array([0, 0, 1, 1]),
            2,
        )
        unitcell = TriclinicUnitCell.from_matrix(jnp.tile(jnp.eye(3) * 10.0, (2, 1, 1)))
        systems = _make_systems(unitcell, 5.0)
        rmax, bins = 5.0, 50
        dr = rmax / bins

        @dataclass
        class MultiSystemNeighborList:
            def __call__(self, *args, **kwargs) -> Edges[Literal[2]]:
                indices = Index(
                    tuple(map(ParticleId, range(4))),
                    jnp.array([[0, 1], [1, 0], [2, 3], [3, 2]]),
                )
                shifts = jnp.zeros((4, 1, 3))
                return Edges(indices, shifts)

        result = radial_distribution_function(
            positions, systems, rmax, bins, MultiSystemNeighborList()
        )

        assert result.shape == (2, bins)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)

        r_centers = jnp.linspace(dr / 2, rmax - dr / 2, bins)
        bin_at_1 = int(jnp.argmin(jnp.abs(r_centers - 1.0)))
        bin_at_2 = int(jnp.argmin(jnp.abs(r_centers - 2.0)))
        assert abs(int(jnp.argmax(result[0])) - bin_at_1) <= 2
        assert abs(int(jnp.argmax(result[1])) - bin_at_2) <= 2

    def test_with_real_neighborlist(self):
        """RDF with real neighbor list: peaks at r=1.0 and r=sqrt(2) for square grid."""
        positions = _make_particles(
            jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            jnp.array([0, 0, 0, 0]),
            1,
        )
        unitcell = TriclinicUnitCell.from_matrix(
            jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])[None],
        )
        rmax, bins = 3.0, 30
        dr = rmax / bins
        systems = _make_systems(unitcell, rmax)

        nnlist = AllDenseNearestNeighborList(
            avg_edges=FixedCapacity(4), avg_image_candidates=FixedCapacity(4)
        )
        result = radial_distribution_function(positions, systems, rmax, bins, nnlist)

        assert result.shape == (1, bins)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)
        assert result[0, 0] == 0.0

        r_centers = jnp.linspace(dr / 2, rmax - dr / 2, bins)
        bin_at_1 = int(jnp.argmin(jnp.abs(r_centers - 1.0)))
        bin_at_sqrt2 = int(jnp.argmin(jnp.abs(r_centers - jnp.sqrt(2))))

        rdf = result[0]
        peak_1 = float(jnp.max(rdf[max(0, bin_at_1 - 2) : min(bins, bin_at_1 + 3)]))
        peak_sqrt2 = float(
            jnp.max(rdf[max(0, bin_at_sqrt2 - 2) : min(bins, bin_at_sqrt2 + 3)])
        )
        assert peak_1 > 0, "Should have a peak near r=1.0"
        assert peak_sqrt2 > 0, "Should have a peak near r=sqrt(2)"
