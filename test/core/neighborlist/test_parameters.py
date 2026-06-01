# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.parameters``."""

import jax.numpy as jnp

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.table import Table
from kups.core.neighborlist.parameters import (
    UniversalNeighborlistParameters,
    _estimate_avg_num_edges,
)
from kups.core.typing import SystemId

from ._builders import cutoff_table, make_systems


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class TestEstimateAvgNumEdges:
    def test_density_estimate_rounds_to_power_of_two(self):
        # density = 100/1000 = 0.1; sphere(2) = 4/3*pi*8 ≈ 33.5; ≈ 3.35 -> 4.
        est = _estimate_avg_num_edges(num_particles=100, volume=1000.0, cutoff=2.0)
        assert est == 4

    def test_scales_with_density(self):
        sparse = _estimate_avg_num_edges(10, 1000.0, 2.0)
        dense = _estimate_avg_num_edges(1000, 1000.0, 2.0)
        assert dense > sparse
        assert _is_power_of_two(sparse) and _is_power_of_two(dense)


class TestUniversalNeighborlistParametersEstimate:
    def _single_system(self, n_particles=100, cutoff=2.0, box=10.0):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * box))
        systems, _ = make_systems(cell, jnp.array([cutoff]))
        particles_per_system = Table((SystemId(0),), jnp.array([n_particles]))
        cutoffs = cutoff_table(jnp.array([cutoff]))
        return particles_per_system, systems, cutoffs

    def test_estimate_single_system(self):
        ppc, systems, cutoffs = self._single_system()
        params = UniversalNeighborlistParameters.estimate(ppc, systems, cutoffs)

        assert isinstance(params, UniversalNeighborlistParameters)
        # 10 Å box, cutoff 2 -> 5 bins/axis -> 125 cells.
        assert params.cells == 125
        # candidates ≈ 100/125*27 ≈ 21.6 -> next power of two = 32.
        assert params.avg_candidates == 32
        assert params.avg_image_candidates == params.avg_candidates
        # edges ≈ 3.35 -> 4.
        assert params.avg_edges == 4

    def test_all_fields_positive_powers_of_two(self):
        ppc, systems, cutoffs = self._single_system(n_particles=512)
        params = UniversalNeighborlistParameters.estimate(ppc, systems, cutoffs)
        assert _is_power_of_two(params.avg_candidates)
        assert _is_power_of_two(params.avg_edges)
        assert params.cells > 0

    def test_estimate_multi_system(self):
        cell = PeriodicCell(
            TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(2, axis=0) * 10.0)
        )
        systems, _ = make_systems(cell, jnp.array([2.0, 2.0]))
        ppc = Table((SystemId(0), SystemId(1)), jnp.array([100, 200]))
        cutoffs = cutoff_table(jnp.array([2.0, 2.0]))
        params = UniversalNeighborlistParameters.estimate(ppc, systems, cutoffs)
        assert params.cells == 125
        assert params.avg_edges > 0
        assert params.avg_candidates > 0
