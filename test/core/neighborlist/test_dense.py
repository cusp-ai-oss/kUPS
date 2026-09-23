# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.dense`` selector and constructors."""

import jax.numpy as jnp
import pytest

from kups.core.capacity import FixedCapacity, LensCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.neighborlist.dense import DenseNearestNeighborList, _dense_subselect
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.typing import ParticleId, SystemId

from ._builders import (
    EvalState,
    SamplePoints,
    cutoff_table,
    make_lh,
    make_systems,
    systems_from_lvecs,
    valid_edge_set,
)


class TestDenseSubselect:
    def test_pairs_never_cross_systems(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.array([0, 0, 1, 1]))
        systems, _ = make_systems(
            PeriodicCell(
                TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(2, 0) * 10.0)
            ),
            jnp.array([1.0, 1.0]),
        )
        candidates = _dense_subselect(lh, lh, systems, FixedCapacity(16))
        sys_ids = lh.data.system.indices
        for a, b in zip(
            candidates.key_idx.indices.tolist(), candidates.query_idx.indices.tolist()
        ):
            if a < 4 and b < 4:
                assert sys_ids[a] == sys_ids[b]

    def test_self_pairs_present(self):
        lh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0)),
            jnp.array([1.0]),
        )
        candidates = _dense_subselect(lh, lh, systems, FixedCapacity(8))
        pairs = set(
            zip(
                candidates.key_idx.indices.tolist(),
                candidates.query_idx.indices.tolist(),
            )
        )
        # Single system of 2 particles -> all 4 ordered pairs including self.
        assert {(0, 0), (0, 1), (1, 0), (1, 1)} <= pairs


class TestFromState:
    def test_from_state_builds_lens_capacities_and_carries_cutoffs(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0)),
            jnp.array([2.0]),
        )
        params = UniversalNeighborlistParameters(
            avg_edges=16, avg_candidates=32, avg_image_candidates=32, cells=64
        )
        state = EvalState(particles=lh, systems=systems, neighborlist_params=params)
        cutoffs = cutoff_table(jnp.array([2.5]))

        nl = DenseNearestNeighborList.from_state(state, cutoffs)

        assert isinstance(nl.avg_candidates, LensCapacity)
        assert isinstance(nl.avg_edges, LensCapacity)
        assert int(nl.avg_candidates.size) == 32
        assert int(nl.avg_edges.size) == 16
        assert float(nl.cutoffs.data[0]) == 2.5


class TestBipartiteQueryKeys:
    @pytest.mark.parametrize("query_system", [0, 1])
    def test_query_meets_its_own_system(self, query_system: int) -> None:
        """A query ``Index`` spanning only some systems is paired by key, not by position.

        ``Index.new`` compacts keys to the systems that occur, so a lone query in system 1
        carries ``keys == (1,)`` and ``indices == [0]``; by position it would meet system 0.
        """
        lh = make_lh(jnp.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]), jnp.array([0, 1]))
        systems, cutoffs = systems_from_lvecs(
            jnp.eye(3)[None] * 20.0, jnp.array([2.0, 2.0])
        )
        system = Index.new([SystemId(query_system)])
        queries = Table(
            (ParticleId(0),),
            SamplePoints(
                positions=jnp.array([[5.0, 5.0, 5.0]]),
                system=system,
                inclusion=system,
                exclusion=Index.integer(jnp.array([2])),
            ),
        )
        nl = DenseNearestNeighborList(
            FixedCapacity(8), FixedCapacity(8), FixedCapacity(8), cutoffs
        )
        assert valid_edge_set(nl(lh, systems, queries=queries), 2) == {
            (query_system, 0)
        }
