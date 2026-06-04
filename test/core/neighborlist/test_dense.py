# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.dense`` selector and constructors."""

import jax.numpy as jnp

from kups.core.capacity import FixedCapacity, LensCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.neighborlist.dense import DenseNearestNeighborList, _dense_subselect
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters

from ._builders import EvalState, cutoff_table, make_lh, make_systems


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
