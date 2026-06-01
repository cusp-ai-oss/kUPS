# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.all_connected``.

``all_connected_neighborlist`` connects every pair sharing an inclusion segment
with differing exclusion ids, ignoring distance. Used by Ewald summation to
enumerate the real-space exclusion list.
"""

import jax.numpy as jnp
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.all_connected import (
    InclusionGroupSelector,
    all_connected_neighborlist,
)

from ._builders import make_lh, make_pipeline_ctx, make_systems, valid_edge_set


def _systems(n=1, box=100.0):
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(n, 0) * box))
    systems, _ = make_systems(cell, jnp.full((n,), 10.0))
    return systems


class TestInclusionGroupSelector:
    def test_pairs_every_member_of_a_segment(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int), inclusion_max_count=3)
        ctx = make_pipeline_ctx(lh)
        batch = InclusionGroupSelector(capacity=FixedCapacity(9))(ctx)
        pairs = {
            (a, b)
            for a, b in zip(
                batch.lh_idx.indices.tolist(), batch.rh_idx.indices.tolist()
            )
            if a < 3 and b < 3
        }
        # All ordered pairs within the single inclusion segment, incl. self.
        assert pairs == {(a, b) for a in range(3) for b in range(3)}
        assert bool(batch.is_minimum_image.all())


class TestAllConnectedNeighborlist:
    def test_connects_all_distinct_exclusion_pairs(self):
        lh = make_lh(
            jnp.zeros((3, 3)),
            jnp.zeros(3, dtype=int),
            exclusion_ids=jnp.array([0, 1, 2]),
            inclusion_max_count=3,
        )
        edges = all_connected_neighborlist(lh, _systems())
        assert valid_edge_set(edges, 3) == {
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        }

    def test_shared_exclusion_segment_is_not_connected(self):
        # Particles 0 and 1 share exclusion id 0; only pairs involving 2 survive.
        lh = make_lh(
            jnp.zeros((3, 3)),
            jnp.zeros(3, dtype=int),
            exclusion_ids=jnp.array([0, 0, 1]),
            inclusion_max_count=3,
        )
        edges = all_connected_neighborlist(lh, _systems())
        assert valid_edge_set(edges, 3) == {(0, 2), (2, 0), (1, 2), (2, 1)}

    def test_separate_inclusion_segments_do_not_connect(self):
        lh = make_lh(
            jnp.zeros((4, 3)),
            jnp.array([0, 0, 1, 1]),
            exclusion_ids=jnp.array([0, 1, 2, 3]),
            inclusion_max_count=2,
        )
        edges = all_connected_neighborlist(lh, _systems(n=2))
        # Only within-segment pairs: {0,1} and {2,3}.
        assert valid_edge_set(edges, 4) == {(0, 1), (1, 0), (2, 3), (3, 2)}

    def test_requires_inclusion_max_count(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))  # no max_count
        with pytest.raises(AssertionError, match="max_count must be set"):
            all_connected_neighborlist(lh, _systems())

    def test_for_indices_returns_only_touched_pairs(self):
        lh = make_lh(
            jnp.zeros((3, 3)),
            jnp.zeros(3, dtype=int),
            exclusion_ids=jnp.array([0, 1, 2]),
            inclusion_max_count=3,
        )
        edges = all_connected_neighborlist(
            lh, _systems(), for_indices=Index(lh.keys, jnp.array([0]))
        )
        # Every surviving edge must touch particle 0.
        valid = valid_edge_set(edges, 3)
        assert valid == {(0, 1), (1, 0), (0, 2), (2, 0)}
