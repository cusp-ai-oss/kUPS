# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for ``neighborlist_changes`` (MCMC / patch-style moves).

A single combined query must reproduce the added/removed edge sets that two
separate before/after neighbor list calls would produce, while respecting
system boundaries and the compaction fraction.
"""

from collections import Counter
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.wrappers import WithIndices
from kups.core.neighborlist import DenseNearestNeighborList, neighborlist_changes
from kups.core.neighborlist.edges import Edges

from .._builders import (
    call_with_retry,
    make_lh,
    make_rh,
    make_systems,
    valid_edge_set,
)

_EdgeKey = tuple[int, int, tuple[int, ...]]


def _edge_counts(edges: Edges[Literal[2]], n_particles: int) -> Counter[_EdgeKey]:
    """Count occurrences of each ``(i, j, shift)`` edge with both ends in range."""
    raw = np.asarray(edges.indices.indices)
    shifts = np.asarray(jnp.round(edges.shifts[:, 0, :]).astype(int))
    counts: Counter[_EdgeKey] = Counter()
    for (a, b), s in zip(raw, shifts):
        if a < n_particles and b < n_particles:
            counts[(int(a), int(b), tuple(int(x) for x in s))] += 1
    return counts


class TestNeighborlistChanges:
    """Tests for the single-call ``neighborlist_changes`` utility."""

    @staticmethod
    def _make_nl(cutoff, capacity=32):
        return DenseNearestNeighborList(
            avg_candidates=FixedCapacity(capacity),
            avg_edges=FixedCapacity(capacity),
            avg_image_candidates=FixedCapacity(capacity),
            cutoff=cutoff,
        )

    def test_matches_separate_calls(self):
        """Combined call produces same edge sets as two separate calls."""
        N, M = 10, 3
        key = jax.random.key(42)
        k1, k2, k3 = jax.random.split(key, 3)

        positions = jax.random.uniform(k1, (N, 3), minval=0.0, maxval=9.0)
        changed_idx = jax.random.choice(k2, N, shape=(M,), replace=False)
        new_positions = jax.random.uniform(k3, (M, 3), minval=0.0, maxval=9.0)

        batch = jnp.zeros(N, dtype=int)
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([3.0]))
        nl = self._make_nl(3.0)

        # --- reference: two separate calls ---
        full_new_pos = positions.at[changed_idx].set(new_positions)
        lh_after = make_lh(full_new_pos, batch)
        queried_keys_after = Index(lh_after.keys, changed_idx)
        ref_after = call_with_retry(nl, lh_after, systems, queried_keys_after)

        lh_before = make_lh(positions, batch)
        queried_keys_before = Index(lh_before.keys, changed_idx)
        ref_removed = call_with_retry(nl, lh_before, systems, queried_keys_before)

        # --- combined call ---
        lh = make_lh(positions, batch)
        rh_table, queried_keys = make_rh(
            lh, new_positions, jnp.zeros(M, dtype=int), changed_idx
        )
        rh_with_indices = WithIndices(queried_keys, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems)

        added_set = valid_edge_set(result.added, N)
        removed_set = valid_edge_set(result.removed, N)
        ref_after_set = valid_edge_set(ref_after, N)
        ref_removed_set = valid_edge_set(ref_removed, N)

        assert added_set == ref_after_set, (
            f"added mismatch:\n  extra={added_set - ref_after_set}\n"
            f"  missing={ref_after_set - added_set}"
        )
        assert removed_set == ref_removed_set, (
            f"removed mismatch:\n  extra={removed_set - ref_removed_set}\n"
            f"  missing={ref_removed_set - removed_set}"
        )

    def test_single_particle_change(self):
        """Changing a single particle produces correct before/after edges."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )
        new_pos = jnp.array([[4.5, 0.0, 0.0]])  # move particle 1 near particle 2
        changed_idx = jnp.array([1])

        batch = jnp.zeros(3, dtype=int)
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([1.5]))
        nl = self._make_nl(1.5)

        lh = make_lh(positions, batch)
        rh_table, queried_keys = make_rh(
            lh, new_pos, jnp.zeros(1, dtype=int), changed_idx
        )
        rh_with_indices = WithIndices(queried_keys, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems)

        removed = valid_edge_set(result.removed, 3)
        added = valid_edge_set(result.added, 3)

        assert (0, 1) in removed
        assert (1, 0) in removed
        assert (1, 2) in added
        assert (2, 1) in added
        assert (1, 2) not in removed
        assert (0, 1) not in added

    def test_multi_system(self):
        """Changes respect system boundaries in multi-system setups."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # system 0
                [5.0, 0.0, 0.0],  # system 0
                [0.0, 0.0, 0.0],  # system 1
                [5.0, 0.0, 0.0],  # system 1
            ]
        )
        new_pos = jnp.array([[0.5, 0.0, 0.0]])
        changed_idx = jnp.array([1])

        batch = jnp.array([0, 0, 1, 1])
        cell = PeriodicCell(
            TriclinicFrame.from_matrix(
                jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 10.0])
            )
        )
        systems, _ = make_systems(cell, jnp.array([1.5, 1.5]))
        nl = self._make_nl(1.5)

        lh = make_lh(positions, batch)
        rh_table, queried_keys = make_rh(
            lh, new_pos, jnp.zeros(1, dtype=int), changed_idx
        )
        rh_with_indices = WithIndices(queried_keys, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems)

        added = valid_edge_set(result.added, 4)
        assert (0, 1) in added
        assert (1, 0) in added
        assert (1, 2) not in added
        assert (2, 1) not in added

    def test_noop_change_in_batch_does_not_duplicate_edges(self):
        """A no-op per-system change must not duplicate another system's edges.

        In batched GCMC one system often contributes an empty change while
        another moves a particle. The empty change's slot is the out-of-bounds
        sentinel; it must not alias an appended particle and cause that
        particle's edges to be emitted more than once.
        """
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # 0 sys0
                [1.0, 0.0, 0.0],  # 1 sys0 (moves next to particle 2)
                [5.0, 0.0, 0.0],  # 2 sys0
                [2.0, 0.0, 0.0],  # 3 sys1
            ]
        )
        batch = jnp.array([0, 0, 0, 1])
        N = 4
        # rh row 0: sys0 real move; rh row 1: sys1 no-op (queried slot == N == OOB).
        new_pos = jnp.array([[4.5, 0.0, 0.0], [8.0, 0.0, 0.0]])
        rh_batch = jnp.array([0, 1])
        changed_idx = jnp.array([1, N])

        cell = PeriodicCell(
            TriclinicFrame.from_matrix(
                jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 10.0])
            )
        )
        systems, cutoffs = make_systems(cell, jnp.array([1.5, 1.5]))
        nl = self._make_nl(cutoffs)

        lh = make_lh(positions, batch)
        rh_table, queried_keys = make_rh(lh, new_pos, rh_batch, changed_idx)
        result = neighborlist_changes(
            nl, lh, WithIndices(queried_keys, rh_table), systems
        )

        counts = _edge_counts(result.added, N)
        assert (1, 2, (0, 0, 0)) in counts, "expected the moved particle's new edge"
        duplicates = {e: c for e, c in counts.items() if c > 1}
        assert not duplicates, f"duplicated added edges: {duplicates}"

    @pytest.mark.parametrize("compaction", [0.5, 0.75, 1.0])
    def test_compaction(self, compaction: float):
        """Different compaction fractions produce identical edge sets."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        new_pos = jnp.array([[4.5, 0.0, 0.0]])
        changed_idx = jnp.array([1])

        batch = jnp.zeros(3, dtype=int)
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([1.5]))
        nl = self._make_nl(1.5)

        lh = make_lh(positions, batch)
        rh_table, queried_keys = make_rh(
            lh, new_pos, jnp.zeros(1, dtype=int), changed_idx
        )
        rh_with_indices = WithIndices(queried_keys, rh_table)
        result = neighborlist_changes(
            nl, lh, rh_with_indices, systems, compaction=compaction
        )

        removed = valid_edge_set(result.removed, 3)
        added = valid_edge_set(result.added, 3)
        assert (0, 1) in removed and (1, 0) in removed
        assert (1, 2) in added and (2, 1) in added

    def test_random_large(self):
        """Stress test with random positions and multiple changed particles."""
        N, M = 20, 5
        key = jax.random.key(123)
        k1, k2, k3 = jax.random.split(key, 3)

        positions = jax.random.uniform(k1, (N, 3), minval=0.0, maxval=9.0)
        changed_idx = jax.random.choice(k2, N, shape=(M,), replace=False)
        new_positions = jax.random.uniform(k3, (M, 3), minval=0.0, maxval=9.0)

        batch = jnp.zeros(N, dtype=int)
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([3.0]))
        nl = self._make_nl(3.0, capacity=64)

        full_new_pos = positions.at[changed_idx].set(new_positions)
        lh_after = make_lh(full_new_pos, batch)
        queried_keys_after = Index(lh_after.keys, changed_idx)
        ref_after = call_with_retry(nl, lh_after, systems, queried_keys_after)
        lh_before = make_lh(positions, batch)
        queried_keys_before = Index(lh_before.keys, changed_idx)
        ref_removed = call_with_retry(nl, lh_before, systems, queried_keys_before)

        lh = make_lh(positions, batch)
        rh_table, queried_keys = make_rh(
            lh, new_positions, jnp.zeros(M, dtype=int), changed_idx
        )
        result = neighborlist_changes(
            nl, lh, WithIndices(queried_keys, rh_table), systems
        )

        assert valid_edge_set(result.added, N) == valid_edge_set(ref_after, N)
        assert valid_edge_set(result.removed, N) == valid_edge_set(ref_removed, N)
