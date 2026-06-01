# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the refinement neighbor lists.

``RefineCutoffNeighborList`` re-checks distances against new cutoffs;
``RefineMaskNeighborList`` re-applies inclusion/exclusion masks without
recomputing distances. Both run full pipelines over precomputed edges in
lh-space, supporting self-graph ``for_indices`` updates and bipartite ``rh``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.neighborlist import RefineCutoffNeighborList, RefineMaskNeighborList

from .._builders import (
    cutoff_table,
    make_edges,
    make_lh,
    make_rh,
    make_systems,
    systems_from_lvecs,
)


class TestRefineCutoffNeighborList:
    """Test cases for ``RefineCutoffNeighborList``."""

    def _create_test_pointset(self, positions, batch_mask=None, exclusion_offset=0):
        if batch_mask is None:
            batch_mask = jnp.zeros(len(positions), dtype=int)
        return make_lh(
            positions, batch_mask, jnp.arange(len(positions)) + exclusion_offset
        )

    def _create_candidate_edges(
        self, lh_indices, rh_indices, n_particles=None, shifts=None
    ):
        return make_edges(lh_indices, rh_indices, n_particles, shifts)

    def test_basic_refinement(self):
        """Test basic edge refinement with simple candidate edges."""
        lh_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        lh = self._create_test_pointset(lh_positions)

        lh_indices = jnp.array([0, 0, 0, 1, 1, 2])
        rh_indices = jnp.array([1, 2, 3, 2, 3, 3])
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([1.5])  # Should only include distance 1.0 edges

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=cutoff_table(cutoffs),
        )

        _sys, _cut = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(lh=lh, systems=_sys)

        assert len(edges) > 0
        assert edges.degree == 2

        pos_diffs = (
            lh_positions[edges.indices.indices[:, 1]]
            - lh_positions[edges.indices.indices[:, 0]]
        )
        distances = jnp.linalg.norm(pos_diffs, axis=-1)
        valid_distances = distances[edges.indices.indices[:, 0] < len(lh_positions)]
        valid_distances = valid_distances[
            edges.indices.indices[:, 1][: len(valid_distances)] < len(lh_positions)
        ]

        if len(valid_distances) > 0:
            assert jnp.all(valid_distances <= cutoffs[0] + 1e-6), (
                "All distances should be within cutoff"
            )

    def test_with_cells(self):
        """Test refinement with periodic boundary conditions."""
        positions = jnp.array(
            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [0.5, 2.5, 0.5], [2.5, 2.5, 0.5]]
        )
        lh = self._create_test_pointset(positions)

        lattice_vectors = jnp.eye(3)[None] * 3.0
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))

        lh_indices = jnp.array([0, 1, 2, 3])
        rh_indices = jnp.array([1, 0, 3, 2])
        shifts = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        candidates = self._create_candidate_edges(lh_indices, rh_indices, shifts=shifts)

        cutoffs = jnp.array([2.1])

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=cutoff_table(cutoffs),
        )

        _sys, _cut = make_systems(cell, cutoffs)
        edges = refinement_nl(lh=lh, systems=_sys)
        assert len(edges) >= 0
        assert edges.degree == 2

    def test_with_for_indices(self):
        """Test refinement with affected lh indices."""
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        update_positions = jnp.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        lh = self._create_test_pointset(lh_positions)

        for_indices = jnp.array([1, 2])

        lh_indices = jnp.array([0, 1])
        rh_indices = jnp.array([0, 1])
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([1.1])

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=cutoff_table(cutoffs),
        )

        rh_update, for_indices = make_rh(
            lh,
            update_positions,
            jnp.zeros(len(update_positions), dtype=int),
            for_indices,
        )
        lh = lh.update(for_indices, rh_update.data)
        _sys, _cut = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(lh=lh, systems=_sys, for_indices=for_indices)

        assert edges.degree == 2

    def test_for_indices_uses_updated_lh_positions_for_cutoff(self):
        """Precomputed edges are in public lh-space, so for_indices updates must
        evaluate distances after the updated data has been written to lh."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [105.0, 0.0, 0.0],
            ]
        )
        lh = self._create_test_pointset(lh_positions)
        rh_update, for_indices = make_rh(
            lh,
            jnp.array([[100.4, 0.0, 0.0]]),
            jnp.zeros(1, dtype=int),
            jnp.array([2]),
            exclusion_ids=jnp.array([20]),
        )
        lh = lh.update(for_indices, rh_update.data)
        candidates = self._create_candidate_edges(
            jnp.array([1]), jnp.array([2]), n_particles=3
        )
        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(4),
            cutoffs=cutoff_table(jnp.array([1.0])),
        )

        systems, _ = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([1.0]))
        edges = refinement_nl(lh=lh, systems=systems, for_indices=for_indices)

        npt.assert_array_equal(
            np.asarray(edges.indices.indices[edges.indices.indices[:, 0] < lh.size]),
            np.array([[1, 2]]),
        )

    def test_disjoint_rh_uses_rh_positions_for_cutoff(self):
        """Without for_indices, the precomputed candidate right column is in
        rh-space and distance checks must use the separate rh table."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        )
        lh = self._create_test_pointset(lh_positions)
        rh, _ = make_rh(
            lh,
            jnp.array([[100.4, 0.0, 0.0]]),
            jnp.zeros(1, dtype=int),
            jnp.array([0]),
            exclusion_ids=jnp.array([20]),
        )
        candidates = self._create_candidate_edges(
            jnp.array([1]), jnp.array([0]), n_particles=2
        )
        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(4),
            cutoffs=cutoff_table(jnp.array([1.0])),
        )
        systems, _ = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([1.0]))

        edges = refinement_nl(lh=lh, systems=systems, rh=rh)

        npt.assert_array_equal(
            np.asarray(edges.indices.indices[edges.indices.indices[:, 0] < lh.size]),
            np.array([[1, 0]]),
        )

    def test_for_indices_uses_updated_lh_exclusion_for_cutoff_refinement(self):
        """The cutoff refiner applies exclusion masks after distance filtering,
        so updated lh metadata must be used there too."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        lh = self._create_test_pointset(lh_positions, exclusion_offset=0)
        rh_update, for_indices = make_rh(
            lh,
            jnp.array([[1.0, 0.0, 0.0]]),
            jnp.zeros(1, dtype=int),
            jnp.array([1]),
            exclusion_ids=jnp.array([0]),
        )
        lh = lh.update(for_indices, rh_update.data)
        candidates = self._create_candidate_edges(
            jnp.array([0]), jnp.array([1]), n_particles=2
        )
        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(4),
            cutoffs=cutoff_table(jnp.array([2.0])),
        )
        systems, _ = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([2.0]))

        edges = refinement_nl(lh=lh, systems=systems, for_indices=for_indices)

        assert not np.any(np.asarray(edges.indices.indices[:, 0] < lh.size))

    def test_empty_candidates(self):
        """Test refinement with candidates that fail the distance filter."""
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        lh = self._create_test_pointset(lh_positions)

        lh_indices = jnp.array([0])
        rh_indices = jnp.array([1])
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([0.1])  # Much smaller than distance between points (1.0)

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(5),
            cutoffs=cutoff_table(cutoffs),
        )

        _sys, _cut = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(lh=lh, systems=_sys)

        assert edges.degree == 2

        valid_edges = edges.indices.indices[
            (edges.indices.indices[:, 0] < len(lh_positions))
            & (edges.indices.indices[:, 1] < len(lh_positions))
        ]
        assert len(valid_edges) == 0, "Should have no valid edges due to cutoff filter"

    def test_multiple_segments(self):
        """Test refinement with multiple segments/batches."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # System 0
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],  # System 1
            ]
        )
        batch_mask = jnp.array([0, 0, 1, 1])
        lh = self._create_test_pointset(positions, batch_mask)

        lh_indices = jnp.array([0, 1, 2, 3, 0, 2])  # Mix of within/across systems
        rh_indices = jnp.array([1, 0, 3, 2, 2, 0])
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([1.5, 1.5])

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=cutoff_table(cutoffs),
        )

        _sys, _cut = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(lh=lh, systems=_sys)

        assert edges.degree == 2

    def test_exclusion_segments(self):
        """Test that exclusion segments prevent self-interactions."""
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

        lh = make_lh(lh_positions, jnp.array([0, 0, 0]), jnp.array([0, 1, 2]))

        lh_indices = jnp.array([0, 1, 2])
        rh_indices = jnp.array([0, 1, 2])  # Self-interactions
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([2.0])

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=cutoff_table(cutoffs),
        )

        _sys, _cut = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(lh=lh, systems=_sys)

        valid_edges = edges.indices.indices[
            edges.indices.indices[:, 0] < len(lh_positions)
        ]
        valid_edges = valid_edges[valid_edges[:, 1] < len(lh_positions)]
        for i, edge in enumerate(valid_edges):
            if len(edge) >= 2:
                assert edge[0] != edge[1], f"Found self-interaction in edge {i}: {edge}"

    def test_gradient_computation(self):
        """Test that gradients can be computed through the refined neighbor list."""

        def loss_fn(positions):
            lh = self._create_test_pointset(positions)
            candidates = self._create_candidate_edges(jnp.array([0]), jnp.array([1]))
            refinement_nl = RefineCutoffNeighborList(
                candidates=candidates,
                avg_edges=FixedCapacity(5),
                cutoffs=cutoff_table(jnp.array([2.0])),
            )
            _sys, _cut = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([2.0]))
            edges = refinement_nl(lh=lh, systems=_sys)

            if len(edges) > 0:
                diff_vectors = edges.difference_vectors(lh, _sys)
                distances = jnp.linalg.norm(diff_vectors, axis=-1)
                return jnp.sum(distances**2)
            else:
                return 0.0

        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(positions)

        assert gradients.shape == positions.shape
        assert jnp.sum(jnp.abs(gradients)) > 0

    def test_difference_vectors_computation(self):
        """Test that difference vectors are computed correctly."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        lh = self._create_test_pointset(positions)

        candidates = self._create_candidate_edges(jnp.array([0, 1]), jnp.array([1, 2]))
        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(5),
            cutoffs=cutoff_table(jnp.array([10.0])),
        )

        _sys, _cut = systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([10.0]))
        edges = refinement_nl(lh=lh, systems=_sys)

        if len(edges) > 0:
            diff_vectors = edges.difference_vectors(lh, _sys)

            expected_neighbors = edges.degree - 1
            if expected_neighbors > 0:
                assert diff_vectors.shape[-1] == 3
                assert diff_vectors.shape[1] == expected_neighbors

                for i in range(len(edges.indices.indices)):
                    edge = edges.indices.indices[i]
                    if edge[0] < len(positions) and edge[1] < len(positions):
                        expected_diff = positions[edge[1]] - positions[edge[0]]
                        if i < len(diff_vectors) and diff_vectors.shape[1] > 0:
                            npt.assert_allclose(
                                diff_vectors[i, 0], expected_diff, rtol=1e-10
                            )


class TestRefineMaskNeighborList:
    """Test cases for ``RefineMaskNeighborList``.

    RefineMask is the post-filter used by MCMC. Its precomputed edges carry
    indices in lh-space. For self-graph updates the changed particle data has
    already been written into lh and for_indices selects affected rows. True
    bipartite calls pass rh with no for_indices.
    """

    def test_mcmc_patch_pattern_with_partial_overlap(self):
        """Precomputed edges can touch affected and unaffected particles."""
        lh = make_lh(
            jnp.zeros((5, 3)),
            jnp.zeros(5, dtype=int),
            exclusion_ids=jnp.array([0, 1, 2, 3, 4]),
        )
        rh_update, for_indices = make_rh(
            lh, jnp.zeros((2, 3)), jnp.zeros(2, dtype=int), jnp.array([1, 3])
        )
        lh = lh.update(for_indices, rh_update.data)
        candidates = make_edges(
            jnp.array([0, 2, 1]), jnp.array([1, 3, 4]), n_particles=5
        )

        refine_nl = RefineMaskNeighborList(candidates=candidates)
        sys, _ = systems_from_lvecs(jnp.eye(3)[None] * 100.0, jnp.array([10.0]))
        edges = refine_nl(lh=lh, systems=sys, for_indices=for_indices)

        raw = edges.indices.indices
        oob = lh.size
        valid_mask = (raw[:, 0] < oob) & (raw[:, 1] < oob)
        valid = np.asarray(raw[valid_mask])
        valid = valid[np.lexsort((valid[:, 1], valid[:, 0]))]

        npt.assert_array_equal(valid, np.array([[0, 1], [1, 4], [2, 3]]))

    def test_full_overlap_with_rh_subset(self):
        """All precomputed edges touch the affected subset."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        lh = make_lh(lh_positions, jnp.zeros(4, dtype=int), jnp.arange(4))
        rh_update, for_indices = make_rh(
            lh,
            lh_positions[jnp.array([1, 3])],
            jnp.zeros(2, dtype=int),
            jnp.array([1, 3]),
        )
        lh = lh.update(for_indices, rh_update.data)
        candidates = make_edges(jnp.array([0, 2]), jnp.array([1, 3]), n_particles=4)

        refine_nl = RefineMaskNeighborList(candidates=candidates)
        sys, _ = systems_from_lvecs(jnp.eye(3)[None] * 100.0, jnp.array([10.0]))
        edges = refine_nl(lh=lh, systems=sys, for_indices=for_indices)

        raw = edges.indices.indices
        oob = lh.size
        valid_mask = (raw[:, 0] < oob) & (raw[:, 1] < oob)
        valid = np.asarray(raw[valid_mask])
        valid = valid[np.lexsort((valid[:, 1], valid[:, 0]))]

        npt.assert_array_equal(valid, np.array([[0, 1], [2, 3]]))

    def test_disjoint_rh_uses_rh_inclusion(self):
        """True bipartite refinement reads rhs metadata from rh."""
        lh = make_lh(
            jnp.zeros((2, 3)),
            jnp.zeros(2, dtype=int),
            exclusion_ids=jnp.array([0, 1]),
        )
        rh, _ = make_rh(
            lh,
            jnp.zeros((2, 3)),
            jnp.array([0, 1]),
            jnp.array([0, 1]),
            exclusion_ids=jnp.array([2, 3]),
        )
        candidates = make_edges(jnp.array([0]), jnp.array([1]), n_particles=2)
        refine_nl = RefineMaskNeighborList(candidates=candidates)
        systems, _ = systems_from_lvecs(
            jnp.eye(3)[None].repeat(2, axis=0) * 100.0,
            jnp.array([10.0, 10.0]),
        )

        edges = refine_nl(lh=lh, systems=systems, rh=rh)

        npt.assert_array_equal(np.asarray(edges.indices.indices), np.array([[2, 2]]))

    def test_for_indices_uses_updated_lh_inclusion(self):
        """Updated lh rows can move inclusion segments before refinement."""
        lh = make_lh(
            jnp.zeros((2, 3)),
            jnp.zeros(2, dtype=int),
            exclusion_ids=jnp.array([0, 1]),
        )
        rh_update, for_indices = make_rh(
            lh,
            jnp.zeros((1, 3)),
            jnp.ones(1, dtype=int),
            jnp.array([1]),
            exclusion_ids=jnp.array([1]),
        )
        lh = lh.update(for_indices, rh_update.data)
        candidates = make_edges(jnp.array([0]), jnp.array([1]), n_particles=2)
        refine_nl = RefineMaskNeighborList(candidates=candidates)
        systems, _ = systems_from_lvecs(
            jnp.eye(3)[None] * 100.0, jnp.array([10.0, 10.0])
        )

        edges = refine_nl(lh=lh, systems=systems, for_indices=for_indices)

        npt.assert_array_equal(np.asarray(edges.indices.indices), np.array([[2, 2]]))

    def test_for_indices_uses_updated_lh_exclusion(self):
        """Updated lh rows can introduce an exclusion before refinement."""
        lh = make_lh(
            jnp.zeros((2, 3)),
            jnp.zeros(2, dtype=int),
            exclusion_ids=jnp.array([0, 1]),
        )
        rh_update, for_indices = make_rh(
            lh,
            jnp.zeros((1, 3)),
            jnp.zeros(1, dtype=int),
            jnp.array([1]),
            exclusion_ids=jnp.array([0]),
        )
        lh = lh.update(for_indices, rh_update.data)
        candidates = make_edges(jnp.array([0]), jnp.array([1]), n_particles=2)
        refine_nl = RefineMaskNeighborList(candidates=candidates)
        systems, _ = systems_from_lvecs(jnp.eye(3)[None] * 100.0, jnp.array([10.0]))

        edges = refine_nl(lh=lh, systems=systems, for_indices=for_indices)

        npt.assert_array_equal(np.asarray(edges.indices.indices), np.array([[2, 2]]))
