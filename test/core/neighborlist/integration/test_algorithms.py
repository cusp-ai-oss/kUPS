# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Full-algorithm correctness suite shared across the cutoff neighbor lists.

The same body runs for ``DenseNearestNeighborList``, ``CellListNeighborList``,
and ``AllDenseNearestNeighborList`` via the ``neighbor_list_impl`` fixture, so
every implementation is held to one behavioural contract: basic search,
capacity management, exact-cutoff handling, periodic images, multi-system
isolation, ``for_indices`` self-graph updates, and agreement with a brute-force
reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import bind
from kups.core.neighborlist import (
    AllDenseNearestNeighborList,
    CellListNeighborList,
    DenseNearestNeighborList,
)
from kups.core.result import as_result_function

from .._builders import make_lh, make_rh, make_systems, systems_from_lvecs


class TestNearestNeighborListImplementations:
    """Test different neighbor list implementations with the same test suite."""

    @pytest.fixture(
        params=[
            {
                "instance_factory": (
                    lambda candidates, edges, cutoffs, image_candidates=None, **kwargs: (
                        DenseNearestNeighborList(
                            avg_candidates=FixedCapacity(candidates),
                            avg_edges=FixedCapacity(edges),
                            avg_image_candidates=FixedCapacity(
                                image_candidates or candidates
                            ),
                            cutoffs=cutoffs,
                        )
                    )
                ),
                "name": "naive",
            },
            {
                "instance_factory": (
                    lambda candidates, edges, cells, cutoffs, image_candidates=None, **kwargs: (
                        CellListNeighborList(
                            avg_candidates=FixedCapacity(candidates),
                            avg_edges=FixedCapacity(edges),
                            cells=FixedCapacity(cells),
                            avg_image_candidates=FixedCapacity(
                                image_candidates or candidates
                            ),
                            cutoffs=cutoffs,
                        )
                    )
                ),
                "name": "cell_list",
            },
            {
                "instance_factory": (
                    lambda candidates, edges, cells, cutoffs, image_candidates=None, **kwargs: (
                        AllDenseNearestNeighborList(
                            avg_edges=FixedCapacity(edges),
                            avg_image_candidates=FixedCapacity(
                                image_candidates or edges
                            ),
                            cutoffs=cutoffs,
                        )
                    )
                ),
                "name": "all_to_all",
            },
        ]
    )
    def neighbor_list_impl(self, request):
        """Different neighbor list implementations with their instance factories."""
        return request.param

    def _run_neighbor_search_test(self, neighbor_list_impl_info, **kwargs):
        """Generic test runner for neighbor search implementations."""
        default_params = {
            "positions": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "batch_mask": jnp.array([0, 0, 0]),
            "cells": jnp.eye(3)[None] * 10.0,
            "cutoffs": jnp.array([1.5]),
            "extras": {"candidates": 9, "edges": 4, "cells": 256},
        }

        params = {**default_params, **kwargs}

        instance_factory = neighbor_list_impl_info["instance_factory"]

        lh = make_lh(
            params["positions"],
            params["batch_mask"],
            jnp.arange(len(params["batch_mask"])),
        )

        for_indices = None
        if (update_positions := params.get("update_positions", None)) is not None:
            for_indices_raw = params.get("for_indices", None)
            assert for_indices_raw is not None, (
                "update_positions requires for_indices in the new API"
            )
            update_batch_mask = params.get("update_batch_mask", params["batch_mask"])
            update, for_indices = make_rh(
                lh, update_positions, update_batch_mask, for_indices_raw
            )
            lh = lh.update(for_indices, update.data)

        if isinstance(params["cells"], Cell):
            systems, cutoffs = make_systems(params["cells"], params["cutoffs"])
        else:
            systems, cutoffs = systems_from_lvecs(params["cells"], params["cutoffs"])
        neighbor_list_instance = instance_factory(cutoffs=cutoffs, **params["extras"])
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            systems=systems,
            for_indices=for_indices,
        )

        return result

    def test_basic_functionality(self, neighbor_list_impl):
        """Test basic neighbor search functionality for any implementation."""
        result = self._run_neighbor_search_test(neighbor_list_impl)
        result.raise_assertion()
        edges = result.value

        # Should find 4 edges: (0,1), (1,0), (1,2), (2,1)
        assert edges.degree == 2

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]

        edge_set = set(tuple(edge.tolist()) for edge in valid_edges)
        expected_edges = {(0, 1), (1, 0), (1, 2), (2, 1)}
        assert expected_edges.issubset(edge_set), (
            f"Missing edges: {expected_edges - edge_set}"
        )

    def test_capacity_management(self, neighbor_list_impl):
        """Test capacity management for any implementation."""
        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=jnp.zeros((10, 3)),
            batch_mask=jnp.zeros(10, dtype=int),
            cutoffs=jnp.array([5.0]),
            capacity=5,
        )

        assert len(result.assertions) > 0

        with pytest.raises(CapacityError):
            result.raise_assertion()

    def test_exact_cutoff_behavior(self, neighbor_list_impl):
        """Test behavior with particles exactly at cutoff distance."""
        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            batch_mask=jnp.array([0, 0]),
            cutoffs=jnp.array([1.0]),
            extras={"candidates": 9, "edges": 4, "cells": 1024},
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 2]
        valid_edges = valid_edges[valid_edges[:, 1] < 2]

        # The implementation uses < cutoff, so particles exactly at cutoff
        # should not be neighbors.
        assert len(valid_edges) == 0, (
            f"Found unexpected edges at exact cutoff: {valid_edges}"
        )

    def test_periodic_boundary_conditions(self, neighbor_list_impl):
        """Test neighbor search with periodic boundary conditions."""
        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=jnp.array([[0.1, 0.0, 0.0], [2.9, 0.0, 0.0]]),
            batch_mask=jnp.array([0, 0]),
            cells=jnp.eye(3)[None] * 3.0,
            cutoffs=jnp.array([1.0]),
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 2]
        assert valid_edges.shape[0] >= 2  # At least bidirectional edge

    def test_multi_batch_isolation(self, neighbor_list_impl):
        """Test that batches are properly isolated."""
        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],  # System 0
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],  # System 1
                ]
            ),
            batch_mask=jnp.array([0, 0, 1, 1]),
            cells=jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
            cutoffs=jnp.array([1.5, 1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 4]

        batch_mask = jnp.array([0, 0, 1, 1])
        for i in range(valid_edges.shape[0]):
            lh_idx, rh_idx = valid_edges[i]
            if lh_idx < 4 and rh_idx < 4:
                assert batch_mask[lh_idx] == batch_mask[rh_idx]

    def test_for_indices_update_positions_basic(self, neighbor_list_impl):
        """Test basic functionality for updated lh positions selected by for_indices."""
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        update_positions = jnp.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        for_indices = jnp.array([2, 0])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            update_positions=update_positions,
            batch_mask=jnp.array([0, 0, 0]),
            update_batch_mask=jnp.array([0, 0]),
            for_indices=for_indices,
            cutoffs=jnp.array([1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]
        edge_set = set(tuple(edge.tolist()) for edge in valid_edges)

        # lh[1] (2,0,0) vs updated lh[2] (1,0,0): dist=1.0<1.5, edge (1,2)
        # lh[1] (2,0,0) vs updated lh[0] (3,0,0): dist=1.0<1.5, edge (1,0)
        assert (1, 2) in edge_set, f"Expected edge (1, 2) not found in {edge_set}"
        assert (2, 1) in edge_set, f"Expected edge (2, 1) not found in {edge_set}"
        assert (1, 0) in edge_set, f"Expected edge (1, 0) not found in {edge_set}"
        assert (0, 1) in edge_set, f"Expected edge (0, 1) not found in {edge_set}"

    def test_for_indices_update_batch_boundaries(self, neighbor_list_impl):
        """Test update batch masks: edges found within systems, blocked across systems."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # system 0, lh[0]
                [5.0, 0.0, 0.0],  # system 0, lh[1]
                [0.0, 0.0, 0.0],  # system 1, lh[2]
                [5.0, 0.0, 0.0],  # system 1, lh[3]
            ]
        )
        lh_batch = jnp.array([0, 0, 1, 1])
        lvecs = jnp.eye(3)[None].repeat(2, axis=0) * 10.0
        cutoffs = jnp.array([1.5, 1.5])

        # Positive case: updates in correct systems, edges should be found
        result_pos = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            update_positions=jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            batch_mask=lh_batch,
            update_batch_mask=jnp.array([0, 1]),
            for_indices=jnp.array([1, 3]),
            cells=lvecs,
            cutoffs=cutoffs,
            capacity=20,
        )
        result_pos.raise_assertion()
        valid_pos = result_pos.value.indices.indices[
            result_pos.value.indices.indices[:, 0] < 4
        ]
        valid_pos = valid_pos[valid_pos[:, 1] < 4]
        assert len(valid_pos) > 0, "Should find edges within same systems"
        for i in range(valid_pos.shape[0]):
            lh_idx, rh_idx = valid_pos[i]
            if lh_idx < 4 and rh_idx < 4:
                assert lh_batch[lh_idx] == lh_batch[rh_idx], (
                    f"Edge ({lh_idx}, {rh_idx}) crosses batch boundary"
                )

        # Negative case: updates in swapped systems, cross-system edges blocked
        result_neg = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            update_positions=jnp.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            batch_mask=lh_batch,
            update_batch_mask=jnp.array([1, 0]),  # swapped
            for_indices=jnp.array([3, 1]),
            cells=lvecs,
            cutoffs=cutoffs,
            capacity=20,
        )
        result_neg.raise_assertion()
        valid_neg = result_neg.value.indices.indices[
            result_neg.value.indices.indices[:, 0] < 4
        ]
        valid_neg = valid_neg[valid_neg[:, 1] < 4]
        for i in range(valid_neg.shape[0]):
            lh_idx, rh_idx = valid_neg[i]
            if lh_idx < 4 and rh_idx < 4:
                assert lh_batch[lh_idx] == lh_batch[rh_idx], (
                    f"Edge ({lh_idx}, {rh_idx}) crosses batch boundary: "
                    f"lhs_batch={lh_batch[lh_idx]}, rhs_batch={lh_batch[rh_idx]}"
                )

    def test_for_indices_update_positions_asymmetric_search(self, neighbor_list_impl):
        """Test affected-index search with a subset of updated lh particles."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        update_positions = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        # update[0] -> lh[0], update[1] -> lh[2]: non-adjacent affected ids
        for_indices = jnp.array([0, 2])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            update_positions=update_positions,
            batch_mask=jnp.array([0, 0, 0]),
            update_batch_mask=jnp.array([0, 0]),
            for_indices=for_indices,
            cutoffs=jnp.array([1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]
        edge_set = set(tuple(edge.tolist()) for edge in valid_edges)

        assert (1, 0) in edge_set, f"Expected edge (1, 0) not found in {edge_set}"
        assert (0, 1) in edge_set, f"Expected edge (0, 1) not found in {edge_set}"
        assert (1, 2) in edge_set, f"Expected edge (1, 2) not found in {edge_set}"
        assert (2, 1) in edge_set, f"Expected edge (2, 1) not found in {edge_set}"

    def test_for_indices_update_for_subsets(self, neighbor_list_impl):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        for_indices = jnp.array([0, 2, 1])
        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=positions,
            update_positions=positions[:3],
            batch_mask=jnp.array([0, 0, 0, 0]),
            update_batch_mask=jnp.array([0, 0, 0]),
            for_indices=for_indices,
            cutoffs=jnp.array([2.5]),
            extras={"candidates": 20, "edges": 12, "cells": 256},
        )
        result.raise_assertion()
        edges = result.value
        sort_idxs = jnp.lexsort(
            [edges.indices.indices[:, 0], edges.indices.indices[:, 1]]
        )
        edges = bind(edges).focus(lambda e: e[sort_idxs]).get()
        relevant_edges = tuple(
            sorted(
                tuple(index_list)
                for index_list in jax.tree.map(
                    lambda x: x[:10], edges.indices.indices
                ).tolist()
                if index_list[0] < 4 and index_list[1] < 4
            )
        )

        assert set(relevant_edges) == set(
            (
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (1, 3),
                (3, 1),
                (2, 3),
                (3, 2),
            )
        )

    def test_for_indices_prevents_self_interaction(self, neighbor_list_impl):
        """Test that self-graph updates still prevent self-interactions."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        for_indices = jnp.array([0, 1])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=positions,
            update_positions=positions,
            batch_mask=jnp.array([0, 0]),
            update_batch_mask=jnp.array([0, 0]),
            for_indices=for_indices,
            cutoffs=jnp.array([1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 2]
        valid_edges = valid_edges[valid_edges[:, 1] < 2]

        assert len(valid_edges) > 0, "Should find edges with for_indices"

        for i in range(valid_edges.shape[0]):
            lh_idx, rh_idx = valid_edges[i]
            assert lh_idx != rh_idx, f"Found self-interaction: ({lh_idx}, {rh_idx})"

    def test_for_indices_update_arguments_combined(self, neighbor_list_impl):
        """Test updated positions, systems, and affected ids together."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # system 0
                [0.0, 0.0, 0.0],  # system 1
                [0.0, 0.0, 0.0],  # system 0
            ]
        )
        update_positions = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        # Affected ids: update[0]->lh[2], update[1]->lh[1], update[2]->lh[0]
        for_indices = jnp.array([2, 1, 0])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            update_positions=update_positions,
            batch_mask=jnp.array([0, 1, 0]),
            update_batch_mask=jnp.array([0, 1, 0]),
            for_indices=for_indices,
            cells=jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
            cutoffs=jnp.array([1.5, 1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]

        lh_batch_mask = jnp.array([0, 1, 0])
        for i in range(valid_edges.shape[0]):
            lh_idx, rh_idx = valid_edges[i]
            if lh_idx < 3 and rh_idx < 3:
                assert lh_batch_mask[lh_idx] == lh_batch_mask[rh_idx], (
                    f"Edge ({lh_idx}, {rh_idx}) violates batch constraint: "
                    f"lhs_batch={lh_batch_mask[lh_idx]}, rhs_batch={lh_batch_mask[rh_idx]}"
                )

    def test_compare_to_naive(self, neighbor_list_impl):
        N = 15
        positions = jax.random.uniform(
            jax.random.key(0), (N, 3), minval=0.0, maxval=10.0
        )
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        cutoff = 3
        instance_factory = neighbor_list_impl["instance_factory"]

        lh = make_lh(positions, jnp.array([0] * N), jnp.arange(N))

        _sys, _cut = make_systems(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            cutoffs=_cut, **{"edges": N, "candidates": N, "cells": 64}
        )
        while (
            result := as_result_function(neighbor_list_instance)(lh, systems=_sys)
        ).failed_assertions:
            neighbor_list_instance = result.fix_or_raise(neighbor_list_instance)
        result.raise_assertion()
        actual_edges = {tuple(map(int, edge)) for edge in result.value.indices.indices}

        diffs = positions[:, None] - positions[None, :]
        diffs = cell.wrap(diffs)
        dists = jnp.linalg.norm(diffs, axis=-1)
        mask = (dists < cutoff) & ~jnp.eye(N, dtype=bool)
        edges = jnp.stack(jnp.where(mask), axis=-1)
        edges = {tuple(map(int, edge)) for edge in edges}

        for edge in edges:
            assert edge in actual_edges, (
                f"Edge {edge} not found in actual edges with distance {dists[edge[0], edge[1]]}."
            )
        for edge in actual_edges:
            if edge not in edges:
                assert edge[0] == edge[1] == N, (
                    f"Unexpected edge {edge} found in actual edges with distance {dists[edge[0], edge[1]]}."
                )

    def test_compare_to_naive_update(self, neighbor_list_impl):
        N = 15
        M = 3
        positions = jax.random.uniform(
            jax.random.key(0), (N, 3), minval=-5.0, maxval=5.0
        )
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        cutoff = 3
        instance_factory = neighbor_list_impl["instance_factory"]

        rh_indices = jax.random.choice(jax.random.key(1), N, shape=(M,), replace=False)
        new_positions = jax.random.uniform(
            jax.random.key(2), (M, 3), minval=-5.0, maxval=5.0
        )
        positions = positions.at[rh_indices].set(new_positions)
        lh = make_lh(positions, jnp.array([0] * N), jnp.arange(N))
        for_indices = Index(lh.keys, rh_indices)

        _sys, _cut = make_systems(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            cutoffs=_cut, **{"edges": N, "candidates": N, "cells": 64}
        )
        while (
            result := jax.jit(as_result_function(neighbor_list_instance))(
                lh=lh,
                systems=_sys,
                for_indices=for_indices,
            )
        ).failed_assertions:
            neighbor_list_instance = result.fix_or_raise(neighbor_list_instance)
        result.raise_assertion()
        actual_edges = {tuple(map(int, edge)) for edge in result.value.indices.indices}

        diffs = positions[:, None] - new_positions[None, :]
        diffs = cell.wrap(diffs)
        dists = jnp.linalg.norm(diffs, axis=-1)
        mask = dists < cutoff
        mask = mask.at[rh_indices].min(~jnp.eye(M, dtype=bool))
        assert (dists[mask] < cutoff).all(), "Distances should be less than cutoff."
        for i in range(N):
            for j in range(M):
                if mask[i, j]:
                    assert (i, int(rh_indices[j])) in actual_edges, (
                        f"Missing edge {(i, int(rh_indices[j]))} with indices {(i, j)} found with distance {dists[i, j]}."
                    )
                else:
                    assert (i, int(rh_indices[j])) not in actual_edges, (
                        f"Unexpected edge {(i, int(rh_indices[j]))} with indices {(i, j)} found with distance {dists[i, j]}."
                    )

    # --- Small cell periodic image tests ---

    @staticmethod
    @jax.jit
    def _compute_neighbor_mask(positions, vectors, cutoff):
        """JIT-compiled helper to compute neighbor mask (max_images=1)."""
        n = positions.shape[0]
        r = jnp.arange(-1, 2)  # [-1, 0, 1]
        image_offsets = jnp.stack(
            jnp.meshgrid(r, r, r, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        real_offsets = image_offsets @ vectors[0]
        deltas = positions[None, :, :] - positions[:, None, :]
        all_deltas = deltas[:, :, None, :] + real_offsets[None, None, :, :]
        dists = jnp.linalg.norm(all_deltas, axis=-1)
        within_cutoff = (dists < cutoff).any(axis=-1)
        return within_cutoff & ~jnp.eye(n, dtype=bool)

    def _compute_naive_neighbors(self, positions, cell, cutoff):
        """Compute neighbors by explicitly checking all periodic images."""
        mask = self._compute_neighbor_mask(positions, cell.vectors, cutoff)
        i_idx, j_idx = jnp.where(mask)
        return {(int(i), int(j)) for i, j in zip(i_idx, j_idx)}

    def test_small_cell_periodic_images(self, neighbor_list_impl):
        """Test with cell smaller than 2*cutoff in one or all directions."""
        instance_factory = neighbor_list_impl["instance_factory"]
        cutoff = 0.8

        # Sub-scenario 1: single direction
        positions_1 = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
        batch_mask_1 = jnp.array([0, 0])
        lv_1 = jnp.diag(jnp.array([1.0, 10.0, 10.0]))[None]
        cell_1 = PeriodicCell(TriclinicFrame.from_matrix(lv_1))
        lh_1 = make_lh(positions_1, batch_mask_1, jnp.arange(len(batch_mask_1)))
        _sys_1, _cut_1 = make_systems(cell_1, jnp.array([cutoff]))
        nl_1 = instance_factory(
            candidates=10, edges=10, cells=256, image_candidates=200, cutoffs=_cut_1
        )
        result_1 = jax.jit(as_result_function(nl_1))(lh=lh_1, systems=_sys_1)
        result_1.raise_assertion()
        valid_1 = {
            (int(e[0]), int(e[1]))
            for e in result_1.value.indices.indices
            if e[0] < len(positions_1) and e[1] < len(positions_1)
        }
        expected_1 = self._compute_naive_neighbors(positions_1, cell_1, cutoff)
        assert expected_1.issubset(valid_1), f"Missing edges: {expected_1 - valid_1}"

        # Sub-scenario 2: all directions
        positions_2 = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]])
        batch_mask_2 = jnp.array([0, 0])
        lv_2 = jnp.eye(3)[None] * 1.0
        cell_2 = PeriodicCell(TriclinicFrame.from_matrix(lv_2))
        lh_2 = make_lh(positions_2, batch_mask_2, jnp.arange(len(batch_mask_2)))
        _sys_2, _cut_2 = make_systems(cell_2, jnp.array([cutoff]))
        nl_2 = instance_factory(
            candidates=4, edges=53, cells=8, image_candidates=600, cutoffs=_cut_2
        )
        result_2 = jax.jit(as_result_function(nl_2))(lh=lh_2, systems=_sys_2)
        result_2.raise_assertion()
        valid_2 = {
            (int(e[0]), int(e[1]))
            for e in result_2.value.indices.indices
            if e[0] < len(positions_2) and e[1] < len(positions_2)
        }
        expected_2 = self._compute_naive_neighbors(positions_2, cell_2, cutoff)
        assert expected_2.issubset(valid_2), f"Missing edges: {expected_2 - valid_2}"

    def test_small_cell_correctness(self, neighbor_list_impl):
        """Verify correctness by comparing with naive brute-force on small cell."""
        N = 5
        positions = jax.random.uniform(
            jax.random.key(42), (N, 3), minval=0.0, maxval=1.0
        )
        batch_mask = jnp.zeros(N, dtype=int)
        lattice_vectors = jnp.eye(3)[None] * 1.5
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = 1.2

        instance_factory = neighbor_list_impl["instance_factory"]

        lh = make_lh(positions, batch_mask, jnp.arange(N))

        _sys, _cut = make_systems(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=50, edges=50, cells=64, image_candidates=10000, cutoffs=_cut
        )
        while (
            result := jax.jit(as_result_function(neighbor_list_instance))(
                lh=lh,
                systems=_sys,
            )
        ).failed_assertions:
            neighbor_list_instance = result.fix_or_raise(neighbor_list_instance)
        result.raise_assertion()

        actual_edges = {
            (int(e[0]), int(e[1]))
            for e in result.value.indices.indices
            if e[0] < N and e[1] < N
        }
        expected = self._compute_naive_neighbors(positions, cell, cutoff)
        for edge in expected:
            assert edge in actual_edges, f"Missing edge {edge}"

    def test_batched_systems_different_cell_sizes(self, neighbor_list_impl):
        """Test batched systems where some need images and others don't."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],  # System 0: small cell
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # System 1: large cell
            ]
        )
        batch_mask = jnp.array([0, 0, 1, 1])
        lattice_vectors = jnp.array(
            [
                jnp.diag(jnp.array([1.0, 10.0, 10.0])),
                jnp.eye(3) * 10.0,
            ]
        )
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoffs = jnp.array([0.8, 1.5])

        instance_factory = neighbor_list_impl["instance_factory"]

        lh = make_lh(positions, batch_mask, jnp.arange(len(batch_mask)))

        _sys, _cut = make_systems(cell, cutoffs)
        neighbor_list_instance = instance_factory(
            candidates=30, edges=20, cells=256, image_candidates=300, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            systems=_sys,
        )
        result.raise_assertion()

        valid_edges = {
            (int(e[0]), int(e[1]))
            for e in result.value.indices.indices
            if e[0] < len(positions) and e[1] < len(positions)
        }

        assert (0, 1) in valid_edges and (1, 0) in valid_edges
        assert (2, 3) in valid_edges and (3, 2) in valid_edges
        for i in [0, 1]:
            for j in [2, 3]:
                assert (i, j) not in valid_edges and (j, i) not in valid_edges

    def test_unsorted_particles_with_images(self, neighbor_list_impl):
        """Unsorted particles work correctly when periodic images are needed.

        Verifies that ``_get_candidate_images`` does not require sorted
        candidates: shuffling particles across systems yields the same edge set.
        """
        positions = jnp.array(
            [
                [0.1, 0.0, 0.0],
                [0.4, 0.0, 0.0],  # System 0
                [0.1, 0.0, 0.0],
                [0.4, 0.0, 0.0],  # System 1
                [0.1, 0.0, 0.0],
                [0.4, 0.0, 0.0],  # System 2
            ]
        )
        batch_mask = jnp.array([0, 0, 1, 1, 2, 2])
        cell = PeriodicCell(
            TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(3, axis=0) * 1.0)
        )
        cutoffs = jnp.array([0.8, 0.8, 0.8])  # > 0.5 triggers images

        # Shuffle particles across systems: [S0, S1, S0, S2, S1, S2]
        shuffle = jnp.array([0, 2, 1, 4, 3, 5])

        instance_factory = neighbor_list_impl["instance_factory"]
        data = make_lh(positions, batch_mask, jnp.arange(6))

        _sys, _cut = make_systems(cell, cutoffs)
        nl = instance_factory(
            candidates=50, edges=53, cells=8, image_candidates=1500, cutoffs=_cut
        )
        nl = jax.jit(as_result_function(nl))

        def get_edges(idx_order):
            rev_order = np.argsort(idx_order)
            reordered_index = tuple(range(len(idx_order)))
            reordered_data = jax.tree.map(
                lambda x: x[jnp.asarray(idx_order)], data.data
            )
            reordered = Table(reordered_index, reordered_data)
            result = nl(reordered, systems=_sys)
            result.raise_assertion()
            mask = (result.value.indices.indices < 6).all(axis=1)
            valid = np.asarray(result.value.indices.indices[mask])
            shifts = np.asarray(result.value.shifts[mask])
            return {
                (int(rev_order[i]), int(rev_order[j]), *map(int, s))
                for i, j, s in zip(valid[:, 0], valid[:, 1], shifts[:, 0])
            }

        assert len(get_edges(jnp.arange(6)).difference(get_edges(shuffle))) == 0

    def test_triclinic_small_cell(self, neighbor_list_impl):
        """Test small triclinic (non-orthogonal) cell."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]])
        batch_mask = jnp.array([0, 0])
        lattice_vectors = jnp.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 0.866, 0.0],
                    [0.0, 0.0, 10.0],
                ]
            ]
        )
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = 0.8

        instance_factory = neighbor_list_impl["instance_factory"]

        lh = make_lh(positions, batch_mask, jnp.arange(len(batch_mask)))

        _sys, _cut = make_systems(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=4, edges=30, cells=12, image_candidates=200, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            systems=_sys,
        )
        result.raise_assertion()

        valid_edges = {
            (int(e[0]), int(e[1]))
            for e in result.value.indices.indices
            if e[0] < len(positions) and e[1] < len(positions)
        }
        expected = self._compute_naive_neighbors(positions, cell, cutoff)
        assert len(expected.difference(valid_edges)) == 0, (
            f"Missing edges: {expected - valid_edges}"
        )

    def test_self_interactions_with_periodic_images(self, neighbor_list_impl):
        """Self-interactions with images are included; the zero-shift self-pair is excluded."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        batch_mask = jnp.array([0])

        # Cell size 0.5 Å, cutoff 0.55 Å => nearest self-images (distance 0.5)
        # are within cutoff.
        cell_size = 0.5
        lattice_vectors = jnp.eye(3)[None] * cell_size
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = 0.55

        instance_factory = neighbor_list_impl["instance_factory"]

        lh = make_lh(positions, batch_mask, jnp.array([0]))

        _sys, _cut = make_systems(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=1, edges=16, cells=8, image_candidates=125, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            systems=_sys,
        )
        result.raise_assertion()
        edges = result.value

        valid_mask = (edges.indices.indices == 0).all(axis=1)
        valid_indices = np.asarray(edges.indices.indices[valid_mask])
        valid_shifts = np.asarray(edges.shifts[valid_mask, 0])
        edge_set = {
            (int(i), int(j), tuple(int(s) for s in shift))
            for (i, j), shift in zip(valid_indices, valid_shifts)
        }

        expected_edges: set[tuple[int, int, tuple[int, ...]]] = {
            (0, 0, (1, 0, 0)),
            (0, 0, (-1, 0, 0)),
            (0, 0, (0, 1, 0)),
            (0, 0, (0, -1, 0)),
            (0, 0, (0, 0, 1)),
            (0, 0, (0, 0, -1)),
        }

        assert (0, 0, (0, 0, 0)) not in edge_set
        assert edge_set == expected_edges, (
            f"Could not find edges {expected_edges - edge_set} "
            f"or found unexpected edges {edge_set - expected_edges}"
        )

    def test_no_replication_with_infinite_cutoff(self, neighbor_list_impl):
        """Verify no periodic images are generated when cutoff is infinite."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        batch_mask = jnp.array([0, 0])

        lattice_vectors = jnp.eye(3)[None] * 1.0
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = jnp.inf

        instance_factory = neighbor_list_impl["instance_factory"]

        lh = make_lh(positions, batch_mask, jnp.array([0, 1]))

        _sys, _cut = make_systems(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=4, edges=4, cells=8, image_candidates=4, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            systems=_sys,
        )
        result.raise_assertion()
        edges = result.value

        valid_mask = (edges.indices.indices < 2).all(axis=1)
        valid_indices = np.asarray(edges.indices.indices[valid_mask])
        valid_shifts = np.asarray(edges.shifts[valid_mask, 0])

        edge_set = {
            (int(i), int(j), tuple(int(s) for s in shift))
            for (i, j), shift in zip(valid_indices, valid_shifts)
        }

        expected_edges = {(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0))}

        assert edge_set == expected_edges, (
            f"Expected edges {expected_edges}, got {edge_set}"
        )

    def test_exclusion_only_applies_to_minimum_image(self, neighbor_list_impl):
        """Exclusion segments only exclude the minimum-image interaction.

        Particles in the same exclusion segment should not interact via the
        minimum image (closest periodic image), but should interact via
        non-minimum periodic images.
        """
        positions = jnp.array([[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]])
        batch_mask = jnp.array([0, 0])

        # Small cell where periodic images are within cutoff.
        # Direct distance (0,0,0): 0.3, Image distance (-1,0,0): 0.2
        cell_size = 0.5
        lattice_vectors = jnp.eye(3)[None] * cell_size
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = 0.4

        instance_factory = neighbor_list_impl["instance_factory"]

        # Same exclusion segment for both particles.
        lh = make_lh(positions, batch_mask, jnp.array([0, 0]))

        _sys, _cut = make_systems(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=4, edges=30, cells=8, image_candidates=200, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            systems=_sys,
        )
        result.raise_assertion()
        edges = result.value

        valid_mask = (edges.indices.indices < 2).all(axis=1)
        valid_indices = np.asarray(edges.indices.indices[valid_mask])
        valid_shifts = np.asarray(edges.shifts[valid_mask, 0])

        edge_set = {
            (int(i), int(j), tuple(int(s) for s in shift))
            for (i, j), shift in zip(valid_indices, valid_shifts)
        }

        # Minimum-image interactions are EXCLUDED (same exclusion segment).
        assert (0, 1, (-1, 0, 0)) not in edge_set
        assert (1, 0, (1, 0, 0)) not in edge_set
        # Non-minimum image interactions are INCLUDED.
        assert (0, 1, (0, 0, 0)) in edge_set
        assert (1, 0, (0, 0, 0)) in edge_set
