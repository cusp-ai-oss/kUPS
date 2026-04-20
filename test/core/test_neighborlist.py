# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.data.wrappers import WithIndices
from kups.core.lens import bind
from kups.core.neighborlist import (
    AllDenseNearestNeighborList,
    CellListNeighborList,
    DenseNearestNeighborList,
    Edges,
    RefineCutoffNeighborList,
    neighborlist_changes,
)
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass


# Override the default class-scoped clear_cache fixture with module scope so
# JAX compilation caches persist across test classes within this file.
@pytest.fixture(autouse=True, scope="module")
def clear_cache():
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()


@dataclass
class SamplePoints:
    """Concrete NeighborListPoints for testing."""

    positions: jax.Array
    system: Index
    inclusion: Index
    exclusion: Index


@dataclass
class SampleSystems:
    """Concrete NeighborListSystems for testing."""

    unitcell: UnitCell


def _make_lh(positions, batch_mask, exclusion_ids=None):
    """Create Table lh from positions and batch mask."""
    n = len(positions)
    n_sys = int(jnp.max(batch_mask)) + 1 if n > 0 else 1
    sys_keys = tuple(range(n_sys))
    pi_keys = tuple(ParticleId(i) for i in range(n))
    if exclusion_ids is None:
        exclusion_ids = jnp.arange(n)
    return Table(
        pi_keys,
        SamplePoints(
            positions=positions,
            system=Index(sys_keys, batch_mask.astype(int)),
            inclusion=Index(sys_keys, batch_mask.astype(int)),
            exclusion=Index.integer(exclusion_ids.astype(int)),
        ),
    )


def _make_systems(lattice_or_uc, cutoffs):
    """Create Table systems from unit cell, alongside cutoffs.

    Returns:
        A tuple of (Table systems, Table cutoffs).
    """
    n = len(cutoffs)
    if isinstance(lattice_or_uc, UnitCell):
        uc = lattice_or_uc
    else:
        lv = jnp.asarray(lattice_or_uc)
        if lv.shape[0] == 1 and n > 1:
            lv = jnp.repeat(lv, n, axis=0)
        uc = TriclinicUnitCell.from_matrix(lv)
    sys_keys = tuple(SystemId(i) for i in range(n))
    indexed_systems = Table(sys_keys, SampleSystems(unitcell=uc))
    indexed_cutoffs = Table(sys_keys, cutoffs)
    return indexed_systems, indexed_cutoffs


def _make_rh(lh, rh_positions, rh_batch_mask, rh_index_remap, exclusion_ids=None):
    """Create rh Table data and index remap for testing."""
    n_rh = len(rh_positions)
    n_sys = int(jnp.max(rh_batch_mask)) + 1
    sys_keys = tuple(range(n_sys))
    rh_pi_keys = tuple(ParticleId(i) for i in range(n_rh))
    if exclusion_ids is None:
        exclusion_ids = rh_index_remap
    rh_points = SamplePoints(
        positions=rh_positions,
        system=Index(sys_keys, rh_batch_mask.astype(int)),
        inclusion=Index(sys_keys, rh_batch_mask.astype(int)),
        exclusion=Index.integer(exclusion_ids.astype(int)),
    )
    rh_indexed = Table(rh_pi_keys, rh_points)
    rh_remap = Index(lh.keys, rh_index_remap.astype(int))
    return rh_indexed, rh_remap


def _make_edges(lh_indices, rh_indices, n_particles=None, shifts=None):
    """Create Edges with Index for testing."""
    raw = jnp.stack([lh_indices, rh_indices], axis=-1)
    if n_particles is None:
        n_particles = int(max(lh_indices.max(), rh_indices.max())) + 1
    if shifts is None:
        shifts = jnp.zeros((len(raw), 1, 3), dtype=int)
    else:
        shifts = shifts.reshape(len(raw), 1, 3)
    return Edges(Index(tuple(ParticleId(i) for i in range(n_particles)), raw), shifts)


def _call_nl(nl_instance, lh, systems, cutoffs, rh=None, rh_index_remap=None):
    """Call a neighbor list with the new API."""
    return nl_instance(
        lh=lh, rh=rh, systems=systems, cutoffs=cutoffs, rh_index_remap=rh_index_remap
    )


class TestEdges:
    """Test cases for the Edges dataclass."""

    def test_edges_creation_binary(self):
        """Test creating binary edges (degree=2)."""
        indices = Index(
            (ParticleId(0), ParticleId(1), ParticleId(2)),
            jnp.array([[0, 1], [1, 2], [2, 0]]),
        )
        shifts = jnp.array([[[0, 0, 0]], [[1, 0, 0]], [[-1, 0, 0]]])
        edges = Edges(indices, shifts)

        assert edges.degree == 2
        assert edges.indices.shape == (3, 2)
        assert edges.shifts.shape == (3, 1, 3)
        npt.assert_array_equal(edges.indices.indices, indices.indices)
        npt.assert_array_equal(edges.shifts, shifts)

    def test_edges_creation_ternary(self):
        """Test creating ternary edges (degree=3)."""
        indices = Index(
            (ParticleId(0), ParticleId(1), ParticleId(2), ParticleId(3)),
            jnp.array([[0, 1, 2], [1, 2, 3]]),
        )
        shifts = jnp.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]])
        edges = Edges(indices, shifts)

        assert edges.degree == 3
        assert edges.indices.shape == (2, 3)
        assert edges.shifts.shape == (2, 2, 3)
        npt.assert_array_equal(edges.indices.indices, indices.indices)
        npt.assert_array_equal(edges.shifts, shifts)

    def test_edges_shape_validation(self):
        """Test that edges validates shape consistency with raw arrays."""
        indices = jnp.array([[0, 1], [1, 2]])
        wrong_shifts = jnp.array([[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]]])

        with pytest.raises(AssertionError):
            Edges(indices, wrong_shifts)  # type: ignore[arg-type]


class TestNumNeighbors:
    """Test cases for the NumNeighbors dataclass."""

    def test_num_neighbors_creation(self):
        """Test creating NumNeighbors with various capacities."""
        nn = FixedCapacity(100)
        assert nn.size == 100

        nn_zero = FixedCapacity(0)
        assert nn_zero.size == 0


# Parametrized test class for testing different neighbor list implementations
class TestNearestNeighborListImplementations:
    """Test different neighbor list implementations with the same test suite."""

    @pytest.fixture(
        params=[
            {
                "instance_factory": (
                    lambda candidates, edges, image_candidates=None, **kwargs: (
                        DenseNearestNeighborList(
                            avg_candidates=FixedCapacity(candidates),
                            avg_edges=FixedCapacity(edges),
                            avg_image_candidates=FixedCapacity(
                                image_candidates or candidates
                            ),
                        )
                    )
                ),
                "name": "naive",
            },
            {
                "instance_factory": (
                    lambda candidates, edges, cells, image_candidates=None, **kwargs: (
                        CellListNeighborList(
                            avg_candidates=FixedCapacity(candidates),
                            avg_edges=FixedCapacity(edges),
                            cells=FixedCapacity(cells),
                            avg_image_candidates=FixedCapacity(
                                image_candidates or candidates
                            ),
                        )
                    )
                ),
                "name": "cell_list",
            },
            {
                "instance_factory": (
                    lambda candidates, edges, cells, image_candidates=None, **kwargs: (
                        AllDenseNearestNeighborList(
                            avg_edges=FixedCapacity(edges),
                            avg_image_candidates=FixedCapacity(
                                image_candidates or edges
                            ),
                        )
                    )
                ),
                "name": "all_to_all",
            },
            # Add more implementations here, e.g.:
            # {
            #     "impl": bruteforce_neighbor_list,
            #     "statics_factory": lambda capacity: (BruteforceStatics(capacity=capacity), BruteforceStatics.capacity_lens),
            #     "name": "bruteforce"
            # },
        ]
    )
    def neighbor_list_impl(self, request):
        """Fixture that provides different neighbor list implementations with their instance factories."""
        return request.param

    def _run_neighbor_search_test(self, neighbor_list_impl_info, **kwargs):
        """Generic test runner for neighbor search implementations."""
        # Default test parameters
        default_params = {
            "positions": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "batch_mask": jnp.array([0, 0, 0]),
            "unitcells": jnp.eye(3)[None] * 10.0,
            "cutoffs": jnp.array([1.5]),
            "extras": {"candidates": 9, "edges": 4, "cells": 256},
        }

        # Update with provided parameters
        params = {**default_params, **kwargs}

        # Get the instance factory for this implementation
        instance_factory = neighbor_list_impl_info["instance_factory"]

        # Create neighbor list instance using the implementation-specific factory
        neighbor_list_instance = instance_factory(**params["extras"])

        # Create PointSet objects for lh and rh
        lh = _make_lh(
            params["positions"],
            params["batch_mask"],
            jnp.arange(
                len(params["batch_mask"]),
            ),
        )

        rh = None
        rh_remap = None
        if (rh_pos := params.get("rh_positions", None)) is not None:
            rh_idx = params.get("rh_index_remap", None)
            assert rh_idx is not None, "rh_positions requires rh_index_remap in new API"
            rh_batch_mask = params.get("rh_batch_mask", params["batch_mask"])
            rh, rh_remap = _make_rh(lh, rh_pos, rh_batch_mask, rh_idx)

        systems, cutoffs = _make_systems(params["unitcells"], params["cutoffs"])
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=rh,
            systems=systems,
            cutoffs=cutoffs,
            rh_index_remap=rh_remap,
        )

        return result

    def test_basic_functionality(self, neighbor_list_impl):
        """Test basic neighbor search functionality for any implementation."""
        result = self._run_neighbor_search_test(neighbor_list_impl)
        result.raise_assertion()
        edges = result.value

        # Should find 4 edges: (0,1), (1,0), (1,2), (2,1)
        assert edges.degree == 2

        # Extract valid edges
        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]

        # Check specific edges exist
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

        # Should contain an assertion about capacity
        assert len(result.assertions) > 0

        # Should raise CapacityError when assertions are checked
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

        # Should NOT find edges since distance == cutoff and we use < not <=
        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 2]
        valid_edges = valid_edges[valid_edges[:, 1] < 2]

        # The implementation uses < cutoff, so particles exactly at cutoff should not be neighbors
        assert len(valid_edges) == 0, (
            f"Found unexpected edges at exact cutoff: {valid_edges}"
        )

    def test_periodic_boundary_conditions(self, neighbor_list_impl):
        """Test neighbor search with periodic boundary conditions."""
        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=jnp.array([[0.1, 0.0, 0.0], [2.9, 0.0, 0.0]]),
            batch_mask=jnp.array([0, 0]),
            unitcells=jnp.eye(3)[None] * 3.0,
            cutoffs=jnp.array([1.0]),
        )
        result.raise_assertion()
        edges = result.value

        # Should find neighbors due to periodic boundary conditions
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
            unitcells=jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
            cutoffs=jnp.array([1.5, 1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        # Should find edges within each batch, not across batches
        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 4]

        # Check that no edges cross batch boundaries
        batch_mask = jnp.array([0, 0, 1, 1])
        for i in range(valid_edges.shape[0]):
            lh_idx, rh_idx = valid_edges[i]
            if lh_idx < 4 and rh_idx < 4:  # Valid indices
                assert batch_mask[lh_idx] == batch_mask[rh_idx]

    def test_rh_positions_basic(self, neighbor_list_impl):
        """Test basic functionality with separate rh_positions."""
        # lh has 3 particles, rh selects a subset with different positions.
        # rh[0] -> lh[2], rh[1] -> lh[0]. Using remap avoids self-interaction
        # exclusion between lh[0] and rh[0], lh[1] and rh[1].
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        rh_positions = jnp.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        rh_index_remap = jnp.array([2, 0])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            rh_positions=rh_positions,
            batch_mask=jnp.array([0, 0, 0]),
            rh_batch_mask=jnp.array([0, 0]),
            rh_index_remap=rh_index_remap,
            cutoffs=jnp.array([1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]
        edge_set = set(tuple(edge.tolist()) for edge in valid_edges)

        # lh[1] (2,0,0) vs rh[0] (1,0,0): dist=1.0<1.5, remap[0]=2, edge (1,2)
        # lh[1] (2,0,0) vs rh[1] (3,0,0): dist=1.0<1.5, remap[1]=0, edge (1,0)
        # Symmetrized: (2,1) and (0,1)
        assert (1, 2) in edge_set, f"Expected edge (1, 2) not found in {edge_set}"
        assert (2, 1) in edge_set, f"Expected edge (2, 1) not found in {edge_set}"
        assert (1, 0) in edge_set, f"Expected edge (1, 0) not found in {edge_set}"
        assert (0, 1) in edge_set, f"Expected edge (0, 1) not found in {edge_set}"

    def test_rh_batch_boundaries(self, neighbor_list_impl):
        """Test rh batch masks: edges found within systems, blocked across systems.

        Same shapes (4 lh, 2 rh, 2 systems), different values for the two
        sub-scenarios.
        """
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # system 0, lh[0]
                [5.0, 0.0, 0.0],  # system 0, lh[1]
                [0.0, 0.0, 0.0],  # system 1, lh[2]
                [5.0, 0.0, 0.0],  # system 1, lh[3]
            ]
        )
        lh_batch = jnp.array([0, 0, 1, 1])
        uc = jnp.eye(3)[None].repeat(2, axis=0) * 10.0
        cutoffs = jnp.array([1.5, 1.5])

        # Positive case: rh in correct systems, edges should be found
        result_pos = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            rh_positions=jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            batch_mask=lh_batch,
            rh_batch_mask=jnp.array([0, 1]),
            rh_index_remap=jnp.array([1, 3]),
            unitcells=uc,
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

        # Negative case: rh in swapped systems, cross-system edges blocked
        result_neg = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            rh_positions=jnp.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            batch_mask=lh_batch,
            rh_batch_mask=jnp.array([1, 0]),  # swapped
            rh_index_remap=jnp.array([3, 1]),
            unitcells=uc,
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
                    f"lh_batch={lh_batch[lh_idx]}, rh_batch={lh_batch[rh_idx]}"
                )

    def test_rh_positions_asymmetric_search(self, neighbor_list_impl):
        """Test asymmetric search with different lh and rh particle sets."""
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        rh_positions = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        # rh[0] -> lh[0], rh[1] -> lh[2]: maps to non-adjacent particles
        rh_index_remap = jnp.array([0, 2])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            rh_positions=rh_positions,
            batch_mask=jnp.array([0, 0, 0]),
            rh_batch_mask=jnp.array([0, 0]),
            rh_index_remap=rh_index_remap,
            cutoffs=jnp.array([1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]
        edge_set = set(tuple(edge.tolist()) for edge in valid_edges)

        # lh[0](0,0,0) vs rh[0](1,0,0): dist=1.0<1.5, remap[0]=0, self-excluded
        # lh[1](2,0,0) vs rh[0](1,0,0): dist=1.0<1.5, remap[0]=0, edge (1,0)
        # lh[1](2,0,0) vs rh[1](3,0,0): dist=1.0<1.5, remap[1]=2, edge (1,2)
        # lh[2](4,0,0) vs rh[1](3,0,0): dist=1.0<1.5, remap[1]=2, self-excluded
        # Symmetrized: (0,1) and (2,1) also appear
        assert (1, 0) in edge_set, f"Expected edge (1, 0) not found in {edge_set}"
        assert (0, 1) in edge_set, f"Expected edge (0, 1) not found in {edge_set}"
        assert (1, 2) in edge_set, f"Expected edge (1, 2) not found in {edge_set}"
        assert (2, 1) in edge_set, f"Expected edge (2, 1) not found in {edge_set}"

    def test_rh_index_remap_for_subsets(self, neighbor_list_impl):
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        rh_index_remap = jnp.array([0, 2, 1, 4])
        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=positions,
            rh_positions=positions,
            batch_mask=jnp.array([0, 0, 0, 0]),
            rh_batch_mask=jnp.array([0, 0, 0, 0]),
            rh_index_remap=rh_index_remap,
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

    def test_rh_index_remap_prevents_self_interaction(self, neighbor_list_impl):
        """Test that rh_index_remap prevents self-interactions."""
        # Same positions for lh and rh
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Identity remap - should prevent self-interactions
        rh_index_remap = jnp.array([0, 1])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=positions,
            rh_positions=positions,
            batch_mask=jnp.array([0, 0]),
            rh_batch_mask=jnp.array([0, 0]),
            rh_index_remap=rh_index_remap,
            cutoffs=jnp.array([1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        # Should not find self-interactions (i, i)
        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 2]
        valid_edges = valid_edges[valid_edges[:, 1] < 2]

        # Absorb basic remap assertion: remapping produces edges
        assert len(valid_edges) > 0, "Should find edges with index remapping"

        for i in range(valid_edges.shape[0]):
            lh_idx, rh_idx = valid_edges[i]
            assert lh_idx != rh_idx, f"Found self-interaction: ({lh_idx}, {rh_idx})"

    def test_rh_arguments_combined(self, neighbor_list_impl):
        """Test all rh_ arguments used together."""
        # Complex scenario combining all rh_ arguments
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # system 0
                [0.0, 0.0, 0.0],  # system 1
                [0.0, 0.0, 0.0],  # system 0
            ]
        )
        rh_positions = jnp.array(
            [
                [1.0, 0.0, 0.0],  # will be remapped
                [1.0, 0.0, 0.0],  # will be remapped
                [1.0, 0.0, 0.0],  # will be remapped
            ]
        )

        # Remap: rh[0]->original[2], rh[1]->original[0], rh[2]->original[1]
        rh_index_remap = jnp.array([2, 1, 0])

        result = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            rh_positions=rh_positions,
            batch_mask=jnp.array([0, 1, 0]),
            rh_batch_mask=jnp.array([0, 1, 0]),
            rh_index_remap=rh_index_remap,
            unitcells=jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
            cutoffs=jnp.array([1.5, 1.5]),
            capacity=20,
        )
        result.raise_assertion()
        edges = result.value

        # Should find edges respecting all constraints
        valid_edges = edges.indices.indices[edges.indices.indices[:, 0] < 3]
        valid_edges = valid_edges[valid_edges[:, 1] < 3]

        # Check that all edges respect batch constraints
        lh_batch_mask = jnp.array([0, 1, 0])
        rh_batch_mask = jnp.array([0, 1, 0])
        for i in range(valid_edges.shape[0]):
            lh_idx, rh_idx = valid_edges[i]
            if lh_idx < 3 and rh_idx < 3:  # Valid indices
                # Check that batch masks match (not remap logic)
                assert lh_batch_mask[lh_idx] == rh_batch_mask[rh_idx], (
                    f"Edge ({lh_idx}, {rh_idx}) violates batch constraint: "
                    f"lh_batch={lh_batch_mask[lh_idx]}, rh_batch={rh_batch_mask[rh_idx]}"
                )

    def test_compare_to_naive(self, neighbor_list_impl):
        N = 15
        positions = jax.random.uniform(
            jax.random.key(0), (N, 3), minval=0.0, maxval=10.0
        )
        unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
        cutoff = 3
        # Get the instance factory
        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            **{"edges": N, "candidates": N, "cells": 64}
        )

        lh = _make_lh(
            positions,
            jnp.array([0] * N),
            jnp.arange(N),
        )

        _sys, _cut = _make_systems(unitcell, jnp.array([cutoff]))
        while (
            result := as_result_function(neighbor_list_instance)(
                lh, None, systems=_sys, cutoffs=_cut
            )
        ).failed_assertions:
            neighbor_list_instance = result.fix_or_raise(neighbor_list_instance)
        result.raise_assertion()
        actual_edges = {tuple(map(int, edge)) for edge in result.value.indices.indices}

        diffs = positions[:, None] - positions[None, :]
        diffs = unitcell.wrap(diffs)
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
        unitcell = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
        cutoff = 3
        # Get the instance factory
        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            **{"edges": N, "candidates": N, "cells": 64}
        )

        rh_indices = jax.random.choice(jax.random.key(1), N, shape=(M,), replace=False)
        new_positions = jax.random.uniform(
            jax.random.key(2), (M, 3), minval=-5.0, maxval=5.0
        )
        positions = positions.at[rh_indices].set(new_positions)
        lh = _make_lh(
            positions,
            jnp.array([0] * N),
            jnp.arange(N),
        )
        rh, rh_remap = _make_rh(lh, new_positions, jnp.array([0] * M), rh_indices)

        _sys, _cut = _make_systems(unitcell, jnp.array([cutoff]))
        while (
            result := jax.jit(as_result_function(neighbor_list_instance))(
                lh=lh,
                rh=rh,
                systems=_sys,
                cutoffs=_cut,
                rh_index_remap=rh_remap,
            )
        ).failed_assertions:
            neighbor_list_instance = result.fix_or_raise(neighbor_list_instance)
        result.raise_assertion()
        actual_edges = {tuple(map(int, edge)) for edge in result.value.indices.indices}

        diffs = positions[:, None] - new_positions[None, :]
        diffs = unitcell.wrap(diffs)
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
    def _compute_neighbor_mask(positions, lattice_vectors, cutoff):
        """JIT-compiled helper to compute neighbor mask (max_images=1)."""
        n = positions.shape[0]
        r = jnp.arange(-1, 2)  # [-1, 0, 1]
        image_offsets = jnp.stack(
            jnp.meshgrid(r, r, r, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        real_offsets = image_offsets @ lattice_vectors[0]
        deltas = positions[None, :, :] - positions[:, None, :]
        all_deltas = deltas[:, :, None, :] + real_offsets[None, None, :, :]
        dists = jnp.linalg.norm(all_deltas, axis=-1)
        within_cutoff = (dists < cutoff).any(axis=-1)
        return within_cutoff & ~jnp.eye(n, dtype=bool)

    def _compute_naive_neighbors(self, positions, unitcell, cutoff):
        """Compute neighbors by explicitly checking all periodic images."""
        mask = self._compute_neighbor_mask(positions, unitcell.lattice_vectors, cutoff)
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
        uc_1 = TriclinicUnitCell.from_matrix(lv_1)
        nl_1 = instance_factory(
            candidates=10, edges=10, cells=256, image_candidates=200
        )
        lh_1 = _make_lh(positions_1, batch_mask_1, jnp.arange(len(batch_mask_1)))
        _sys_1, _cut_1 = _make_systems(uc_1, jnp.array([cutoff]))
        result_1 = jax.jit(as_result_function(nl_1))(
            lh=lh_1, rh=None, systems=_sys_1, cutoffs=_cut_1
        )
        result_1.raise_assertion()
        valid_1 = {
            (int(e[0]), int(e[1]))
            for e in result_1.value.indices.indices
            if e[0] < len(positions_1) and e[1] < len(positions_1)
        }
        expected_1 = self._compute_naive_neighbors(positions_1, uc_1, cutoff)
        assert expected_1.issubset(valid_1), f"Missing edges: {expected_1 - valid_1}"

        # Sub-scenario 2: all directions
        positions_2 = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]])
        batch_mask_2 = jnp.array([0, 0])
        lv_2 = jnp.eye(3)[None] * 1.0
        uc_2 = TriclinicUnitCell.from_matrix(lv_2)
        nl_2 = instance_factory(candidates=4, edges=53, cells=8, image_candidates=600)
        lh_2 = _make_lh(positions_2, batch_mask_2, jnp.arange(len(batch_mask_2)))
        _sys_2, _cut_2 = _make_systems(uc_2, jnp.array([cutoff]))
        result_2 = jax.jit(as_result_function(nl_2))(
            lh=lh_2, rh=None, systems=_sys_2, cutoffs=_cut_2
        )
        result_2.raise_assertion()
        valid_2 = {
            (int(e[0]), int(e[1]))
            for e in result_2.value.indices.indices
            if e[0] < len(positions_2) and e[1] < len(positions_2)
        }
        expected_2 = self._compute_naive_neighbors(positions_2, uc_2, cutoff)
        assert expected_2.issubset(valid_2), f"Missing edges: {expected_2 - valid_2}"

    def test_small_cell_correctness(self, neighbor_list_impl):
        """Verify correctness by comparing with naive brute-force on small cell."""
        N = 5
        positions = jax.random.uniform(
            jax.random.key(42), (N, 3), minval=0.0, maxval=1.0
        )
        batch_mask = jnp.zeros(N, dtype=int)
        lattice_vectors = jnp.eye(3)[None] * 1.5
        unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
        cutoff = 1.2

        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            candidates=50, edges=50, cells=64, image_candidates=10000
        )

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.arange(N),
        )

        _sys, _cut = _make_systems(unitcell, jnp.array([cutoff]))
        while (
            result := jax.jit(as_result_function(neighbor_list_instance))(
                lh=lh,
                rh=None,
                systems=_sys,
                cutoffs=_cut,
            )
        ).failed_assertions:
            neighbor_list_instance = result.fix_or_raise(neighbor_list_instance)
        result.raise_assertion()

        actual_edges = {
            (int(e[0]), int(e[1]))
            for e in result.value.indices.indices
            if e[0] < N and e[1] < N
        }
        expected = self._compute_naive_neighbors(positions, unitcell, cutoff)
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
        unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
        cutoffs = jnp.array([0.8, 1.5])

        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            candidates=30, edges=20, cells=256, image_candidates=300
        )

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.arange(
                len(batch_mask),
            ),
        )

        _sys, _cut = _make_systems(unitcell, cutoffs)
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )
        result.raise_assertion()

        valid_edges = {
            (int(e[0]), int(e[1]))
            for e in result.value.indices.indices
            if e[0] < len(positions) and e[1] < len(positions)
        }

        # System 0 and System 1 edges
        assert (0, 1) in valid_edges and (1, 0) in valid_edges
        assert (2, 3) in valid_edges and (3, 2) in valid_edges
        # No cross-system edges
        for i in [0, 1]:
            for j in [2, 3]:
                assert (i, j) not in valid_edges and (j, i) not in valid_edges

    def test_unsorted_particles_with_images(self, neighbor_list_impl):
        """Test that unsorted particles work correctly when periodic images are needed.

        This verifies that _get_candidate_images doesn't require sorted candidates,
        contradicting the misleading comment on line 461 of neighborlist.py.
        """
        # Small cells with cutoff > 0.5*lattice to trigger image generation in _get_candidate_images
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
        unitcell = TriclinicUnitCell.from_matrix(
            jnp.eye(3)[None].repeat(3, axis=0) * 1.0
        )
        cutoffs = jnp.array([0.8, 0.8, 0.8])  # > 0.5 triggers images

        # Shuffle particles across systems: [S0, S1, S0, S2, S1, S2]
        shuffle = jnp.array([0, 2, 1, 4, 3, 5])

        instance_factory = neighbor_list_impl["instance_factory"]
        nl = instance_factory(candidates=50, edges=53, cells=8, image_candidates=1500)
        nl = jax.jit(as_result_function(nl))
        data = _make_lh(
            positions,
            batch_mask,
            jnp.arange(6),
        )

        _sys, _cut = _make_systems(unitcell, cutoffs)

        def get_edges(idx_order):
            rev_order = np.argsort(idx_order)
            reordered_index = tuple(range(len(idx_order)))
            reordered_data = jax.tree.map(
                lambda x: x[jnp.asarray(idx_order)], data.data
            )
            reordered = Table(reordered_index, reordered_data)
            result = nl(reordered, None, systems=_sys, cutoffs=_cut)
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
        unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
        cutoff = 0.8

        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            candidates=4, edges=30, cells=12, image_candidates=200
        )

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.arange(
                len(batch_mask),
            ),
        )

        _sys, _cut = _make_systems(unitcell, jnp.array([cutoff]))
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )
        result.raise_assertion()

        valid_edges = {
            (int(e[0]), int(e[1]))
            for e in result.value.indices.indices
            if e[0] < len(positions) and e[1] < len(positions)
        }
        expected = self._compute_naive_neighbors(positions, unitcell, cutoff)
        assert len(expected.difference(valid_edges)) == 0, (
            f"Missing edges: {expected - valid_edges}"
        )

    def test_self_interactions_with_periodic_images(self, neighbor_list_impl):
        """Verify self-interactions with images are included while non-image self-interactions are excluded."""
        # Single particle at origin in a small cell where self-images are within cutoff
        positions = jnp.array([[0.0, 0.0, 0.0]])
        batch_mask = jnp.array([0])

        # Cell size 0.5 Å, cutoff 0.55 Å => nearest self-images (distance 0.5) are within cutoff
        cell_size = 0.5
        lattice_vectors = jnp.eye(3)[None] * cell_size
        unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
        cutoff = 0.55

        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            candidates=1, edges=16, cells=8, image_candidates=27
        )

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.array([0]),
        )

        _sys, _cut = _make_systems(unitcell, jnp.array([cutoff]))
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )
        result.raise_assertion()
        edges = result.value

        # Extract valid edges (indices within valid range) as set of (i, j, shift_tuple)
        valid_mask = (edges.indices.indices == 0).all(
            axis=1
        )  # Only 1 particle with index 0
        valid_indices = np.asarray(edges.indices.indices[valid_mask])
        valid_shifts = np.asarray(edges.shifts[valid_mask, 0])
        edge_set = {
            (int(i), int(j), tuple(int(s) for s in shift))
            for (i, j), shift in zip(valid_indices, valid_shifts)
        }

        # Expected: self-interactions with 6 nearest periodic images (±1 along each axis)
        # distance = cell_size = 0.5 < cutoff = 0.55
        expected_edges: set[tuple[int, int, tuple[int, ...]]] = {
            (0, 0, (1, 0, 0)),
            (0, 0, (-1, 0, 0)),
            (0, 0, (0, 1, 0)),
            (0, 0, (0, -1, 0)),
            (0, 0, (0, 0, 1)),
            (0, 0, (0, 0, -1)),
        }

        # Self-interaction with zero shift must be excluded (drop_diagonal=True)
        assert (0, 0, (0, 0, 0)) not in edge_set

        # All expected self-image edges should be present
        assert edge_set == expected_edges, (
            f"Could not find edges {expected_edges - edge_set} or found unexpected edges {edge_set - expected_edges}"
        )

    def test_no_replication_with_infinite_cutoff(self, neighbor_list_impl):
        """Verify no periodic images are generated when cutoff is infinite."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        batch_mask = jnp.array([0, 0])

        lattice_vectors = jnp.eye(3)[None] * 1.0
        unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
        cutoff = jnp.inf

        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            candidates=4, edges=4, cells=8, image_candidates=4
        )

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.array([0, 1]),
        )

        _sys, _cut = _make_systems(unitcell, jnp.array([cutoff]))
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )
        result.raise_assertion()
        edges = result.value

        # Extract valid edges
        valid_mask = (edges.indices.indices < 2).all(axis=1)
        valid_indices = np.asarray(edges.indices.indices[valid_mask])
        valid_shifts = np.asarray(edges.shifts[valid_mask, 0])

        edge_set = {
            (int(i), int(j), tuple(int(s) for s in shift))
            for (i, j), shift in zip(valid_indices, valid_shifts)
        }

        # Expected: only direct neighbors (0,1) and (1,0) with zero shift, no periodic images
        expected_edges = {(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0))}

        assert edge_set == expected_edges, (
            f"Expected edges {expected_edges}, got {edge_set}"
        )

    def test_exclusion_only_applies_to_minimum_image(self, neighbor_list_impl):
        """Verify exclusion segments only exclude the minimum image convention interaction.

        Particles in the same exclusion segment should not interact via the minimum image
        (closest periodic image), but should interact via non-minimum periodic images.
        """
        # Two particles where the minimum image is NOT the direct (0,0,0) interaction
        positions = jnp.array([[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]])
        batch_mask = jnp.array([0, 0])

        # Small cell where periodic images are within cutoff
        # Direct distance (0,0,0): 0.3, Image distance (-1,0,0): 0.2
        # The minimum image is (-1,0,0) with distance 0.2
        cell_size = 0.5
        lattice_vectors = jnp.eye(3)[None] * cell_size
        unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
        cutoff = 0.4

        instance_factory = neighbor_list_impl["instance_factory"]
        neighbor_list_instance = instance_factory(
            candidates=4, edges=30, cells=8, image_candidates=200
        )

        # Same exclusion segment for both particles
        lh = _make_lh(positions, batch_mask, jnp.array([0, 0]))

        _sys, _cut = _make_systems(unitcell, jnp.array([cutoff]))
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )
        result.raise_assertion()
        edges = result.value

        # Extract valid edges
        valid_mask = (edges.indices.indices < 2).all(axis=1)
        valid_indices = np.asarray(edges.indices.indices[valid_mask])
        valid_shifts = np.asarray(edges.shifts[valid_mask, 0])

        edge_set = {
            (int(i), int(j), tuple(int(s) for s in shift))
            for (i, j), shift in zip(valid_indices, valid_shifts)
        }

        # Minimum image interactions should be EXCLUDED due to same exclusion segment
        # The minimum image for (0→1) is (-1,0,0) with distance 0.2
        # The minimum image for (1→0) is (1,0,0) with distance 0.2
        assert (0, 1, (-1, 0, 0)) not in edge_set
        assert (1, 0, (1, 0, 0)) not in edge_set

        # Non-minimum image interactions should be INCLUDED
        # The direct (0,0,0) interaction with distance 0.3 is not the minimum image
        assert (0, 1, (0, 0, 0)) in edge_set
        assert (1, 0, (0, 0, 0)) in edge_set


class TestRefineCutoffNeighborList:
    """Test cases for the RefinementNeighborList dataclass."""

    def _create_test_pointset(self, positions, batch_mask=None, exclusion_offset=0):
        if batch_mask is None:
            batch_mask = jnp.zeros(len(positions), dtype=int)
        return _make_lh(
            positions, batch_mask, jnp.arange(len(positions)) + exclusion_offset
        )

    def _create_candidate_edges(
        self, lh_indices, rh_indices, n_particles=None, shifts=None
    ):
        return _make_edges(lh_indices, rh_indices, n_particles, shifts)

    def test_basic_refinement(self):
        """Test basic edge refinement with simple candidate edges."""
        # Create simple linear positions
        lh_positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        lh = self._create_test_pointset(lh_positions)

        # Create candidate edges - all pairs
        lh_indices = jnp.array([0, 0, 0, 1, 1, 2])
        rh_indices = jnp.array([1, 2, 3, 2, 3, 3])
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        # Set cutoff to only allow nearest neighbors
        cutoffs = jnp.array([1.5])  # Should only include distance 1.0 edges

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates, avg_edges=FixedCapacity(10)
        )

        _sys, _cut = _make_systems(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )

        # Should only get edges with distance <= 1.5
        assert len(edges) > 0
        assert edges.degree == 2

        # Verify distances are within cutoff
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

    def test_with_unitcells(self):
        """Test refinement with periodic boundary conditions."""
        # Create a 2x2x2 grid in a 3x3x3 unit cell
        positions = jnp.array(
            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [0.5, 2.5, 0.5], [2.5, 2.5, 0.5]]
        )
        lh = self._create_test_pointset(positions)

        # Lattice vectors for 3x3x3 unit cell
        lattice_vectors = jnp.eye(3)[None] * 3.0
        unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)

        # Create candidates including some that need PBC
        lh_indices = jnp.array([0, 1, 2, 3])
        rh_indices = jnp.array([1, 0, 3, 2])
        # Add some shifts for PBC
        shifts = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        candidates = self._create_candidate_edges(lh_indices, rh_indices, shifts=shifts)

        cutoffs = jnp.array([2.1])  # Should include nearest neighbors

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates, avg_edges=FixedCapacity(10)
        )

        _sys, _cut = _make_systems(unitcell, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )
        assert len(edges) >= 0
        assert edges.degree == 2

    def test_with_rh_index_remap(self):
        """Test refinement with index remapping."""
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        rh_positions = jnp.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])

        lh = self._create_test_pointset(lh_positions)

        # Remap indices: only use subset of rh
        rh_index_remap = jnp.array([1, 2])  # Map to original lh indices

        # Create candidates
        lh_indices = jnp.array([0, 1])
        rh_indices = jnp.array([0, 1])  # These will be remapped
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([1.1])

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates, avg_edges=FixedCapacity(10)
        )

        rh, rh_remap = _make_rh(
            lh, rh_positions, jnp.zeros(len(rh_positions), dtype=int), rh_index_remap
        )
        _sys, _cut = _make_systems(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=rh,
            systems=_sys,
            cutoffs=_cut,
            rh_index_remap=rh_remap,
        )

        assert edges.degree == 2
        # Should handle remapping correctly

    def test_empty_candidates(self):
        """Test refinement with no candidate edges."""
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        lh = self._create_test_pointset(lh_positions)

        # Create candidates that won't pass the filters (too far apart)
        # This effectively tests the "no valid candidates" case
        lh_indices = jnp.array([0])
        rh_indices = jnp.array([1])
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        # Set very small cutoff so no edges pass the distance filter
        cutoffs = jnp.array([0.1])  # Much smaller than distance between points (1.0)

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates, avg_edges=FixedCapacity(5)
        )

        _sys, _cut = _make_systems(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )

        assert edges.degree == 2

        # Check that no valid edges exist (indices should be out of bounds fill values)
        # Valid indices should be < len(lh_positions) = 2
        valid_edges = edges.indices.indices[
            (edges.indices.indices[:, 0] < len(lh_positions))
            & (edges.indices.indices[:, 1] < len(lh_positions))
        ]
        assert len(valid_edges) == 0, "Should have no valid edges due to cutoff filter"

    def test_multiple_segments(self):
        """Test refinement with multiple segments/batches."""
        # Create two separate systems
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # System 0
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],  # System 1
            ]
        )
        batch_mask = jnp.array([0, 0, 1, 1])  # Two separate systems
        lh = self._create_test_pointset(positions, batch_mask)

        # Create candidates within and across systems
        lh_indices = jnp.array([0, 1, 2, 3, 0, 2])  # Mix of within/across systems
        rh_indices = jnp.array([1, 0, 3, 2, 2, 0])
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([1.5, 1.5])  # One cutoff per system

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates, avg_edges=FixedCapacity(10)
        )

        _sys, _cut = _make_systems(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )

        assert edges.degree == 2
        # Should only include edges within the same system (same inclusion segment)

    def test_exclusion_segments(self):
        """Test that exclusion segments prevent self-interactions."""
        lh_positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Same inclusion segment but different exclusion segments
        lh = _make_lh(lh_positions, jnp.array([0, 0, 0]), jnp.array([0, 1, 2]))

        # Create self-interaction candidates (should be excluded)
        lh_indices = jnp.array([0, 1, 2])
        rh_indices = jnp.array([0, 1, 2])  # Self-interactions
        candidates = self._create_candidate_edges(lh_indices, rh_indices)

        cutoffs = jnp.array([2.0])

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates, avg_edges=FixedCapacity(10)
        )

        _sys, _cut = _make_systems(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
            cutoffs=_cut,
        )

        # Should exclude all self-interactions due to exclusion segments
        valid_edges = edges.indices.indices[
            edges.indices.indices[:, 0] < len(lh_positions)
        ]
        valid_edges = valid_edges[valid_edges[:, 1] < len(lh_positions)]
        for i, edge in enumerate(valid_edges):
            if len(edge) >= 2:
                assert edge[0] != edge[1], f"Found self-interaction in edge {i}: {edge}"

    def test_gradient_computation(self):
        """Test that gradients can be computed through refined_neighborlist."""

        def loss_fn(positions):
            lh = self._create_test_pointset(positions)
            candidates = self._create_candidate_edges(jnp.array([0]), jnp.array([1]))
            refinement_nl = RefineCutoffNeighborList(
                candidates=candidates, avg_edges=FixedCapacity(5)
            )
            _sys, _cut = _make_systems(jnp.eye(3)[None] * 1000.0, jnp.array([2.0]))
            edges = refinement_nl(lh=lh, rh=None, systems=_sys, cutoffs=_cut)

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
            candidates=candidates, avg_edges=FixedCapacity(5)
        )

        _sys, _cut = _make_systems(jnp.eye(3)[None] * 1000.0, jnp.array([10.0]))
        edges = refinement_nl(lh=lh, rh=None, systems=_sys, cutoffs=_cut)

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


def _extract_valid_edge_set(edges: Edges, n_particles: int) -> set[tuple[int, int]]:
    """Extract the set of valid (non-padding) edges."""
    raw = edges.indices.indices
    mask = (raw[:, 0] < n_particles) & (raw[:, 1] < n_particles)
    return {(int(raw[i, 0]), int(raw[i, 1])) for i in range(len(raw)) if mask[i]}


def _run_nl_with_retry(nl, lh, rh, systems, cutoffs, rh_remap):
    """Run neighborlist call, retrying on capacity errors."""
    while (
        result := jax.jit(as_result_function(nl))(
            lh=lh, rh=rh, systems=systems, cutoffs=cutoffs, rh_index_remap=rh_remap
        )
    ).failed_assertions:
        nl = result.fix_or_raise(nl)
    result.raise_assertion()
    return result.value


class TestNeighborlistChanges:
    """Tests for the single-call neighborlist_changes utility."""

    @staticmethod
    def _make_nl(capacity=32):
        return DenseNearestNeighborList(
            avg_candidates=FixedCapacity(capacity),
            avg_edges=FixedCapacity(capacity),
            avg_image_candidates=FixedCapacity(capacity),
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
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
        systems, cutoffs = _make_systems(uc, jnp.array([3.0]))
        nl = self._make_nl()

        # --- reference: two separate calls ---
        # "after" lh: original positions with changes applied
        full_new_pos = positions.at[changed_idx].set(new_positions)
        lh_after = _make_lh(full_new_pos, batch)
        rh_after, remap_after = _make_rh(
            lh_after, new_positions, jnp.zeros(M, dtype=int), changed_idx
        )
        ref_after = _run_nl_with_retry(
            nl, lh_after, rh_after, systems, cutoffs, remap_after
        )

        # "before" lh: original positions
        lh_before = _make_lh(positions, batch)
        old_data = positions[changed_idx]
        rh_before, remap_before = _make_rh(
            lh_before, old_data, jnp.zeros(M, dtype=int), changed_idx
        )
        ref_removed = _run_nl_with_retry(
            nl, lh_before, rh_before, systems, cutoffs, remap_before
        )

        # --- combined call ---
        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(
            lh, new_positions, jnp.zeros(M, dtype=int), changed_idx
        )
        rh_with_indices = WithIndices(rh_remap, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems, cutoffs)

        added_set = _extract_valid_edge_set(result.added, N)
        removed_set = _extract_valid_edge_set(result.removed, N)
        ref_after_set = _extract_valid_edge_set(ref_after, N)
        ref_removed_set = _extract_valid_edge_set(ref_removed, N)

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
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
        systems, cutoffs = _make_systems(uc, jnp.array([1.5]))
        nl = self._make_nl()

        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(lh, new_pos, jnp.zeros(1, dtype=int), changed_idx)
        rh_with_indices = WithIndices(rh_remap, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems, cutoffs)

        removed = _extract_valid_edge_set(result.removed, 3)
        added = _extract_valid_edge_set(result.added, 3)

        # Before: particle 1 at (1,0,0) is near particle 0 at (0,0,0)
        assert (0, 1) in removed
        assert (1, 0) in removed
        # After: particle 1 at (4.5,0,0) is near particle 2 at (5,0,0)
        assert (1, 2) in added
        assert (2, 1) in added
        # Particle 1 should NOT be near particle 2 before, or near 0 after
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
        # Move particle 1 near particle 0 (same system) and near particle 2 (diff system)
        new_pos = jnp.array([[0.5, 0.0, 0.0]])
        changed_idx = jnp.array([1])

        batch = jnp.array([0, 0, 1, 1])
        uc = TriclinicUnitCell.from_matrix(
            jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 10.0])
        )
        systems, cutoffs = _make_systems(uc, jnp.array([1.5, 1.5]))
        nl = self._make_nl()

        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(lh, new_pos, jnp.zeros(1, dtype=int), changed_idx)
        rh_with_indices = WithIndices(rh_remap, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems, cutoffs)

        added = _extract_valid_edge_set(result.added, 4)
        # Should find edge (0,1) and (1,0) in system 0
        assert (0, 1) in added
        assert (1, 0) in added
        # Should NOT find cross-system edges (1,2) or (2,1)
        assert (1, 2) not in added
        assert (2, 1) not in added

    @pytest.mark.parametrize("compaction", [0.5, 0.75, 1.0])
    def test_compaction(self, compaction: float):
        """Different compaction fractions produce identical edge sets."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        new_pos = jnp.array([[4.5, 0.0, 0.0]])
        changed_idx = jnp.array([1])

        batch = jnp.zeros(3, dtype=int)
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
        systems, cutoffs = _make_systems(uc, jnp.array([1.5]))
        nl = self._make_nl()

        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(lh, new_pos, jnp.zeros(1, dtype=int), changed_idx)
        rh_with_indices = WithIndices(rh_remap, rh_table)
        result = neighborlist_changes(
            nl, lh, rh_with_indices, systems, cutoffs, compaction=compaction
        )

        removed = _extract_valid_edge_set(result.removed, 3)
        added = _extract_valid_edge_set(result.added, 3)
        assert (0, 1) in removed and (1, 0) in removed
        assert (1, 2) in added and (2, 1) in added

    def test_random_large(self):
        """Stress test with random positions and multiple changed particles."""
        N, M = 30, 5
        key = jax.random.key(123)
        k1, k2, k3 = jax.random.split(key, 3)

        positions = jax.random.uniform(k1, (N, 3), minval=0.0, maxval=9.0)
        changed_idx = jax.random.choice(k2, N, shape=(M,), replace=False)
        new_positions = jax.random.uniform(k3, (M, 3), minval=0.0, maxval=9.0)

        batch = jnp.zeros(N, dtype=int)
        uc = TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)
        systems, cutoffs = _make_systems(uc, jnp.array([3.0]))
        nl = self._make_nl(capacity=64)

        # reference
        full_new_pos = positions.at[changed_idx].set(new_positions)
        lh_after = _make_lh(full_new_pos, batch)
        rh_after, remap_after = _make_rh(
            lh_after, new_positions, jnp.zeros(M, dtype=int), changed_idx
        )
        ref_after = _run_nl_with_retry(
            nl, lh_after, rh_after, systems, cutoffs, remap_after
        )
        lh_before = _make_lh(positions, batch)
        rh_before, remap_before = _make_rh(
            lh_before, positions[changed_idx], jnp.zeros(M, dtype=int), changed_idx
        )
        ref_removed = _run_nl_with_retry(
            nl, lh_before, rh_before, systems, cutoffs, remap_before
        )

        # combined
        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(
            lh, new_positions, jnp.zeros(M, dtype=int), changed_idx
        )
        result = neighborlist_changes(
            nl, lh, WithIndices(rh_remap, rh_table), systems, cutoffs
        )

        assert _extract_valid_edge_set(result.added, N) == _extract_valid_edge_set(
            ref_after, N
        )
        assert _extract_valid_edge_set(result.removed, N) == _extract_valid_edge_set(
            ref_removed, N
        )
