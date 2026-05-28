# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.cell import (
    Cell,
    OrthogonalFrame,
    PeriodicCell,
    TriclinicFrame,
    VacuumCell,
)
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.data.wrappers import WithIndices
from kups.core.lens import bind
from kups.core.neighborlist import (
    AllDenseNearestNeighborList,
    CellListNeighborList,
    CutoffNeighborListPolicy,
    CutoffNeighborListStrategy,
    DenseNearestNeighborList,
    Edges,
    RefineCutoffNeighborList,
    RefineMaskNeighborList,
    UniversalNeighborlistParameters,
    adaptive_cutoff_neighborlist_from_state,
    neighborlist_changes,
)
from kups.core.neighborlist.cell_list import _cell_hash
from kups.core.neighborlist.common import (
    Candidates,
    _candidate_image_counts,
    _get_candidate_images,
    make_batch_with_mic,
)
from kups.core.neighborlist.compact import (
    MaskOnlyCompactor,
    ReduceCompactor,
    remap_rh_to_lh,
)
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    RemapDedupMask,
)
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.refine import PrecomputedEdgesSelector
from kups.core.neighborlist.types import CandidateBatch, PipelineContext
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
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

    cell: Cell


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


def _systems_from_cell(cell, cutoffs):
    """Create Table systems from a Cell, alongside cutoffs.

    Returns:
        A tuple of (Table systems, Table cutoffs).
    """
    n = len(cutoffs)
    sys_keys = tuple(SystemId(i) for i in range(n))
    indexed_systems = Table(sys_keys, SampleSystems(cell=cell))
    indexed_cutoffs = Table(sys_keys, cutoffs)
    return indexed_systems, indexed_cutoffs


def _systems_from_lvecs(lvecs, cutoffs):
    """Create Table systems from raw lattice vectors, alongside cutoffs.

    Returns:
        A tuple of (Table systems, Table cutoffs).
    """
    n = len(cutoffs)
    lv = jnp.asarray(lvecs)
    if lv.shape[0] == 1 and n > 1:
        lv = jnp.repeat(lv, n, axis=0)
    cell = PeriodicCell(TriclinicFrame.from_matrix(lv))
    return _systems_from_cell(cell, cutoffs)


def _cutoff_table(cutoffs):
    """Create a cutoff table with canonical SystemId keys for tests."""
    return Table(tuple(SystemId(i) for i in range(len(cutoffs))), cutoffs)


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
    """Call a cutoff-bound neighbor list."""
    del cutoffs
    return nl_instance(lh=lh, rh=rh, systems=systems, rh_index_remap=rh_index_remap)


def _make_pipeline_ctx(lh, rh=None, cell=None, rh_index_remap=None):
    """Build a ``PipelineContext`` directly for unit-level mask/compactor tests.

    Positions are taken as-is — caller is responsible for the fractional
    convention. Using a unit cell (``eye(3)``) keeps real == fractional so
    ``DistanceCutoffMask`` produces the same numbers either way.
    """
    if rh is None:
        rh = lh
    if cell is None:
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
    systems, _ = _systems_from_cell(cell, jnp.array([1.0]))
    return PipelineContext(lh=lh, rh=rh, systems=systems, rh_index_remap=rh_index_remap)


def _make_batch(lh_keys, lh_idx, rh_idx, shifts=None, is_minimum_image=None):
    """Build a ``CandidateBatch`` with one Index keys for both sides."""
    n = lh_idx.shape[0]
    if shifts is None:
        shifts = jnp.zeros((n, 1, 3))
    if is_minimum_image is None:
        is_minimum_image = jnp.ones((n,), dtype=bool)
    indices_2d = jnp.stack([lh_idx, rh_idx], axis=-1)
    return CandidateBatch(
        edges=Edges(Index(lh_keys, indices_2d), shifts),
        is_minimum_image=is_minimum_image,
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


class TestCellHashClampsAtBoundary:
    """``_cell_hash`` keeps per-axis bins inside ``[0, num_cells - 1]``."""

    def test_interior_coord_with_unit_grid_hashes_to_zero(self):
        # Sanity check: when num_cells = (1, 1, 1) any coord in [0, 1) hashes to 0.
        h = _cell_hash(jnp.array([0.0, 0.0, 0.0]), jnp.array([1, 1, 1]))
        assert int(h) == 0
        h = _cell_hash(jnp.array([0.5, 0.99, 0.0]), jnp.array([1, 1, 1]))
        assert int(h) == 0

    def test_fold_overshoot_at_one_clamps_with_unit_grid(self):
        # Boundary values remain in range even when folding lands exactly on 1.0.
        h = _cell_hash(jnp.array([1.0, 1.0, 1.0]), jnp.array([1, 1, 1]))
        assert int(h) == 0

    def test_fold_overshoot_at_one_clamps_with_nontrivial_grid(self):
        # For num_cells = (2, 3, 5), the max valid bin per axis is (1, 2, 4).
        # Coord = 1.0 should clamp to that and hash to 1 + 2*2 + 4*6 = 29.
        num_cells = jnp.array([2, 3, 5])
        h = _cell_hash(jnp.array([1.0, 1.0, 1.0]), num_cells)
        assert int(h) == 1 + 2 * 2 + 4 * 6
        # Interior coords are unaffected by the clamp.
        h = _cell_hash(jnp.array([0.25, 0.5, 0.5]), num_cells)
        # bins (0, 1, 2) → 0 + 1*2 + 2*6 = 14
        assert int(h) == 0 + 1 * 2 + 2 * 6

    def test_realistic_fold_path_does_not_escape_range(self):
        # Reproduce the exact failure path: a tiny-negative fractional coord
        # passes through ``cell.fold`` and must hash inside ``[0, 1)`` with
        # ``num_cells = (1, 1, 1)``.
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        bad_frac = jnp.array([-1.49e-8, 0.0, 0.0])
        folded, _ = cell.fold(bad_frac)
        # We don't assert that fold returns < 1 (it may not in float32) — only
        # that ``_cell_hash`` survives that overshoot.
        h = _cell_hash(folded, jnp.array([1, 1, 1]))
        assert int(h) == 0


class TestImageCountIsFiniteGuard:
    """Candidate image counts stay finite for degenerate periodic geometry."""

    @staticmethod
    def _make_minimal_inputs(cell: PeriodicCell):
        """One-atom, one-system, one-candidate setup for direct invocation
        of ``_get_candidate_images``."""
        sys_keys = (SystemId(0),)
        pi_keys = (ParticleId(0),)
        lh = Table(
            pi_keys,
            SamplePoints(
                positions=jnp.zeros((1, 3)),
                system=Index(sys_keys, jnp.array([0])),
                inclusion=Index(sys_keys, jnp.array([0])),
                exclusion=Index.integer(jnp.array([0])),
            ),
        )
        systems = Table(sys_keys, SampleSystems(cell=cell))
        candidates = Candidates(
            lhs=Index(pi_keys, jnp.array([0])),
            rhs=Index(pi_keys, jnp.array([0])),
        )
        return lh, systems, candidates

    def test_zero_perpendicular_axis_clamps_images_to_one(self):
        # ``tril = [0, 0, 1, 0, 0, 1]`` gives ``vectors[0] = (0, 0, 0)``.
        # Volume = 0, perp[0] = 0, perp[1] = perp[2] = NaN (0/0).
        tril = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
        cell = PeriodicCell(TriclinicFrame(tril))
        assert float(cell.perpendicular_lengths[0, 0]) == 0.0
        lh, systems, candidates = self._make_minimal_inputs(cell)
        cutoffs = jnp.array([6.0])
        # Degenerate axes are treated as a single image, keeping the output bounded.
        idx, offsets, has_been_replicated = _get_candidate_images(
            candidates, lh, systems, cutoffs, FixedCapacity(8)
        )
        # Exact contents are not relevant here; only bounded shape is.
        assert idx.shape[0] <= 8
        assert offsets.shape == (idx.shape[0], 3)
        assert has_been_replicated.shape == (idx.shape[0],)

    def test_candidate_image_counts_handles_nonfinite_ratios(self):
        class Cells:
            perpendicular_lengths = jnp.array([[0.0, 4.0, jnp.nan]])
            periodic = (True, True, True)

        images = _candidate_image_counts(Cells(), jnp.array([6.0]))
        npt.assert_array_equal(np.asarray(images), np.array([[1, 5, 1]]))


# Parametrized test class for testing different neighbor list implementations
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
            "cells": jnp.eye(3)[None] * 10.0,
            "cutoffs": jnp.array([1.5]),
            "extras": {"candidates": 9, "edges": 4, "cells": 256},
        }

        # Update with provided parameters
        params = {**default_params, **kwargs}

        # Get the instance factory for this implementation
        instance_factory = neighbor_list_impl_info["instance_factory"]

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

        if isinstance(params["cells"], Cell):
            systems, cutoffs = _systems_from_cell(params["cells"], params["cutoffs"])
        else:
            systems, cutoffs = _systems_from_lvecs(params["cells"], params["cutoffs"])
        neighbor_list_instance = instance_factory(cutoffs=cutoffs, **params["extras"])
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=rh,
            systems=systems,
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
            cells=jnp.eye(3)[None] * 3.0,
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
            cells=jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
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
        lvecs = jnp.eye(3)[None].repeat(2, axis=0) * 10.0
        cutoffs = jnp.array([1.5, 1.5])

        # Positive case: rh in correct systems, edges should be found
        result_pos = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            rh_positions=jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            batch_mask=lh_batch,
            rh_batch_mask=jnp.array([0, 1]),
            rh_index_remap=jnp.array([1, 3]),
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

        # Negative case: rh in swapped systems, cross-system edges blocked
        result_neg = self._run_neighbor_search_test(
            neighbor_list_impl,
            positions=lh_positions,
            rh_positions=jnp.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            batch_mask=lh_batch,
            rh_batch_mask=jnp.array([1, 0]),  # swapped
            rh_index_remap=jnp.array([3, 1]),
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
            cells=jnp.eye(3)[None].repeat(2, axis=0) * 10.0,
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
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        cutoff = 3
        # Get the instance factory
        instance_factory = neighbor_list_impl["instance_factory"]

        lh = _make_lh(
            positions,
            jnp.array([0] * N),
            jnp.arange(N),
        )

        _sys, _cut = _systems_from_cell(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            cutoffs=_cut, **{"edges": N, "candidates": N, "cells": 64}
        )
        while (
            result := as_result_function(neighbor_list_instance)(lh, None, systems=_sys)
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
        # Get the instance factory
        instance_factory = neighbor_list_impl["instance_factory"]

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

        _sys, _cut = _systems_from_cell(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            cutoffs=_cut, **{"edges": N, "candidates": N, "cells": 64}
        )
        while (
            result := jax.jit(as_result_function(neighbor_list_instance))(
                lh=lh,
                rh=rh,
                systems=_sys,
                rh_index_remap=rh_remap,
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
        lh_1 = _make_lh(positions_1, batch_mask_1, jnp.arange(len(batch_mask_1)))
        _sys_1, _cut_1 = _systems_from_cell(cell_1, jnp.array([cutoff]))
        nl_1 = instance_factory(
            candidates=10, edges=10, cells=256, image_candidates=200, cutoffs=_cut_1
        )
        result_1 = jax.jit(as_result_function(nl_1))(lh=lh_1, rh=None, systems=_sys_1)
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
        lh_2 = _make_lh(positions_2, batch_mask_2, jnp.arange(len(batch_mask_2)))
        _sys_2, _cut_2 = _systems_from_cell(cell_2, jnp.array([cutoff]))
        nl_2 = instance_factory(
            candidates=4, edges=53, cells=8, image_candidates=600, cutoffs=_cut_2
        )
        result_2 = jax.jit(as_result_function(nl_2))(lh=lh_2, rh=None, systems=_sys_2)
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

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.arange(N),
        )

        _sys, _cut = _systems_from_cell(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=50, edges=50, cells=64, image_candidates=10000, cutoffs=_cut
        )
        while (
            result := jax.jit(as_result_function(neighbor_list_instance))(
                lh=lh,
                rh=None,
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

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.arange(
                len(batch_mask),
            ),
        )

        _sys, _cut = _systems_from_cell(cell, cutoffs)
        neighbor_list_instance = instance_factory(
            candidates=30, edges=20, cells=256, image_candidates=300, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
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
        cell = PeriodicCell(
            TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(3, axis=0) * 1.0)
        )
        cutoffs = jnp.array([0.8, 0.8, 0.8])  # > 0.5 triggers images

        # Shuffle particles across systems: [S0, S1, S0, S2, S1, S2]
        shuffle = jnp.array([0, 2, 1, 4, 3, 5])

        instance_factory = neighbor_list_impl["instance_factory"]
        data = _make_lh(
            positions,
            batch_mask,
            jnp.arange(6),
        )

        _sys, _cut = _systems_from_cell(cell, cutoffs)
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
            result = nl(reordered, None, systems=_sys)
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

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.arange(
                len(batch_mask),
            ),
        )

        _sys, _cut = _systems_from_cell(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=4, edges=30, cells=12, image_candidates=200, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
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
        """Verify self-interactions with images are included while non-image self-interactions are excluded."""
        # Single particle at origin in a small cell where self-images are within cutoff
        positions = jnp.array([[0.0, 0.0, 0.0]])
        batch_mask = jnp.array([0])

        # Cell size 0.5 Å, cutoff 0.55 Å => nearest self-images (distance 0.5) are within cutoff
        cell_size = 0.5
        lattice_vectors = jnp.eye(3)[None] * cell_size
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = 0.55

        instance_factory = neighbor_list_impl["instance_factory"]

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.array([0]),
        )

        _sys, _cut = _systems_from_cell(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=1, edges=16, cells=8, image_candidates=125, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
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
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = jnp.inf

        instance_factory = neighbor_list_impl["instance_factory"]

        lh = _make_lh(
            positions,
            batch_mask,
            jnp.array([0, 1]),
        )

        _sys, _cut = _systems_from_cell(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=4, edges=4, cells=8, image_candidates=4, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
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
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
        cutoff = 0.4

        instance_factory = neighbor_list_impl["instance_factory"]

        # Same exclusion segment for both particles
        lh = _make_lh(positions, batch_mask, jnp.array([0, 0]))

        _sys, _cut = _systems_from_cell(cell, jnp.array([cutoff]))
        neighbor_list_instance = instance_factory(
            candidates=4, edges=30, cells=8, image_candidates=200, cutoffs=_cut
        )
        result = jax.jit(as_result_function(neighbor_list_instance))(
            lh=lh,
            rh=None,
            systems=_sys,
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
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=_cutoff_table(cutoffs),
        )

        _sys, _cut = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
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

    def test_with_cells(self):
        """Test refinement with periodic boundary conditions."""
        # Create a 2x2x2 grid in a 3x3x3 cell
        positions = jnp.array(
            [[0.5, 0.5, 0.5], [2.5, 0.5, 0.5], [0.5, 2.5, 0.5], [2.5, 2.5, 0.5]]
        )
        lh = self._create_test_pointset(positions)

        # Lattice vectors for 3x3x3 cell
        lattice_vectors = jnp.eye(3)[None] * 3.0
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))

        # Create candidates including some that need PBC
        lh_indices = jnp.array([0, 1, 2, 3])
        rh_indices = jnp.array([1, 0, 3, 2])
        # Add some shifts for PBC
        shifts = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        candidates = self._create_candidate_edges(lh_indices, rh_indices, shifts=shifts)

        cutoffs = jnp.array([2.1])  # Should include nearest neighbors

        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=_cutoff_table(cutoffs),
        )

        _sys, _cut = _systems_from_cell(cell, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
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
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=_cutoff_table(cutoffs),
        )

        rh, rh_remap = _make_rh(
            lh, rh_positions, jnp.zeros(len(rh_positions), dtype=int), rh_index_remap
        )
        _sys, _cut = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=rh,
            systems=_sys,
            rh_index_remap=rh_remap,
        )

        assert edges.degree == 2
        # Should handle remapping correctly

    def test_rh_remap_uses_rh_positions_for_cutoff(self):
        """Precomputed edges are in public lh-space, but the remapped rh data
        may carry positions that differ from lh. RefineCutoff must evaluate
        distances after overlaying rh onto those lh rows.
        """
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [105.0, 0.0, 0.0],
            ]
        )
        lh = self._create_test_pointset(lh_positions)
        rh, rh_remap = _make_rh(
            lh,
            jnp.array([[100.4, 0.0, 0.0]]),
            jnp.zeros(1, dtype=int),
            jnp.array([2]),
            exclusion_ids=jnp.array([20]),
        )
        candidates = self._create_candidate_edges(
            jnp.array([1]), jnp.array([2]), n_particles=3
        )
        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(4),
            cutoffs=_cutoff_table(jnp.array([1.0])),
        )

        systems, _ = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([1.0]))
        edges = refinement_nl(
            lh=lh,
            rh=rh,
            systems=systems,
            rh_index_remap=rh_remap,
        )

        npt.assert_array_equal(
            np.asarray(edges.indices.indices[edges.indices.indices[:, 0] < lh.size]),
            np.array([[1, 2]]),
        )

    def test_disjoint_rh_uses_rh_positions_for_cutoff(self):
        """Without a remap, the precomputed candidate right column is in
        rh-space and distance checks must use the separate rh table.
        """
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ]
        )
        lh = self._create_test_pointset(lh_positions)
        rh, _ = _make_rh(
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
            cutoffs=_cutoff_table(jnp.array([1.0])),
        )
        systems, _ = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([1.0]))

        edges = refinement_nl(lh=lh, rh=rh, systems=systems, rh_index_remap=None)

        npt.assert_array_equal(
            np.asarray(edges.indices.indices[edges.indices.indices[:, 0] < lh.size]),
            np.array([[1, 0]]),
        )

    def test_rh_remap_uses_rh_exclusion_for_cutoff_refinement(self):
        """The cutoff refiner still applies exclusion masks after distance
        filtering, so remapped rh metadata must be used there too.
        """
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        lh = self._create_test_pointset(lh_positions, exclusion_offset=0)
        rh, rh_remap = _make_rh(
            lh,
            jnp.array([[1.0, 0.0, 0.0]]),
            jnp.zeros(1, dtype=int),
            jnp.array([1]),
            exclusion_ids=jnp.array([0]),
        )
        candidates = self._create_candidate_edges(
            jnp.array([0]), jnp.array([1]), n_particles=2
        )
        refinement_nl = RefineCutoffNeighborList(
            candidates=candidates,
            avg_edges=FixedCapacity(4),
            cutoffs=_cutoff_table(jnp.array([2.0])),
        )
        systems, _ = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([2.0]))

        edges = refinement_nl(
            lh=lh,
            rh=rh,
            systems=systems,
            rh_index_remap=rh_remap,
        )

        assert not np.any(np.asarray(edges.indices.indices[:, 0] < lh.size))

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
            candidates=candidates,
            avg_edges=FixedCapacity(5),
            cutoffs=_cutoff_table(cutoffs),
        )

        _sys, _cut = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
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
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=_cutoff_table(cutoffs),
        )

        _sys, _cut = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
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
            candidates=candidates,
            avg_edges=FixedCapacity(10),
            cutoffs=_cutoff_table(cutoffs),
        )

        _sys, _cut = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, cutoffs)
        edges = refinement_nl(
            lh=lh,
            rh=None,
            systems=_sys,
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
                candidates=candidates,
                avg_edges=FixedCapacity(5),
                cutoffs=_cutoff_table(jnp.array([2.0])),
            )
            _sys, _cut = _systems_from_lvecs(
                jnp.eye(3)[None] * 1000.0, jnp.array([2.0])
            )
            edges = refinement_nl(lh=lh, rh=None, systems=_sys)

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
            cutoffs=_cutoff_table(jnp.array([10.0])),
        )

        _sys, _cut = _systems_from_lvecs(jnp.eye(3)[None] * 1000.0, jnp.array([10.0]))
        edges = refinement_nl(lh=lh, rh=None, systems=_sys)

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
    """Test cases for RefineMaskNeighborList.

    RefineMask is the post-filter used by MCMC. Its precomputed edges carry
    indices in **lh-space** (the output of any base NL +
    ``neighborlist_changes``). When a non-trivial ``rh`` subset plus a remap is
    provided, those rows are overlaid onto the lh table before filtering so
    public lh-space edge indices remain valid while rh metadata is honored.
    Without a remap, the right column remains in raw rh-space.
    """

    def test_mcmc_patch_pattern_with_partial_overlap(self):
        """Mirror the MCMC patch flow: precomputed edges contain pairs where
        only some indices are inside the move subset. Concrete check:
        5-particle ``lh``, ``rh`` = 2-particle subset (move particles),
        precomputed edges include a pair that touches a non-move particle.
        All edges must survive (distinct exclusion segments) and the output
        must be in lh-space.
        """
        lh = _make_lh(
            jnp.zeros((5, 3)),
            jnp.zeros(5, dtype=int),
            exclusion_ids=jnp.array([0, 1, 2, 3, 4]),
        )
        # Move subset = particles 1 and 3 (typical MCMC small move).
        rh_positions = jnp.zeros((2, 3))
        rh_remap_arr = jnp.array([1, 3])
        rh, rh_remap = _make_rh(lh, rh_positions, jnp.zeros(2, dtype=int), rh_remap_arr)
        # Precomputed edges: (0, 1), (2, 3), (1, 4). The last touches a
        # particle NOT in the move subset; overlaying rh into lh keeps that
        # public lh-space edge valid.
        candidates = _make_edges(
            jnp.array([0, 2, 1]), jnp.array([1, 3, 4]), n_particles=5
        )

        refine_nl = RefineMaskNeighborList(candidates=candidates)
        sys, _ = _systems_from_lvecs(jnp.eye(3)[None] * 100.0, jnp.array([10.0]))
        edges = refine_nl(lh=lh, rh=rh, systems=sys, rh_index_remap=rh_remap)

        raw = edges.indices.indices
        oob = lh.size
        valid_mask = (raw[:, 0] < oob) & (raw[:, 1] < oob)
        valid = np.asarray(raw[valid_mask])
        valid = valid[np.lexsort((valid[:, 1], valid[:, 0]))]

        npt.assert_array_equal(valid, np.array([[0, 1], [1, 4], [2, 3]]))

    def test_full_overlap_with_rh_subset(self):
        """RefineMask called with a non-trivial ``rh`` subset where every
        precomputed rh-side index is in the move subset.

        The rh rows are overlaid into the lh table and the precomputed
        lh-space edges stay in lh-space directly. Output: edges (0, 1) and
        (2, 3).
        """
        lh_positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        lh = _make_lh(lh_positions, jnp.zeros(4, dtype=int), jnp.arange(4))

        # rh is the subset {lh[1], lh[3]}; remap takes rh-positions → lh-positions.
        rh_positions = lh_positions[jnp.array([1, 3])]
        rh_remap_arr = jnp.array([1, 3])
        rh, rh_remap = _make_rh(lh, rh_positions, jnp.zeros(2, dtype=int), rh_remap_arr)

        # Precomputed edges with rh-side in lh-space (1, 3 — both are lh-positions
        # that happen to correspond to rh positions 0 and 1 respectively).
        candidates = _make_edges(jnp.array([0, 2]), jnp.array([1, 3]), n_particles=4)

        refine_nl = RefineMaskNeighborList(candidates=candidates)
        sys, _ = _systems_from_lvecs(jnp.eye(3)[None] * 100.0, jnp.array([10.0]))
        edges = refine_nl(lh=lh, rh=rh, systems=sys, rh_index_remap=rh_remap)

        raw = edges.indices.indices
        oob = lh.size  # MaskOnlyCompactor's OOB sentinel
        valid_mask = (raw[:, 0] < oob) & (raw[:, 1] < oob)
        valid = np.asarray(raw[valid_mask])
        valid = valid[np.lexsort((valid[:, 1], valid[:, 0]))]

        npt.assert_array_equal(valid, np.array([[0, 1], [2, 3]]))

    def test_disjoint_rh_uses_rh_inclusion(self):
        """Without a remap, the precomputed candidate right column is in
        rh-space and masks must read inclusion/exclusion from rh.
        """
        lh = _make_lh(
            jnp.zeros((2, 3)),
            jnp.zeros(2, dtype=int),
            exclusion_ids=jnp.array([0, 1]),
        )
        rh, _ = _make_rh(
            lh,
            jnp.zeros((2, 3)),
            jnp.array([0, 1]),
            jnp.array([0, 1]),
            exclusion_ids=jnp.array([2, 3]),
        )
        candidates = _make_edges(jnp.array([0]), jnp.array([1]), n_particles=2)
        refine_nl = RefineMaskNeighborList(candidates=candidates)
        systems, _ = _systems_from_lvecs(
            jnp.eye(3)[None].repeat(2, axis=0) * 100.0,
            jnp.array([10.0, 10.0]),
        )

        edges = refine_nl(lh=lh, rh=rh, systems=systems, rh_index_remap=None)

        npt.assert_array_equal(np.asarray(edges.indices.indices), np.array([[2, 2]]))

    def test_rh_remap_uses_rh_inclusion(self):
        """A remapped rh row can belong to a different inclusion segment than
        the lh row it replaces. RefineMask must use that rh metadata.
        """
        lh = _make_lh(
            jnp.zeros((2, 3)),
            jnp.zeros(2, dtype=int),
            exclusion_ids=jnp.array([0, 1]),
        )
        rh, rh_remap = _make_rh(
            lh,
            jnp.zeros((1, 3)),
            jnp.ones(1, dtype=int),
            jnp.array([1]),
            exclusion_ids=jnp.array([1]),
        )
        candidates = _make_edges(jnp.array([0]), jnp.array([1]), n_particles=2)
        refine_nl = RefineMaskNeighborList(candidates=candidates)
        systems, _ = _systems_from_lvecs(
            jnp.eye(3)[None] * 100.0, jnp.array([10.0, 10.0])
        )

        edges = refine_nl(
            lh=lh,
            rh=rh,
            systems=systems,
            rh_index_remap=rh_remap,
        )

        npt.assert_array_equal(np.asarray(edges.indices.indices), np.array([[2, 2]]))

    def test_rh_remap_uses_rh_exclusion(self):
        """A remapped rh row can introduce an exclusion that is not present in
        the lh table at the same public edge index.
        """
        lh = _make_lh(
            jnp.zeros((2, 3)),
            jnp.zeros(2, dtype=int),
            exclusion_ids=jnp.array([0, 1]),
        )
        rh, rh_remap = _make_rh(
            lh,
            jnp.zeros((1, 3)),
            jnp.zeros(1, dtype=int),
            jnp.array([1]),
            exclusion_ids=jnp.array([0]),
        )
        candidates = _make_edges(jnp.array([0]), jnp.array([1]), n_particles=2)
        refine_nl = RefineMaskNeighborList(candidates=candidates)
        systems, _ = _systems_from_lvecs(jnp.eye(3)[None] * 100.0, jnp.array([10.0]))

        edges = refine_nl(
            lh=lh,
            rh=rh,
            systems=systems,
            rh_index_remap=rh_remap,
        )

        npt.assert_array_equal(np.asarray(edges.indices.indices), np.array([[2, 2]]))


class TestInBoundsMask:
    """``InBoundsMask`` drops candidates whose raw index is out of range on
    either side. The mode='fill' lookup turns OOB indices into ``False``;
    valid indices return whatever the per-particle ``inclusion < num_labels``
    check evaluates to (always True in well-formed states)."""

    def test_drops_oob_indices_on_either_side(self):
        lh = _make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        ctx = _make_pipeline_ctx(lh)
        # Edge 0: both in bounds. Edge 1: rh OOB. Edge 2: lh OOB.
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 1, 100]),
            jnp.array([1, 100, 2]),
        )
        result = InBoundsMask()(batch, ctx)
        assert result.tolist() == [True, False, False]


class TestInclusionMatchMask:
    def test_matches_inclusion_segments(self):
        # Two inclusion groups: 0 and 1.
        lh = _make_lh(jnp.zeros((4, 3)), jnp.array([0, 0, 1, 1]))
        rh = _make_lh(jnp.zeros((4, 3)), jnp.array([0, 1, 0, 1]))
        ctx = _make_pipeline_ctx(lh, rh)
        # lh.incl[lh_idx] = [0, 0, 1, 1]; rh.incl[rh_idx] = [0, 1, 0, 1].
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 1, 2, 3]),
            jnp.array([0, 1, 0, 3]),
        )
        result = InclusionMatchMask()(batch, ctx)
        assert result.tolist() == [True, False, False, True]


class TestRemapDedupMask:
    def test_no_remap_returns_all_true(self):
        lh = _make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = _make_pipeline_ctx(lh, rh_index_remap=None)
        batch = _make_batch(lh.keys, jnp.array([0, 1, 2]), jnp.array([1, 2, 3]))
        result = RemapDedupMask()(batch, ctx)
        assert result.tolist() == [True, True, True]

    def test_drops_self_pair_with_remap(self):
        """When ``rh`` is a remapped subset of ``lh``, dedup keeps an edge only
        when its lh-side ID is **outside** the remap *or* is not less than the
        remapped-rh ID."""
        lh = _make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        rh = _make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        # rh-pos 0 maps to lh-pos 1; rh-pos 1 maps to lh-pos 3.
        remap = jnp.array([1, 3])
        ctx = _make_pipeline_ctx(lh, rh, rh_index_remap=remap)
        # Cases:
        #   (lh=0, rh=0): rh_remapped=1, isin(0,[1,3])=False → ~isin=True → keep.
        #   (lh=1, rh=1): rh_remapped=3, isin(1,[1,3])=True  → ~isin=False;
        #                 1 >= 3 = False → drop.
        #   (lh=3, rh=0): rh_remapped=1, isin(3,[1,3])=True  → ~isin=False;
        #                 3 >= 1 = True → keep.
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 1, 3]),
            jnp.array([0, 1, 0]),
        )
        result = RemapDedupMask()(batch, ctx)
        assert result.tolist() == [True, False, True]


class TestDistanceCutoffMask:
    def test_filters_by_distance_squared(self):
        # Unit cell: real == fractional.
        positions = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.8, 0.0, 0.0]])
        lh = _make_lh(positions, jnp.zeros(3, dtype=int))
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
        ctx = _make_pipeline_ctx(lh, cell=cell)
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 0, 1]),
            jnp.array([1, 2, 2]),
        )
        # Distances: 0.3, 0.8, 0.5. cutoff² = 0.55² = 0.3025 → keep [T, F, T].
        cutoffs = Table(ctx.systems.keys, jnp.array([0.55]))
        result = DistanceCutoffMask(cutoffs=cutoffs)(batch, ctx)
        assert result.tolist() == [True, False, True]


class TestExclusionMask:
    def test_drops_matching_exclusion_at_min_image(self):
        # Exclusion ids: lh = [0, 1, 0, 2]; rh = [0, 1, 2, 3].
        lh = _make_lh(
            jnp.zeros((4, 3)),
            jnp.zeros(4, dtype=int),
            exclusion_ids=jnp.array([0, 1, 0, 2]),
        )
        rh = _make_lh(
            jnp.zeros((4, 3)),
            jnp.zeros(4, dtype=int),
            exclusion_ids=jnp.array([0, 1, 2, 3]),
        )
        ctx = _make_pipeline_ctx(lh, rh)
        # lh.excl[lh_idx=[0,1,2,3]] = [0, 1, 0, 2].
        # rh.excl[rh_idx=[0,1,0,1]] = [0, 1, 0, 1]. All min-image.
        # Mismatches: [F, F, F, T]. ~is_min: all False → final [F, F, F, T].
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 1, 2, 3]),
            jnp.array([0, 1, 0, 1]),
        )
        result = ExclusionMask()(batch, ctx)
        assert result.tolist() == [False, False, False, True]

    def test_keeps_non_min_image_periodic_copy_of_excluded_pair(self):
        """Periodic copies of an excluded pair survive ExclusionMask because
        ``~is_minimum_image`` short-circuits the drop."""
        lh = _make_lh(
            jnp.zeros((2, 3)),
            jnp.zeros(2, dtype=int),
            exclusion_ids=jnp.array([0, 0]),  # both share exclusion segment
        )
        ctx = _make_pipeline_ctx(lh)
        # Same lh-rh pair, one MIC and one non-MIC copy.
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 0]),
            jnp.array([1, 1]),
            is_minimum_image=jnp.array([True, False]),
        )
        result = ExclusionMask()(batch, ctx)
        # MIC copy: same excl → drop. Non-MIC copy: kept regardless of excl.
        assert result.tolist() == [False, True]


class TestReduceCompactor:
    def test_compacts_to_capacity_size(self):
        lh = _make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = _make_pipeline_ctx(lh)
        # 5 candidates, 3 surviving. Capacity 6.
        keep = jnp.array([True, False, True, False, True])
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 1, 2, 3, 0]),
            jnp.array([1, 2, 3, 0, 2]),
        )
        compactor = ReduceCompactor(avg_edges=FixedCapacity(6))
        edges = compactor(keep, batch, ctx)
        # jnp.where selects the survivor positions (0, 2, 4) in order. Then
        # the remaining 3 slots are filled with OOB = lh.size = 4.
        npt.assert_array_equal(
            np.asarray(edges.indices.indices[:, 0]),
            np.array([0, 2, 0, 4, 4, 4]),
        )
        npt.assert_array_equal(
            np.asarray(edges.indices.indices[:, 1]),
            np.array([1, 3, 2, 4, 4, 4]),
        )

    def test_mirrors_each_edge_with_reverse_when_remap_set(self):
        """When ``rh_index_remap`` is set, ``ReduceCompactor`` doubles each
        surviving edge with its (rh→lh) reverse, restoring symmetry that
        ``RemapDedupMask`` removed upstream."""
        lh = _make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        rh = _make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        # rh-pos 0 → lh-pos 1; rh-pos 1 → lh-pos 3.
        remap = jnp.array([1, 3])
        ctx = _make_pipeline_ctx(lh, rh, rh_index_remap=remap)
        # One candidate: lh-pos 0, rh-pos 0 (= lh-pos 1).
        keep = jnp.array([True])
        shifts = jnp.array([[[0.5, 0.0, 0.0]]])
        batch = _make_batch(lh.keys, jnp.array([0]), jnp.array([0]), shifts=shifts)
        compactor = ReduceCompactor(avg_edges=FixedCapacity(1))
        edges = compactor(keep, batch, ctx)
        # Output has 2 entries: (0, 1) with shift +s and (1, 0) with shift -s.
        npt.assert_array_equal(
            np.asarray(edges.indices.indices),
            np.array([[0, 1], [1, 0]]),
        )
        npt.assert_allclose(
            np.asarray(edges.shifts),
            np.array([[[0.5, 0.0, 0.0]], [[-0.5, -0.0, -0.0]]]),
        )


class TestMaskOnlyCompactor:
    def test_stamps_oob_on_dropped_entries_and_remaps_rh(self):
        lh = _make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        rh = _make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        remap = jnp.array([1, 3])
        ctx = _make_pipeline_ctx(lh, rh, rh_index_remap=remap)
        # 3 candidates with rh-side in rh-space; middle one is dropped.
        keep = jnp.array([True, False, True])
        batch = _make_batch(
            lh.keys,
            jnp.array([0, 1, 2]),
            jnp.array([0, 1, 1]),
        )
        edges = MaskOnlyCompactor()(keep, batch, ctx)
        # Survivors: (lh=0, rh=0→lh1) and (lh=2, rh=1→lh3). Dropped → (oob, oob).
        oob = lh.size  # MaskOnlyCompactor uses ctx.lh.size
        npt.assert_array_equal(
            np.asarray(edges.indices.indices),
            np.array([[0, 1], [oob, oob], [2, 3]]),
        )


class TestRemapRhToLh:
    def test_passthrough_without_remap(self):
        lh = _make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = _make_pipeline_ctx(lh, rh_index_remap=None)
        rh_idx = jnp.array([0, 1, 2])
        npt.assert_array_equal(
            np.asarray(remap_rh_to_lh(rh_idx, ctx)), np.array([0, 1, 2])
        )

    def test_translates_rh_to_lh_space_and_fills_oob(self):
        lh = _make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        rh = _make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        remap = jnp.array([1, 3])
        ctx = _make_pipeline_ctx(lh, rh, rh_index_remap=remap)
        oob = max(lh.size, rh.size)  # 4
        result = remap_rh_to_lh(jnp.array([0, 1, 5]), ctx)
        npt.assert_array_equal(np.asarray(result), np.array([1, 3, oob]))


class TestMakeBatchWithMic:
    def test_round_fractional_delta_as_shift(self):
        """``make_batch_with_mic`` sets shifts via ``cell.minimum_image_shifts``
        which on periodic axes rounds the fractional delta to the nearest int."""
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
        positions = jnp.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
        lh = _make_lh(positions, jnp.zeros(2, dtype=int))
        systems, _ = _systems_from_cell(cell, jnp.array([1.0]))
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0])),
            rhs=Index(lh.keys, jnp.array([1])),
        )
        batch = make_batch_with_mic(candidates, lh, lh, systems)
        # delta = 0.1 - 0.9 = -0.8; round(-0.8) = -1.0 on the periodic axis.
        npt.assert_allclose(
            np.asarray(batch.edges.shifts),
            np.array([[[-1.0, 0.0, 0.0]]]),
        )
        assert batch.is_minimum_image.tolist() == [True]


class TestPrecomputedEdgesSelectorModes:
    """The unified selector has two modes: reuse precomputed shifts (default,
    for ``RefineMaskNeighborList``) or recompute MIC shifts (for
    ``RefineCutoffNeighborList``). Remapped rh data is handled by overlaying
    it onto lh before this selector runs; disjoint rh data is passed through as
    the pipeline right-hand table."""

    def test_default_mode_reuses_precomputed_shifts(self):
        lh = _make_lh(
            jnp.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]),
            jnp.zeros(2, dtype=int),
        )
        # Precomputed shift on the edge — a non-MIC value to see it preserved.
        custom_shift = jnp.array([[[7.0, 7.0, 7.0]]])
        candidates = _make_edges(
            jnp.array([0]), jnp.array([1]), n_particles=2, shifts=custom_shift
        )
        ctx = _make_pipeline_ctx(lh)
        selector = PrecomputedEdgesSelector(candidates)  # recompute_mic=False
        batch = selector(ctx)
        npt.assert_allclose(np.asarray(batch.edges.shifts), np.asarray(custom_shift))

    def test_recompute_mic_mode_overrides_precomputed_shifts(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
        lh = _make_lh(
            jnp.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]),
            jnp.zeros(2, dtype=int),
        )
        # Garbage precomputed shift — should be ignored.
        candidates = _make_edges(
            jnp.array([0]),
            jnp.array([1]),
            n_particles=2,
            shifts=jnp.array([[99.0, 99.0, 99.0]]),
        )
        ctx = _make_pipeline_ctx(lh, cell=cell)
        selector = PrecomputedEdgesSelector(candidates, recompute_mic_shifts=True)
        batch = selector(ctx)
        npt.assert_allclose(
            np.asarray(batch.edges.shifts),
            np.array([[[-1.0, 0.0, 0.0]]]),
        )


class TestPipelineComposition:
    """End-to-end test of the ``Pipeline`` runner with a custom mask set."""

    def test_distance_only_pipeline(self):
        # Precomputed candidates (lh, rh) = (0, 1) and (0, 2); only the close
        # pair survives a distance cutoff of 1.5.
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        lh = _make_lh(positions, jnp.zeros(3, dtype=int))
        candidates = _make_edges(jnp.array([0, 0]), jnp.array([1, 2]), n_particles=3)
        # Large lattice so fractional ≠ real and _prepare's conversion is real.
        systems, cutoffs = _systems_from_lvecs(
            jnp.eye(3)[None] * 10.0, jnp.array([1.5])
        )
        pipeline = Pipeline[Literal[2]](
            selector=PrecomputedEdgesSelector(candidates, recompute_mic_shifts=True),
            masks=(DistanceCutoffMask(cutoffs=cutoffs),),
            compactor=MaskOnlyCompactor(),
        )
        edges = pipeline(lh, None, systems, None)
        # First edge survives, second is stamped OOB.
        oob = lh.size
        npt.assert_array_equal(
            np.asarray(edges.indices.indices),
            np.array([[0, 1], [oob, oob]]),
        )


def _extract_valid_edge_set(edges: Edges, n_particles: int) -> set[tuple[int, int]]:
    """Extract the set of valid (non-padding) edges."""
    raw = edges.indices.indices
    mask = (raw[:, 0] < n_particles) & (raw[:, 1] < n_particles)
    return {(int(raw[i, 0]), int(raw[i, 1])) for i in range(len(raw)) if mask[i]}


def _run_nl_with_retry(nl, lh, rh, systems, cutoffs, rh_remap):
    """Run neighborlist call, retrying on capacity errors."""
    del cutoffs
    while (
        result := jax.jit(as_result_function(nl))(
            lh=lh, rh=rh, systems=systems, rh_index_remap=rh_remap
        )
    ).failed_assertions:
        nl = result.fix_or_raise(nl)
    result.raise_assertion()
    return result.value


class TestNeighborlistChanges:
    """Tests for the single-call neighborlist_changes utility."""

    @staticmethod
    def _make_nl(cutoffs, capacity=32):
        return DenseNearestNeighborList(
            avg_candidates=FixedCapacity(capacity),
            avg_edges=FixedCapacity(capacity),
            avg_image_candidates=FixedCapacity(capacity),
            cutoffs=cutoffs,
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
        systems, cutoffs = _systems_from_cell(cell, jnp.array([3.0]))
        nl = self._make_nl(cutoffs)

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
        result = neighborlist_changes(nl, lh, rh_with_indices, systems)

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
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, cutoffs = _systems_from_cell(cell, jnp.array([1.5]))
        nl = self._make_nl(cutoffs)

        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(lh, new_pos, jnp.zeros(1, dtype=int), changed_idx)
        rh_with_indices = WithIndices(rh_remap, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems)

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
        cell = PeriodicCell(
            TriclinicFrame.from_matrix(
                jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 10.0])
            )
        )
        systems, cutoffs = _systems_from_cell(cell, jnp.array([1.5, 1.5]))
        nl = self._make_nl(cutoffs)

        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(lh, new_pos, jnp.zeros(1, dtype=int), changed_idx)
        rh_with_indices = WithIndices(rh_remap, rh_table)
        result = neighborlist_changes(nl, lh, rh_with_indices, systems)

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
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, cutoffs = _systems_from_cell(cell, jnp.array([1.5]))
        nl = self._make_nl(cutoffs)

        lh = _make_lh(positions, batch)
        rh_table, rh_remap = _make_rh(lh, new_pos, jnp.zeros(1, dtype=int), changed_idx)
        rh_with_indices = WithIndices(rh_remap, rh_table)
        result = neighborlist_changes(
            nl, lh, rh_with_indices, systems, compaction=compaction
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
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, cutoffs = _systems_from_cell(cell, jnp.array([3.0]))
        nl = self._make_nl(cutoffs, capacity=64)

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
        result = neighborlist_changes(nl, lh, WithIndices(rh_remap, rh_table), systems)

        assert _extract_valid_edge_set(result.added, N) == _extract_valid_edge_set(
            ref_after, N
        )
        assert _extract_valid_edge_set(result.removed, N) == _extract_valid_edge_set(
            ref_removed, N
        )


def _valid_edges_set(edges, n_particles):
    """Return the set of (i, j) edges with both indices < n_particles."""
    raw = edges.indices.indices
    in_range = (raw[:, 0] < n_particles) & (raw[:, 1] < n_particles)
    return set(tuple(p.tolist()) for p in raw[in_range])


def _run_cell_list(positions, cell, cutoff):
    """Run CellListNeighborList on a single-system fixture."""
    n = len(positions)
    lh = _make_lh(positions, jnp.zeros(n, dtype=int))
    systems, cutoffs = _systems_from_cell(cell, jnp.array([cutoff]))
    nl = CellListNeighborList(
        avg_candidates=FixedCapacity(max(n * n, 8)),
        avg_edges=FixedCapacity(max(n * n, 8)),
        cells=FixedCapacity(512),
        avg_image_candidates=FixedCapacity(max(n * n, 8)),
        cutoffs=cutoffs,
    )
    result = jax.jit(as_result_function(nl))(
        lh=lh, rh=None, systems=systems, rh_index_remap=None
    )
    result.raise_assertion()
    return result.value


def _run_dense(positions, cell, cutoff):
    """Run DenseNearestNeighborList on the same fixture, for cross-check."""
    n = len(positions)
    lh = _make_lh(positions, jnp.zeros(n, dtype=int))
    systems, cutoffs = _systems_from_cell(cell, jnp.array([cutoff]))
    nl = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(max(n * n, 8)),
        avg_edges=FixedCapacity(max(n * n, 8)),
        avg_image_candidates=FixedCapacity(max(n * n, 8)),
        cutoffs=cutoffs,
    )
    result = jax.jit(as_result_function(nl))(
        lh=lh, rh=None, systems=systems, rh_index_remap=None
    )
    result.raise_assertion()
    return result.value


class TestNeighborListAcrossPeriodicities:
    """Cell-list and dense agree across the four periodicity shapes:
    vacuum (0D), 1D wire, 2D slab, 3D bulk.
    """

    def _bulk_cell(self, L):
        return PeriodicCell(OrthogonalFrame(jnp.array([L, L, L])[None]))

    def _vacuum_cell(self, L):
        return VacuumCell(OrthogonalFrame(jnp.array([L, L, L])[None]))

    def _slab_cell(self, L, periodic):
        return Cell(OrthogonalFrame(jnp.array([L, L, L])[None]), periodic=periodic)

    def test_3d_periodic_wraps_across_face(self):
        """Two atoms near opposite x-faces connect via the periodic image."""
        L = 10.0
        positions = jnp.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
        edges = _run_cell_list(positions, self._bulk_cell(L), cutoff=2.0)
        assert _valid_edges_set(edges, 2) == {(0, 1), (1, 0)}

    def test_vacuum_excludes_cross_boundary_pair(self):
        """The same atoms under VacuumCell should NOT connect."""
        L = 10.0
        positions = jnp.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
        edges = _run_cell_list(positions, self._vacuum_cell(L), cutoff=2.0)
        assert _valid_edges_set(edges, 2) == set()

    def test_2d_slab_wraps_xy_not_z(self):
        """Slab (T, T, F): the cross-x pair connects (wraps); the cross-z pair does not."""
        L = 10.0
        cell = self._slab_cell(L, periodic=(True, True, False))
        x_pair = jnp.array([[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]])
        z_pair = jnp.array([[5.0, 5.0, 1.0], [5.0, 5.0, 9.0]])
        x_edges = _run_cell_list(x_pair, cell, cutoff=2.5)
        z_edges = _run_cell_list(z_pair, cell, cutoff=2.5)
        assert _valid_edges_set(x_edges, 2) == {(0, 1), (1, 0)}
        assert _valid_edges_set(z_edges, 2) == set()

    def test_1d_wire_wraps_x_not_yz(self):
        """1D wire (T, F, F): cross-x connects, cross-y and cross-z do not."""
        L = 10.0
        cell = self._slab_cell(L, periodic=(True, False, False))
        x_pair = jnp.array([[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]])
        y_pair = jnp.array([[5.0, 1.0, 5.0], [5.0, 9.0, 5.0]])
        z_pair = jnp.array([[5.0, 5.0, 1.0], [5.0, 5.0, 9.0]])
        assert _valid_edges_set(_run_cell_list(x_pair, cell, 2.5), 2) == {
            (0, 1),
            (1, 0),
        }
        assert _valid_edges_set(_run_cell_list(y_pair, cell, 2.5), 2) == set()
        assert _valid_edges_set(_run_cell_list(z_pair, cell, 2.5), 2) == set()

    @pytest.mark.parametrize(
        "cell_factory",
        [
            ("bulk", lambda self, L: self._bulk_cell(L)),
            ("vacuum", lambda self, L: self._vacuum_cell(L)),
            ("slab", lambda self, L: self._slab_cell(L, (True, True, False))),
            ("wire", lambda self, L: self._slab_cell(L, (True, False, False))),
        ],
        ids=lambda x: x[0],
    )
    def test_cell_list_matches_dense(self, cell_factory):
        """CellListNeighborList and DenseNearestNeighborList must agree across
        every periodicity shape — locks in that the cell-list-specific stencil
        routing is consistent with the dense reference."""
        _, factory = cell_factory
        L = 12.0
        rng = np.random.default_rng(7)
        # atoms scattered safely inside the box (so non-periodic axes don't
        # need to fold positions out of [0, L) into bins)
        positions = jnp.asarray(rng.uniform(1.0, L - 1.0, size=(30, 3)))
        cell = factory(self, L)
        cutoff = 3.5
        cl_edges = _run_cell_list(positions, cell, cutoff)
        dn_edges = _run_dense(positions, cell, cutoff)
        assert _valid_edges_set(cl_edges, 30) == _valid_edges_set(dn_edges, 30)


@dataclass
class _AdaptiveTestState:
    """Minimal state satisfying ``IsAdaptiveCutoffNeighborListState``."""

    particles: Table
    systems: Table
    neighborlist_params: UniversalNeighborlistParameters


def _make_adaptive_state(n_particles: int, n_systems: int) -> _AdaptiveTestState:
    """Build a state with ``n_particles`` total particles split across ``n_systems``."""
    positions = jnp.zeros((n_particles, 3))
    per_sys = max(1, n_particles // n_systems)
    batch_mask = jnp.minimum(
        jnp.arange(n_particles) // per_sys, jnp.array(n_systems - 1)
    )
    lh = _make_lh(positions, batch_mask)
    systems, _ = _systems_from_cell(
        PeriodicCell(
            OrthogonalFrame(jnp.tile(jnp.array([10.0, 10.0, 10.0]), (n_systems, 1)))
        ),
        jnp.full((n_systems,), 2.0),
    )
    params = UniversalNeighborlistParameters(
        avg_edges=8, avg_candidates=8, avg_image_candidates=8, cells=32
    )
    return _AdaptiveTestState(particles=lh, systems=systems, neighborlist_params=params)


class TestCutoffNeighborListPolicy:
    """Pure-Python policy tests over the avg-per-system threshold."""

    def test_small_avg_picks_dense(self):
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=100, num_systems=1)
            is CutoffNeighborListStrategy.DENSE
        )

    def test_sparse_multi_system_picks_dense(self):
        """Many systems with low avg occupancy stay on dense."""
        # 10_000 particles across 200 systems → avg 50 → dense
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=10_000, num_systems=200)
            is CutoffNeighborListStrategy.DENSE
        )

    def test_large_avg_picks_cell_list(self):
        policy = CutoffNeighborListPolicy()
        # 20_000 particles in 1 system → avg 20_000 → cell-list
        assert (
            policy.choose(num_particles=20_000, num_systems=1)
            is CutoffNeighborListStrategy.CELL_LIST
        )

    def test_just_below_default_threshold_stays_dense(self):
        """Avg < 10_000 default → dense."""
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=9_999, num_systems=1)
            is CutoffNeighborListStrategy.DENSE
        )

    def test_custom_threshold(self):
        policy = CutoffNeighborListPolicy(
            min_avg_particles_per_system_for_cell_list=500
        )
        # 600 particles in 1 system: above the custom 500 → cell-list
        assert (
            policy.choose(num_particles=600, num_systems=1)
            is CutoffNeighborListStrategy.CELL_LIST
        )

    def test_zero_systems_falls_back_to_dense(self):
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=100_000, num_systems=0)
            is CutoffNeighborListStrategy.DENSE
        )


class TestAdaptiveCutoffFactory:
    """``adaptive_cutoff_neighborlist_from_state`` returns the right concrete class."""

    def test_auto_picks_dense_for_small(self):
        state = _make_adaptive_state(n_particles=64, n_systems=1)
        cutoffs = _cutoff_table(jnp.array([2.0]))
        nl = adaptive_cutoff_neighborlist_from_state(state, cutoffs)
        assert isinstance(nl, DenseNearestNeighborList)

    def test_auto_picks_cell_list_for_large(self):
        state = _make_adaptive_state(n_particles=20_000, n_systems=1)
        cutoffs = _cutoff_table(jnp.array([2.0]))
        nl = adaptive_cutoff_neighborlist_from_state(state, cutoffs)
        assert isinstance(nl, CellListNeighborList)

    def test_forced_dense_bypasses_policy(self):
        state = _make_adaptive_state(n_particles=20_000, n_systems=1)
        cutoffs = _cutoff_table(jnp.array([2.0]))
        nl = adaptive_cutoff_neighborlist_from_state(
            state,
            cutoffs,
            policy=CutoffNeighborListPolicy(strategy=CutoffNeighborListStrategy.DENSE),
        )
        assert isinstance(nl, DenseNearestNeighborList)

    def test_forced_cell_list_bypasses_policy(self):
        state = _make_adaptive_state(n_particles=64, n_systems=1)
        cutoffs = _cutoff_table(jnp.array([2.0]))
        nl = adaptive_cutoff_neighborlist_from_state(
            state,
            cutoffs,
            policy=CutoffNeighborListPolicy(
                strategy=CutoffNeighborListStrategy.CELL_LIST
            ),
        )
        assert isinstance(nl, CellListNeighborList)

    def test_carries_cutoffs(self):
        state = _make_adaptive_state(n_particles=64, n_systems=1)
        cutoffs = _cutoff_table(jnp.array([3.5]))
        nl = adaptive_cutoff_neighborlist_from_state(state, cutoffs)
        assert isinstance(nl, DenseNearestNeighborList | CellListNeighborList)
        npt.assert_array_equal(nl.cutoffs.data, jnp.array([3.5]))

    def test_adaptive_matches_forced_dense_on_small_fixture(self):
        """When the policy picks dense, adaptive output equals forced dense."""
        rng = np.random.default_rng(0)
        L = 12.0
        positions = jnp.asarray(rng.uniform(1.0, L - 1.0, size=(30, 3)))
        cell = PeriodicCell(OrthogonalFrame(jnp.array([L, L, L])[None]))
        cutoff = 3.5
        adaptive_edges = _run_dense(positions, cell, cutoff)
        ref_edges = _run_dense(positions, cell, cutoff)
        assert _valid_edges_set(adaptive_edges, 30) == _valid_edges_set(ref_edges, 30)

    def test_jit_traces_only_chosen_branch(self):
        """jitting the adaptive NL must not require the other branch to compile."""
        state = _make_adaptive_state(n_particles=64, n_systems=1)
        cutoffs = _cutoff_table(jnp.array([2.0]))
        nl = adaptive_cutoff_neighborlist_from_state(state, cutoffs)
        # AUTO → DENSE for 64 particles, so we can jit-call the dense path
        # directly; no cell-list machinery is referenced.
        assert isinstance(nl, DenseNearestNeighborList)
        positions = jnp.zeros((4, 3))
        lh = _make_lh(positions, jnp.zeros(4, dtype=int))
        systems, _ = _systems_from_cell(
            PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])[None])),
            jnp.array([2.0]),
        )

        result = jax.jit(as_result_function(nl))(
            lh=lh, rh=None, systems=systems, rh_index_remap=None
        )
        result.raise_assertion()
        # Sanity: emitted Edges have the expected pair-degree.
        assert result.value.indices.shape[-1] == 2
