# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.common`` algorithmic helpers."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.common import (
    Candidates,
    _candidate_image_counts,
    _generate_image_offsets,
    _get_candidate_images,
    _minimum_image_shifts,
    candidates_to_batch,
    edge_rhs_table,
    lift_query_candidates,
    make_batch_with_mic,
    num_cells,
    query_table,
    real_distance_sq,
    replicate_for_images,
)
from kups.core.result import as_result_function
from kups.core.typing import SystemId

from ._builders import (
    cutoff_table,
    make_lh,
    make_pipeline_ctx,
    make_systems,
    systems_from_lvecs,
)


class TestNumCells:
    def test_face_length_over_cutoff(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([2.0]))
        bins = num_cells(systems.data, jnp.array([2.0]))
        npt.assert_array_equal(np.asarray(bins), np.array([[5, 5, 5]]))

    def test_clamps_to_at_least_one_when_cutoff_exceeds_box(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([100.0]))
        bins = num_cells(systems.data, jnp.array([100.0]))
        npt.assert_array_equal(np.asarray(bins), np.array([[1, 1, 1]]))


class TestGenerateImageOffsets:
    def test_matches_documented_example(self):
        images = jnp.array([[3, 3, 1], [1, 1, 1]])
        coords = _generate_image_offsets(images, FixedCapacity(10))
        expected = np.array(
            [
                [0, 0, 0],  # center first
                [1, 0, 0],
                [-1, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [0, -1, 0],
                [1, -1, 0],
                [-1, 0, 0],
                [0, 0, 0],  # second 1x1x1 grid
            ]
        )
        npt.assert_array_equal(np.asarray(coords), expected)


class TestCandidateImageCounts:
    def test_minimum_image_uses_single_image(self):
        # ratio <= 0.5 -> one image per axis.
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([2.0]))
        images = _candidate_image_counts(systems.data.cell, jnp.array([2.0]))
        npt.assert_array_equal(np.asarray(images), np.array([[1, 1, 1]]))

    def test_wide_cutoff_uses_symmetric_stencil(self):
        # ratio = 0.8 -> 2*ceil(0.8)+1 = 3 images per axis.
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 1.0))
        systems, _ = make_systems(cell, jnp.array([0.8]))
        images = _candidate_image_counts(systems.data.cell, jnp.array([0.8]))
        npt.assert_array_equal(np.asarray(images), np.array([[3, 3, 3]]))

    def test_handles_nonfinite_ratios(self):
        class Cells:
            perpendicular_lengths = jnp.array([[0.0, 4.0, jnp.nan]])
            periodic = (True, True, True)

        images = _candidate_image_counts(Cells(), jnp.array([6.0]))
        npt.assert_array_equal(np.asarray(images), np.array([[1, 5, 1]]))


class TestGetCandidateImagesIsFinite:
    """Candidate image counts stay finite for degenerate periodic geometry."""

    def _minimal_inputs(self, cell):
        lh = make_lh(jnp.zeros((1, 3)), jnp.array([0]))
        systems, _ = make_systems(cell, jnp.array([6.0]))
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0])),
            rhs=Index(lh.keys, jnp.array([0])),
        )
        return lh, systems, candidates

    def test_zero_perpendicular_axis_clamps_images_to_one(self):
        tril = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
        cell = PeriodicCell(TriclinicFrame(tril))
        assert float(cell.perpendicular_lengths[0, 0]) == 0.0
        lh, systems, candidates = self._minimal_inputs(cell)
        idx, offsets, has_been_replicated = _get_candidate_images(
            candidates, lh, systems, jnp.array([6.0]), FixedCapacity(8)
        )
        assert idx.shape[0] <= 8
        assert offsets.shape == (idx.shape[0], 3)
        assert has_been_replicated.shape == (idx.shape[0],)


def _shift_set(batch) -> set[tuple[int, int, int]]:
    """Set of integer shift tuples from a degree-2 candidate batch."""
    return {tuple(int(c) for c in s) for s in np.asarray(batch.edges.shifts[:, 0, :])}


class TestReplicateForImages:
    """``replicate_for_images`` is the PBC core: below half the perpendicular
    length it emits one minimum-image copy per candidate; above it, it expands
    each candidate over the per-system integer image stencil and flags the copy
    that coincides with the minimum image (the only one ``ExclusionMask`` may
    drop)."""

    def _one_candidate(self, box, cutoff, p0, p1):
        """Single candidate (0, 1) at fractional positions ``p0``/``p1``."""
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * box))
        lh = make_lh(jnp.array([p0, p1]), jnp.zeros(2, dtype=int))
        systems, _ = make_systems(cell, jnp.array([cutoff]))
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0])), rhs=Index(lh.keys, jnp.array([1]))
        )
        return candidates, lh, systems

    def test_no_replication_emits_single_mic_copy(self):
        # ratio 0.2 <= 0.5; the wrapping pair's minimum image is shift [-1,0,0],
        # so the single emitted copy must carry the MIC shift (not [0,0,0]).
        candidates, lh, systems = self._one_candidate(
            box=10.0, cutoff=2.0, p0=[0.1, 0.0, 0.0], p1=[0.9, 0.0, 0.0]
        )
        batch = replicate_for_images(
            candidates, lh, lh, systems, cutoff_table(jnp.array([2.0])), None
        )
        assert len(batch.edges) == 1
        assert bool(batch.is_minimum_image.all())
        npt.assert_array_equal(np.asarray(batch.edges.shifts), [[[-1.0, 0.0, 0.0]]])

    def test_replication_count_and_index_integrity(self):
        # ratio 0.8 -> 3 images/axis -> 27 copies, every one of the same pair.
        candidates, lh, systems = self._one_candidate(
            box=1.0, cutoff=0.8, p0=[0.0, 0.0, 0.0], p1=[0.3, 0.0, 0.0]
        )
        batch = replicate_for_images(
            candidates,
            lh,
            lh,
            systems,
            cutoff_table(jnp.array([0.8])),
            FixedCapacity(27),
        )
        assert len(batch.edges) == 27
        npt.assert_array_equal(
            np.asarray(batch.lh_idx.indices), np.zeros(27, dtype=int)
        )
        npt.assert_array_equal(np.asarray(batch.rh_idx.indices), np.ones(27, dtype=int))

    def test_replication_spans_full_integer_stencil(self):
        candidates, lh, systems = self._one_candidate(
            box=1.0, cutoff=0.8, p0=[0.0, 0.0, 0.0], p1=[0.3, 0.0, 0.0]
        )
        batch = replicate_for_images(
            candidates,
            lh,
            lh,
            systems,
            cutoff_table(jnp.array([0.8])),
            FixedCapacity(27),
        )
        expected = {
            (i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
        }
        assert _shift_set(batch) == expected

    def test_minimum_image_flag_tracks_mic_when_nonzero(self):
        # Pair whose minimum image is the [-1,0,0] copy (delta = -0.8 -> round -1),
        # not the in-cell [0,0,0] copy. Exactly that stencil copy is flagged.
        candidates, lh, systems = self._one_candidate(
            box=1.0, cutoff=0.8, p0=[0.1, 0.0, 0.0], p1=[0.9, 0.0, 0.0]
        )
        batch = replicate_for_images(
            candidates,
            lh,
            lh,
            systems,
            cutoff_table(jnp.array([0.8])),
            FixedCapacity(27),
        )
        assert int(batch.is_minimum_image.sum()) == 1
        min_shift = np.asarray(batch.edges.shifts[batch.is_minimum_image][:, 0, :])
        npt.assert_array_equal(min_shift, [[-1.0, 0.0, 0.0]])

    def test_multi_system_replicates_per_system(self):
        # System 0 (box 10, cutoff 2): ratio 0.2 -> 1 copy. System 1 (box 1,
        # cutoff 0.8): ratio 0.8 -> 27 copies. One call, mixed per-system depth.
        systems, _ = systems_from_lvecs(
            jnp.stack([jnp.eye(3) * 10.0, jnp.eye(3) * 1.0]), jnp.array([2.0, 0.8])
        )
        lh = make_lh(
            jnp.array(
                [
                    [0.05, 0.0, 0.0],
                    [0.15, 0.0, 0.0],  # system 0, MIC shift 0
                    [0.1, 0.0, 0.0],
                    [0.9, 0.0, 0.0],  # system 1, MIC shift -1
                ]
            ),
            jnp.array([0, 0, 1, 1]),
        )
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0, 2])),
            rhs=Index(lh.keys, jnp.array([1, 3])),
        )
        batch = replicate_for_images(
            candidates,
            lh,
            lh,
            systems,
            cutoff_table(jnp.array([2.0, 0.8])),
            FixedCapacity(28),
        )
        assert len(batch.edges) == 28
        # First row is the lone system-0 copy; the rest expand the system-1 pair.
        assert (int(batch.lh_idx.indices[0]), int(batch.rh_idx.indices[0])) == (0, 1)
        npt.assert_array_equal(np.asarray(batch.lh_idx.indices[1:]), np.full(27, 2))
        npt.assert_array_equal(np.asarray(batch.rh_idx.indices[1:]), np.full(27, 3))
        # One minimum image per candidate: the system-0 copy plus system-1's.
        assert int(batch.is_minimum_image.sum()) == 2
        assert bool(batch.is_minimum_image[0])

    def test_uses_rh_positions_for_minimum_image(self):
        # Bipartite: the rhs endpoint lives in a separate ``rh`` table, so the
        # minimum-image delta must read rh's position (here giving MIC [-1,0,0]),
        # not lh's. The emitted rh column is keyed by ``rh``.
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        lh = make_lh(jnp.array([[0.1, 0.0, 0.0]]), jnp.zeros(1, dtype=int))
        rh = make_lh(jnp.array([[0.9, 0.0, 0.0]]), jnp.zeros(1, dtype=int))
        systems, _ = make_systems(cell, jnp.array([2.0]))
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0])), rhs=Index(rh.keys, jnp.array([0]))
        )
        batch = replicate_for_images(
            candidates, lh, rh, systems, cutoff_table(jnp.array([2.0])), None
        )
        npt.assert_array_equal(np.asarray(batch.edges.shifts), [[[-1.0, 0.0, 0.0]]])
        assert batch.rhs_keys == rh.keys

    def test_missing_capacity_with_replication_needed_asserts(self):
        # max_image_candidates=None falls back to a non-growable capacity sized
        # for the candidate count; when images are actually needed it must raise.
        candidates, lh, systems = self._one_candidate(
            box=1.0, cutoff=0.8, p0=[0.0, 0.0, 0.0], p1=[0.3, 0.0, 0.0]
        )
        result = as_result_function(
            lambda: replicate_for_images(
                candidates, lh, lh, systems, cutoff_table(jnp.array([0.8])), None
            )
        )()
        with pytest.raises(CapacityError):
            result.raise_assertion()


class TestMakeBatchWithMic:
    def test_rounds_fractional_delta_as_shift(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
        positions = jnp.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
        lh = make_lh(positions, jnp.zeros(2, dtype=int))
        systems, _ = make_systems(cell, jnp.array([1.0]))
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0])),
            rhs=Index(lh.keys, jnp.array([1])),
        )
        batch = make_batch_with_mic(candidates, lh, lh, systems)
        # delta = 0.1 - 0.9 = -0.8; round(-0.8) = -1.0 on the periodic axis.
        npt.assert_allclose(
            np.asarray(batch.edges.shifts), np.array([[[-1.0, 0.0, 0.0]]])
        )
        assert batch.is_minimum_image.tolist() == [True]


class TestCandidatesToBatch:
    def test_packs_indices_shifts_and_rhs_keys(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0, 1])),
            rhs=Index(lh.keys, jnp.array([1, 2])),
        )
        shifts = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        batch = candidates_to_batch(candidates, shifts, jnp.array([True, False]))
        npt.assert_array_equal(
            np.asarray(batch.edges.indices.indices), np.array([[0, 1], [1, 2]])
        )
        npt.assert_array_equal(np.asarray(batch.edges.shifts[:, 0]), np.asarray(shifts))
        assert batch.rhs_keys == lh.keys


class TestMinimumImageShifts:
    def test_zero_for_in_cell_pair(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
        lh = make_lh(
            jnp.array([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]), jnp.zeros(2, dtype=int)
        )
        systems, _ = make_systems(cell, jnp.array([1.0]))
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0])), rhs=Index(lh.keys, jnp.array([1]))
        )
        shifts = _minimum_image_shifts(candidates, lh, lh, systems)
        # delta = -0.2, nearest image is the in-cell one (shift 0).
        npt.assert_allclose(np.asarray(shifts), np.array([[0.0, 0.0, 0.0]]))


class TestRealDistanceSq:
    def test_scales_fractional_delta_into_real(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        systems, _ = make_systems(cell, jnp.array([1.0]))
        frame_table = systems.map_data(lambda s: s.cell.frame.materialize())
        frames = frame_table[Index((SystemId(0),), jnp.array([0]))]
        # fractional delta 0.3 along x -> real 3.0 -> squared 9.0.
        dist_sq = real_distance_sq(
            jnp.array([[0.0, 0.0, 0.0]]),
            jnp.array([[0.3, 0.0, 0.0]]),
            frames,
            jnp.zeros((1, 3)),
        )
        npt.assert_allclose(np.asarray(dist_sq), np.array([9.0]), atol=1e-5)


class TestContextTableHelpers:
    def test_edge_rhs_table_prefers_rh(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        rh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        assert edge_rhs_table(make_pipeline_ctx(lh)) is lh
        assert edge_rhs_table(make_pipeline_ctx(lh, rh)) is rh

    def test_query_table_self_full_vs_subset_vs_bipartite(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        rh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        assert query_table(make_pipeline_ctx(lh)).size == 4
        assert (
            query_table(make_pipeline_ctx(lh, for_indices=jnp.array([1, 3]))).size == 2
        )
        assert query_table(make_pipeline_ctx(lh, rh)) is rh

    def test_lift_query_candidates_remaps_rhs_to_lh_ids(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh, for_indices=jnp.array([2, 0]))
        # query-local rhs ids [0, 1] map through for_indices -> lh ids [2, 0].
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([1, 3])),
            rhs=Index(lh.keys, jnp.array([0, 1])),
        )
        lifted = lift_query_candidates(candidates, ctx)
        npt.assert_array_equal(np.asarray(lifted.lhs.indices), np.array([1, 3]))
        npt.assert_array_equal(np.asarray(lifted.rhs.indices), np.array([2, 0]))

    def test_lift_query_candidates_noop_without_for_indices(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh)
        candidates = Candidates(
            lhs=Index(lh.keys, jnp.array([0])), rhs=Index(lh.keys, jnp.array([1]))
        )
        assert lift_query_candidates(candidates, ctx) is candidates
