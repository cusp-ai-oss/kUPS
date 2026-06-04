# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.cell_list`` internals + constructors."""

import jax.numpy as jnp
import numpy as np

from kups.core.capacity import FixedCapacity, LensCapacity
from kups.core.cell import OrthogonalFrame, PeriodicCell, TriclinicFrame
from kups.core.neighborlist.cell_list import (
    CellListNeighborList,
    _cell_hash,
    _cell_list_subselect,
    _cell_stencil,
)
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters

from ._builders import EvalState, cutoff_table, make_lh, make_systems


class TestCellHashClampsAtBoundary:
    """``_cell_hash`` keeps per-axis bins inside ``[0, num_cells - 1]``."""

    def test_interior_coord_with_unit_grid_hashes_to_zero(self):
        h = _cell_hash(jnp.array([0.0, 0.0, 0.0]), jnp.array([1, 1, 1]))
        assert int(h) == 0
        h = _cell_hash(jnp.array([0.5, 0.99, 0.0]), jnp.array([1, 1, 1]))
        assert int(h) == 0

    def test_fold_overshoot_at_one_clamps_with_unit_grid(self):
        h = _cell_hash(jnp.array([1.0, 1.0, 1.0]), jnp.array([1, 1, 1]))
        assert int(h) == 0

    def test_fold_overshoot_at_one_clamps_with_nontrivial_grid(self):
        num_cells = jnp.array([2, 3, 5])
        h = _cell_hash(jnp.array([1.0, 1.0, 1.0]), num_cells)
        assert int(h) == 1 + 2 * 2 + 4 * 6
        h = _cell_hash(jnp.array([0.25, 0.5, 0.5]), num_cells)
        assert int(h) == 0 + 1 * 2 + 2 * 6

    def test_realistic_fold_path_does_not_escape_range(self):
        cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])))
        bad_frac = jnp.array([-1.49e-8, 0.0, 0.0])
        folded, _ = cell.fold(bad_frac)
        h = _cell_hash(folded, jnp.array([1, 1, 1]))
        assert int(h) == 0


class TestCellStencil:
    def test_3d_stencil_is_27_centered_offsets(self):
        stencil = _cell_stencil(3)
        assert stencil.shape == (27, 3)
        assert set(np.unique(np.asarray(stencil)).tolist()) == {-1, 0, 1}
        rows = {tuple(r) for r in np.asarray(stencil).tolist()}
        assert (0, 0, 0) in rows
        assert (1, -1, 0) in rows

    def test_2d_stencil_is_9_offsets(self):
        assert _cell_stencil(2).shape == (9, 2)


class TestCellListSubselect:
    def test_local_pair_is_candidate_far_pair_is_not(self):
        # Fractional positions in a 4 A box, cutoff 1.0 -> 4 bins/axis (64 cells).
        # p0 (bin 0) and p1 (bin 1) are adjacent; p2 (bin 2) is not (bin 0's
        # periodic neighbours are bins {3, 0, 1}). A small grid keeps the
        # spatial-hash arrays tiny so this stays a fast unit test.
        lh = make_lh(
            jnp.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            jnp.zeros(3, dtype=int),
        )
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 4.0)),
            jnp.array([1.0]),
        )
        candidates = _cell_list_subselect(
            lh,
            lh,
            systems,
            cutoffs=jnp.array([1.0]),
            max_num_cells=FixedCapacity(128),
            max_num_candidates=FixedCapacity(32),
        )
        pairs = {
            (a, b)
            for a, b in zip(
                candidates.key_idx.indices.tolist(),
                candidates.query_idx.indices.tolist(),
            )
            if a < 3 and b < 3
        }
        assert (0, 1) in pairs or (1, 0) in pairs
        assert (0, 2) not in pairs and (2, 0) not in pairs


class TestFromState:
    def test_from_state_wires_lens_capacities(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0)),
            jnp.array([2.0]),
        )
        params = UniversalNeighborlistParameters(
            avg_edges=16, avg_candidates=32, avg_image_candidates=32, cells=64
        )
        state = EvalState(particles=lh, systems=systems, neighborlist_params=params)
        nl = CellListNeighborList.from_state(state, cutoff_table(jnp.array([2.0])))

        assert isinstance(nl.cells, LensCapacity)
        assert isinstance(nl.avg_candidates, LensCapacity)
        assert int(nl.cells.size) == 64
        assert int(nl.avg_candidates.size) == 32
