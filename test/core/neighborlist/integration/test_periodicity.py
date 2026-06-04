# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Cell-list and dense neighbor lists agree across periodicity shapes.

Covers the four periodicity shapes — vacuum (0D), 1D wire, 2D slab, 3D bulk —
checking that periodic faces wrap only on periodic axes and that the cell-list
stencil routing stays consistent with the dense reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import Cell, OrthogonalFrame, PeriodicCell, VacuumCell
from kups.core.neighborlist import CellListNeighborList, DenseNearestNeighborList
from kups.core.result import as_result_function

from .._builders import make_lh, make_systems, valid_edge_set


def _run_cell_list(positions, cell, cutoff):
    """Run ``CellListNeighborList`` on a single-system fixture."""
    n = len(positions)
    lh = make_lh(positions, jnp.zeros(n, dtype=int))
    systems, cutoffs = make_systems(cell, jnp.array([cutoff]))
    nl = CellListNeighborList(
        avg_candidates=FixedCapacity(max(n * n, 8)),
        avg_edges=FixedCapacity(max(n * n, 8)),
        cells=FixedCapacity(512),
        avg_image_candidates=FixedCapacity(max(n * n, 8)),
        cutoffs=cutoffs,
    )
    result = jax.jit(as_result_function(nl))(keys=lh, systems=systems)
    result.raise_assertion()
    return result.value


def _run_dense(positions, cell, cutoff):
    """Run ``DenseNearestNeighborList`` on the same fixture, for cross-check."""
    n = len(positions)
    lh = make_lh(positions, jnp.zeros(n, dtype=int))
    systems, cutoffs = make_systems(cell, jnp.array([cutoff]))
    nl = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(max(n * n, 8)),
        avg_edges=FixedCapacity(max(n * n, 8)),
        avg_image_candidates=FixedCapacity(max(n * n, 8)),
        cutoffs=cutoffs,
    )
    result = jax.jit(as_result_function(nl))(keys=lh, systems=systems)
    result.raise_assertion()
    return result.value


class TestNeighborListAcrossPeriodicities:
    """Cell-list and dense agree across vacuum (0D), 1D wire, 2D slab, 3D bulk."""

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
        assert valid_edge_set(edges, 2) == {(0, 1), (1, 0)}

    def test_vacuum_excludes_cross_boundary_pair(self):
        """The same atoms under ``VacuumCell`` should NOT connect."""
        L = 10.0
        positions = jnp.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
        edges = _run_cell_list(positions, self._vacuum_cell(L), cutoff=2.0)
        assert valid_edge_set(edges, 2) == set()

    def test_2d_slab_wraps_xy_not_z(self):
        """Slab (T, T, F): the cross-x pair wraps; the cross-z pair does not."""
        L = 10.0
        cell = self._slab_cell(L, periodic=(True, True, False))
        x_pair = jnp.array([[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]])
        z_pair = jnp.array([[5.0, 5.0, 1.0], [5.0, 5.0, 9.0]])
        x_edges = _run_cell_list(x_pair, cell, cutoff=2.5)
        z_edges = _run_cell_list(z_pair, cell, cutoff=2.5)
        assert valid_edge_set(x_edges, 2) == {(0, 1), (1, 0)}
        assert valid_edge_set(z_edges, 2) == set()

    def test_1d_wire_wraps_x_not_yz(self):
        """1D wire (T, F, F): cross-x connects, cross-y and cross-z do not."""
        L = 10.0
        cell = self._slab_cell(L, periodic=(True, False, False))
        x_pair = jnp.array([[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]])
        y_pair = jnp.array([[5.0, 1.0, 5.0], [5.0, 9.0, 5.0]])
        z_pair = jnp.array([[5.0, 5.0, 1.0], [5.0, 5.0, 9.0]])
        assert valid_edge_set(_run_cell_list(x_pair, cell, 2.5), 2) == {(0, 1), (1, 0)}
        assert valid_edge_set(_run_cell_list(y_pair, cell, 2.5), 2) == set()
        assert valid_edge_set(_run_cell_list(z_pair, cell, 2.5), 2) == set()

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
        """``CellListNeighborList`` and ``DenseNearestNeighborList`` must agree
        across every periodicity shape — locking in that the cell-list-specific
        stencil routing is consistent with the dense reference."""
        _, factory = cell_factory
        L = 12.0
        rng = np.random.default_rng(7)
        # atoms scattered safely inside the box (so non-periodic axes don't need
        # to fold positions out of [0, L) into bins)
        positions = jnp.asarray(rng.uniform(1.0, L - 1.0, size=(30, 3)))
        cell = factory(self, L)
        cutoff = 3.5
        cl_edges = _run_cell_list(positions, cell, cutoff)
        dn_edges = _run_dense(positions, cell, cutoff)
        assert valid_edge_set(cl_edges, 30) == valid_edge_set(dn_edges, 30)
