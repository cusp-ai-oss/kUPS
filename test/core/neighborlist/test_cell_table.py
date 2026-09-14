# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.cell import Cell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.lens import lens
from kups.core.neighborlist.cell_table import (
    CellTableNeighborList,
    CellTableParameters,
    CellTableSelector,
    CellTableUpdatePatch,
    build_cell_table,
)
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
)
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.result import as_result_function
from kups.core.typing import ParticleId

from ._builders import make_lh, make_systems, systems_from_lvecs


def _edge_rows(edges, n_keys, n_queries):
    indices = np.asarray(edges.indices.indices)
    valid = (indices >= 0).all(axis=-1)
    valid &= (indices[:, 0] < n_keys) & (indices[:, 1] < n_queries)
    return Counter(
        map(
            tuple,
            np.concatenate(
                [indices[valid], np.asarray(edges.shifts)[valid, 0]], axis=-1
            ),
        )
    )


class TestCellTableNeighborList:
    @pytest.mark.parametrize("mode", ["full", "queried_keys", "queries"])
    @pytest.mark.parametrize("cutoff", [1.4, 5.4])
    @pytest.mark.parametrize(
        "periodic", [(True, True, True), (False, True, True), (False, False, False)]
    )
    def test_matches_dense_with_images_and_original_particle_ids(
        self, mode, cutoff, periodic
    ):
        matrices = jnp.array(
            [
                [[4.0, 0, 0], [0.7, 4.5, 0], [0.2, 0.4, 5.0]],
                [[9.0, 0, 0], [0.8, 8.5, 0], [0.3, 0.6, 10.0]],
            ]
        )
        systems, cutoffs = make_systems(
            Cell(TriclinicFrame.from_matrix(matrices), periodic=periodic),
            jnp.array([cutoff, 1.4]),
        )
        system = jnp.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1])
        frac = jnp.array(
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.2, 0.1, 0.1],
                [0.3, 0.2, 0.1],
                [0.95, 0.1, 0.1],
                [0.6, 0.5, 0.5],
                [0.65, 0.55, 0.5],
                [0.65, 0.6, 0.5],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        # Preserve integer image offsets outside the primary periodic box.
        frac += jnp.asarray(periodic) * jnp.arange(10)[:, None] % 3
        points = make_lh(
            jnp.einsum("ni,nij->nj", frac, matrices[system]),
            system,
            jnp.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 7]),
        )
        points = Table(
            tuple(ParticleId(i) for i in (12, 17, 21, 30, 44, 59, 65, 73, 86, 91)),
            replace(
                points.data,
                positions=points.data.positions.at[-2:].set(jnp.inf),
                inclusion=points.data.inclusion.apply_mask(jnp.arange(10) < 8),
            ),
        )
        layout = CellTableParameters.estimate(
            points, systems, cutoffs, chunk_size=4, occupancy_headroom=1
        )
        neighborlist = CellTableNeighborList(FixedCapacity(256), cutoffs, layout)
        dense = DenseNearestNeighborList(
            FixedCapacity(16), FixedCapacity(256), FixedCapacity(16 * 27), cutoffs
        )
        kwargs = {}
        n_queries = points.size
        if mode == "queried_keys":
            # A call must rebuild from updated positions, including crossed bins.
            points = points.map_data(
                lambda p: replace(p, positions=p.positions.at[2].add(1.1))
            )
            kwargs[mode] = Index(
                (points.keys[2], points.keys[5], points.keys[8]), jnp.array([2, 1, 0])
            )
        elif mode == "queries":
            query = points.subset(Index(points.keys, jnp.array([1, 5, 8])))
            vocabulary = (-1, *query.data.inclusion.keys)
            query = Table(
                tuple(ParticleId(i) for i in (801, 802, 803)),
                replace(
                    query.data,
                    positions=query.data.positions + jnp.array([0.13, 0.09, 0]),
                    inclusion=Index(
                        vocabulary, query.data.inclusion.indices_in(vocabulary)
                    ),
                ),
            )
            kwargs[mode] = query
            n_queries = query.size
        actual = jax.jit(as_result_function(neighborlist))(points, systems, **kwargs)
        expected = jax.jit(as_result_function(dense))(points, systems, **kwargs)
        actual.raise_assertion()
        expected.raise_assertion()
        assert actual.value.indices.keys == points.keys
        assert _edge_rows(actual.value, points.size, n_queries) == _edge_rows(
            expected.value, points.size, n_queries
        )

    @pytest.mark.parametrize("mode", ["full", "queried_keys", "queries", "empty_keys"])
    def test_empty_query(self, mode):
        points = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, int))
        queries = points
        if mode in ("full", "empty_keys"):
            points = points.subset(Index(points.keys, jnp.zeros(0, int)))
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 4, jnp.array([2.5]))
        layout = CellTableParameters.estimate(points, systems, cutoffs)
        neighborlist = CellTableNeighborList(FixedCapacity(8), cutoffs, layout)
        empty = Index(points.keys, jnp.zeros(0, int), _cls=ParticleId)
        kwargs = {}
        if mode == "queried_keys":
            kwargs[mode] = empty
        elif mode == "queries":
            kwargs[mode] = points.subset(empty)
        elif mode == "empty_keys":
            kwargs["queries"] = queries
        result = jax.jit(as_result_function(neighborlist))(points, systems, **kwargs)
        result.raise_assertion()
        assert not _edge_rows(result.value, points.size, queries.size)

    def test_selector_reuses_an_updated_table(self):
        points = make_lh(
            jnp.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [6.0, 1.0, 1.0]]),
            jnp.zeros(3, int),
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 12, jnp.array([3.2]))
        layout = CellTableParameters.estimate(
            points, systems, cutoffs, occupancy_factor=1, occupancy_headroom=1
        )
        table = build_cell_table(points, systems, cutoffs, (), layout)
        moved = points.map_data(
            lambda p: replace(p, positions=p.positions.at[2].set(jnp.array([2, 1, 1])))
        )
        index = Index(points.keys, jnp.array([2]))
        rows = table.bin_rows(moved.subset(index), systems, ())
        patch = CellTableUpdatePatch(
            table.slot_of_row[index.indices], rows.system, rows, lens(lambda t: t)
        )
        updated = as_result_function(patch)(table, systems.set_data(jnp.array([True])))
        updated.raise_assertion()
        pipeline = Pipeline(
            selector=CellTableSelector(
                updated.value, cutoffs, layout.max_images_per_pair
            ),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                DistanceCutoffMask(cutoffs),
                ExclusionMask(),
            ),
            compactor=ReduceCompactor(FixedCapacity(6)),
        )
        result = jax.jit(as_result_function(pipeline))(moved, systems)
        result.raise_assertion()
        assert _edge_rows(result.value, 3, 3) == Counter(
            (i, j, 0, 0, 0) for i in range(3) for j in range(3) if i != j
        )

    def test_edge_capacity_and_query_contract(self):
        points = make_lh(
            jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]), jnp.zeros(2, int)
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 4, jnp.array([5.0]))
        layout = CellTableParameters.estimate(points, systems, cutoffs)
        neighborlist = CellTableNeighborList(FixedCapacity(1), cutoffs, layout)
        result = jax.jit(as_result_function(neighborlist))(points, systems)
        with pytest.raises(CapacityError):
            result.raise_assertion()
        with pytest.raises(AssertionError, match="cannot combine"):
            neighborlist(points, systems, queries=points, queried_keys=points.index)


class TestCellTable:
    @pytest.mark.parametrize(
        "counts,expected_layout",
        [((83,) + (52,) * 7, "slots"), ((142,) * 8, "cells")],
    )
    def test_auto_layout_requires_smaller_slot_blocks(self, counts, expected_layout):
        centers = np.array(list(np.ndindex(2, 2, 2))) * 6.0 + 3.0
        particles = make_lh(
            jnp.asarray(np.repeat(centers, counts, axis=0)),
            jnp.zeros(sum(counts), int),
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 12.0, jnp.array([6.0]))
        parameters = CellTableParameters.estimate(
            particles,
            systems,
            cutoffs,
            key_layout="auto",
            key_chunk_size="auto",
        )
        assert parameters.key_layout == expected_layout

    @pytest.mark.parametrize(
        "length,n_systems,periodic,expected_layout",
        [
            (18.0, 1, (True, True, True), "slots"),
            (18.0, 2, (True, True, True), "cells"),
            (24.0, 1, (True, True, True), "cells"),
            (18.0, 1, (False, True, True), "cells"),
        ],
    )
    def test_auto_slots_require_complete_single_system_stencil(
        self, length, n_systems, periodic, expected_layout
    ):
        particles = make_lh(
            jax.random.uniform(jax.random.key(74), (5 * n_systems, 3)) * length,
            jnp.repeat(jnp.arange(n_systems), 5),
        )
        frame = TriclinicFrame.from_matrix(
            jnp.tile(jnp.eye(3) * length, (n_systems, 1, 1))
        )
        systems, cutoffs = make_systems(
            Cell(frame, periodic=periodic), jnp.full(n_systems, 6.0)
        )
        parameters = CellTableParameters.estimate(
            particles,
            systems,
            cutoffs,
            key_layout="auto",
            key_chunk_size="auto",
        )
        assert parameters.key_layout == expected_layout

    def test_rejection_same_cell_and_overflow(self):
        particles = make_lh(
            jnp.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [4.0, 1.0, 1.0]]),
            jnp.zeros(3, int),
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 12.0, jnp.array([3.0]))
        parameters = CellTableParameters(
            chunk_size=8, max_cells_per_system=64, cell_capacity=2
        )
        table = build_cell_table(particles, systems, cutoffs, jnp.zeros(3), parameters)
        index = Index(particles.keys, jnp.array([2]))
        slot = table.slot_of_row[index.indices]

        def proposed(position):
            query = particles.subset(index).map_data(
                lambda p: replace(p, positions=jnp.array([position]))
            )
            return table.bin_rows(query, systems, jnp.zeros(1))

        @jax.jit
        @as_result_function
        def update(rows, accepted):
            patch = CellTableUpdatePatch(slot, rows.system, rows, lens(lambda t: t))
            return patch(table, systems.set_data(accepted))

        # A rejected relocation into a full cell must leave the entire table intact
        # and must not report the overflow from the uncommitted proposal.
        overflow = proposed([2.0, 1.0, 1.0])
        rejected = update(overflow, jnp.array([False]))
        rejected.raise_assertion()
        for before, after in zip(
            jax.tree.leaves(table), jax.tree.leaves(rejected.value)
        ):
            npt.assert_array_equal(before, after)
        with pytest.raises(AssertionError, match="cell_capacity exceeded"):
            update(overflow, jnp.array([True])).raise_assertion()

        # Accepted motion within a cell still updates coordinates and payload.
        same_cell = proposed([4.5, 1.0, 1.0])._replace(data=jnp.array([0.75]))
        accepted = update(same_cell, jnp.array([True]))
        accepted.raise_assertion()
        npt.assert_array_equal(accepted.value.cells, table.cells)
        npt.assert_allclose(accepted.value.rows.frac[slot], same_cell.frac)
        npt.assert_array_equal(accepted.value.rows.data[slot], same_cell.data)

        # Crossing into a free cell transfers membership exactly once.
        crossing = proposed([7.0, 1.0, 1.0])
        accepted = update(crossing, jnp.array([True]))
        accepted.raise_assertion()
        cells = accepted.value.cells
        assert (cells[table.rows.cell[slot]] != slot[:, None]).all()
        assert (cells[crossing.cell] == slot[:, None]).sum() == 1
