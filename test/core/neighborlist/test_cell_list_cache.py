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
from kups.core.lens import bind, lens
from kups.core.neighborlist.cell_list import CellListNeighborList
from kups.core.neighborlist.cell_list_cache import (
    BlockBounds,
    CellListCacheParameters,
    CellListCacheUpdatePatch,
    _bisection_levels,
    build_cell_list_cache,
)
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.result import as_result_function
from kups.core.typing import Label, ParticleId, SystemId

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


def _neighborlist(cache, cutoffs, layout, avg_edges):
    width = layout.stencil_width * layout.cell_capacity
    return CellListNeighborList(
        avg_candidates=FixedCapacity(width),
        avg_edges=FixedCapacity(avg_edges),
        cells=FixedCapacity(layout.max_cells_per_system),
        avg_image_candidates=FixedCapacity(width * layout.max_images_per_pair),
        cutoffs=cutoffs,
        cache=cache,
    )


class TestCachedCellList:
    @pytest.mark.parametrize("cached", [False, True])
    @pytest.mark.parametrize("periodic", [(False, False, False), (False, True, True)])
    def test_open_boundaries_match_dense(self, cached, periodic):
        # Include neighbors inside/outside both faces and entirely outside the box.
        x = jnp.array([-21.0, -20.0, -0.5, 0.5, 9.5, 10.5, 20.0, 21.0])
        points = make_lh(jnp.stack((x, x * 0, x * 0), -1), jnp.zeros(len(x), int))
        systems, cutoffs = make_systems(
            Cell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10), periodic),
            jnp.array([2.0]),
        )
        layout = CellListCacheParameters.estimate(points, systems, cutoffs)
        cache = build_cell_list_cache(points, systems, cutoffs, (), layout)
        neighborlist = _neighborlist(cache if cached else None, cutoffs, layout, 8)
        dense = DenseNearestNeighborList(
            FixedCapacity(8), FixedCapacity(8), FixedCapacity(8), cutoffs
        )
        actual = jax.jit(as_result_function(neighborlist))(points, systems)
        expected = jax.jit(as_result_function(dense))(points, systems)
        actual.raise_assertion()
        expected.raise_assertion()
        assert _edge_rows(actual.value, 8, 8) == _edge_rows(expected.value, 8, 8)

    @pytest.mark.parametrize("cached", [False, True])
    def test_roundoff_at_periodic_boundary_preserves_neighbors(self, cached: bool):
        tiny = jnp.finfo(jnp.float64).eps / 4
        points = make_lh(
            jnp.array([[-tiny, 0.0, 0.0], [0.1, 0.0, 0.0]]), jnp.zeros(2, int)
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None], jnp.array([0.25]))
        layout = CellListCacheParameters.estimate(points, systems, cutoffs)
        cache = build_cell_list_cache(points, systems, cutoffs, (), layout)
        neighborlist = _neighborlist(cache if cached else None, cutoffs, layout, 2)
        result = jax.jit(as_result_function(neighborlist))(points, systems)
        result.raise_assertion()
        assert _edge_rows(result.value, 2, 2) == Counter(
            [(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)]
        )

    @pytest.mark.parametrize("cached", [False, True])
    @pytest.mark.parametrize("mode", ["full", "queried_keys", "queries"])
    @pytest.mark.parametrize("cutoff", [1.4, 5.4])
    @pytest.mark.parametrize(
        "periodic", [(True, True, True), (False, True, True), (False, False, False)]
    )
    def test_matches_dense_with_images_and_original_particle_ids(
        self, mode, cutoff, periodic, cached
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
        layout = CellListCacheParameters.estimate(
            points, systems, cutoffs, chunk_size=4, occupancy_headroom=1
        )
        # The allocated stride can exceed every system's actual bin count.
        layout = replace(layout, max_cells_per_system=layout.max_cells_per_system + 1)
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
        cache = build_cell_list_cache(points, systems, cutoffs, (), layout)
        neighborlist = _neighborlist(cache if cached else None, cutoffs, layout, 256)
        actual = jax.jit(as_result_function(neighborlist))(points, systems, **kwargs)
        expected = jax.jit(as_result_function(dense))(points, systems, **kwargs)
        actual.raise_assertion()
        expected.raise_assertion()
        assert actual.value.indices.keys == points.keys
        assert _edge_rows(actual.value, points.size, n_queries) == _edge_rows(
            expected.value, points.size, n_queries
        )

    @pytest.mark.parametrize("cached", [False, True])
    @pytest.mark.parametrize("mode", ["full", "queried_keys", "queries", "empty_keys"])
    def test_empty_query(self, mode, cached):
        points = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, int))
        queries = points
        if mode in ("full", "empty_keys"):
            points = points.subset(Index(points.keys, jnp.zeros(0, int)))
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 4, jnp.array([2.5]))
        layout = CellListCacheParameters.estimate(points, systems, cutoffs)
        cache = build_cell_list_cache(points, systems, cutoffs, (), layout)
        neighborlist = _neighborlist(cache if cached else None, cutoffs, layout, 8)
        empty = Index(points.keys, jnp.zeros(0, int), _cls=ParticleId)
        kwargs = {}
        if mode == "queried_keys":
            kwargs[mode] = empty
        elif mode == "queries":
            kwargs[mode] = points.subset(empty)
        elif mode == "empty_keys":
            kwargs["queries"] = queries
        # Empty queries must compile with the optimizations used in production.
        result = jax.jit(
            as_result_function(neighborlist),
            compiler_options={
                "xla_backend_optimization_level": 3,
                "xla_llvm_disable_expensive_passes": False,
            },
        )(points, systems, **kwargs)
        result.raise_assertion()
        assert not _edge_rows(result.value, points.size, queries.size)

    def test_selector_reuses_an_updated_table(self):
        points = make_lh(
            jnp.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [6.0, 1.0, 1.0]]),
            jnp.zeros(3, int),
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 12, jnp.array([3.2]))
        layout = CellListCacheParameters.estimate(
            points, systems, cutoffs, occupancy_factor=1, occupancy_headroom=1
        )
        table = build_cell_list_cache(points, systems, cutoffs, (), layout)
        moved = points.map_data(
            lambda p: replace(p, positions=p.positions.at[2].set(jnp.array([2, 1, 1])))
        )
        index = Index(points.keys, jnp.array([2]))
        rows = table.bin_rows(moved.subset(index), systems, ())
        patch = CellListCacheUpdatePatch(
            table.slot_of_row[index.indices], rows.system, rows, lens(lambda t: t)
        )
        updated = as_result_function(patch)(table, systems.set_data(jnp.array([True])))
        updated.raise_assertion()
        neighborlist = _neighborlist(updated.value, cutoffs, layout, 6)
        result = jax.jit(as_result_function(neighborlist))(moved, systems)
        result.raise_assertion()
        assert _edge_rows(result.value, 3, 3) == Counter(
            (i, j, 0, 0, 0) for i in range(3) for j in range(3) if i != j
        )

    def test_edge_capacity_and_query_contract(self):
        points = make_lh(
            jnp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]), jnp.zeros(2, int)
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 4, jnp.array([5.0]))
        layout = CellListCacheParameters.estimate(points, systems, cutoffs)
        cache = build_cell_list_cache(points, systems, cutoffs, (), layout)
        neighborlist = _neighborlist(cache, cutoffs, layout, 1)
        result = jax.jit(as_result_function(neighborlist))(points, systems)
        with pytest.raises(CapacityError):
            result.raise_assertion()
        with pytest.raises(AssertionError, match="cannot combine"):
            neighborlist(points, systems, queries=points, queried_keys=points.index)


class TestCellListCache:
    @pytest.mark.parametrize("max_count", [None, 2])
    def test_updates_align_index_vocabularies_and_preserve_bounds(
        self, max_count: int | None
    ):
        particles = make_lh(
            jnp.array([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]]), jnp.zeros(2, int)
        )
        particles = bind(particles, lambda p: p.data.inclusion.max_count).set(max_count)
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 12, jnp.array([3.0]))
        features = Index(
            (Label("A"), Label("B")), jnp.array([0, 1]), max_count=max_count
        )
        table = build_cell_list_cache(
            particles, systems, cutoffs, features, CellListCacheParameters()
        )
        query = particles.subset(Index(particles.keys, jnp.array([0])))
        query = bind(query, lambda p: p.data.inclusion.max_count).set(1)
        rows = table.bin_rows(
            query, systems, Index((Label("B"),), jnp.array([0]), max_count=1)
        )
        patch = CellListCacheUpdatePatch(
            table.slot_of_row[:1], rows.system, rows, lens(lambda t: t)
        )
        result = jax.jit(as_result_function(patch))(
            table, systems.set_data(jnp.ones(1, bool))
        )
        result.raise_assertion()
        updated = result.value.rows
        assert updated.inclusion.max_count == max_count
        assert updated.data.keys == features.keys
        assert updated.data.max_count == max_count
        npt.assert_array_equal(updated.data.indices[table.slot_of_row], [1, 1])

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
        parameters = CellListCacheParameters.estimate(
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
        parameters = CellListCacheParameters.estimate(
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
        parameters = CellListCacheParameters(
            chunk_size=8, max_cells_per_system=64, cell_capacity=2
        )
        table = build_cell_list_cache(
            particles, systems, cutoffs, jnp.zeros(3), parameters
        )
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
            patch = CellListCacheUpdatePatch(slot, rows.system, rows, lens(lambda t: t))
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


class TestSpatialOrder:
    def test_blocks_are_compact_per_system_with_exact_bounds(self):
        rng = np.random.default_rng(7)
        lengths = np.array([20.0, 12.0])
        system = np.repeat([0, 1], (256, 64))
        positions = rng.random((320, 3)) * lengths[system, None]
        # Eight inactive rows per system keep active runs block aligned.
        inactive = np.zeros(320, bool)
        inactive[rng.choice(256, 8, replace=False)] = True
        inactive[256 + rng.choice(64, 8, replace=False)] = True
        particles = make_lh(jnp.asarray(positions), jnp.asarray(system))
        inclusion = Index(particles.data.inclusion.keys, np.where(inactive, 2, system))
        particles = particles.map_data(lambda p: replace(p, inclusion=inclusion))
        systems, cutoffs = systems_from_lvecs(
            jnp.asarray(lengths[:, None, None] * np.eye(3)), jnp.array([3.0, 3.0])
        )
        parameters = CellListCacheParameters(
            chunk_size=8, max_cells_per_system=216, cell_capacity=64, key_block_size=8
        )
        table = jax.jit(build_cell_list_cache, static_argnums=4)(
            particles, systems, cutoffs, (), parameters
        )

        slot = np.asarray(table.slot_of_row)
        n_blocks = table.sentinel_slot // 8
        frac = np.empty((n_blocks * 8, 3))
        frac[slot] = positions / lengths[system, None]
        active = np.zeros(n_blocks * 8, bool)
        active[slot] = ~inactive
        npt.assert_array_equal(
            np.asarray(table.rows.cell[: n_blocks * 8]) != table.sentinel_cell, active
        )
        # Inactive slots come last and no block mixes systems.
        assert np.all(active[: active.sum()])
        block_system = np.full(n_blocks * 8, -1)
        block_system[slot] = np.where(inactive, -1, system)
        block_system = block_system.reshape(n_blocks, 8)
        assert np.all(block_system == block_system[:, :1])

        frac = frac.reshape(n_blocks, 8, 3)
        mask = active.reshape(n_blocks, 8, 1)
        lower = np.where(mask, frac, np.inf).min(1)
        upper = np.where(mask, frac, -np.inf).max(1)
        npt.assert_allclose(table.block_bounds.lower, lower.T, atol=1e-6)
        npt.assert_allclose(table.block_bounds.upper, upper.T, atol=1e-6)
        # Each block's system range is its one system; empty blocks have none.
        occupied = block_system[:, 0] >= 0
        lo, hi = np.asarray(table.block_bounds.systems.indices)
        npt.assert_array_equal(lo[occupied], block_system[occupied, 0])
        npt.assert_array_equal(hi[occupied], block_system[occupied, 0])
        assert np.all(lo[~occupied] > hi[~occupied])

        # Mean block extent relative to the edge of a cube holding one block.
        owner = block_system[occupied, 0]
        extent = (upper - lower)[occupied] * lengths[owner, None]
        density = np.array([248 / 20.0**3, 56 / 12.0**3])
        edge = (8 / density[owner]) ** (1 / 3)
        assert np.mean(extent / edge[:, None]) < 0.9


def _interleaved_points(positions, system, active):
    """Particles of interleaved systems; inactive rows keep their system."""
    points = make_lh(jnp.asarray(positions), jnp.asarray(system))
    inclusion = points.data.inclusion
    n_systems = len(inclusion.keys)
    inclusion = Index(inclusion.keys, jnp.where(active, system, n_systems))
    return points.map_data(lambda p: replace(p, inclusion=inclusion))


def _assert_exact_bounds(table, block_size):
    n = table.sentinel_slot
    frac = np.asarray(table.rows.frac[:n]).reshape(-1, block_size, 3)
    active = np.asarray(table.rows.cell[:n] != table.sentinel_cell)
    active = active.reshape(-1, block_size)
    system = table.rows.system[:n]
    keys = system.keys
    system = np.asarray(system.indices_in(keys)).reshape(-1, block_size)
    bounds = table.block_bounds
    npt.assert_array_equal(
        bounds.lower, np.where(active[..., None], frac, np.inf).min(1).T
    )
    npt.assert_array_equal(
        bounds.upper, np.where(active[..., None], frac, -np.inf).max(1).T
    )
    npt.assert_array_equal(
        bounds.systems.indices_in(keys),
        [
            np.where(active, system, len(keys)).min(1),
            np.where(active, system, 0).max(1),
        ],
    )


class TestBlockBounds:
    def test_empty_single_and_interleaved_blocks(self):
        # Blocks: empty, one active row, interleaved systems, one system with an
        # inactive row of another system.
        active = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1], [1, 0, 1, 1]], bool
        )
        ids = np.array([[0, 1, 2, 0], [2, 1, 1, 0], [2, 0, 2, 0], [1, 2, 1, 1]])
        frac = np.random.default_rng(3).random((16, 3))
        keys = tuple(SystemId(i) for i in range(3))
        bounds = BlockBounds.of(
            jnp.asarray(frac), Index(keys, ids.ravel()), active.ravel(), 4
        )
        frac = frac.reshape(4, 4, 3)
        npt.assert_array_equal(
            bounds.lower, np.where(active[..., None], frac, np.inf).min(1).T
        )
        npt.assert_array_equal(
            bounds.upper, np.where(active[..., None], frac, -np.inf).max(1).T
        )
        # The empty block's range is out of bounds above its first system.
        npt.assert_array_equal(
            bounds.systems.indices_in(keys), [[3, 1, 0, 1], [0, 1, 2, 1]]
        )
        holds = [bounds.holds(Index(keys, jnp.array(s))) for s in range(4)]
        npt.assert_array_equal(
            holds,
            [
                [False, False, True, False],
                [False, True, True, True],
                [False, False, True, False],
                # Out-of-bounds systems are in no block.
                [False, False, False, False],
            ],
        )

    def test_parameters_reject_non_positive_block_size(self):
        with pytest.raises(ValueError, match="key_block_size"):
            CellListCacheParameters(key_block_size=0)
        assert (
            CellListCacheParameters(chunk_size=6, key_block_size=4).slot_multiple == 12
        )

    @pytest.mark.parametrize("n_blocks", [1, 2, 5, 8])
    def test_bisection_splits_at_block_boundaries(self, n_blocks):
        levels = _bisection_levels(n_blocks, 3)
        assert len(levels) == int(np.ceil(np.log2(n_blocks)))
        for depth, level in enumerate(levels):
            assert level.shape == (3 * n_blocks,)
            steps = np.flatnonzero(np.diff(level)) + 1
            assert np.all(np.diff(level) >= 0) and np.all(steps % 3 == 0)
            assert level[-1] + 1 == min(2**depth, n_blocks)
        # Every leaf of the last split is a single block.
        if levels:
            sizes = np.bincount(levels[-1]) // 3
            assert np.all(sizes <= 2)

    def test_interleaved_systems_build_and_update_exactly(self):
        # Rows of three systems interleaved in input order, a slot count that is
        # not a block multiple, and a third system with no active rows.
        rng = np.random.default_rng(5)
        lengths = np.array([10.0, 14.0, 9.0])
        n, block_size = 61, 4
        system = rng.permutation(np.arange(n) % 3)
        positions = rng.random((n, 3)) * lengths[system, None]
        active = (system != 2) & (rng.random(n) > 0.15)
        systems, cutoffs = systems_from_lvecs(
            jnp.asarray(lengths[:, None, None] * np.eye(3)), jnp.full(3, 3.0)
        )
        parameters = CellListCacheParameters(
            chunk_size=block_size,
            max_cells_per_system=64,
            cell_capacity=16,
            key_block_size=block_size,
        )
        points = _interleaved_points(positions, system, active)
        table = jax.jit(build_cell_list_cache, static_argnums=4)(
            points, systems, cutoffs, (), parameters
        )
        slot = np.asarray(table.slot_of_row)
        assert np.unique(slot).size == n and slot.max() < table.sentinel_slot
        # Active rows come first, grouped by system: only the block at the one
        # boundary between the two active systems may mix them.
        slot_active = np.zeros(table.sentinel_slot, bool)
        slot_active[slot] = active
        assert np.all(slot_active[: active.sum()])
        slot_system = np.full(table.sentinel_slot, -1)
        slot_system[slot[active]] = system[active]
        mixed = [
            np.unique(block[block >= 0]).size > 1
            for block in slot_system.reshape(-1, block_size)
        ]
        assert sum(mixed) <= 1
        _assert_exact_bounds(table, block_size)
        # block_rows returns whole blocks of slots.
        blocks = table.block_rows(jnp.array([[1, 0], [2, 2]]))
        npt.assert_array_equal(
            blocks.frac[1, 0], table.rows.frac[2 * block_size : 3 * block_size]
        )
        assert blocks.cell.shape == (2, 2, block_size)

        # Moves across blocks, deletions and insertions, including into the
        # empty system and into blocks of other systems.
        @jax.jit
        @as_result_function
        def update(table, rows, proposal, accept):
            query = points.subset(Index(points.keys, rows)).map_data(
                lambda p: replace(
                    p,
                    positions=proposal.positions,
                    system=proposal.system,
                    inclusion=proposal.inclusion,
                )
            )
            binned = table.bin_rows(query, systems, ())
            patch = CellListCacheUpdatePatch(
                table.slot_of_row[rows], proposal.system, binned, lens(lambda t: t)
            )
            return patch(table, systems.set_data(accept))

        for step in range(6):
            rows = rng.choice(n, 8, replace=False)
            new_system = np.where(active[rows], system[rows], rng.integers(0, 3, 8))
            new_active = np.where(active[rows], rng.random(8) > 0.3, True)
            new_positions = rng.random((8, 3)) * lengths[new_system, None]
            accept = rng.random(3) > 0.3
            proposal = _interleaved_points(new_positions, new_system, new_active).data
            proposal = replace(
                proposal,
                system=Index(points.data.system.keys, jnp.asarray(new_system)),
                inclusion=Index(
                    points.data.inclusion.keys,
                    jnp.where(new_active, new_system, 3),
                ),
            )
            result = update(table, jnp.asarray(rows), proposal, jnp.asarray(accept))
            result.raise_assertion()
            table = result.value
            ok = accept[new_system]
            positions[rows[ok]] = new_positions[ok]
            system[rows[ok]] = new_system[ok]
            active[rows[ok]] = new_active[ok]
            _assert_exact_bounds(table, block_size)

        # The updated table still finds exactly the dense neighbors.
        points = _interleaved_points(positions, system, active)
        dense = DenseNearestNeighborList(
            FixedCapacity(64), FixedCapacity(1024), FixedCapacity(64 * 27), cutoffs
        )
        cached = _neighborlist(table, cutoffs, parameters, 1024)
        actual = jax.jit(as_result_function(cached))(points, systems)
        expected = jax.jit(as_result_function(dense))(points, systems)
        actual.raise_assertion()
        expected.raise_assertion()
        assert _edge_rows(actual.value, n, n) == _edge_rows(expected.value, n, n)
