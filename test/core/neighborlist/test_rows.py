# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""``neighbor_rows``: each particle's row holds exactly its pairs, in spatial order."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table, cast_indices
from kups.core.lens import bind
from kups.core.neighborlist import (
    CellScanNeighborList,
    DenseNearestNeighborList,
    Edges,
)
from kups.core.neighborlist.rows import NeighborRows, neighbor_rows
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId

from ._builders import SamplePoints, make_systems

CUTOFF = 3.0
LATTICES = np.array(
    [
        [[9.0, 0.0, 0.0], [3.6, 8.1, 0.0], [-2.7, 1.8, 9.0]],
        [[12.0, 0.0, 0.0], [4.8, 10.8, 0.0], [-3.6, 2.4, 12.0]],
        [[5.5, 0.0, 0.0], [1.0, 5.2, 0.0], [0.5, -0.8, 5.6]],  # < 2 x cutoff
    ]
)
COUNTS = (30, 60, 20)


def _tables(seed: int = 0, periodic=(True, True, True)) -> tuple[Table, Table, Table]:
    rng = np.random.default_rng(seed)
    system = rng.permutation(np.repeat(np.arange(len(COUNTS)), COUNTS))
    frac = rng.uniform(0, 1, (system.size, 3))
    positions = np.einsum("ni,nij->nj", frac, LATTICES[system])
    particles = Table(
        tuple(ParticleId(i) for i in range(system.size)),
        SamplePoints(
            positions=jnp.asarray(positions),
            system=Index.integer(system, label=SystemId),
            inclusion=Index.integer(system, label=SystemId),
            exclusion=Index.integer(np.arange(system.size)),
        ),
    )
    cell = Cell.from_pbc(TriclinicFrame.from_matrix(jnp.asarray(LATTICES)), periodic)
    systems, cutoffs = make_systems(cell, jnp.full(len(COUNTS), CUTOFF))
    return particles, systems, cutoffs


def _dense(cutoffs: Table) -> DenseNearestNeighborList:
    n = sum(COUNTS)
    return DenseNearestNeighborList(
        FixedCapacity(n), FixedCapacity(128), FixedCapacity(32 * n), cutoffs
    )


def _edge_rows(edges: Edges, n: int) -> list[set[tuple[int, ...]]]:
    """Per particle ``i``: ``(j, shift from i to j)`` for every edge ``(j, i, s)``."""
    valid = np.asarray(edges.indices.valid_mask).all(-1)
    ids = np.asarray(edges.indices.raw_indices)[valid]
    shifts = np.rint(np.asarray(edges.shifts)[valid, 0]).astype(int)
    rows: list[set[tuple[int, ...]]] = [set() for _ in range(n)]
    for (j, i), s in zip(ids, shifts):
        rows[i].add((int(j), *map(int, -s)))
    return rows


def _row_sets(rows: NeighborRows) -> list[set[tuple[int, ...]]]:
    order, offsets = np.asarray(rows.order), np.asarray(rows.offsets)
    neighbors, shifts = np.asarray(rows.neighbors), np.asarray(rows.shifts)
    out: list[set[tuple[int, ...]]] = [set() for _ in range(order.size)]
    for k, particle in enumerate(order):
        for slot in range(offsets[k], offsets[k + 1]):
            out[particle].add((int(order[neighbors[slot]]), *map(int, shifts[slot])))
    return out


@pytest.mark.parametrize("periodic", [(True, True, True), (True, True, False)])
@pytest.mark.parametrize("builder", ["dense", "cell_scan"])
def test_rows_hold_each_particles_pairs(builder, periodic):
    particles, systems, cutoffs = _tables(periodic=periodic)
    if builder == "dense":
        nl = _dense(cutoffs)
    else:
        nl = CellScanNeighborList(FixedCapacity(128), FixedCapacity(512), cutoffs)
    edges = nl(particles, systems)
    rows = neighbor_rows(edges, particles, systems)
    expected = _edge_rows(edges, particles.size)
    assert _row_sets(rows) == expected
    assert sum(map(len, expected)) == int(rows.offsets[-1])
    assert rows.neighbors.dtype == jnp.int32 and rows.shifts.dtype == jnp.int8


def test_centers_name_each_slots_row():
    particles, systems, cutoffs = _tables()
    rows = neighbor_rows(_dense(cutoffs)(particles, systems), particles, systems)
    offsets, centers = np.asarray(rows.offsets), np.asarray(rows.centers)
    filled = np.repeat(np.arange(particles.size), np.diff(offsets))
    np.testing.assert_array_equal(centers[: offsets[-1]], filled)
    assert offsets[-1] < centers.size  # padding names no row
    np.testing.assert_array_equal(centers[offsets[-1] :], particles.size)
    assert centers.dtype == jnp.int32


def test_rows_are_symmetric_with_several_images():
    particles, systems, cutoffs = _tables()
    rows = neighbor_rows(_dense(cutoffs)(particles, systems), particles, systems)
    sets = _row_sets(rows)
    for i, row in enumerate(sets):
        for j, *s in row:
            assert (i, *(-np.asarray(s))) in sets[j]
    # the small cell is narrower than twice the cutoff: some pairs appear twice
    small = np.nonzero(np.asarray(particles.data.system.raw_indices) == 2)[0]
    assert any(len({j for j, *_ in sets[i]}) < len(sets[i]) for i in small)


def test_edge_order_does_not_change_rows():
    particles, systems, cutoffs = _tables()
    edges = _dense(cutoffs)(particles, systems)
    perm = np.random.default_rng(1).permutation(len(edges))
    shuffled = Edges(edges.indices[perm], edges.shifts[perm])
    a = neighbor_rows(edges, particles, systems)
    b = neighbor_rows(shuffled, particles, systems)
    np.testing.assert_array_equal(a.order, b.order)
    np.testing.assert_array_equal(a.offsets, b.offsets)
    assert _row_sets(a) == _row_sets(b)


def test_spatial_order_is_z_order_within_systems():
    # a 4 x 4 x 4 grid in one cubic cell: the order interleaves the bits of the cell
    grid = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
    n = len(grid)
    particles = Table(
        tuple(ParticleId(i) for i in range(n)),
        SamplePoints(
            positions=jnp.asarray((grid + 0.5) * 2.0),
            system=Index.integer(np.zeros(n, int), label=SystemId),
            inclusion=Index.integer(np.zeros(n, int), label=SystemId),
            exclusion=Index.integer(np.arange(n)),
        ),
    )
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 8.0))
    systems, cutoffs = make_systems(cell, jnp.array([1.0]))
    rows = neighbor_rows(_dense(cutoffs)(particles, systems), particles, systems)
    code = sum(
        ((grid[:, a] >> b) & 1) << (3 * b + a) for b in range(2) for a in range(3)
    )
    np.testing.assert_array_equal(rows.order, np.argsort(code, kind="stable"))
    # several systems: grouped by system, rank inverts order
    particles, systems, cutoffs = _tables()
    rows = neighbor_rows(_dense(cutoffs)(particles, systems), particles, systems)
    system = np.asarray(particles.data.system.raw_indices)[np.asarray(rows.order)]
    assert np.all(np.diff(system) >= 0)
    np.testing.assert_array_equal(
        np.asarray(rows.rank)[np.asarray(rows.order)], np.arange(particles.size)
    )


def test_int32_indices_give_the_same_rows():
    particles, systems, cutoffs = _tables()
    a = neighbor_rows(_dense(cutoffs)(particles, systems), particles, systems)
    particles32, systems32 = cast_indices((particles, systems), jnp.int32)
    b = neighbor_rows(_dense(cutoffs)(particles32, systems32), particles32, systems32)
    np.testing.assert_array_equal(a.order, b.order)
    assert _row_sets(a) == _row_sets(b)


def test_shifts_beyond_int8_fail():
    """A particle 130 cells away from its pairs (unwrapped input)."""
    particles, systems, cutoffs = _tables()
    lattice = LATTICES[int(particles.data.system.raw_indices[0])]
    far = bind(particles, lambda p: p.data.positions).apply(
        lambda x: x.at[0].add(130 * lattice[0])
    )
    edges = _dense(cutoffs)(far, systems)
    result = jax.jit(as_result_function(neighbor_rows))(edges, far, systems)
    with pytest.raises(AssertionError, match="exceed int8"):
        result.raise_assertion()
