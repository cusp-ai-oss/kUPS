# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""``SkinList``: the stored rows and the pairs within a cutoff always equal a direct
build.

Three random systems in one batch, one narrower than twice the cutoff so pairs
appear through several images; the reference is a dense list with shifts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table, cast_indices
from kups.core.lens import bind, lens, view
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    CellListNeighborList,
    CellScanNeighborList,
    DenseNearestNeighborList,
    Edges,
    NeighborRows,
    SkinList,
    SkinListPropagator,
    SkinNeighborList,
    SkinReference,
    neighbor_rows,
    wraps_since_build,
)
from kups.core.propagator import propagate_and_fix
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass

from ._builders import SamplePoints, SampleSystems, make_systems

CUTOFF, SKIN = 3.0, 0.5
LATTICES = np.array(
    [
        [[12.0, 0.0, 0.0], [1.5, 11.0, 0.0], [-1.0, 2.0, 12.5]],
        [[5.5, 0.0, 0.0], [1.0, 5.2, 0.0], [0.5, -0.8, 5.6]],  # < 2 x cutoff
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    ]
)
COUNTS = (60, 30, 45)


@dataclass
class _State:
    particles: Table
    systems: Table
    skin_list: SkinList


def _tables(seed: int = 0) -> tuple[Table, Table, Table]:
    rng = np.random.default_rng(seed)
    system = np.repeat(np.arange(len(COUNTS)), COUNTS)
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
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.asarray(LATTICES)))
    systems, cutoffs = make_systems(cell, jnp.full(len(COUNTS), CUTOFF))
    return particles, systems, cutoffs


def _skin(*values: float) -> Table[SystemId, Array]:
    """One skin for all systems, or one per system."""
    return Table.arange(jnp.array(values), label=SystemId)


def _state(builder=CellListNeighborList.new, skin=(SKIN,)) -> _State:
    particles, systems, cutoffs = _tables()
    skin_list = SkinList.create(
        particles, systems, cutoffs, _skin(*skin), builder=builder
    )
    return _State(particles, systems, skin_list)


def _step():
    propagator = SkinListPropagator(
        skin_list=lens(lambda s: s.skin_list, cls=_State),
        particles=view(lambda s: s.particles),
        systems=view(lambda s: s.systems),
    )
    return jax.jit(as_result_function(propagator))


def _pairs(edges: Edges) -> set[tuple[int, ...]]:
    valid = np.asarray(edges.indices.valid_mask).all(-1)
    ids = np.asarray(edges.indices.raw_indices)[valid]
    shifts = np.rint(np.asarray(edges.shifts)[valid, 0]).astype(int)
    return {(*map(int, i), *map(int, s)) for i, s in zip(ids, shifts)}


def _row_pairs(rows: NeighborRows) -> set[tuple[int, ...]]:
    """``(i, j, shift)`` of every slot, ``i`` its row's particle, as in ``_pairs``."""
    order, centers = np.asarray(rows.order), np.asarray(rows.centers)
    filled = centers < order.size
    i = order[centers[filled]]
    j = order[np.asarray(rows.neighbors)[filled]]
    shifts = np.asarray(rows.shifts)[filled]
    return {(int(a), int(b), *map(int, s)) for a, b, s in zip(i, j, shifts)}


def _dense(cutoffs: Table) -> DenseNearestNeighborList:
    n = sum(COUNTS)
    return DenseNearestNeighborList(
        FixedCapacity(n), FixedCapacity(128), FixedCapacity(32 * n), cutoffs
    )


def _assert_complete(state: _State, queried_keys: Index[ParticleId] | None = None):
    cutoffs = Table(state.systems.keys, jnp.full(len(COUNTS), CUTOFF))
    skin = SkinNeighborList(state.skin_list, FixedCapacity(128), cutoffs)
    args = (state.particles, state.systems)
    got = _pairs(skin(*args, queried_keys=queried_keys))
    expected = _pairs(_dense(cutoffs)(*args, queried_keys=queried_keys))
    assert got == expected and expected


def _assert_rows_built_at_reference(state: _State):
    """The stored rows are those of a dense build at the stored positions."""
    skin_list = state.skin_list
    built = bind(state, lambda s: s.particles.data.positions).set(
        skin_list.reference.positions
    )
    edges = _dense(skin_list.cutoffs + skin_list.skin)(built.particles, built.systems)
    fresh = neighbor_rows(edges, built.particles, built.systems)
    np.testing.assert_array_equal(skin_list.rows.order, fresh.order)
    np.testing.assert_array_equal(skin_list.rows.offsets, fresh.offsets)
    assert _row_pairs(skin_list.rows) == _row_pairs(fresh)


def _wrapped(state: _State, positions: Array, lattices: Array | None = None) -> _State:
    if lattices is not None:
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattices))
        systems = state.systems.set_data(SampleSystems(cell=cell))
        state = bind(state, lambda s: s.systems).set(systems)
    cell = state.systems[state.particles.data.system].cell
    # integrators wrap positions into the cell
    return bind(state, lambda s: s.particles.data.positions).set(cell.wrap(positions))


def _pulled(state: _State) -> Array:
    """The large system's atoms pulled to its centre: more pairs than stored."""
    system = state.particles.data.system.raw_indices
    centre = jnp.asarray(LATTICES[0].sum(0) / 2)
    positions = state.particles.data.positions
    return jnp.where(
        (system == 0)[:, None], centre + 0.2 * (positions - centre), positions
    )


def _rebuilt(before: _State, after: _State) -> bool:
    old, new = before.skin_list.reference.positions, after.skin_list.reference.positions
    return not np.array_equal(old, new)


def test_create_matches_direct_builds():
    state = _state()
    _assert_rows_built_at_reference(state)
    _assert_complete(state)


def test_random_walk_with_wrapping_stays_complete():
    step, state = _step(), _state()
    rng = np.random.default_rng(1)
    rebuilds = 0
    for i in range(20):
        noise = rng.normal(0, 0.05, (state.particles.size, 3))
        moved = _wrapped(state, state.particles.data.positions + noise)
        new = propagate_and_fix(step, jax.random.key(i), moved)
        rebuilds += _rebuilt(moved, new)
        state = new
        _assert_complete(state)
    assert 0 < rebuilds < 20


def test_strain_and_shear_stay_complete():
    step, state = _step(), _state()
    rng = np.random.default_rng(2)
    lattices = jnp.asarray(LATTICES)
    for i in range(12):
        strain = jnp.eye(3) + 0.01 * jnp.array(
            [[-1.0, 0.3, 0.0], [0.0, -0.5, 0.2], [0.0, 0.0, 0.8]]
        )
        frac = state.systems[state.particles.data.system].cell.frame.to_fractional(
            state.particles.data.positions
        )
        lattices = lattices @ strain
        positions = jnp.einsum(
            "ni,nij->nj", frac, lattices[state.particles.data.system.raw_indices]
        )
        noise = rng.normal(0, 0.02, positions.shape)
        state = _wrapped(state, positions + noise, lattices)
        state = propagate_and_fix(step, jax.random.key(i), state)
        _assert_complete(state)


def test_no_rebuild_within_the_margin():
    step, state = _step(), _state()
    nudged = _wrapped(state, state.particles.data.positions + 0.2 * SKIN / 2)
    after = propagate_and_fix(step, jax.random.key(0), nudged)
    assert not _rebuilt(nudged, after)
    moved = _wrapped(after, after.particles.data.positions + SKIN)
    assert _rebuilt(moved, propagate_and_fix(step, jax.random.key(1), moved))


def test_capacity_growth_rebuilds_on_the_host():
    step, state = _step(), _state()
    stored = len(state.skin_list.rows.neighbors)
    state = propagate_and_fix(step, jax.random.key(0), _wrapped(state, _pulled(state)))
    assert len(state.skin_list.rows.neighbors) > stored
    _assert_complete(state)


def test_rows_follow_rebuilds():
    step, state = _step(), _state()
    moved = _wrapped(state, state.particles.data.positions + SKIN)
    state = propagate_and_fix(step, jax.random.key(0), moved)
    assert _rebuilt(moved, state)
    _assert_rows_built_at_reference(state)
    # rebuilt on the host with larger capacities
    state = propagate_and_fix(step, jax.random.key(1), _wrapped(state, _pulled(state)))
    _assert_rows_built_at_reference(state)


def test_each_system_has_its_own_skin():
    """Each system lists the pairs within its own cutoff + skin, and the list is
    rebuilt once motion uses up the skin of one system."""
    step, state = _step(), _state(skin=(0.2, 0.5, 0.8))
    _assert_rows_built_at_reference(state)
    system = state.particles.data.system.raw_indices[:, None]
    # moving the atoms of one system by 0.15 Å uses up 0.3 Å of its skin
    for moved_system, rebuilt in ((2, False), (1, False), (0, True)):
        shift = jnp.where(system == moved_system, jnp.array([0.15, 0.0, 0.0]), 0.0)
        moved = _wrapped(state, state.particles.data.positions + shift)
        after = propagate_and_fix(step, jax.random.key(0), moved)
        assert _rebuilt(moved, after) is rebuilt
    _assert_rows_built_at_reference(after)
    _assert_complete(after)


def test_edges_view_matches_rows():
    """At the build, the edges within the cutoff are the rows' slots within it."""
    state = _state()
    cutoffs = Table(state.systems.keys, jnp.full(len(COUNTS), CUTOFF))
    skin = SkinNeighborList(state.skin_list, FixedCapacity(128), cutoffs)
    got = _pairs(skin(state.particles, state.systems))
    positions = np.asarray(state.particles.data.positions)
    lattices = LATTICES[np.asarray(state.particles.data.system.raw_indices)]
    expected = {
        (i, j, *s)
        for i, j, *s in _row_pairs(state.skin_list.rows)
        if np.linalg.norm(positions[j] - positions[i] + np.dot(s, lattices[i])) < CUTOFF
    }
    assert got == expected and expected


@pytest.mark.parametrize(
    "periodic", [(True, True, True), (True, True, False)], ids=["bulk", "slab"]
)
def test_wraps_since_build_under_shear(periodic):
    """Wraps applied since the build come back after a large shear; open axes have
    none."""
    particles, systems, cutoffs = _tables()
    system = particles.data.system
    reference = SkinReference(particles.data.positions, systems.data.cell)
    built = systems[system].cell.frame.to_fractional(particles.data.positions)
    rng = np.random.default_rng(3)
    wraps = rng.integers(-2, 3, built.shape) * np.asarray(periodic)
    fractional = built + rng.uniform(-0.3, 0.3, built.shape) - wraps
    # lower triangular like the lattices, which TriclinicFrame keeps
    shear = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.4, 0.6, 1.0]])
    frame = TriclinicFrame.from_matrix(jnp.asarray(LATTICES @ shear))
    sheared, _ = make_systems(Cell.from_pbc(frame, periodic), cutoffs.data)
    got = wraps_since_build(reference, fractional, system, sheared)
    np.testing.assert_array_equal(got, wraps)


@pytest.mark.parametrize(
    "builder",
    [
        DenseNearestNeighborList.new,
        CellListNeighborList.new,
        CellScanNeighborList.new,
        AdaptiveNeighborList.new,
    ],
    ids=["dense", "cell_list", "cell_scan", "adaptive"],
)
def test_builders_agree(builder):
    _assert_complete(_state(builder))


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64], ids=["int32", "int64"])
def test_index_dtypes(dtype):
    """Rows are int32 and int8; edges follow the particles' index dtype."""
    particles, systems, cutoffs = cast_indices(_tables(), dtype)
    skin_list = SkinList.create(particles, systems, cutoffs, _skin(SKIN))
    rows = skin_list.rows
    for x in (rows.order, rows.rank, rows.offsets, rows.centers, rows.neighbors):
        assert x.dtype == jnp.int32
    assert rows.shifts.dtype == jnp.int8
    edges = SkinNeighborList(skin_list, FixedCapacity(128), cutoffs)(particles, systems)
    assert edges.indices.raw_indices.dtype == dtype
    assert edges.shifts.dtype == particles.data.positions.dtype
    _assert_complete(_State(particles, systems, skin_list))


def test_queried_keys_match_direct_builds():
    state = _state()
    queried = Index.integer(
        np.arange(0, sum(COUNTS), 7), n=sum(COUNTS), label=ParticleId
    )
    _assert_complete(state, queried)


def test_cutoffs_above_the_list_raise():
    state = _state()
    cutoffs = Table(state.systems.keys, jnp.full(len(COUNTS), CUTOFF + 2 * SKIN))
    skin = SkinNeighborList(state.skin_list, FixedCapacity(128), cutoffs)
    result = jax.jit(as_result_function(skin))(state.particles, state.systems)
    with pytest.raises(AssertionError, match="exceed the skin list"):
        result.raise_assertion()


def test_cells_narrower_than_the_skin_raise():
    particles, systems, cutoffs = _tables()
    with pytest.raises(AssertionError, match="wider than the skin"):
        SkinList.create(particles, systems, cutoffs, _skin(6.0))
