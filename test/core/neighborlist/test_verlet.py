# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Verlet skin list (kups.core.neighborlist.verlet).

``skin_margin`` is covered on its own: motion threshold, compression, expansion,
pure shear, triclinic boundary crossing, non-periodic axes, per-system accounting
and cells narrower than twice the build radius.

``SkinList`` is compared with a dense list with shifts on three random systems in
one batch, one of them narrower than twice the cutoff, so that pairs appear through
several images. Whichever list the state holds and whichever of its patches
landed, a read returns exactly the pairs within the cutoff.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from kups.core.assertion import runtime_assert
from kups.core.capacity import FixedCapacity
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table, cast_indices
from kups.core.lens import bind, identity_lens, lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    CellListNeighborList,
    CellScanNeighborList,
    DenseNearestNeighborList,
    Edges,
    IsUniversalNeighborlistParams,
    NeighborListFactory,
    NeighborRows,
    SkinList,
    SkinMargin,
    SkinNeighborList,
    SkinReference,
    UniversalNeighborlistParameters,
    neighbor_rows,
    skin_margin,
    wraps_since_build,
)
from kups.core.neighborlist.verlet import _sym3_min_eigenvalue
from kups.core.patch import ComposedPatch, Patch, WithPatch
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass

from ._builders import SamplePoints, SampleSystems, make_lh, make_systems

CUTOFF, SKIN = 3.5, 1.0


def _cell(lvecs: Array, periodic=(True, True, True), n_sys: int = 1) -> Cell:
    """An ``(n_sys,)``-batched cell."""
    lv = jnp.asarray(lvecs)
    if lv.ndim == 2:
        lv = jnp.repeat(lv[None], n_sys, axis=0)
    return Cell.from_pbc(TriclinicFrame.from_matrix(lv), periodic)


def _margins(
    positions: Array,
    reference_positions: Array,
    cell_now: Cell,
    cell_ref: Cell,
    system: Array | None = None,
    cutoff: float = CUTOFF,
    skin: float | Array = SKIN,
) -> Table[SystemId, SkinMargin]:
    if system is None:
        system = jnp.zeros(positions.shape[0], dtype=int)
    n_sys = int(jnp.max(system)) + 1
    particles = make_lh(positions, system)
    systems, cutoffs = make_systems(cell_now, jnp.full((n_sys,), cutoff))
    skins = cutoffs.set_data(jnp.broadcast_to(skin, (n_sys,)))
    reference = SkinReference(reference_positions, cell_ref, system)
    return skin_margin(particles, systems, reference, cutoffs, skins)


def _margin_fires(*args, **kwargs) -> bool:
    return bool(jnp.min(_margins(*args, **kwargs).data.headroom) < 0.0)


class TestSkinMargin:
    def test_motion_threshold_at_half_skin(self):
        """One atom moving just under/over ``skin/2`` sits on the margin boundary."""
        cell = _cell(20.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        for factor, expected in [(0.99, False), (1.01, True)]:
            moved = pos.at[0, 0].add(factor * SKIN / 2)
            assert _margin_fires(moved, pos, cell, cell) is expected

    def test_compression_consumes_margin(self):
        """Isotropic compression consumes ``r_build * (1 - sigma_min)``."""
        ref = _cell(20.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        for factor, expected in [(0.99, False), (1.01, True)]:
            f = 1.0 - factor * SKIN / (CUTOFF + SKIN)
            now = _cell(20.0 * f * jnp.eye(3))
            # Atoms ride the cell (pure affine motion): zero residual, all
            # margin consumption comes from the deformation term.
            assert _margin_fires(pos * f, pos, now, ref) is expected

    def test_expansion_consumes_no_margin(self):
        """Expansion moves non-listed pairs farther away; sigma_min > 1 is free."""
        ref = _cell(20.0 * jnp.eye(3))
        now = _cell(30.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        assert _margin_fires(pos * 1.5, pos, now, ref) is False

    def test_pure_shear_consumes_margin(self):
        """An off-diagonal cell move must consume margin (sigma_min < 1) even
        though it is invisible to any per-axis length ratio."""
        ref = _cell(20.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        for gamma, expected in [(1.0, False), (12.0, True)]:
            lvecs = jnp.array([[20.0, 0.0, 0.0], [gamma, 20.0, 0.0], [0.0, 0.0, 20.0]])
            now = _cell(lvecs)
            # Atoms at fixed fractional coordinates: pure affine, zero residual.
            frac = ref.frame.to_fractional(pos)
            assert _margin_fires(now.frame.to_real(frac), pos, now, ref) is expected

    def test_triclinic_boundary_crossing(self):
        """A wrap along a sheared lattice vector must not distort the residual.

        The atom crosses the ``a2`` boundary, so its stored (wrapped) position
        jumps by a lattice vector with off-diagonal Cartesian components. Undoing
        that per axis with the perpendicular lengths (the old formula) misreads
        the ~0.4 A true displacement as ~3.9 A and rebuilds spuriously; the
        minimum-image residual through the cell stays exact.
        """
        lvecs = jnp.array([[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 10.0]])
        cell = _cell(lvecs)
        f_ref = jnp.array([[0.5, 0.98, 0.5]])
        pos_ref = cell.frame.to_real(f_ref)
        for df, expected in [(0.04, False), (0.07, True)]:
            # |a2| = 10, so a fractional move of df along a2 is a 10*df true step.
            pos_new = cell.wrap(cell.frame.to_real(f_ref + jnp.array([[0, df, 0]])))
            raw = jnp.linalg.norm(pos_new - pos_ref)
            assert float(raw) > 5.0  # the raw difference jumped by ~a lattice vector
            assert _margin_fires(pos_new, pos_ref, cell, cell) is expected

    def test_per_system_accounting(self):
        """One hot system must not charge a cold system's margin: pairs never
        span systems, so the motion term reduces per system."""
        cell = _cell(20.0 * jnp.eye(3), n_sys=2)
        pos = jnp.array(
            [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]
        )
        system = jnp.array([0, 0, 1, 1])
        moved = pos.at[2, 0].add(0.6)  # only system 1 moves
        margin = _margins(moved, pos, cell, cell, system=system).data
        assert float(margin.consumed[0]) == pytest.approx(0.0)
        assert float(margin.consumed[1]) == pytest.approx(1.2)
        assert bool(margin.headroom[0] >= 0.0) and bool(margin.headroom[1] < 0.0)

    def test_skin_per_system(self):
        """Each system's budget is its own skin."""
        cell = _cell(20.0 * jnp.eye(3), n_sys=2)
        pos = jnp.array(
            [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]
        )
        system = jnp.array([0, 0, 1, 1])
        moved = pos.at[0, 0].add(0.6).at[2, 0].add(0.6)  # consumes 1.2 in both
        skin = jnp.array([1.0, 1.5])
        margin = _margins(moved, pos, cell, cell, system=system, skin=skin).data
        np.testing.assert_allclose(margin.budget, skin)
        assert bool(margin.headroom[0] < 0.0) and bool(margin.headroom[1] >= 0.0)

    def test_nonperiodic_axes_are_not_unwrapped(self):
        """In a vacuum cell nothing wraps, so a large real move must trigger even
        when a periodic un-wrap of the bounding box would shrink it below the
        threshold (6 A move, 10 A box: un-wrapping would misread it as 4 A)."""
        cell = _cell(10.0 * jnp.eye(3), periodic=(False, False, False))
        pos_ref = jnp.array([[2.0, 5.0, 5.0]])
        pos_new = pos_ref + jnp.array([[6.0, 0.0, 0.0]])
        # 2 * 6 >= 10 triggers; the misread 2 * 4 would not.
        assert _margin_fires(pos_new, pos_ref, cell, cell, cutoff=1.0, skin=10.0)

    def test_budget_is_the_skin_in_small_cells(self):
        """Images are followed, so a cell narrower than ``2 (cutoff + skin)`` keeps the
        whole skin."""
        cell = _cell(6.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0]])
        margin = _margins(pos, pos, cell, cell).data
        np.testing.assert_allclose(margin.budget, [SKIN])
        np.testing.assert_allclose(margin.consumed, [0.0])


def test_sym3_min_eigenvalue_matches_eigvalsh():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((64, 3, 3))
    gram = m @ np.swapaxes(m, -1, -2)
    near_identity = np.eye(3) + 1e-7 * (gram[:8] + np.swapaxes(gram[:8], -1, -2))
    a = np.concatenate([gram, near_identity, np.eye(3)[None] * 2.0])
    expected = np.linalg.eigvalsh(a)[:, 0]
    np.testing.assert_allclose(
        _sym3_min_eigenvalue(jnp.asarray(a)), expected, rtol=1e-10, atol=1e-12
    )


LATTICES = np.array(
    [
        [[12.0, 0.0, 0.0], [1.5, 11.0, 0.0], [-1.0, 2.0, 12.5]],
        [[5.5, 0.0, 0.0], [1.0, 5.2, 0.0], [0.5, -0.8, 5.6]],  # < 2 x cutoff
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    ]
)
COUNTS = (60, 30, 45)
LIST_CUTOFF, LIST_SKIN = 3.0, 0.5


@dataclass
class _State:
    particles: Table[ParticleId, SamplePoints]
    systems: Table[SystemId, SampleSystems]
    neighborlist_params: UniversalNeighborlistParameters
    skin_list: SkinList


@dataclass
class _Outer:
    """Nests ``_State``, so lenses and fixes must start at the root."""

    inner: _State


STATE = identity_lens(_State)


def _tables() -> tuple[
    Table[ParticleId, SamplePoints],
    Table[SystemId, SampleSystems],
    Table[SystemId, Array],
]:
    rng = np.random.default_rng(0)
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
    systems, cutoffs = make_systems(cell, jnp.full(len(COUNTS), LIST_CUTOFF))
    return particles, systems, cutoffs


def _skin(*values: float) -> Table[SystemId, Array]:
    """One skin for all systems, or one per system."""
    return Table.arange(jnp.array(values), label=SystemId)


def _state(
    builder: NeighborListFactory[
        IsUniversalNeighborlistParams
    ] = CellListNeighborList.new,
    skin: tuple[float, ...] = (LIST_SKIN,),
) -> _State:
    particles, systems, cutoffs = _tables()
    skin_list = SkinList.create(
        particles, systems, cutoffs, _skin(*skin), builder=builder
    )
    params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoffs
    )
    return _State(particles, systems, params, skin_list)


def _cutoffs(state: _State, cutoff: float = LIST_CUTOFF) -> Table[SystemId, Array]:
    return Table(state.systems.keys, jnp.full(len(COUNTS), cutoff))


def _read(
    state: _State,
    queried_keys: Index[ParticleId] | None = None,
    cutoff: float = LIST_CUTOFF,
) -> WithPatch[Edges[Literal[2]], Patch[_State]]:
    reader = SkinNeighborList.from_state(state, STATE, _cutoffs(state, cutoff))
    return reader(state.particles, state.systems, queried_keys=queried_keys)


_checked_read = jax.jit(as_result_function(_read))


def _read_and_fix(
    state: _State, queried_keys: Index[ParticleId] | None = None
) -> tuple[_State, WithPatch[Edges[Literal[2]], Patch[_State]]]:
    """Read, after applying the fixes of the assertions a read fails."""
    while (result := _checked_read(state, queried_keys)).failed_assertions:
        state = result.fix_or_raise(state)
    return state, result.value


def _commit(state: _State, patch: Patch[_State], *accepted: bool) -> _State:
    """Apply ``patch``, by default with every system accepting."""
    accept = accepted or (True,) * len(COUNTS)
    return patch(state, Table(state.systems.keys, jnp.asarray(accept)))


def _rebuilt(before: _State, after: _State) -> bool:
    """Whether ``after`` holds a list built after the one ``before`` holds."""
    old, new = before.skin_list.reference.positions, after.skin_list.reference.positions
    return not np.array_equal(old, new)


def _pairs(edges: Edges[Literal[2]]) -> set[tuple[int, ...]]:
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


def _dense(cutoffs: Table[SystemId, Array]) -> DenseNearestNeighborList:
    n = sum(COUNTS)
    return DenseNearestNeighborList(
        FixedCapacity(n), FixedCapacity(128), FixedCapacity(32 * n), cutoffs
    )


def _expected(
    state: _State,
    queried_keys: Index[ParticleId] | None = None,
    cutoff: float = LIST_CUTOFF,
) -> set[tuple[int, ...]]:
    """The pairs within ``cutoff`` from a dense build."""
    found = _dense(_cutoffs(state, cutoff))(
        state.particles, state.systems, queried_keys=queried_keys
    )
    return _pairs(found.data)


def _assert_rows_built_at_reference(skin_list: SkinList, state: _State) -> None:
    """The rows of ``skin_list`` are those of a dense build at its reference
    positions and cells."""
    reference = skin_list.reference
    built = bind(
        state, lambda s: (s.particles.data.positions, s.systems.data.cell)
    ).set((reference.positions, reference.cell))
    edges = _dense(skin_list.cutoffs + skin_list.skin)(built.particles, built.systems)
    fresh = neighbor_rows(edges.data, built.particles, built.systems)
    np.testing.assert_array_equal(skin_list.rows.order, fresh.order)
    np.testing.assert_array_equal(skin_list.rows.offsets, fresh.offsets)
    assert _row_pairs(skin_list.rows) == _row_pairs(fresh)


def _wrapped(state: _State, positions: Array, lattices: Array | None = None) -> _State:
    if lattices is not None:
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattices))
        systems = state.systems.set_data(SampleSystems(cell=cell))
        state = bind(state, lambda s: s.systems).set(systems)
    cell = state.systems[state.particles.data.system].cell
    # Integrators wrap positions into the cell.
    return bind(state, lambda s: s.particles.data.positions).set(cell.wrap(positions))


def _pulled(state: _State) -> Array:
    """The large system's atoms pulled to its center: more pairs than stored."""
    system = state.particles.data.system.raw_indices
    center = jnp.asarray(LATTICES[0].sum(0) / 2)
    positions = state.particles.data.positions
    return jnp.where(
        (system == 0)[:, None], center + 0.2 * (positions - center), positions
    )


def _strained(state: _State, rng: np.random.Generator) -> _State:
    """Positions riding a random strain and shear of every cell."""
    system = state.particles.data.system
    frac = state.systems[system].cell.frame.to_fractional(
        state.particles.data.positions
    )
    # Lower triangular, like the lattices, which TriclinicFrame keeps.
    lattices = state.systems.data.cell.vectors @ (
        jnp.eye(3) + jnp.tril(jnp.asarray(rng.normal(0, 0.02, (3, 3))))
    )
    positions = jnp.einsum("ni,nij->nj", frac, lattices[system.raw_indices])
    return _wrapped(state, positions, lattices)


def test_create_matches_direct_builds():
    state = _state()
    _assert_rows_built_at_reference(state.skin_list, state)
    _, found = _read_and_fix(state)
    assert _pairs(found.data) == _expected(state)


@pytest.mark.parametrize("seed", range(4))
def test_no_stale_read(seed: int):
    """Whichever list the state holds and whichever patches landed, a read returns
    exactly the pairs within the cutoff."""
    rng = np.random.default_rng(seed)
    state = _state()
    held = [state.skin_list]
    for _ in range(16):
        scale = rng.choice([0.01, 0.1, LIST_SKIN, 3 * LIST_SKIN])
        noise = rng.normal(0, scale, (state.particles.size, 3))
        state = _wrapped(state, state.particles.data.positions + noise)
        if rng.random() < 0.3:
            state = _strained(state, rng)
        # Any list the state ever held.
        state = bind(state, lambda s: s.skin_list).set(held[rng.integers(len(held))])
        queried = None
        if rng.random() < 0.3:
            ids = np.sort(rng.choice(sum(COUNTS), sum(COUNTS) // 5, replace=False))
            queried = Index.integer(ids, n=sum(COUNTS), label=ParticleId)
        state, found = _read_and_fix(state, queried)
        assert _pairs(found.data) == _expected(state, queried)
        # The patch may be dropped or applied with any acceptance.
        if rng.random() < 0.7:
            state = _commit(state, found.patch, *(rng.random(len(COUNTS)) < 0.5))
        held.append(state.skin_list)


def test_random_walk_with_wrapping_stays_complete():
    """Reading and committing at every step, the list is reused and rebuilt."""
    state = _state()
    rng = np.random.default_rng(1)
    rebuilds = 0
    for _ in range(20):
        noise = rng.normal(0, 0.05, (state.particles.size, 3))
        moved = _wrapped(state, state.particles.data.positions + noise)
        moved, found = _read_and_fix(moved)
        assert _pairs(found.data) == _expected(moved)
        state = _commit(moved, found.patch)
        rebuilds += _rebuilt(moved, state)
    assert 0 < rebuilds < 20


def test_strain_and_shear_stay_complete():
    state = _state()
    rng = np.random.default_rng(2)
    lattices = jnp.asarray(LATTICES)
    strain = jnp.eye(3) + 0.01 * jnp.array(
        [[-1.0, 0.3, 0.0], [0.0, -0.5, 0.2], [0.0, 0.0, 0.8]]
    )
    for _ in range(12):
        system = state.particles.data.system
        frac = state.systems[system].cell.frame.to_fractional(
            state.particles.data.positions
        )
        lattices = lattices @ strain
        positions = jnp.einsum("ni,nij->nj", frac, lattices[system.raw_indices])
        noise = rng.normal(0, 0.02, positions.shape)
        moved = _wrapped(state, positions + noise, lattices)
        moved, found = _read_and_fix(moved)
        assert _pairs(found.data) == _expected(moved)
        state = _commit(moved, found.patch)


def test_no_rebuild_within_the_margin():
    state = _state()
    nudged = _wrapped(state, state.particles.data.positions + 0.2 * LIST_SKIN / 2)
    nudged, found = _read_and_fix(nudged)
    after = _commit(nudged, found.patch)
    assert not _rebuilt(nudged, after)
    moved = _wrapped(after, after.particles.data.positions + LIST_SKIN)
    moved, found = _read_and_fix(moved)
    assert _rebuilt(moved, _commit(moved, found.patch))


def test_each_system_has_its_own_skin():
    """Each system lists the pairs within its own cutoff and skin, and the list is
    rebuilt once one system has used up its skin."""
    state = _state(skin=(0.2, 0.5, 0.8))
    _assert_rows_built_at_reference(state.skin_list, state)
    system = state.particles.data.system.raw_indices[:, None]
    # Moving the atoms of one system by 0.15 Å uses up 0.3 Å of its skin.
    for moved_system, rebuilt in ((2, False), (1, False), (0, True)):
        shift = jnp.where(system == moved_system, jnp.array([0.15, 0.0, 0.0]), 0.0)
        moved = _wrapped(state, state.particles.data.positions + shift)
        moved, found = _read_and_fix(moved)
        assert _pairs(found.data) == _expected(moved)
        after = _commit(moved, found.patch)
        assert _rebuilt(moved, after) is rebuilt
    _assert_rows_built_at_reference(after.skin_list, after)


def test_changed_systems_rebuild():
    """A read rebuilds once a particle leaves its system, as a deletion does."""
    state = _state()
    system = state.particles.data.system
    removed = bind(state, lambda s: s.particles.data.system).set(
        system.with_raw_indices(system.raw_indices.at[0].set(len(COUNTS)))
    )
    assert not state.skin_list.is_complete(removed.particles, removed.systems)
    removed, found = _read_and_fix(removed)
    assert _pairs(found.data) == _expected(removed)
    stored = _commit(removed, found.patch).skin_list
    assert stored.is_complete(removed.particles, removed.systems)


def test_nan_positions_are_not_complete():
    state = _state()
    positions = state.particles.data.positions.at[0].set(jnp.nan)
    nan = bind(state, lambda s: s.particles.data.positions).set(positions)
    assert not state.skin_list.is_complete(nan.particles, nan.systems)


def test_an_earlier_failure_flags_a_stale_read():
    """After an assertion failed earlier in the same traced function, a read keeps
    the stale list, and the computation stays flagged."""
    state = _state()
    stale = _wrapped(state, state.particles.data.positions + 3 * LIST_SKIN)

    def read_after_failure(s: _State) -> WithPatch[Edges[Literal[2]], Patch[_State]]:
        runtime_assert(jnp.asarray(False), "An earlier assertion failed.")
        return _read(s)

    result = jax.jit(as_result_function(read_after_failure))(stale)
    with pytest.raises(AssertionError, match="An earlier assertion failed"):
        result.raise_assertion()
    assert not _rebuilt(stale, _commit(stale, result.value.patch))


def test_zero_patch_without_acceptance_keeps_the_list():
    """Monte Carlo applies a zero-filled patch, accepted by no system, for no-op
    moves."""
    state = _state()
    moved = _wrapped(state, state.particles.data.positions + LIST_SKIN)
    moved, found = _read_and_fix(moved)
    zeros = jax.tree.map(jnp.zeros_like, found.patch)
    kept = _commit(moved, zeros, False, False, False)
    for a, b in zip(jax.tree.leaves(kept.skin_list), jax.tree.leaves(moved.skin_list)):
        np.testing.assert_array_equal(a, b)


def test_patch_stores_the_list_read_unless_no_system_accepts():
    state = _state()
    moved = _wrapped(state, state.particles.data.positions + LIST_SKIN)
    moved, found = _read_and_fix(moved)
    assert not _rebuilt(moved, _commit(moved, found.patch, False, False, False))
    stored = _commit(moved, found.patch, False, True, False)
    np.testing.assert_array_equal(
        stored.skin_list.reference.positions, moved.particles.data.positions
    )
    _assert_rows_built_at_reference(stored.skin_list, stored)


def test_an_overflowed_rebuild_is_not_stored():
    """Applied without a rollback, the patch of a read whose rebuild outgrew its
    capacities keeps the stored list."""
    state = _state()
    pulled = _wrapped(state, _pulled(state))
    result = _checked_read(pulled, None)
    assert result.failed_assertions
    stored = _commit(pulled, result.value.patch)
    assert not _rebuilt(pulled, stored)
    _assert_rows_built_at_reference(stored.skin_list, state)


def test_capacity_growth_pads_and_the_next_read_rebuilds():
    state = _state()
    pulled = _wrapped(state, _pulled(state))
    result = _checked_read(pulled, None)
    assert result.failed_assertions
    grown = result.fix_or_raise(pulled)
    # Padded: the pairs of the old reference, in more slots.
    slots = len(pulled.skin_list.rows.neighbors)
    assert len(grown.skin_list.rows.neighbors) > slots
    _assert_rows_built_at_reference(grown.skin_list, pulled)
    grown, found = _read_and_fix(grown)
    assert _pairs(found.data) == _expected(grown)
    stored = _commit(grown, found.patch)
    assert _rebuilt(grown, stored)
    _assert_rows_built_at_reference(stored.skin_list, stored)


def test_fixes_and_patches_apply_to_a_nested_state():
    state = _state()
    outer = _Outer(_wrapped(state, _pulled(state)))
    inner = lens(lambda o: o.inner, cls=_Outer)

    def read(o: _Outer) -> WithPatch[Edges[Literal[2]], Patch[_Outer]]:
        reader = SkinNeighborList.from_state(o, inner, _cutoffs(o.inner))
        return reader(o.inner.particles, o.inner.systems)

    checked_read = jax.jit(as_result_function(read))
    while (result := checked_read(outer)).failed_assertions:
        outer = result.fix_or_raise(outer)
    grown = outer.inner.skin_list.params.avg_edges
    assert grown > state.skin_list.params.avg_edges
    assert _pairs(result.value.data) == _expected(outer.inner)
    accept = Table(outer.inner.systems.keys, jnp.ones(len(COUNTS), bool))
    committed = result.value.patch(outer, accept)
    _assert_rows_built_at_reference(committed.inner.skin_list, committed.inner)


@pytest.mark.parametrize(
    "periodic", [(True, True, True), (True, True, False)], ids=["bulk", "slab"]
)
def test_wraps_since_build_under_shear(periodic: tuple[bool, bool, bool]):
    """Wraps applied since the build come back after a large shear; open axes have
    none."""
    particles, systems, cutoffs = _tables()
    system = particles.data.system
    reference = SkinReference(
        particles.data.positions, systems.data.cell, system.raw_indices
    )
    built = systems[system].cell.frame.to_fractional(particles.data.positions)
    rng = np.random.default_rng(3)
    wraps = rng.integers(-2, 3, built.shape) * np.asarray(periodic)
    fractional = built + rng.uniform(-0.3, 0.3, built.shape) - wraps
    # Lower triangular like the lattices, which TriclinicFrame keeps.
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
def test_builders_agree(builder: NeighborListFactory[IsUniversalNeighborlistParams]):
    """Each builder creates the list and rebuilds it in a read."""
    state = _state(builder)
    _assert_rows_built_at_reference(state.skin_list, state)
    moved = _wrapped(state, state.particles.data.positions + LIST_SKIN)
    moved, found = _read_and_fix(moved)
    assert _pairs(found.data) == _expected(moved)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64], ids=["int32", "int64"])
def test_index_dtypes(dtype: DTypeLike):
    """Rows are int32 and int8; edges follow the particles' index dtype."""
    particles, systems, cutoffs = cast_indices(_tables(), dtype)
    skin_list = SkinList.create(particles, systems, cutoffs, _skin(LIST_SKIN))
    rows = skin_list.rows
    for x in (rows.order, rows.rank, rows.offsets, rows.centers, rows.neighbors):
        assert x.dtype == jnp.int32
    assert rows.shifts.dtype == jnp.int8
    params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoffs
    )
    _, found = _read_and_fix(_State(particles, systems, params, skin_list))
    assert found.data.indices.raw_indices.dtype == dtype
    assert found.data.shifts.dtype == particles.data.positions.dtype


def test_queried_keys_match_direct_builds():
    state = _state()
    queried = Index.integer(
        np.arange(0, sum(COUNTS), 7), n=sum(COUNTS), label=ParticleId
    )
    _, found = _read_and_fix(state, queried)
    assert _pairs(found.data) == _expected(state, queried)


def test_readers_return_the_pairs_within_their_own_cutoffs():
    """Two readers share one list, one below its cutoffs. Each returns the pairs
    within its own cutoffs, from the stored list and from a rebuilt one."""
    cutoffs = (LIST_CUTOFF - 1.0, LIST_CUTOFF)

    @jax.jit
    @as_result_function
    def read(s: _State) -> tuple[WithPatch[Edges[Literal[2]], Patch[_State]], ...]:
        return tuple(_read(s, cutoff=cutoff) for cutoff in cutoffs)

    state = _state()
    for shift, rebuilt in ((0.0, False), (LIST_SKIN, True)):
        moved = _wrapped(state, state.particles.data.positions + shift)
        while (result := read(moved)).failed_assertions:
            moved = result.fix_or_raise(moved)
        for found, cutoff in zip(result.value, cutoffs):
            assert _pairs(found.data) == _expected(moved, cutoff=cutoff)
        state = _commit(moved, ComposedPatch(tuple(r.patch for r in result.value)))
        assert _rebuilt(moved, state) is rebuilt


def test_cutoffs_above_the_list_raise():
    state = _state()
    above = LIST_CUTOFF + 2 * LIST_SKIN
    result = jax.jit(as_result_function(lambda s: _read(s, cutoff=above)))(state)
    with pytest.raises(AssertionError, match="exceed the skin list"):
        result.raise_assertion()


def test_cells_narrower_than_the_skin_raise():
    particles, systems, cutoffs = _tables()
    with pytest.raises(AssertionError, match="wider than the skin"):
        SkinList.create(particles, systems, cutoffs, _skin(6.0))
