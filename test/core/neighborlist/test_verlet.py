# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet-skin margins, build radii, cache refresh, and the cached neighbor list."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.core.cell import Cell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.lens import bind, lens
from kups.core.capacity import FixedCapacity
from kups.core.neighborlist import (
    SKIN_PARAMS,
    CellListNeighborList,
    DenseNearestNeighborList,
    Edges,
    SkinMargin,
    SkinReference,
    UniversalNeighborlistParameters,
    VerletSkinState,
    effective_build_radii,
    refresh_skin,
    skin_covers,
    skin_margin,
)
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, tree_copy

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
    skin: float = SKIN,
) -> Table[SystemId, SkinMargin]:
    if system is None:
        system = jnp.zeros(positions.shape[0], dtype=int)
    n_sys = int(jnp.max(system)) + 1
    particles = make_lh(positions, system)
    systems, cutoffs = make_systems(cell_now, jnp.full((n_sys,), cutoff))
    built, _ = make_systems(cell_ref, jnp.full((n_sys,), cutoff))
    reference = SkinReference.new(make_lh(reference_positions, system), built)
    radii = Table(systems.keys, effective_build_radii(cutoffs.data, skin, cell_ref))
    return skin_margin(particles, systems, reference, radii, cutoffs)


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

    def test_nonperiodic_axes_are_not_unwrapped(self):
        """In a vacuum cell nothing wraps, so a large real move must trigger even
        when a periodic un-wrap of the bounding box would shrink it below the
        threshold (6 A move, 10 A box: un-wrapping would misread it as 4 A)."""
        cell = _cell(10.0 * jnp.eye(3), periodic=(False, False, False))
        pos_ref = jnp.array([[2.0, 5.0, 5.0]])
        pos_new = pos_ref + jnp.array([[6.0, 0.0, 0.0]])
        # 2 * 6 >= 10 triggers; the misread 2 * 4 would not.
        assert _margin_fires(pos_new, pos_ref, cell, cell, cutoff=1.0, skin=10.0)

    def test_margin_uses_system_labels_with_an_empty_system(self):
        """Particles map onto systems by label, so an empty system is unconstrained."""
        positions = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        particles = (
            bind(make_lh(positions, jnp.array([0, 1])))
            .focus(lambda p: p.data.system)
            .set(Index((SystemId(0), SystemId(2)), jnp.array([0, 1])))
        )
        systems, cutoffs = make_systems(
            _cell(20.0 * jnp.eye(3), n_sys=3), jnp.full(3, CUTOFF)
        )
        reference = SkinReference.new(particles, systems)
        moved = (
            bind(particles)
            .focus(lambda p: p.data.positions)
            .apply(lambda x: x.at[1, 0].add(0.2))
        )
        radii = cutoffs.map_data(lambda c: c + SKIN)
        margin = skin_margin(moved, systems, reference, radii, cutoffs).data
        np.testing.assert_allclose(margin.consumed, [0.0, 0.0, 0.4], atol=1e-12)

    def test_zero_build_radius_leaves_no_budget(self):
        """An unbuilt cache (zero radius) never covers, even without motion."""
        cell = _cell(20.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        particles = make_lh(pos, jnp.zeros(2, dtype=int))
        systems, cutoffs = make_systems(cell, jnp.array([CUTOFF]))
        reference = SkinReference.new(particles, systems)
        radii = Table(systems.keys, jnp.zeros(1))
        margin = skin_margin(particles, systems, reference, radii, cutoffs).data
        assert float(margin.budget[0]) == pytest.approx(-CUTOFF)
        assert bool(margin.headroom[0] < 0.0)

    def test_float32_geometry_keeps_a_float32_margin(self):
        """Float64 cutoffs are compared in the float32 dtype of the build radii."""
        cell = jax.tree.map(lambda x: x.astype(jnp.float32), _cell(20.0 * jnp.eye(3)))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]], dtype=jnp.float32)
        particles = make_lh(pos, jnp.zeros(2, dtype=int))
        systems, cutoffs = make_systems(cell, jnp.array([3.3]))
        reference = SkinReference.new(particles, systems)
        radii = Table(systems.keys, jnp.array([3.3], dtype=jnp.float32))
        margin = skin_margin(particles, systems, reference, radii, cutoffs).data
        assert margin.budget.dtype == jnp.float32
        assert float(margin.budget[0]) == 0.0


class TestEffectiveBuildRadii:
    def test_unclamped_below_the_limit(self):
        """A radius inside the single-image regime passes through untouched."""
        cell = _cell(8.0 * jnp.eye(3))
        radii = effective_build_radii(jnp.array([2.0]), 1.5, cell)
        assert float(radii[0]) == pytest.approx(3.5)  # 2 + 1.5 < 8 / 2

    def test_clamped_to_half_perpendicular_length(self):
        cell = _cell(8.0 * jnp.eye(3))
        radii = effective_build_radii(jnp.array([3.0]), 2.0, cell)
        assert float(radii[0]) == pytest.approx(4.0)  # min(3+2, 8/2)

    def test_unclamped_in_vacuum(self):
        cell = _cell(8.0 * jnp.eye(3), periodic=(False, False, False))
        radii = effective_build_radii(jnp.array([3.0]), 2.0, cell)
        assert float(radii[0]) == pytest.approx(5.0)


PARAMS = UniversalNeighborlistParameters(16, 32, 32, 64)


def _inputs(dtype=jnp.float64):
    positions = jax.random.uniform(jax.random.key(0), (12, 3)) * 10.0
    # make_lh shares one buffer between labels; copy so no leaf aliases another.
    particles = tree_copy(make_lh(positions.astype(dtype), jnp.zeros(12, dtype=int)))
    systems, cutoffs = make_systems(_cell(10.0 * jnp.eye(3)), jnp.array([CUTOFF]))
    return particles, systems, cutoffs


def _built(cache: VerletSkinState) -> VerletSkinState:
    """``cache`` with nonzero radii, as after a build."""
    return bind(cache).focus(lambda c: c.radii.data).apply(lambda r: r + CUTOFF + SKIN)


class TestVerletSkinState:
    def test_new_allocates_an_unbuilt_cache_of_builder_size(self):
        particles, systems, _ = _inputs()
        cache = VerletSkinState.new(particles, systems, PARAMS)
        assert len(cache.edges) == PARAMS.avg_edges * particles.size
        assert not bool(cache.edges.indices.valid_mask.any())
        np.testing.assert_array_equal(cache.radii.data, 0.0)
        np.testing.assert_array_equal(
            cache.reference.particles.data.positions, particles.data.positions
        )

    def test_new_copies_the_reference_so_the_state_can_be_donated(self):
        @dataclass
        class Holder:
            particles: Table
            cache: VerletSkinState

        particles, systems, _ = _inputs()
        holder = Holder(particles, VerletSkinState.new(particles, systems, PARAMS))
        jax.jit(lambda h: h, donate_argnums=0)(holder)

    def test_float32_positions_allocate_the_builder_shift_dtype(self):
        particles, systems, cutoffs = _inputs(jnp.float32)
        builder = DenseNearestNeighborList(
            avg_candidates=FixedCapacity(PARAMS.avg_candidates),
            avg_edges=FixedCapacity(PARAMS.avg_edges),
            avg_image_candidates=FixedCapacity(PARAMS.avg_image_candidates),
            cutoffs=cutoffs,
        )
        built = builder(particles, systems)
        cache = VerletSkinState.new(particles, systems, PARAMS)
        assert cache.edges.shifts.dtype == built.shifts.dtype
        assert cache.edges.shifts.shape == built.shifts.shape

    def test_edge_rows_must_match_the_capacities(self):
        particles, systems, _ = _inputs()
        cache = VerletSkinState.new(particles, systems, PARAMS)
        grown = UniversalNeighborlistParameters(32, 32, 32, 64)
        with pytest.raises(AssertionError):
            VerletSkinState(grown, cache.edges, cache.reference, cache.radii)


class TestSkinParams:
    def test_unchanged_capacities_keep_the_build(self):
        particles, systems, _ = _inputs()
        cache = _built(VerletSkinState.new(particles, systems, PARAMS))
        assert SKIN_PARAMS.set(cache, SKIN_PARAMS.get(cache)) is cache

    def test_grown_capacities_discard_the_build(self):
        particles, systems, _ = _inputs()
        cache = _built(VerletSkinState.new(particles, systems, PARAMS))
        grown = UniversalNeighborlistParameters(32, 64, 64, 64)
        resized = SKIN_PARAMS.set(cache, grown)
        assert SKIN_PARAMS.get(resized) == grown
        assert len(resized.edges) == grown.avg_edges * particles.size
        assert not bool(resized.edges.indices.valid_mask.any())
        np.testing.assert_array_equal(resized.radii.data, 0.0)


@dataclass
class SkinState:
    particles: Table[ParticleId, SamplePoints]
    systems: Table[SystemId, SampleSystems]
    verlet_skin: VerletSkinState


SKIN_CAPACITIES = lens(lambda s: s.verlet_skin, cls=SkinState).nest(SKIN_PARAMS)


def _skin_state(lvecs, positions, params=PARAMS, cutoff=CUTOFF):
    particles = tree_copy(
        make_lh(jnp.asarray(positions), jnp.zeros(len(positions), dtype=int))
    )
    systems, cutoffs = make_systems(_cell(lvecs), jnp.array([cutoff]))
    return SkinState(
        particles, systems, VerletSkinState.new(particles, systems, params)
    ), cutoffs


def _refresh(state, cutoffs, skin=SKIN, factory=DenseNearestNeighborList.new):
    def refreshed(s):
        return refresh_skin(
            s.verlet_skin,
            s.particles,
            s.systems,
            cutoffs,
            skin,
            lambda radii: factory(s, SKIN_CAPACITIES, radii),
        )

    evaluate = jax.jit(as_result_function(refreshed))
    for _ in range(10):
        result = evaluate(state)
        if result.all_assertions_pass:
            return bind(state).focus(lambda s: s.verlet_skin).set(result.value)
        state = result.fix_or_raise(state)
    raise AssertionError("capacity repair did not converge")


def _covers(state, cutoffs):
    return bool(skin_covers(state.verlet_skin, state.particles, state.systems, cutoffs))


def _move(state, positions):
    return bind(state).focus(lambda s: s.particles.data.positions).set(positions)


def _edge_set(edges: Edges, particles, systems) -> set:
    """Directed (i, j, quantized separation) tuples of the in-bounds rows."""
    n = particles.size
    idx = np.asarray(edges.indices.indices)
    diff = np.asarray(edges.difference_vectors(particles, systems))[:, 0, :]
    valid = (idx[:, 0] < n) & (idx[:, 1] < n)
    return {
        (int(i), int(j), tuple(np.round(d / 1e-6).astype(np.int64)))
        for (i, j), d in zip(idx[valid], diff[valid])
    }


def _fresh(state, radius):
    radii = Table(state.systems.keys, jnp.array([radius]))
    params = state.verlet_skin.params
    return DenseNearestNeighborList(
        avg_candidates=FixedCapacity(params.avg_candidates),
        avg_edges=FixedCapacity(params.avg_edges),
        avg_image_candidates=FixedCapacity(params.avg_image_candidates),
        cutoffs=radii,
    )(state.particles, state.systems)


BOX = 15.0 * jnp.eye(3)
POSITIONS = jax.random.uniform(jax.random.key(1), (32, 3)) * 15.0


class TestSkinCovers:
    def test_unbuilt_cache_covers_nothing(self):
        state, cutoffs = _skin_state(BOX, POSITIONS)
        assert not _covers(state, cutoffs)

    def test_covers_within_the_skin_and_not_beyond(self):
        state, cutoffs = _skin_state(BOX, POSITIONS)
        state = _refresh(state, cutoffs)
        assert _covers(state, cutoffs)
        assert _covers(_move(state, POSITIONS.at[0, 0].add(0.49 * SKIN)), cutoffs)
        assert not _covers(_move(state, POSITIONS.at[0, 0].add(0.51 * SKIN)), cutoffs)

    def test_label_change_is_not_covered(self):
        state, cutoffs = _skin_state(BOX, POSITIONS)
        state = _refresh(state, cutoffs)
        merged = (
            bind(state)
            .focus(lambda s: s.particles.data.exclusion.indices)
            .apply(lambda x: x.at[1].set(x[0]))
        )
        assert not _covers(merged, cutoffs)

    def test_longer_cutoff_than_the_build_is_not_covered(self):
        state, cutoffs = _skin_state(BOX, POSITIONS)
        state = _refresh(state, cutoffs)
        assert not _covers(state, cutoffs.map_data(lambda c: c + 2 * SKIN))


class TestRefreshSkin:
    @pytest.mark.parametrize(
        "factory", [DenseNearestNeighborList.new, CellListNeighborList.new]
    )
    def test_build_holds_every_pair_within_the_build_radius(self, factory):
        state, cutoffs = _skin_state(BOX, POSITIONS)
        state = _refresh(state, cutoffs, factory=factory)
        np.testing.assert_allclose(state.verlet_skin.radii.data, [CUTOFF + SKIN])
        assert _edge_set(state.verlet_skin.edges, state.particles, state.systems) == (
            _edge_set(_fresh(state, CUTOFF + SKIN), state.particles, state.systems)
        )

    def test_keeps_a_covering_cache_and_rebuilds_a_stale_one(self):
        state, cutoffs = _skin_state(BOX, POSITIONS)
        state = _refresh(state, cutoffs)
        near = _refresh(_move(state, POSITIONS.at[0, 0].add(0.1)), cutoffs)
        np.testing.assert_array_equal(
            near.verlet_skin.reference.particles.data.positions, POSITIONS
        )
        far_positions = POSITIONS.at[0, 0].add(0.6)
        far = _refresh(_move(state, far_positions), cutoffs)
        np.testing.assert_array_equal(
            far.verlet_skin.reference.particles.data.positions, far_positions
        )

    def test_label_change_rebuilds(self):
        state, cutoffs = _skin_state(BOX, POSITIONS)
        state = _refresh(state, cutoffs)
        merged = (
            bind(state)
            .focus(lambda s: s.particles.data.exclusion.indices)
            .apply(lambda x: x.at[1].set(x[0]))
        )
        rebuilt = _refresh(merged, cutoffs)
        np.testing.assert_array_equal(
            rebuilt.verlet_skin.reference.particles.data.exclusion.indices,
            merged.particles.data.exclusion.indices,
        )

    @pytest.mark.parametrize(
        ("lvecs", "cutoff", "skin"),
        [(BOX, CUTOFF, 0.0), (4.0 * jnp.eye(3), 4.5, SKIN)],
        ids=["zero_skin", "multiple_images"],
    )
    def test_builds_without_budget_are_not_stored(self, lvecs, cutoff, skin):
        state, cutoffs = _skin_state(lvecs, POSITIONS[:8] * 4.0 / 15.0, cutoff=cutoff)
        refreshed = _refresh(state, cutoffs, skin=skin)
        np.testing.assert_array_equal(refreshed.verlet_skin.radii.data, 0.0)
        assert not _covers(refreshed, cutoffs)

    def test_build_radius_is_clamped_to_a_single_image(self):
        lvecs = 8.0 * jnp.eye(3)
        state, cutoffs = _skin_state(lvecs, POSITIONS * 8.0 / 15.0, cutoff=3.5)
        state = _refresh(state, cutoffs, skin=2.0)
        np.testing.assert_allclose(state.verlet_skin.radii.data, [4.0])
        assert _edge_set(state.verlet_skin.edges, state.particles, state.systems) == (
            _edge_set(_fresh(state, 4.0), state.particles, state.systems)
        )

    def test_overflowing_build_is_repaired_and_complete(self):
        tiny = UniversalNeighborlistParameters(1, 1, 1, 1)
        state, cutoffs = _skin_state(BOX, POSITIONS, params=tiny)
        state = _refresh(state, cutoffs)
        assert state.verlet_skin.params.avg_edges > tiny.avg_edges
        assert _edge_set(state.verlet_skin.edges, state.particles, state.systems) == (
            _edge_set(_fresh(state, CUTOFF + SKIN), state.particles, state.systems)
        )

    def test_float32_geometry_keeps_float32_storage(self):
        state, cutoffs = _skin_state(
            BOX.astype(jnp.float32), POSITIONS.astype(jnp.float32)
        )
        state = _refresh(state, cutoffs)
        assert state.verlet_skin.radii.data.dtype == jnp.float32
        assert state.verlet_skin.edges.shifts.dtype == jnp.float32
        assert _covers(state, cutoffs)
