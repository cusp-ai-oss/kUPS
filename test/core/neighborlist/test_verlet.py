# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Verlet-skin neighbor list (kups.core.neighborlist.verlet).

Covers the pure pieces: the deformation-aware ``skin_margin`` bound (motion
threshold, compression, expansion, pure shear, triclinic boundary crossing,
non-periodic axes), the single-image clamp on the build radius, the
completeness of the refine-based reuse path against a fresh dense build,
``VerletSkinState.seed``, and the ``VerletSkinPropagator`` rebuild/backstop
(fix flips the rebuild flag; an unabsorbable step escalates). MD integration
lives in ``test/application/md/test_verlet.py``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data import Table
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    Edges,
    SkinMargin,
    SkinReference,
    UniversalNeighborlistParameters,
    VerletSkinPropagator,
    VerletSkinState,
    build_skin_edges,
    effective_build_radii,
    estimate_skin_params,
    skin_margin,
    skin_neighborlist,
)
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass

from ._builders import SamplePoints, SampleSystems, make_lh, make_systems

CUTOFF, SKIN = 3.5, 1.0


@dataclass
class SkinState:
    """Minimal state satisfying ``IsVerletState``."""

    particles: Table[ParticleId, SamplePoints]
    systems: Table[SystemId, SampleSystems]
    neighborlist_params: UniversalNeighborlistParameters
    verlet_skin: VerletSkinState | None = None


def _system_cell(lvecs: Array, periodic=(True, True, True)) -> Cell:
    """A ``(1,)``-batched cell."""
    frame = TriclinicFrame.from_matrix(jnp.asarray(lvecs)[None])
    return Cell.from_pbc(frame, periodic)


def _identity_step(key: Array, state: SkinState) -> SkinState:
    del key
    return state


def _make_state(lvecs: Array, positions: Array, cutoff: float) -> SkinState:
    n = positions.shape[0]
    particles = make_lh(positions, jnp.zeros(n, dtype=int))
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.asarray(lvecs)[None]))
    systems, cutoffs = make_systems(cell, jnp.array([cutoff]))
    counts = Table(systems.keys, jnp.array([n]))
    return SkinState(
        particles=particles,
        systems=systems,
        neighborlist_params=UniversalNeighborlistParameters.estimate(
            counts, systems, cutoffs
        ),
    )


def _seeded(state: SkinState, cutoffs, skin: float, rebuild: bool) -> SkinState:
    """Populate the Verlet-skin group so the cond-based propagator can trace."""
    group = VerletSkinState.seed(state.particles, state.systems, cutoffs, skin)
    group = dataclasses.replace(group, should_rebuild=jnp.array(rebuild))
    return dataclasses.replace(state, verlet_skin=group)


def _edges_fixing(nl_factory, state):
    """Build edges, growing lens-backed capacities via the assertion fixes."""
    result = None
    for _ in range(3):
        nl = nl_factory(state)
        result = jax.jit(as_result_function(nl))(state.particles, state.systems)
        if not result.failed_assertions:
            break
        state = result.fix_or_raise(state)
    assert result is not None
    result.raise_assertion()
    return result.value, state


def _skin_edges_fixing(state: SkinState, cutoffs, skin: float):
    result = None
    for _ in range(3):
        result = jax.jit(as_result_function(build_skin_edges))(state, cutoffs, skin)
        if not result.failed_assertions:
            break
        state = result.fix_or_raise(state)
    assert result is not None
    result.raise_assertion()
    return result.value, state


def _edge_set(edges: Edges, state: SkinState) -> set:
    """Directed (i, j, quantized difference vector) tuples of the valid rows."""
    n = state.particles.size
    idx = np.asarray(edges.indices.indices)
    diff = np.asarray(edges.difference_vectors(state.particles, state.systems))[:, 0, :]
    valid = (idx[:, 0] < n) & (idx[:, 1] < n)
    return {
        (int(i), int(j), tuple(np.round(d / 1e-6).astype(np.int64)))
        for (i, j), d in zip(idx[valid], diff[valid])
    }


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
    reference = SkinReference(reference_positions, cell_ref)
    return skin_margin(particles, systems, reference, cutoffs, skin)


def _margin_fires(*args, **kwargs) -> bool:
    return bool(jnp.min(_margins(*args, **kwargs).data.headroom) < 0.0)


class TestSkinMargin:
    def test_motion_threshold_at_half_skin(self):
        """One atom moving just under/over ``skin/2`` sits on the margin boundary."""
        cell = _system_cell(20.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        for factor, expected in [(0.99, False), (1.01, True)]:
            moved = pos.at[0, 0].add(factor * SKIN / 2)
            assert _margin_fires(moved, pos, cell, cell) is expected

    def test_compression_consumes_margin(self):
        """Isotropic compression consumes ``r_build * (1 - sigma_min)``."""
        ref = _system_cell(20.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        for factor, expected in [(0.99, False), (1.01, True)]:
            f = 1.0 - factor * SKIN / (CUTOFF + SKIN)
            now = _system_cell(20.0 * f * jnp.eye(3))
            # Atoms ride the cell (pure affine motion): zero residual, all
            # margin consumption comes from the deformation term.
            assert _margin_fires(pos * f, pos, now, ref) is expected

    def test_expansion_consumes_no_margin(self):
        """Expansion moves non-listed pairs farther away; sigma_min > 1 is free."""
        ref = _system_cell(20.0 * jnp.eye(3))
        now = _system_cell(30.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        assert _margin_fires(pos * 1.5, pos, now, ref) is False

    def test_pure_shear_consumes_margin(self):
        """An off-diagonal cell move must consume margin (sigma_min < 1) even
        though it is invisible to any per-axis length ratio."""
        ref = _system_cell(20.0 * jnp.eye(3))
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        for gamma, expected in [(1.0, False), (12.0, True)]:
            lvecs = jnp.array([[20.0, 0.0, 0.0], [gamma, 20.0, 0.0], [0.0, 0.0, 20.0]])
            now = _system_cell(lvecs)
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
        cell = _system_cell(lvecs)
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
        frame = TriclinicFrame.from_matrix(jnp.repeat(20.0 * jnp.eye(3)[None], 2, 0))
        cell = Cell.from_pbc(frame, (True, True, True))
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
        cell = _system_cell(10.0 * jnp.eye(3), periodic=(False, False, False))
        pos_ref = jnp.array([[2.0, 5.0, 5.0]])
        pos_new = pos_ref + jnp.array([[6.0, 0.0, 0.0]])
        # 2 * 6 >= 10 triggers; the misread 2 * 4 would not.
        assert _margin_fires(pos_new, pos_ref, cell, cell, cutoff=1.0, skin=10.0)


class TestEffectiveBuildRadii:
    def test_unclamped_below_the_limit(self):
        """A radius inside the single-image regime passes through untouched."""
        cell = _system_cell(8.0 * jnp.eye(3))
        radii = effective_build_radii(jnp.array([2.0]), 1.5, cell)
        assert float(radii[0]) == pytest.approx(3.5)  # 2 + 1.5 < 8 / 2

    def test_clamped_to_half_perpendicular_length(self):
        cell = _system_cell(8.0 * jnp.eye(3))
        radii = effective_build_radii(jnp.array([3.0]), 2.0, cell)
        assert float(radii[0]) == pytest.approx(4.0)  # min(3+2, 8/2)

    def test_unclamped_in_vacuum(self):
        cell = _system_cell(8.0 * jnp.eye(3), periodic=(False, False, False))
        radii = effective_build_radii(jnp.array([3.0]), 2.0, cell)
        assert float(radii[0]) == pytest.approx(5.0)


class TestReuseReproducesDense:
    @pytest.mark.parametrize(
        "lvecs",
        [
            pytest.param(15.0 * jnp.eye(3), id="cubic"),
            pytest.param(
                jnp.array([[12.0, 0.0, 0.0], [6.0, 10.0, 0.0], [1.0, 2.0, 11.0]]),
                id="triclinic",
            ),
        ],
    )
    def test_refined_skin_list_matches_fresh_dense(self, lvecs):
        """Refining the stored skin list to the true cutoff yields exactly the
        edge set of a fresh dense build at the true cutoff (no dropped pairs,
        including across sheared boundaries)."""
        frac = jax.random.uniform(jax.random.key(0), (64, 3))
        state = _make_state(lvecs, frac @ lvecs, CUTOFF)
        cutoffs = Table(state.systems.keys, jnp.array([CUTOFF]))
        state = _seeded(state, cutoffs, SKIN, rebuild=True)

        # Replace the untrusted eager seed content with a traced (assertion
        # covered) build, growing capacities as needed.
        skin_edges, state = _skin_edges_fixing(state, cutoffs, SKIN)
        group = state.verlet_skin
        assert group is not None
        state = dataclasses.replace(
            state, verlet_skin=dataclasses.replace(group, edges=skin_edges)
        )

        reuse, state = _edges_fixing(lambda s: skin_neighborlist(s, cutoffs), state)
        fresh, state = _edges_fixing(
            lambda s: DenseNearestNeighborList.from_state(s, cutoffs), state
        )
        reuse_set = _edge_set(reuse, state)
        assert len(reuse_set) > 0
        assert reuse_set == _edge_set(fresh, state)


class TestVerletSkinStateSeed:
    def test_contract(self):
        """The seed schedules an assertion-covered first rebuild and deep-copies
        the reference fields (aliased buffers break state donation)."""
        lvecs = 15.0 * jnp.eye(3)
        frac = jax.random.uniform(jax.random.key(5), (32, 3))
        state = _make_state(lvecs, frac @ lvecs, CUTOFF)
        cutoffs = Table(state.systems.keys, jnp.array([CUTOFF]))
        group = VerletSkinState.seed(state.particles, state.systems, cutoffs, SKIN)
        assert bool(group.should_rebuild)
        assert group.headroom.shape == (1,)
        assert group.reference.positions is not state.particles.data.positions

    def test_overgrown_eager_seed_is_refit_to_the_static_params(self):
        """An eager build silently outgrows undersized static capacities (its
        capacity assertions cannot surface outside a trace); the seed must refit
        the stored edges to the params-implied shape, or the first traced step's
        ``lax.cond`` dies with mismatched branch shapes."""
        lvecs = 15.0 * jnp.eye(3)
        frac = jax.random.uniform(jax.random.key(5), (64, 3))
        state = _make_state(lvecs, frac @ lvecs, CUTOFF)
        cutoffs = Table(state.systems.keys, jnp.array([CUTOFF]))
        counts = Table(state.systems.keys, jnp.array([64]))
        params = dataclasses.replace(
            estimate_skin_params(counts, state.systems, cutoffs, SKIN), avg_edges=1
        )
        group = VerletSkinState.seed(
            state.particles, state.systems, cutoffs, SKIN, params=params
        )
        state = dataclasses.replace(state, verlet_skin=group)
        target = jax.eval_shape(lambda s: build_skin_edges(s, cutoffs, SKIN), state)
        assert group.edges.indices.indices.shape == target.indices.indices.shape


def _move_atom(state: SkinState, index: int, by: float) -> SkinState:
    moved = state.particles.data.positions.at[index, 0].add(by)
    return dataclasses.replace(
        state,
        particles=dataclasses.replace(
            state.particles,
            data=dataclasses.replace(state.particles.data, positions=moved),
        ),
    )


def _with_headroom(state: SkinState, headroom) -> SkinState:
    group = state.verlet_skin
    assert group is not None
    return dataclasses.replace(
        state,
        verlet_skin=dataclasses.replace(group, headroom=jnp.array(headroom)),
    )


class TestVerletSkinPropagatorRebuild:
    def test_small_cell_clamps_instead_of_failing(self):
        """(cutoff + skin) beyond the single-image limit shrinks the effective
        skin rather than failing: the rebuild succeeds with the clamped budget."""
        lvecs = 8.0 * jnp.eye(3)
        frac = jax.random.uniform(jax.random.key(1), (16, 3))
        state = _make_state(lvecs, frac @ lvecs, 3.0)  # 3 + 2 > 8 / 2
        cutoffs = Table(state.systems.keys, jnp.array([3.0]))
        state = _seeded(state, cutoffs, 2.0, rebuild=True)
        prop = VerletSkinPropagator(_identity_step, cutoffs, 2.0)
        result = jax.jit(as_result_function(prop))(jax.random.key(2), state)
        result.raise_assertion()
        group = result.value.verlet_skin
        assert group is not None
        assert float(group.headroom[0]) == pytest.approx(1.0)  # 4.0 - 3.0
        assert not bool(group.should_rebuild)

    def test_cutoff_beyond_limit_has_no_fix(self):
        """A cutoff that itself needs more than one periodic image cannot be
        repaired by any skin choice and must fail the rebuild assertion."""
        lvecs = 8.0 * jnp.eye(3)
        frac = jax.random.uniform(jax.random.key(1), (16, 3))
        state = _make_state(lvecs, frac @ lvecs, 4.5)  # 4.5 > 8 / 2
        cutoffs = Table(state.systems.keys, jnp.array([4.5]))
        state = _seeded(state, cutoffs, 1.0, rebuild=True)
        prop = VerletSkinPropagator(_identity_step, cutoffs, 1.0)
        result = jax.jit(as_result_function(prop))(jax.random.key(2), state)
        assert result.failed_assertions
        with pytest.raises(AssertionError, match="verlet_skin = 0"):
            result.fix_or_raise(state)

    def test_flag_off_leaves_the_reference_untouched(self):
        """With ``should_rebuild`` unset the stored list and its reference
        survive the step (only the measured headroom is refreshed)."""
        lvecs = 15.0 * jnp.eye(3)
        frac = jax.random.uniform(jax.random.key(3), (16, 3))
        positions = frac @ lvecs
        state = _make_state(lvecs, positions, CUTOFF)
        cutoffs = Table(state.systems.keys, jnp.array([CUTOFF]))
        state = _seeded(state, cutoffs, SKIN, rebuild=False)
        state = _move_atom(state, index=0, by=0.1)  # well within the margin
        prop = VerletSkinPropagator(_identity_step, cutoffs, SKIN)
        result = jax.jit(as_result_function(prop))(jax.random.key(4), state)
        result.raise_assertion()
        group = result.value.verlet_skin
        assert group is not None
        # No rebuild: the reference still holds the seed-time positions.
        np.testing.assert_array_equal(group.reference.positions, positions)
        assert not bool(group.should_rebuild)


def _make_two_system_state(lvecs0, lvecs1, pos0, pos1, cutoff: float):
    """Two systems batched into one state (indices [0]*n0 + [1]*n1)."""
    positions = jnp.concatenate([pos0, pos1])
    mask = jnp.concatenate(
        [jnp.zeros(len(pos0), dtype=int), jnp.ones(len(pos1), dtype=int)]
    )
    particles = make_lh(positions, mask)
    frame = TriclinicFrame.from_matrix(jnp.stack([lvecs0, lvecs1]))
    systems, cutoffs = make_systems(PeriodicCell(frame), jnp.full((2,), cutoff))
    counts = Table(systems.keys, jnp.array([len(pos0), len(pos1)]))
    state = SkinState(
        particles=particles,
        systems=systems,
        neighborlist_params=UniversalNeighborlistParameters.estimate(
            counts, systems, cutoffs
        ),
    )
    return state, Table(systems.keys, jnp.full((2,), cutoff))


class TestVerletSkinPropagatorBackstop:
    def _state(self, moved_by: float, headroom_prev: float):
        lvecs = 20.0 * jnp.eye(3)
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        state = _make_state(lvecs, pos, CUTOFF)
        cutoffs = Table(state.systems.keys, jnp.array([CUTOFF]))
        state = _seeded(state, cutoffs, SKIN, rebuild=False)
        state = _with_headroom(_move_atom(state, 0, moved_by), [headroom_prev])
        return state, cutoffs

    def test_exhausted_margin_fix_flips_flag(self):
        """An overshoot within one budget of the recorded headroom is repairable:
        the fix requests a rebuild for the block replay."""
        state, cutoffs = self._state(moved_by=0.55, headroom_prev=0.5)  # consumed 1.1
        prop = VerletSkinPropagator(_identity_step, cutoffs, SKIN)
        result = jax.jit(as_result_function(prop))(jax.random.key(0), state)
        assert result.failed_assertions
        fixed = result.fix_or_raise(state)
        assert fixed.verlet_skin is not None
        assert bool(fixed.verlet_skin.should_rebuild)

    def test_unabsorbable_step_escalates(self):
        """A single step consuming more than the whole budget cannot be fixed by
        any rebuild schedule and must raise with the configuration hint."""
        state, cutoffs = self._state(moved_by=1.0, headroom_prev=1.0)  # consumed 2.0
        prop = VerletSkinPropagator(_identity_step, cutoffs, SKIN)
        result = jax.jit(as_result_function(prop))(jax.random.key(0), state)
        assert result.failed_assertions
        with pytest.raises(ValueError, match="cannot absorb"):
            result.fix_or_raise(state)

    def test_healthy_step_extrapolates_the_flag(self):
        """Well inside the margin nothing fails; the flag anticipates one more
        step of consumption."""
        state, cutoffs = self._state(moved_by=0.2, headroom_prev=1.0)  # consumed 0.4
        prop = VerletSkinPropagator(_identity_step, cutoffs, SKIN)
        result = jax.jit(as_result_function(prop))(jax.random.key(0), state)
        result.raise_assertion()
        group = result.value.verlet_skin
        assert group is not None
        # headroom 0.6, single step 0.4: another such step fits, no flag.
        assert not bool(group.should_rebuild)
        assert float(group.headroom[0]) == pytest.approx(0.6)

    def _two_system_state(self, move0: float, move1: float, headroom_prev):
        lvecs = 20.0 * jnp.eye(3)
        pos = jnp.array([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
        state, cutoffs = _make_two_system_state(lvecs, lvecs, pos, pos, CUTOFF)
        state = _seeded(state, cutoffs, SKIN, rebuild=False)
        state = _move_atom(_move_atom(state, 0, move0), 2, move1)
        return _with_headroom(state, headroom_prev), cutoffs

    def test_per_system_headroom_only_exhausted_system_fails(self):
        """One exhausted system must trip the backstop even when the other is
        fresh (pins the all/any reductions), and per-system headroom must be
        recorded per system (a hot system must not charge the cold one)."""
        state, cutoffs = self._two_system_state(
            move0=0.0, move1=0.55, headroom_prev=[1.0, 0.5]
        )
        prop = VerletSkinPropagator(_identity_step, cutoffs, SKIN)
        result = jax.jit(as_result_function(prop))(jax.random.key(0), state)
        assert result.failed_assertions  # system 1 is over; system 0 must not mask it
        fixed = result.fix_or_raise(state)
        assert fixed.verlet_skin is not None
        assert bool(fixed.verlet_skin.should_rebuild)

    def test_per_system_headroom_is_not_coupled(self):
        """A hot-but-healthy system leaves the cold system's headroom untouched."""
        state, cutoffs = self._two_system_state(
            move0=0.0, move1=0.4, headroom_prev=[1.0, 1.0]
        )
        prop = VerletSkinPropagator(_identity_step, cutoffs, SKIN)
        result = jax.jit(as_result_function(prop))(jax.random.key(0), state)
        result.raise_assertion()
        group = result.value.verlet_skin
        assert group is not None
        headroom = group.headroom
        assert float(headroom[0]) == pytest.approx(1.0)  # cold system: full budget
        assert float(headroom[1]) == pytest.approx(0.2)  # hot system: 1.0 - 0.8
