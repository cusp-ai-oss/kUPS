# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Verlet-skin margin bound (kups.core.neighborlist.verlet).

Covers the pure pieces: the deformation-aware ``skin_margin`` bound (motion
threshold, compression, expansion, pure shear, triclinic boundary crossing,
non-periodic axes, per-system accounting) and the single-image clamp on the
build radius (``effective_build_radii``).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from kups.core.cell import Cell, TriclinicFrame
from kups.core.neighborlist import (
    effective_build_radii,
    skin_margin,
)


def _system_cell(lvecs: Array, periodic=(True, True, True)) -> Cell:
    """A ``(1,)``-batched cell."""
    frame = TriclinicFrame.from_matrix(jnp.asarray(lvecs)[None])
    return Cell.from_pbc(frame, periodic)


CUTOFF, SKIN = 3.5, 1.0


def _margin_fires(
    positions: Array,
    reference_positions: Array,
    cell_now: Cell,
    cell_ref: Cell,
    cutoff: float = CUTOFF,
    skin: float = SKIN,
) -> bool:
    n = positions.shape[0]
    consumed, budget = skin_margin(
        positions,
        reference_positions,
        cell_now,
        cell_ref,
        jnp.zeros(n, dtype=int),
        jnp.array([cutoff]),
        skin,
    )
    return bool(jnp.min(budget - consumed) < 0.0)


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
        consumed, budget = skin_margin(
            moved, pos, cell, cell, system, jnp.full((2,), CUTOFF), SKIN
        )
        assert float(consumed[0]) == pytest.approx(0.0)
        assert float(consumed[1]) == pytest.approx(1.2)
        assert bool(consumed[0] <= budget[0]) and bool(consumed[1] > budget[1])

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
    def test_clamped_to_half_perpendicular_length(self):
        cell = _system_cell(8.0 * jnp.eye(3))
        radii = effective_build_radii(jnp.array([3.0]), 2.0, cell)
        assert float(radii[0]) == pytest.approx(4.0)  # min(3+2, 8/2)

    def test_unclamped_in_vacuum(self):
        cell = _system_cell(8.0 * jnp.eye(3), periodic=(False, False, False))
        radii = effective_build_radii(jnp.array([3.0]), 2.0, cell)
        assert float(radii[0]) == pytest.approx(5.0)
