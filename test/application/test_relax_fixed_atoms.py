# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Relaxation with atoms held in place by ASE constraints."""

import os
from pathlib import Path

import ase.build
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.calculators.lj import LennardJones as AseLJ
from ase.constraints import (
    FixAtoms,
    FixBondLength,
    FixCartesian,
    FixConstraint,
    FixScaled,
)
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.data import (
    RelaxState,
    relax_gradients,
    relax_index_prefix,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import make_relax_propagator
from kups.application.utils.particles import fixed_from_ase
from kups.core import units
from kups.core.geometry import Geometry, PositionsAndCell
from kups.core.lens import Lens, identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import TransformationConfig, make_optimizer
from kups.relaxation.convergence import fmax_per_system
from test.units import EV, FORCE, A

jax.config.update("jax_enable_x64", True)

SIGMA, EPS, RC = 3.405, 0.010326, 4.5
FIXED = [0, 1, 2, 5]
LBFGS_CONFIG: TransformationConfig = [
    {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
    {"transform": "max_step_size", "max_step_size": 0.2},
    {"transform": "scale", "step_size": -1.0},
]
OPTIMIZERS: dict[str, TransformationConfig] = {
    "fire": [
        {"transform": "scale", "step_size": -1.0},
        {"transform": "scale_by_fire", "dt_start": 0.1},
    ],
    "fire2": [
        {"transform": "scale", "step_size": -1.0},
        {"transform": "scale_by_fire2", "dt_start": 0.1},
    ],
    "lbfgs": LBFGS_CONFIG,
    "lbfgs_more_thuente": [
        {"transform": "scale_by_ase_lbfgs", "memory_size": 10},
        {"transform": "scale", "step_size": -1.0},
        {"transform": "scale_by_more_thuente_linesearch"},
    ],
    "lbfgs_backtracking": [
        {"transform": "scale_by_ase_lbfgs", "memory_size": 10},
        {"transform": "scale", "step_size": -1.0},
        {"transform": "scale_by_backtracking_linesearch"},
    ],
    "adam": [
        {"transform": "scale_by_adam"},
        {"transform": "scale", "step_size": -0.01},
    ],
}


def _atoms() -> ase.Atoms:
    """A rattled fcc-Ar supercell in a lower-triangular cell, four atoms fixed."""
    atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=True) * (2, 2, 2)
    atoms.rattle(0.2, seed=1)
    atoms.set_cell(
        np.array([[10.6, 0.0, 0.0], [0.3, 10.6, 0.0], [0.2, 0.25, 10.6]]),
        scale_atoms=True,
    )
    atoms.set_constraint(FixAtoms(indices=FIXED))
    return atoms


def _relax(
    atoms: ase.Atoms,
    optimizer: TransformationConfig,
    gradient: Lens[Geometry, PositionsAndCell[units.Length]],
    steps: int,
) -> tuple[RelaxState, RelaxState]:
    """Relax ``atoms`` with Lennard-Jones for ``steps`` steps; returns (first, last)."""
    lj = LennardJonesParameters.from_dict(
        cutoff=RC * A,
        parameters={"Ar": (SIGMA * A, EPS * EV)},
        mixing_rule="lorentz_berthelot",
    )
    state_lens = identity_lens(RelaxState)
    potential = make_lennard_jones_from_state(
        state_lens, parameters=lj, gradient=gradient
    )
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, make_optimizer(optimizer), gradient
    )
    particles, systems = relax_state_from_ase(atoms)
    nlp = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, lj.cutoff
    )
    first = RelaxState(
        particles, systems, nlp, opt_init(particles, systems), jnp.array([0])
    )
    step = jax.jit(lambda k, s: propagator(k, s))
    key = jax.random.key(0)
    last = first
    for _ in range(steps):
        last = step(key, last)
    return first, last


class TestLoading:
    def test_fix_atoms(self) -> None:
        expected = np.isin(np.arange(32), FIXED)
        np.testing.assert_array_equal(fixed_from_ase(_atoms()), expected)
        particles, _ = relax_state_from_ase(_atoms())
        np.testing.assert_array_equal(particles.data.fixed, expected)

    def test_no_constraint_fixes_nothing(self) -> None:
        atoms = _atoms()
        atoms.set_constraint()
        assert not fixed_from_ase(atoms).any()

    @pytest.mark.parametrize("columns", [1, 3])
    def test_extxyz_move_mask(self, tmp_path: Path, columns: int) -> None:
        # Written by hand: older ASE extxyz writers do not emit move_mask.
        atoms = _atoms()
        lattice = " ".join(f"{x:.6f}" for x in atoms.cell.array.ravel())
        properties = f"species:S:1:pos:R:3:move_mask:L:{columns}"
        lines = [str(len(atoms)), f'Lattice="{lattice}" Properties={properties}']
        for i, (x, y, z) in enumerate(atoms.positions):
            movable = " ".join(["F" if i in FIXED else "T"] * columns)
            lines.append(f"Ar {x:.6f} {y:.6f} {z:.6f} {movable}")
        path = tmp_path / "slab.extxyz"
        path.write_text("\n".join(lines) + "\n")
        assert np.flatnonzero(fixed_from_ase(path)).tolist() == FIXED

    def test_poscar_selective_dynamics(self, tmp_path: Path) -> None:
        atoms = _atoms()
        atoms.set_constraint(FixScaled(FIXED, mask=[True] * 3))
        path = tmp_path / "POSCAR"
        ase.io.write(path, atoms, format="vasp")
        assert np.flatnonzero(fixed_from_ase(path)).tolist() == FIXED

    @pytest.mark.parametrize(
        "constraint",
        [
            FixCartesian(0, mask=[True, False, False]),
            FixScaled([0], mask=[False, False, True]),
            FixBondLength(0, 1),
        ],
    )
    def test_other_constraints_raise(self, constraint: FixConstraint) -> None:
        atoms = _atoms()
        atoms.set_constraint(constraint)
        with pytest.raises(ValueError, match="only whole atoms"):
            fixed_from_ase(atoms)


class TestRelaxation:
    @pytest.mark.parametrize("name", OPTIMIZERS)
    def test_fixed_atoms_do_not_move(self, name: str) -> None:
        first, last = _relax(_atoms(), OPTIMIZERS[name], POSITIONS_ONLY, steps=5)
        r0 = np.asarray(first.particles.data.positions.to(A))
        r1 = np.asarray(last.particles.data.positions.to(A))
        np.testing.assert_array_equal(r1[FIXED], r0[FIXED])
        free = np.setdiff1d(np.arange(len(r0)), FIXED)
        assert np.abs(r1[free] - r0[free]).max() > 1e-4

    def test_fixed_atoms_ride_the_cell(self) -> None:
        first, last = _relax(_atoms(), LBFGS_CONFIG, FRECHET_FILTER, steps=5)

        def fractional(s: RelaxState) -> tuple[np.ndarray, np.ndarray]:
            r = np.asarray(s.particles.data.positions.to(A))
            h = np.asarray(s.systems.data.cell.vectors[0].to(A))
            return r @ np.linalg.inv(h), h

        s0, h0 = fractional(first)
        s1, h1 = fractional(last)
        assert np.abs(h1 - h0).max() > 1e-4
        np.testing.assert_allclose(s1[FIXED], s0[FIXED], rtol=0, atol=1e-12)

    def test_cache_keeps_the_force_and_convergence_ignores_it(self) -> None:
        _, last = _relax(_atoms(), LBFGS_CONFIG, POSITIONS_ONLY, steps=1)
        cached = np.asarray(last.particles.data.position_gradients.to(FORCE))
        seen = np.asarray(relax_gradients(last).positions.data.to(FORCE))
        assert np.abs(cached[FIXED]).max() > 0
        assert not seen[FIXED].any()
        free = np.setdiff1d(np.arange(len(cached)), FIXED)
        np.testing.assert_array_equal(seen[free], cached[free])


class TestAseParity:
    @pytest.mark.parametrize("n_steps", [1, 3])
    def test_frechet_cell_filter_with_fix_atoms(self, n_steps: int) -> None:
        atoms = _atoms()
        reference = atoms.copy()
        reference.calc = AseLJ(sigma=SIGMA, epsilon=EPS, rc=RC)
        LBFGS(
            FrechetCellFilter(reference),  # pyrefly: ignore[bad-argument-type]
            logfile=os.devnull,
            alpha=70,
            maxstep=0.2,
            damping=1.0,
            memory=10,
        ).run(steps=n_steps, fmax=1e-30)
        _, last = _relax(atoms, LBFGS_CONFIG, FRECHET_FILTER, steps=n_steps)
        np.testing.assert_allclose(
            np.asarray(last.systems.data.cell.vectors[0].to(A)),
            np.asarray(reference.cell),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(last.particles.data.positions.to(A)),
            reference.get_positions(),
            rtol=0,
            atol=1e-6,
        )

    def test_fmax_is_the_constrained_fmax(self) -> None:
        atoms = _atoms()
        reference = atoms.copy()
        reference.calc = AseLJ(sigma=SIGMA, epsilon=EPS, rc=RC)
        # A zero-length step fills the gradient cache at the input geometry.
        zero_step: TransformationConfig = [{"transform": "scale", "step_size": 0.0}]
        _, last = _relax(atoms, zero_step, POSITIONS_ONLY, steps=1)
        fmax = fmax_per_system(
            relax_gradients(last),
            relax_index_prefix(last.particles, last.systems),
            include_cell=False,
        ).data[0]
        ase_fmax = np.linalg.norm(reference.get_forces(), axis=1).max()
        np.testing.assert_allclose(float(fmax.to(FORCE)), ase_fmax, rtol=1e-6)
