# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""FixAtoms constraints in relaxation: loading, pinning, convergence, ASE parity."""

import os
import tempfile

import ase.build
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.calculators.lj import LennardJones as AseLJ
from ase.constraints import FixAtoms, FixBondLength, FixCartesian, FixScaled
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
from kups.core.lens import identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import make_optimizer
from kups.relaxation.convergence import fmax_per_system
from test.units import EV, FORCE, A

jax.config.update("jax_enable_x64", True)

SIGMA, EPS, RC = 3.405, 0.010326, 4.5
FIXED = [0, 1, 2, 5]


def _atoms() -> ase.Atoms:
    atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=True) * (2, 2, 2)
    atoms.rattle(0.2, seed=1)
    atoms.set_cell(
        np.array([[10.6, 0.0, 0.0], [0.3, 10.6, 0.0], [0.2, 0.25, 10.6]]),
        scale_atoms=True,
    )
    atoms.set_constraint(FixAtoms(indices=FIXED))
    return atoms


# ---------------------------------------------------------------- loading


def test_fix_atoms_is_read() -> None:
    expected = np.zeros(32, bool)
    expected[FIXED] = True
    np.testing.assert_array_equal(fixed_from_ase(_atoms()), expected)
    particles, _ = relax_state_from_ase(_atoms())
    np.testing.assert_array_equal(particles.data.fixed, expected)


def test_no_constraint_fixes_nothing() -> None:
    atoms = _atoms()
    atoms.set_constraint()
    assert not fixed_from_ase(atoms).any()


@pytest.mark.parametrize("columns", [1, 3])
def test_extxyz_move_mask(columns: int) -> None:
    atoms = _atoms()
    if columns == 3:  # written as one FixCartesian per atom, all-False for free atoms
        atoms.set_constraint(
            [FixCartesian(i, mask=[i in FIXED] * 3) for i in range(len(atoms))]
        )
    with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False) as f:
        path = f.name
    ase.io.write(path, atoms, format="extxyz")
    try:
        assert np.flatnonzero(fixed_from_ase(path)).tolist() == FIXED
    finally:
        os.unlink(path)


def test_poscar_selective_dynamics() -> None:
    atoms = _atoms()
    atoms.set_constraint(FixScaled(FIXED, mask=[True] * 3))
    with tempfile.NamedTemporaryFile(suffix="POSCAR", delete=False) as f:
        path = f.name
    ase.io.write(path, atoms, format="vasp")
    try:
        assert np.flatnonzero(fixed_from_ase(path)).tolist() == FIXED
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    "constraint",
    [
        FixCartesian(0, mask=[True, False, False]),
        FixScaled([0], mask=[False, False, True]),
        FixBondLength(0, 1),
    ],
)
def test_unsupported_constraints_raise(constraint) -> None:
    atoms = _atoms()
    atoms.set_constraint(constraint)
    with pytest.raises(ValueError, match="only whole atoms"):
        fixed_from_ase(atoms)


# ---------------------------------------------------------------- pinning

OPTIMIZERS = {
    "fire": [
        {"transform": "scale", "step_size": -1.0},
        {"transform": "scale_by_fire", "dt_start": 0.1},
    ],
    "fire2": [
        {"transform": "scale", "step_size": -1.0},
        {"transform": "scale_by_fire2", "dt_start": 0.1},
    ],
    "lbfgs": [
        {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
        {"transform": "max_step_size", "max_step_size": 0.2},
        {"transform": "scale", "step_size": -1.0},
    ],
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
    # Weight decay moves an atom with zero gradient: only the pinned set holds it.
    "adamw": [
        {"transform": "scale_by_adam"},
        {"transform": "add_decayed_weights", "weight_decay": 0.1},
        {"transform": "scale", "step_size": -0.01},
    ],
}


def _relax(atoms: ase.Atoms, optimizer, gradient, steps: int):
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
    state = RelaxState(
        particles, systems, nlp, opt_init(particles, systems), jnp.array([0])
    )
    initial = state
    step = jax.jit(lambda k, s: propagator(k, s))
    key = jax.random.key(0)
    for _ in range(steps):
        state = step(key, state)
    return initial, state


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_fixed_atoms_do_not_move(name: str) -> None:
    initial, final = _relax(_atoms(), OPTIMIZERS[name], POSITIONS_ONLY, steps=5)
    r0 = np.asarray(initial.particles.data.positions.to(A))
    r1 = np.asarray(final.particles.data.positions.to(A))
    np.testing.assert_array_equal(r1[FIXED], r0[FIXED])  # bit-identical
    free = np.setdiff1d(np.arange(len(r0)), FIXED)
    assert np.abs(r1[free] - r0[free]).max() > 1e-4


def test_fixed_atoms_are_left_out_of_fmax() -> None:
    _, final = _relax(_atoms(), OPTIMIZERS["lbfgs"], POSITIONS_ONLY, steps=1)
    g = np.asarray(final.particles.data.position_gradients.to(FORCE))
    assert not g[FIXED].any()
    assert np.abs(g).max() > 0


def test_fixed_atoms_ride_the_cell() -> None:
    initial, final = _relax(_atoms(), OPTIMIZERS["lbfgs"], FRECHET_FILTER, steps=5)

    def fractional(s):
        r = np.asarray(s.particles.data.positions.to(A))
        h = np.asarray(s.systems.data.cell.vectors[0].to(A))
        return r @ np.linalg.inv(h), h

    s0, h0 = fractional(initial)
    s1, h1 = fractional(final)
    assert np.abs(h1 - h0).max() > 1e-4
    np.testing.assert_allclose(s1[FIXED], s0[FIXED], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- ASE parity


@pytest.mark.parametrize("n_steps", [1, 3])
def test_matches_ase_fix_atoms_with_frechet_filter(n_steps: int) -> None:
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
    _, final = _relax(atoms, OPTIMIZERS["lbfgs"], FRECHET_FILTER, steps=n_steps)
    np.testing.assert_allclose(
        np.asarray(final.systems.data.cell.vectors[0].to(A)),
        np.asarray(reference.cell),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(final.particles.data.positions.to(A)),
        reference.get_positions(),
        rtol=0,
        atol=1e-6,
    )


def test_fmax_matches_ase_constrained_forces() -> None:
    atoms = _atoms()
    reference = atoms.copy()
    reference.calc = AseLJ(sigma=SIGMA, epsilon=EPS, rc=RC)
    # A zero-length step fills the gradient cache at the input geometry.
    zero_step = [{"transform": "scale", "step_size": 0.0}]
    _, stepped = _relax(atoms, zero_step, POSITIONS_ONLY, steps=1)
    fmax = fmax_per_system(
        relax_gradients(stepped),
        relax_index_prefix(stepped.particles, stepped.systems),
        include_cell=False,
    ).data[0]
    ase_fmax = np.linalg.norm(reference.get_forces(), axis=1).max()
    np.testing.assert_allclose(float(fmax.to(FORCE)), ase_fmax, rtol=1e-6)
