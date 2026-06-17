# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Exact parity with ASE's ``UnitCellFilter`` and ``FrechetCellFilter``.

The relaxation filters reproduce ASE's two cell filters to machine precision:
the deformation-undone atomic forces and the Cauchy stress match the real ASE
filter objects, and the defining cell-gradient transforms match ASE's exactly
(``FrechetCellFilter``'s ``scipy.expm_frechet`` adjoint, ``UnitCellFilter``'s
linear deformation-gradient pullback).
"""

import dataclasses
import functools
import os
from itertools import product
from typing import cast, override

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones as AseLJ
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.geometry import cell_to_cellpar
from ase.optimize import LBFGS
from scipy.linalg import expm_frechet

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_AND_CELL
from kups.application.relaxation.data import relax_state_from_ase
from kups.application.simulations.relax_lj import RelaxLjState
from kups.core.cell import DeformedFrame, LogTriclinicFrame, TriclinicFrame
from kups.core.lens import NestedLens, SimpleLens, identity_lens, lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.utils.math import triangular_3x3_from_tril
from kups.observables.stress import stress_via_virial_theorem
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndSystemIndex,
)
from kups.relaxation.optimizer import chain
from kups.relaxation.propagator import RelaxationPropagator
from kups.relaxation.transforms import MaxStepSize, ScaleByAseLbfgs

jax.config.update("jax_enable_x64", True)

ase_build = pytest.importorskip("ase.build")
SIGMA, EPS, RC = 3.405, 0.010326, 4.5


def _ase_system():
    """A rattled FCC-Ar supercell with a lower-triangular triclinic cell.

    The cell is already lower-triangular so ``relax_state_from_ase`` does not
    rotate it; ASE and kUPS then share a frame and forces compare directly.
    ``rc`` is well below the half-box so the minimum image is unambiguous.
    """
    atoms = ase_build.bulk("Ar", "fcc", a=5.3, cubic=True) * (2, 2, 2)
    atoms.rattle(0.2, seed=1)
    atoms.set_cell(
        np.array([[10.6, 0.0, 0.0], [0.3, 10.6, 0.0], [0.2, 0.25, 10.6]]),
        scale_atoms=True,
    )
    atoms.calc = AseLJ(sigma=SIGMA, epsilon=EPS, rc=RC)
    return atoms


def _kups_state(atoms: Atoms) -> RelaxLjState:
    particles, systems = relax_state_from_ase(atoms)
    lj = LennardJonesParameters.from_dict(
        cutoff=RC, parameters={"Ar": (SIGMA, EPS)}, mixing_rule="lorentz_berthelot"
    )
    nlp = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, lj.cutoff
    )
    return RelaxLjState(particles, systems, nlp, jnp.zeros(()), jnp.array([0]), lj)


def test_atomic_forces_match_ase_cell_filters() -> None:
    """``CELL_FILTER``'s position-DOF gradient equals the deformation-undone atomic
    forces of ASE's real ``UnitCellFilter`` and ``FrechetCellFilter``."""
    atoms = _ase_system()
    n = len(atoms)
    ucf = UnitCellFilter(atoms).get_forces()[:n]
    # get_forces avoids ASE 3.28's get_positions typo (only reached by an optimiser).
    fcf = FrechetCellFilter(atoms).get_forces()[:n]
    state_lens = identity_lens(RelaxLjState)
    out = make_lennard_jones_from_state(state_lens, gradient=FRECHET_FILTER)(
        _kups_state(atoms)
    ).data
    forces = -np.asarray(out.gradients.positions.data)
    assert np.allclose(ucf, forces, atol=1e-10)
    assert np.allclose(fcf, forces, atol=1e-10)


def test_stress_matches_ase():
    """The partial-gradient virial stress equals ASE's (kUPS uses the opposite
    sign convention, ``σ = -1/V·sym[...]``)."""
    atoms = _ase_system()
    state = _kups_state(atoms)
    out = make_lennard_jones_from_state(
        identity_lens(RelaxLjState), gradient=POSITIONS_AND_CELL
    )(state).data
    particles = state.particles.map_data(
        lambda p: dataclasses.replace(
            p, position_gradients=out.gradients.positions.data
        )
    )
    systems = state.systems.map_data(
        lambda s: dataclasses.replace(s, cell_gradients=out.gradients.cell.data)
    )
    sigma = np.asarray(stress_via_virial_theorem(particles, systems).data[0])
    assert np.allclose(atoms.get_stress(voigt=False), -sigma, atol=1e-12)


def test_frechet_transform_matches_expm_frechet() -> None:
    """``FrechetCellFilter``'s defining transform — the ``expm_frechet`` adjoint
    applied to the cell gradient — equals ``DeformedFrame.parameter_gradient``
    (the vjp of ``base @ expm``)."""
    a = jnp.array([0.05, 0.10, -0.03, 0.02, 0.07, 0.04])
    log_deform = np.array(triangular_3x3_from_tril(a))
    cell_grad = np.array(
        [[0.20, -0.07, 0.11], [-0.07, 0.05, 0.15], [0.11, 0.15, -0.20]]
    )
    expected = np.zeros((3, 3))
    for mu, nu in product(range(3), repeat=2):
        direction = np.zeros((3, 3))
        direction[mu, nu] = 1.0
        expected[mu, nu] = np.sum(
            expm_frechet(log_deform, direction, compute_expm=False) * cell_grad
        )
    base = TriclinicFrame.from_matrix(jnp.eye(3)[None])
    frame = DeformedFrame(base, LogTriclinicFrame(a[None], jnp.ones(1)))
    grad = frame.parameter_gradient(jnp.asarray(cell_grad)[None])
    ours = np.asarray(cast(LogTriclinicFrame, grad.deformation).tril[0])
    assert np.allclose(ours, expected[np.tril_indices(3)], atol=1e-9)


def test_unitcell_transform_is_linear():
    """``UnitCellFilter``'s deformation-gradient pullback is linear: a linear
    ``DeformedFrame(base, TriclinicFrame)`` round-trips parameter<->cartesian cell
    gradients exactly (orthonormal Jacobian), unlike the Frechet exp map."""
    base = TriclinicFrame.from_matrix(jnp.eye(3)[None])
    frame = DeformedFrame(
        base, TriclinicFrame(jnp.array([1.05, 0.02, 0.98, -0.01, 0.03, 1.10])[None])
    )
    cell_grad = jnp.array(
        [[0.20, -0.07, 0.11], [-0.07, 0.05, 0.15], [0.11, 0.15, -0.20]]
    )[None]
    roundtrip = frame.vectors_gradient(frame.parameter_gradient(cell_grad))
    assert np.allclose(roundtrip, jnp.tril(cell_grad), atol=1e-12)


ALPHA = 70.0
FMAX = 1e-4


class _FrechetCellFilterFixed(FrechetCellFilter):
    """ASE 3.28's ``FrechetCellFilter`` with the ``get_positions`` typo fixed
    (``self.logm(args=...)`` -> positional), so an optimiser can drive it."""

    @override
    def get_positions(self):
        pos = UnitCellFilter.get_positions(self)
        n = len(self.atoms)
        pos[n:] = np.asarray(self.logm(pos[n:])) * self.exp_cell_factor
        return pos


def _sheared_shear_free_system() -> Atoms:
    """A *sheared* (rhombohedral FCC primitive) cell under purely hydrostatic load.

    The cell is genuinely non-orthogonal but lower-triangular; because the atoms
    sit at ideal FCC sites the shear stress is zero, so the cell deformation stays
    diagonal (no rotation) during relaxation. That is the regime where kUPS's
    lower-triangular ``DeformedFrame`` reproduces ASE's symmetric-strain filters
    exactly (general triclinic shear differs in basis and only matches at the
    minimum).
    """
    atoms = ase_build.bulk("Ar", "fcc", a=5.3) * (4, 4, 4)
    q, r = np.linalg.qr(atoms.cell.array.T)  # lower-triangularise: cell == r.T @ q.T
    lower, rotation = r.T, q.T
    sign = np.where(np.diag(lower) < 0.0, -1.0, 1.0)  # positive-diagonal convention
    lower, rotation = lower * sign[None, :], rotation * sign[:, None]
    atoms.set_positions(atoms.get_positions() @ rotation.T)
    atoms.set_cell(lower, scale_atoms=False)
    atoms.set_cell(atoms.cell.array * 1.02, scale_atoms=True)  # hydrostatic load only
    atoms.calc = AseLJ(sigma=SIGMA, epsilon=EPS, rc=RC)
    return atoms


def _relax_ase(
    ase_filter: type[UnitCellFilter],
    atoms: Atoms,
    *,
    steps: int | None = None,
    fmax: float = FMAX,
    max_step: float = 0.5,
):
    """Relax ``atoms`` with an ASE cell filter; returns (n_steps, cell, positions)."""
    relaxed = atoms.copy()
    relaxed.calc = AseLJ(sigma=SIGMA, epsilon=EPS, rc=RC)
    opt = LBFGS(
        # pyrefly: ignore [bad-argument-type]  # Filter is Atoms-like
        ase_filter(relaxed),
        logfile=os.devnull,
        alpha=ALPHA,
        maxstep=max_step,
        damping=1.0,
        memory=100,
    )
    if steps is not None:
        opt.run(steps=steps, fmax=1e-30)  # fixed step count
    else:
        opt.run(steps=500, fmax=fmax)
    return opt.nsteps, np.array(relaxed.cell), relaxed.get_positions()


@functools.lru_cache(maxsize=None)
def _kups_relaxer(max_step: float | None):
    """Build the state-independent CELL_FILTER + ScaleByAseLbfgs pieces once per
    optimizer config.

    The potential, optimizer, propagator and the jitted ``step`` / fmax functions
    depend only on ``max_step`` (not the atoms), so caching them lets JAX reuse its
    per-shape compilation cache across every ``_relax_kups`` call instead of
    recompiling the (multi-second) traces on each call.
    """
    filter_lens = FRECHET_FILTER
    geometry = SimpleLens[RelaxLjState, Geometry](
        lambda s: Geometry(
            s.particles.map_data(
                lambda p: PositionsAndSystemIndex(p.positions, p.system)
            ),
            s.systems.map_data(lambda sy: sy.cell),
        )
    )
    property_lens = NestedLens(geometry, filter_lens)
    potential = make_lennard_jones_from_state(
        identity_lens(RelaxLjState), gradient=filter_lens
    )
    lbfgs = ScaleByAseLbfgs(memory_size=100, alpha=ALPHA)
    if max_step is None:
        optimizer = chain(lbfgs, optax.scale(-1.0))
    else:
        optimizer = chain(lbfgs, MaxStepSize(max_step_size=max_step), optax.scale(-1.0))
    propagator = RelaxationPropagator(
        potential=potential,
        property=property_lens,
        opt_state=lens(lambda x: x.opt_state),
        optimizer=optimizer,  # pyrefly: ignore [bad-argument-type]  # ChainOptState
    )
    key = jax.random.key(0)
    step = jax.jit(lambda s: propagator(key, s))

    @jax.jit
    def max_dof_gradient(s: RelaxLjState):  # ASE-fmax in the filter's DOFs
        grad = potential(s).data.gradients
        force = jnp.max(jnp.linalg.norm(grad.positions.data, axis=-1))
        # Frame-agnostic max|∂E/∂u_cell|: the only non-zero cell-gradient leaves are
        # the deformation parameters (LogTriclinic tril / MatrixLog log_matrix).
        cell_leaves = jax.tree.leaves(grad.cell)
        cell_dof = jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in cell_leaves]))
        return jnp.maximum(force, cell_dof)

    return optimizer, property_lens, step, max_dof_gradient


def _relax_kups(
    atoms: Atoms,
    *,
    steps: int | None = None,
    fmax: float = FMAX,
    max_step: float | None = None,
):
    """Relax ``atoms`` with CELL_FILTER + ScaleByAseLbfgs; returns (n_steps, cell, pos).

    With ``steps`` it runs a fixed count; otherwise it converges on
    ``max(|dE/dq|, |dE/dA|) < fmax`` -- the ASE-fmax criterion in the filter's
    reference-cartesian / log-deformation DOFs. ``max_step`` adds the matching
    ``MaxStepSize`` clamp (as ASE's ``maxstep``) for stability on stiff systems.
    """
    state = _kups_state(atoms)
    optimizer, property_lens, step, max_dof_gradient = _kups_relaxer(max_step)
    index_prefix = PositionsAndCell(state.particles.data.system, state.systems.index)  # type: ignore[arg-type]  # Index leaves; init only reads the per-system keys
    state = dataclasses.replace(
        state, opt_state=optimizer.init(property_lens.get(state), index_prefix)
    )
    n, limit = 0, steps if steps is not None else 500
    while n < limit:
        if steps is None and float(max_dof_gradient(state)) < fmax:
            break
        state = step(state)
        n += 1
    cell = np.asarray(state.systems.data.cell.vectors[0])
    return n, cell, np.asarray(state.particles.data.positions)


@pytest.mark.parametrize("n_steps", [1, 2])
def test_frechet_filter_matches_ase_per_step(n_steps: int) -> None:
    """kUPS's ``CELL_FILTER`` *is* the exp-map (Frechet) parameterisation, so on a
    shear-free cell it reproduces ASE's ``FrechetCellFilter`` cell and positions
    step-for-step (``ScaleByAseLbfgs`` == ``ase.optimize.LBFGS``)."""
    atoms = _sheared_shear_free_system()
    _, ase_cell, ase_pos = _relax_ase(_FrechetCellFilterFixed, atoms, steps=n_steps)
    _, cell, pos = _relax_kups(atoms, steps=n_steps)
    assert np.allclose(ase_cell, cell, rtol=0, atol=1e-7)
    assert np.allclose(ase_pos, pos, rtol=0, atol=1e-7)


def _lattice(cell: np.ndarray, positions: np.ndarray):
    """Rotation-invariant structure descriptor: (cell parameters, fractional coords).

    Two cells related by a global rotation (e.g. ASE's symmetric strain vs kUPS's
    lower-triangular basis) share these, so comparing them tests "same structure up
    to a rotation".
    """
    fractional = (positions @ np.linalg.inv(cell)) % 1.0
    return cell_to_cellpar(cell), fractional


def _max_frac_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(((a - b + 0.5) % 1.0) - 0.5)))


@pytest.mark.parametrize(
    "ase_filter", [_FrechetCellFilterFixed, UnitCellFilter], ids=["frechet", "unitcell"]
)
def test_converges_in_same_steps_as_ase(ase_filter: type[UnitCellFilter]) -> None:
    """kUPS converges to a tight bound in the *same number of L-BFGS steps* as ASE's
    ``FrechetCellFilter`` and ``UnitCellFilter``, to the same structure (compared up
    to a rotation). Shear-stress-free, so the deformation stays lower-triangular and
    the lower-triangular and symmetric parameterisations coincide step-for-step."""
    atoms = _sheared_shear_free_system()
    n_ase, ase_cell, ase_pos = _relax_ase(ase_filter, atoms, fmax=1e-6)
    n_kups, cell, pos = _relax_kups(atoms, fmax=1e-6)
    ase_par, ase_frac = _lattice(ase_cell, ase_pos)
    kups_par, kups_frac = _lattice(cell, pos)
    assert n_kups == n_ase
    assert np.allclose(ase_par, kups_par, rtol=0, atol=1e-5)
    assert _max_frac_diff(ase_frac, b=kups_frac) < 1e-6


@pytest.mark.parametrize("n_steps", [1, 10])
def test_frechet_matches_ase_per_step_under_shear(n_steps: int) -> None:
    """The full-3x3 matrix-log ``CELL_FILTER`` reproduces ASE's ``FrechetCellFilter``
    step-for-step even under *genuine shear stress* (rattled triclinic cell). kUPS now
    optimises the same full deformation-gradient log as ASE, so the cell vectors and
    positions match *directly* -- not merely up to a global rotation."""
    atoms = _ase_system()  # rattled + triclinic -> nonzero shear stress
    _, ase_cell, ase_pos = _relax_ase(
        _FrechetCellFilterFixed, atoms, steps=n_steps, max_step=0.5
    )
    _, cell, pos = _relax_kups(atoms, steps=n_steps, max_step=0.5)
    assert np.allclose(ase_cell, cell, rtol=0, atol=1e-7)
    assert np.allclose(ase_pos, pos, rtol=0, atol=1e-7)


def test_frechet_matches_ase_at_convergence_under_shear() -> None:
    """Relaxing to convergence under shear, kUPS and ASE ``FrechetCellFilter`` reach
    the *same minimum directly* (the full-matrix basis coincides with ASE's, so no
    rotation gauge remains). Step counts agree closely; they differ by only a few
    steps because the ~1e-10/step round-off accumulated over the long flat tail
    shifts the exact ``fmax`` crossing."""
    atoms = _ase_system()
    n_ase, ase_cell, ase_pos = _relax_ase(
        _FrechetCellFilterFixed, atoms, fmax=1e-5, max_step=0.2
    )
    n_kups, cell, pos = _relax_kups(atoms, fmax=1e-5, max_step=0.2)
    assert abs(n_kups - n_ase) <= 10
    assert np.allclose(ase_cell, cell, rtol=0, atol=1e-4)
    assert np.allclose(ase_pos, pos, rtol=0, atol=1e-4)
