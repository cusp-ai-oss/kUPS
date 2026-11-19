# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones relaxation entry point."""

import dataclasses
import tempfile

import ase.build
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_AND_CELL
from kups.application.relaxation.analysis import analyze_relax_file
from kups.application.relaxation.data import (
    RelaxRunConfig,
    RelaxState,
    relax_gradients,
    relax_index_prefix,
    relax_state_from_ase,
)
from kups.application.relaxation.logging import RelaxStepData
from kups.application.relaxation.simulation import make_relax_propagator, run_relax
from kups.application.simulations.potentials import LjPotentialConfig
from kups.application.simulations.relax import Config, run
from kups.core import units
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.units import catalog as u
from kups.core.units import testing
from kups.observables.stress import stress_via_virial_theorem
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import make_optimizer
from kups.relaxation.convergence import fmax_per_system
from kups.relaxation.optimizer import ChainOptState
from test.units import FORCE, A


def _ar_cif(rattle: float) -> str:
    """Write a rattled fcc-argon supercell as a P1 CIF with uniform ``Ar`` labels.

    The rattle gives the optimizer nonzero forces to act on; uniform labels
    keep them matching the LJ parameter table (ASE's writer would uniquify).
    """
    atoms = ase.build.bulk("Ar", "fcc", a=5.3) * (2, 2, 2)
    atoms.rattle(rattle, seed=1)
    a, b, c, al, be, ga = atoms.cell.cellpar()
    rows = "\n".join(
        f"Ar Ar {x:.6f} {y:.6f} {z:.6f}" for x, y, z in atoms.get_scaled_positions()
    )
    cif = f"""data_ar
_cell_length_a {a:.6f}
_cell_length_b {b:.6f}
_cell_length_c {c:.6f}
_cell_angle_alpha {al:.6f}
_cell_angle_beta {be:.6f}
_cell_angle_gamma {ga:.6f}
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{rows}
"""
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.write(cif)
    f.close()
    return f.name


def _tmp_h5() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    f.close()
    return f.name


def _config(out_file: str, inp_file: str, *, optimize_cell: bool = False) -> Config:
    return Config(
        run=RelaxRunConfig(
            out_file=out_file,
            max_steps=5,
            seed=42,
            force_tolerance="0.5 eV/angstrom",
            optimizer=[
                {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
                {"transform": "max_step_size", "max_step_size": 0.2},
                {"transform": "scale", "step_size": -1},
            ],
            optimize_cell=optimize_cell,
        ),
        potential=LjPotentialConfig(
            cutoff="5 angstrom",
            parameters={"Ar": ("3.405 angstrom", "0.010326 eV")},
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(inp_file,),
    )


class TestRun:
    """A short relaxation writes an HDF5 file the analyzer can read."""

    @pytest.fixture(scope="class")
    def out_file(self) -> str:
        out = _tmp_h5()
        run(_config(out, _ar_cif(rattle=0.1)))
        return out

    def test_analyzer_reads_back_physical_outputs(self, out_file: str):
        results = analyze_relax_file(out_file)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(result.final_energy.to(u.eV)).item()
        assert jnp.isfinite(result.final_max_force.to(FORCE)).item()
        assert result.n_steps >= 1


def _lj(config: Config) -> LennardJonesParameters:
    assert isinstance(config.potential, LjPotentialConfig)
    return LennardJonesParameters.from_dict(
        cutoff=config.potential.cutoff,
        parameters=config.potential.parameters,
        mixing_rule=config.potential.mixing_rule,
    )


def _build_propagator(optimize_cell: bool, optimizer=None):
    """Build an LJ relaxation propagator and its initial state."""
    config = _config(_tmp_h5(), _ar_cif(rattle=0.1), optimize_cell=optimize_cell)
    lj = _lj(config)
    state_lens = identity_lens(RelaxState[ChainOptState])
    potential = make_lennard_jones_from_state(
        state_lens, parameters=lj, gradient=FRECHET_FILTER
    )
    propagator, opt_init = make_relax_propagator(
        state_lens,
        potential,
        make_optimizer(optimizer or config.run.optimizer),
        FRECHET_FILTER,
    )
    particles, systems = relax_state_from_ase(
        config.inp_files[0], fixed_cell=not optimize_cell
    )
    nlp = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, lj.cutoff
    )
    opt_state = opt_init(particles, systems)
    state = RelaxState(particles, systems, nlp, opt_state, jnp.array([0]))
    return propagator, state


def test_nonfinite_gradients_abort_run() -> None:
    _, state = _build_propagator(optimize_cell=False)
    state = (
        bind(state)
        .focus(lambda s: s.particles.data.position_gradients)
        .apply(lambda gradients: gradients * jnp.nan)
    )
    config = _config(_tmp_h5(), "unused").run
    with pytest.raises(AssertionError, match="Non-finite relaxation gradients"):
        run_relax(jax.random.key(0), lambda key, s: s, state, config)


class TestCellRelaxation:
    """``optimize_cell`` drives the lattice vectors via the total cell gradient."""

    @pytest.fixture(scope="class")
    def position_step(self) -> tuple[RelaxState, RelaxState]:
        prop, state = _build_propagator(optimize_cell=False)
        return state, jax.jit(lambda s: prop(jax.random.key(0), s))(state)

    @pytest.fixture(scope="class")
    def cell_step(self) -> tuple[RelaxState, RelaxState]:
        prop, state = _build_propagator(optimize_cell=True)
        return state, jax.jit(lambda s: prop(jax.random.key(0), s))(state)

    def test_cell_moves_only_when_optimizing_cell(self, cell_step, position_step):
        # optimize_cell=True: the cell relaxes (total lattice gradient is non-zero
        # for the off-equilibrium fcc-Ar cell).
        state, stepped = cell_step
        assert not jnp.allclose(
            stepped.systems.data.cell.vectors.to(A),
            state.systems.data.cell.vectors.to(A),
        )

        # optimize_cell=False: positions move but the cell is held fixed.
        state0, stepped0 = position_step
        npt.assert_array_equal(
            stepped0.systems.data.cell.vectors.to(A),
            state0.systems.data.cell.vectors.to(A),
        )
        assert not jnp.allclose(
            stepped0.particles.data.positions.to(A),
            state0.particles.data.positions.to(A),
        )

    def test_converged_value_matches_fmax(self, position_step):
        # Fixed cell: the reference basis is the cell, so ∂E/∂q == ∂E/∂r and
        # max|∂E/∂u| is ASE's fmax. The cached cell gradient (kept for the
        # stress) is hidden from convergence.
        _, stepped = position_step
        gradients = relax_gradients(stepped)
        prefix = relax_index_prefix(stepped.particles, stepped.systems)
        fmax = units.max(units.norm(stepped.particles.data.forces, axis=-1))
        testing.assert_allclose(
            fmax_per_system(gradients, prefix).data[0], fmax, FORCE, atol=1e-12
        )
        log_matrix = stepped.systems.data.cell_gradients.frame.deformation.log_matrix
        assert jnp.abs(log_matrix.to(FORCE)).max() > 0
        hidden = gradients.cell.data.frame.deformation.log_matrix
        npt.assert_array_equal(hidden.to(FORCE), 0.0)

    def test_cell_gradients_enter_per_system_fmax(self, cell_step):
        _, stepped = cell_step
        # Zero the position gradients so only the cell gradient remains; otherwise
        # they dominate the maximum and hide a dropped cell term.
        stepped = (
            bind(stepped)
            .focus(lambda s: s.particles.data.position_gradients)
            .apply(units.zeros_like)
        )
        prefix = relax_index_prefix(stepped.particles, stepped.systems)
        # The only non-zero cell-gradient leaf is the matrix-log deformation; ASE's
        # rows are its columns.
        deformation = stepped.systems.data.cell_gradients.frame.deformation
        cell_max = jnp.max(jnp.linalg.norm(deformation.log_matrix.to(FORCE), axis=-2))
        assert cell_max > 0
        gradients = relax_gradients(stepped)
        npt.assert_allclose(
            fmax_per_system(gradients, prefix).data.to(FORCE), [cell_max]
        )
        npt.assert_array_equal(
            fmax_per_system(gradients, prefix, include_cell=False).data.to(FORCE), [0.0]
        )

    @pytest.mark.parametrize("optimize_cell", [False, True])
    def test_logged_stress_is_the_virial_stress(self, cell_step, optimize_cell):
        # Start from a moved cell (non-identity deformation) and take a
        # zero-length step, so the cached gradients belong to the logged geometry.
        _, moved = cell_step
        moved = (
            bind(moved)
            .focus(lambda s: s.systems.data.fixed_cell)
            .set(jnp.full(1, not optimize_cell))
        )
        prop, state = _build_propagator(
            optimize_cell, optimizer=[{"transform": "scale", "step_size": 0.0}]
        )
        opt_state = state.opt_state
        state = dataclasses.replace(moved, opt_state=opt_state)
        evaluated = jax.jit(lambda s: prop(jax.random.key(0), s))(state)
        logged = RelaxStepData.from_state(evaluated).stress_tensor

        # Reference: the virial theorem on raw ∂E/∂r and partial ∂E/∂h|_r.
        config = _config(_tmp_h5(), "unused")
        raw = make_lennard_jones_from_state(
            identity_lens(RelaxState),
            parameters=_lj(config),
            gradient=POSITIONS_AND_CELL,
        )
        g = raw(evaluated).data.gradients
        particles = evaluated.particles.map_data(
            lambda p: dataclasses.replace(p, position_gradients=g.positions.data)
        )
        systems = evaluated.systems.map_data(
            lambda s: dataclasses.replace(s, cell_gradients=g.cell.data)
        )
        expected = stress_via_virial_theorem(particles, systems).data
        assert jnp.abs(expected.to(u.GPa)).max() > 1e-3
        testing.assert_allclose(logged, expected, u.GPa, atol=1e-10)
