# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones relaxation entry point."""

import tempfile

import ase.build
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.relaxation.analysis import analyze_relax_file
from kups.application.relaxation.data import RelaxParameters, RelaxRunConfig
from kups.application.relaxation.simulation import make_relax_propagator
from kups.application.simulations.relax_lj import (
    Config,
    LjConfig,
    RelaxLjState,
    init_state,
    run,
)
from kups.core.lens import identity_lens
from kups.potential.classical.lennard_jones import make_lennard_jones_from_state
from kups.relaxation.config import make_optimizer


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
            out_file=out_file, max_steps=5, seed=42, force_tolerance=0.5
        ),
        relax=RelaxParameters(
            optimizer=[
                {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
                {"transform": "max_step_size", "max_step_size": 0.2},
                {"transform": "scale", "step_size": -1},
            ],
            optimize_cell=optimize_cell,
        ),
        lj=LjConfig(
            tail_correction=False,
            cutoff=5.0,
            parameters={"Ar": (3.405, 0.010326)},  # (sigma [Å], epsilon [eV])
            mixing_rule="lorentz_berthelot",
        ),
        inp_file=inp_file,
    )


class TestRun:
    """A short relaxation writes an HDF5 file the analyzer can read."""

    @pytest.fixture(scope="class")
    def out_file(self) -> str:
        out = _tmp_h5()
        run(_config(out, _ar_cif(rattle=0.1)))
        return out

    def test_analyzer_reads_back_physical_outputs(self, out_file):
        results = analyze_relax_file(out_file)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(jnp.asarray(result.final_energy)).item()
        assert jnp.isfinite(jnp.asarray(result.final_max_force)).item()
        assert result.n_steps >= 1


def _build_propagator(optimize_cell: bool):
    """Build an LJ relaxation propagator and its initial state."""
    config = _config(_tmp_h5(), _ar_cif(rattle=0.1), optimize_cell=optimize_cell)
    state_lens = identity_lens(RelaxLjState)
    optimizer = make_optimizer(config.relax.optimizer)
    potential = make_lennard_jones_from_state(
        state_lens, compute_position_and_cell_gradients=True
    )
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, optimizer, optimize_cell
    )
    return propagator, init_state(config, opt_init)


class TestCellRelaxation:
    """``optimize_cell`` drives the lattice vectors via the total cell gradient."""

    def test_cell_moves_only_when_optimizing_cell(self):
        # optimize_cell=True: the cell relaxes (total lattice gradient is non-zero
        # for the off-equilibrium fcc-Ar cell).
        prop, state = _build_propagator(optimize_cell=True)
        stepped = prop(jax.random.key(0), state)
        assert not jnp.allclose(
            stepped.systems.data.cell.vectors, state.systems.data.cell.vectors
        )

        # optimize_cell=False: positions move but the cell is held fixed.
        prop0, state0 = _build_propagator(optimize_cell=False)
        stepped0 = prop0(jax.random.key(0), state0)
        npt.assert_array_equal(
            stepped0.systems.data.cell.vectors, state0.systems.data.cell.vectors
        )
        assert not jnp.allclose(
            stepped0.particles.data.positions, state0.particles.data.positions
        )
