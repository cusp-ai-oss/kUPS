# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones MD entry point."""

import tempfile

import ase.build
import jax.numpy as jnp
import pytest

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import MdParameters, MdRunConfig
from kups.application.simulations.md_lj import Config, LjConfig, run

from ..clear_cache import clear_cache  # noqa: F401


def _ar_cif(rattle: float = 0.0) -> str:
    """Write a small fcc-argon supercell (32 atoms, ~10.6 Å cube) as a P1 CIF.

    Labels are kept uniform (``Ar``) so they match the LJ parameter table;
    ASE's CIF writer would otherwise uniquify them to ``Ar1``, ``Ar2``, ...
    """
    atoms = ase.build.bulk("Ar", "fcc", a=5.3) * (2, 2, 2)
    if rattle:
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


def _config(out_file: str, inp_file: str) -> Config:
    return Config(
        run=MdRunConfig(out_file=out_file, num_steps=5, num_warmup_steps=0, seed=42),
        md=MdParameters(
            temperature=100.0,
            time_step=2.0,
            friction_coefficient=1.0,
            thermostat_time_constant=100.0,
            target_pressure=1.0,
            pressure_coupling_time=1.0e10,
            compressibility=4.5e-5,
            minimum_scale_factor=1.0,
            integrator="baoab_langevin",
            initialize_momenta=True,
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
    """A short NVT MD run writes an HDF5 file the analyzer can read."""

    @pytest.fixture(scope="class")
    def out_file(self) -> str:
        out = _tmp_h5()
        run(_config(out, _ar_cif()))
        return out

    def test_analyzer_reads_back_physical_outputs(self, out_file):
        results = analyze_md_file(out_file, n_blocks=2)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(result.total_energy.mean).all().item()
        assert jnp.isfinite(result.temperature.mean).all().item()
        assert (result.temperature.mean >= 0.0).all().item()
