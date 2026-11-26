# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones MD entry point."""

import tempfile

import ase.build
import ase.io
import jax.numpy as jnp
import numpy as np
import pytest
from ase.calculators.lj import LennardJones

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import MdParameters, MdRunConfig
from kups.application.simulations import md
from kups.application.simulations.md import Config, run
from kups.application.simulations.potentials import LjPotentialConfig
from kups.core import units
from kups.core.units import catalog as u
from kups.core.units import testing
from test.units import AMU, EV, FORCE, A


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
            temperature="100 K",
            time_step="2 fs",
            friction_coefficient="1 1/fs",
            thermostat_time_constant="100 fs",
            target_pressure="1 Pa",
            pressure_coupling_time="1.0e10 fs",
            compressibility="4.5e-5 1/Pa",
            minimum_scale_factor=1.0,
            integrator="baoab_langevin",
            initialize_momenta=True,
        ),
        potential=LjPotentialConfig(
            cutoff="5 angstrom",
            parameters={"Ar": ("3.405 angstrom", "0.010326 eV")},
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(inp_file,),
    )


class TestRun:
    """A short NVT MD run writes an HDF5 file the analyzer can read."""

    @pytest.fixture(scope="class")
    def out_file(self) -> str:
        out = _tmp_h5()
        run(_config(out, _ar_cif()))
        return out

    def test_analyzer_reads_back_physical_outputs(self, out_file: str) -> None:
        results = analyze_md_file(out_file, n_blocks=2)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(result.total_energy.mean.to(EV)).all().item()
        kelvin = result.temperature.mean.to(u.K)
        assert jnp.isfinite(kelvin).all().item()
        assert (kelvin >= 0.0).all().item()


def test_replicas_get_independent_momenta(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same input file passed twice must not start with identical momenta."""
    captured = {}
    monkeypatch.setattr(
        md, "run_md", lambda key, propagator, state, config: captured.update(s=state)
    )
    inp = _ar_cif()
    run(_config(_tmp_h5(), inp).model_copy(update={"inp_files": (inp, inp)}))
    first, second = np.split(np.asarray(captured["s"].particles.data.momenta.value), 2)
    assert not np.allclose(first, second)


def test_first_step_uses_initial_forces(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first Verlet step of two rattled systems kicks with the forces of the
    input positions, as ASE's Lennard-Jones computes them."""
    captured = {}
    run_md = md.run_md
    monkeypatch.setattr(md, "run_md", lambda *args: captured.update(s=run_md(*args)))
    inp_files = (_ar_cif(rattle=0.1), _ar_cif(rattle=0.2))
    config = _config(_tmp_h5(), inp_files[0])
    config = config.model_copy(
        update={
            "run": config.run.model_copy(update={"num_steps": 1}),
            "md": config.md.model_copy(
                update={"integrator": "verlet", "initialize_momenta": False}
            ),
            "inp_files": inp_files,
        }
    )
    run(config)
    state = captured["s"]
    time_step = config.md.time_step
    positions, momenta = [], []
    for path in inp_files:
        atoms = ase.io.read(path)
        atoms.calc = LennardJones(sigma=3.405, epsilon=0.010326, rc=5.0)  # _config
        forces, masses = atoms.get_forces() * FORCE, atoms.get_masses()[:, None] * AMU
        atoms.positions += (forces / masses * time_step**2 / 2).to(A)
        positions.append(atoms.positions)
        momenta.append((forces + atoms.get_forces() * FORCE) * time_step / 2)
    cells = state.systems.map_data(lambda s: s.cell)[state.particles.data.system]
    delta = state.particles.data.positions - np.concatenate(positions) * A
    assert np.abs(cells.wrap(delta).to(A)).max() < 1e-10
    testing.assert_allclose(
        state.particles.data.momenta,
        units.concatenate(momenta),
        FORCE * u.fs,
        rtol=0,
        atol=1e-10,
    )
