# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The Lennard-Jones plus Ewald backend of the MD entry point."""

import tempfile

import ase.build
import jax
import numpy as np
import pytest

from kups.application.md.data import MdParameters, MdRunConfig, MdState
from kups.application.simulations import md
from kups.application.simulations.mcmc_rigid import EwaldConfig
from kups.application.simulations.md import Config, run
from kups.application.simulations.potentials import LjEwaldPotentialConfig
from kups.application.utils.propagate import make_cycle_function
from kups.core.propagator import LoopPropagator, Propagator, propagate_and_fix
from kups.core.units import catalog as u
from test.units import EV, A


def _salt_cif(charge: float, seed: int = 0) -> str:
    """Write rattled rock salt (64 atoms, 11.3 Å cube) as a P1 CIF with charges."""
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
    atoms.rattle(0.1, seed=seed)
    rows = "\n".join(
        f"{s} {s} {x:.6f} {y:.6f} {z:.6f} {charge if s == 'Na' else -charge}"
        for s, (x, y, z) in zip(
            atoms.get_chemical_symbols(), atoms.get_scaled_positions(), strict=True
        )
    )
    a = atoms.cell.cellpar()[0]
    cif = f"""data_salt
_cell_length_a {a:.6f}
_cell_length_b {a:.6f}
_cell_length_c {a:.6f}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_charge
{rows}
"""
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.write(cif)
    f.close()
    return f.name


def _config(reciprocal: str, *inp_files: str) -> Config:
    out = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    out.close()
    return Config(
        run=MdRunConfig(out_file=out.name, num_steps=5, num_warmup_steps=0, seed=42),
        md=MdParameters(
            temperature="300 K",
            time_step="1 fs",
            friction_coefficient="1 1/fs",
            thermostat_time_constant="100 fs",
            target_pressure="1 Pa",
            pressure_coupling_time="1.0e10 fs",
            compressibility="4.5e-5 1/Pa",
            minimum_scale_factor=1.0,
            integrator="verlet",
            initialize_momenta=True,
        ),
        potential=LjEwaldPotentialConfig(
            cutoff="5 angstrom",
            parameters={
                "Na": ("2.4 angstrom", "0.005 eV"),
                "Cl": ("4.4 angstrom", "0.004 eV"),
            },
            mixing_rule="lorentz_berthelot",
            ewald=EwaldConfig(real_cutoff="5 angstrom", precision=1e-6),
            reciprocal=reciprocal,
        ),
        inp_files=inp_files,
    )


def _propagator_and_state(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> tuple[Propagator[MdState], MdState]:
    """What ``run`` hands to ``run_md``."""
    captured = {}
    monkeypatch.setattr(
        md,
        "run_md",
        lambda key, propagator, state, config: captured.update(p=propagator, s=state),
    )
    run(config)
    return captured["p"], captured["s"]


def test_mesh_and_direct_sum_give_the_same_steps(monkeypatch: pytest.MonkeyPatch):
    """Two systems step alike with the reciprocal sum on a mesh or over k-vectors."""
    inp_files = (_salt_cif(0.4, seed=0), _salt_cif(0.4, seed=1))
    energies, positions = [], []
    for reciprocal in ("pme", "ewald"):
        propagator, state = _propagator_and_state(
            monkeypatch, _config(reciprocal, *inp_files)
        )
        cycle = make_cycle_function(LoopPropagator(propagator, 3))
        state = propagate_and_fix(cycle, jax.random.key(0), state)
        energies.append(np.asarray(state.systems.data.potential_energy.to(EV)))
        positions.append(np.asarray(state.particles.data.positions.to(A)))
    assert energies[0].shape == (2,)
    assert abs(energies[0][0] - energies[0][1]) > 1e-3
    np.testing.assert_allclose(energies[0], energies[1], rtol=1e-6)
    np.testing.assert_allclose(positions[0], positions[1], atol=1e-7)


def test_run_writes_a_trajectory():
    config = _config("pme", _salt_cif(0.4))
    run(config)
    results = md.analyze_md_file(config.run.out_file, n_blocks=2)
    energy = next(iter(results.values())).total_energy.mean.to(EV)
    assert np.isfinite(energy).all()


def test_input_without_charges_raises():
    with pytest.raises(ValueError, match="needs charges"):
        run(_config("pme", _salt_cif(0.0)))


def test_skin_does_not_change_the_trajectory(monkeypatch: pytest.MonkeyPatch):
    """A list kept over many steps gives the positions of one built at every step."""
    short = {
        "cutoff": 4.0 * u.angstrom,
        "ewald": EwaldConfig(real_cutoff="4 angstrom", precision=1e-4),
    }
    finals = []
    for skin in (0.0, 1.0):
        config = _config("pme", _salt_cif(0.4))
        hot = {"temperature": 1000.0 * u.K, "skin": skin * u.angstrom}
        propagator, state = _propagator_and_state(
            monkeypatch,
            config.model_copy(
                update={
                    "md": config.md.model_copy(update=hot),
                    "potential": config.potential.model_copy(update=short),
                }
            ),
        )
        built_at = np.asarray(state.skin_list.reference.positions.to(A))
        cycle = make_cycle_function(LoopPropagator(propagator, 40))
        for key in jax.random.split(jax.random.key(0), 2):
            state = propagate_and_fix(cycle, key, state)
        finals.append(np.asarray(state.particles.data.positions.to(A)))
        reference = np.asarray(state.skin_list.reference.positions.to(A))
        assert np.abs(reference - built_at).max() > 0, "the list was never built again"
    assert np.abs(finals[1] - reference).max() > 0, "the list was built at every step"
    np.testing.assert_allclose(finals[1], finals[0], rtol=1e-9, atol=1e-9)
