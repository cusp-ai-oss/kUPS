# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones MD entry point."""

import dataclasses
import tempfile

import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import MdParameters, MdRunConfig, MdState
from kups.application.simulations import md
from kups.application.simulations.md import Config, run
from kups.application.simulations.potentials import LjPotentialConfig
from kups.application.utils.propagate import make_cycle_function
from kups.core.neighborlist import SkinList
from kups.core.propagator import LoopPropagator, Propagator, propagate_and_fix
from kups.core.units import catalog as u
from test.units import EV


def _ar_cif(rattle: float = 0.0, cubic: bool = False) -> str:
    """Write a small fcc-argon supercell (32 atoms, ~10.6 Å cube) as a P1 CIF.

    Labels are kept uniform (``Ar``) so they match the LJ parameter table;
    ASE's CIF writer would otherwise uniquify them to ``Ar1``, ``Ar2``, ...
    """
    atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=cubic) * (2, 2, 2)
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


def _propagator_and_state(
    monkeypatch: pytest.MonkeyPatch, skin: float = 1.0
) -> tuple[Propagator[MdState], MdState]:
    """What ``run`` hands to ``run_md`` for hot, rattled argon under NVE."""
    captured = {}
    monkeypatch.setattr(
        md,
        "run_md",
        lambda key, propagator, state, config: captured.update(p=propagator, s=state),
    )
    config = _config(_tmp_h5(), _ar_cif(rattle=0.1, cubic=True))
    hot = {
        "integrator": "verlet",
        "temperature": 300.0 * u.K,
        "skin": skin * u.angstrom,
    }
    short = {"cutoff": 4.0 * u.angstrom}
    run(
        config.model_copy(
            update={
                "md": config.md.model_copy(update=hot),
                "potential": config.potential.model_copy(update=short),
            }
        )
    )
    return captured["p"], captured["s"]


def test_skin_does_not_change_the_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    """A list kept over many steps gives the positions of one built at every step."""
    finals = []
    for skin in (0.0, 1.0):
        propagator, state = _propagator_and_state(monkeypatch, skin)
        built_at = np.asarray(state.skin_list.reference.positions.to(u.angstrom))
        cycle = make_cycle_function(LoopPropagator(propagator, 50))
        for key in jax.random.split(jax.random.key(0), 4):
            state = propagate_and_fix(cycle, key, state)
        finals.append(np.asarray(state.particles.data.positions.to(u.angstrom)))
        reference = np.asarray(state.skin_list.reference.positions.to(u.angstrom))
        assert np.abs(reference - built_at).max() > 0, "the list was never built again"
    assert np.abs(finals[1] - reference).max() > 0, "the list was built at every step"
    np.testing.assert_allclose(finals[1], finals[0], rtol=1e-9, atol=1e-9)


def test_step_repairs_a_skin_list_too_small_to_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild that outgrows its capacities fails the step, which is then redone."""
    propagator, state = _propagator_and_state(monkeypatch)
    cycle = make_cycle_function(propagator)
    key = jax.random.key(0)
    # A list of the same particles three times as far apart holds few pairs.
    dilute = SkinList.create(
        state.particles.map_data(
            lambda p: dataclasses.replace(p, positions=3.0 * p.positions)
        ),
        state.systems.map_data(lambda s: dataclasses.replace(s, cell=s.cell * 3.0)),
        state.skin_list.cutoffs,
        state.skin_list.skin,
    )
    assert dilute.parameters.avg_edges < state.skin_list.parameters.avg_edges
    small = dataclasses.replace(jax.tree.map(jnp.copy, state), skin_list=dilute)
    failed = cycle(key, jax.tree.map(jnp.copy, small))
    assert failed.failed_assertions
    np.testing.assert_array_equal(
        failed.value.particles.data.positions.to(u.angstrom),
        state.particles.data.positions.to(u.angstrom),
    )
    expected = propagate_and_fix(cycle, key, jax.tree.map(jnp.copy, state))
    repaired = propagate_and_fix(cycle, key, small)
    assert repaired.skin_list.parameters.avg_edges > dilute.parameters.avg_edges
    np.testing.assert_allclose(
        repaired.particles.data.positions.to(u.angstrom),
        expected.particles.data.positions.to(u.angstrom),
        rtol=1e-12,
    )
