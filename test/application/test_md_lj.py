# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones MD entry point."""

import tempfile
import types

import ase.build
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.calculators.lj import LennardJones

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import MdParameters, MdRunConfig
from kups.application.simulations import md
from kups.application.simulations.md import Config, run
from kups.application.simulations.potentials import LjPotentialConfig
from kups.core.constants import FEMTO_SECOND
from kups.core.data import Index


def _ar_cif(rattle: float = 0.0, repeat: int = 2) -> str:
    """Write ``repeat**3`` primitive fcc-argon cells (one atom each) as a P1 CIF.

    Labels are kept uniform (``Ar``) so they match the LJ parameter table;
    ASE's CIF writer would otherwise uniquify them to ``Ar1``, ``Ar2``, ...
    """
    atoms = ase.build.bulk("Ar", "fcc", a=5.3) * (repeat, repeat, repeat)
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
        potential=LjPotentialConfig(
            cutoff=5.0,
            parameters={"Ar": (3.405, 0.010326)},  # (sigma [Å], epsilon [eV])
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
        assert jnp.isfinite(result.total_energy.mean).all().item()
        assert jnp.isfinite(result.temperature.mean).all().item()
        assert (result.temperature.mean >= 0.0).all().item()


def test_replicas_get_independent_momenta(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same input file passed twice must not start with identical momenta."""
    captured = {}
    monkeypatch.setattr(
        md, "run_md", lambda key, propagator, state, config: captured.update(s=state)
    )
    inp = _ar_cif()
    run(_config(_tmp_h5(), inp).model_copy(update={"inp_files": (inp, inp)}))
    first, second = np.split(np.asarray(captured["s"].particles.data.momenta), 2)
    assert not np.allclose(first, second)


def test_seed_zero_is_not_time_based(monkeypatch: pytest.MonkeyPatch) -> None:
    """``seed: 0`` is a fixed seed and does not read the clock."""
    clock = types.SimpleNamespace(time_ns=lambda: pytest.fail("seed 0 read the clock"))
    monkeypatch.setattr(md, "time", clock)
    monkeypatch.setattr(md, "run_md", lambda key, propagator, state, *_: state)
    config = _config(_tmp_h5(), _ar_cif())
    run(config.model_copy(update={"run": config.run.model_copy(update={"seed": 0})}))


def test_first_step_uses_initial_forces() -> None:
    """The first Verlet step of two rattled systems kicks with the forces of the
    input positions, as ASE's Lennard-Jones computes them."""
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
    state = run(config)
    time_step = config.md.time_step * FEMTO_SECOND
    positions, momenta = [], []
    for path in inp_files:
        atoms = ase.io.read(path)
        atoms.calc = LennardJones(sigma=3.405, epsilon=0.010326, rc=5.0)  # _config
        forces, masses = atoms.get_forces(), atoms.get_masses()[:, None]
        atoms.positions += forces / masses * time_step**2 / 2
        positions.append(atoms.positions)
        momenta.append((forces + atoms.get_forces()) * time_step / 2)
    cells = state.systems.map_data(lambda s: s.cell)[state.particles.data.system]
    delta = state.particles.data.positions - np.concatenate(positions)
    assert np.abs(cells.wrap(delta)).max() < 1e-10
    np.testing.assert_allclose(
        state.particles.data.momenta, np.concatenate(momenta), rtol=0, atol=1e-10
    )


def test_int32_indices_match_default() -> None:
    states = []
    for index_dtype in ("int64", "int32"):
        # fresh input files per run: particles_from_ase caches by path
        config = _config(_tmp_h5(), _ar_cif(rattle=0.1))
        config = config.model_copy(
            update={
                "run": config.run.model_copy(update={"index_dtype": index_dtype}),
                "inp_files": (*config.inp_files, _ar_cif(rattle=0.2)),
            }
        )
        states.append(run(config))
    leaves = jax.tree.leaves(states[1], is_leaf=lambda x: isinstance(x, Index))
    dtypes = {x.raw_indices.dtype for x in leaves if isinstance(x, Index)}
    assert dtypes == {jnp.dtype(jnp.int32)}
    np.testing.assert_allclose(
        states[1].particles.data.positions,
        states[0].particles.data.positions,
        rtol=1e-10,
        atol=1e-10,
    )


def _float_dtypes(state: md.MdState) -> set[jnp.dtype]:
    leaves = jax.tree.leaves((state.particles, state.systems))
    return {x.dtype for x in leaves if jnp.issubdtype(x.dtype, jnp.floating)}


def test_float32_matches_float64() -> None:
    """``run.float_dtype: float32`` keeps the tables in float32 and follows the
    float64 trajectory of two systems within float32 round-off."""
    states = []
    for float_dtype in ("float64", "float32"):
        # fresh input files per run: particles_from_ase caches by path
        config = _config(_tmp_h5(), _ar_cif(rattle=0.1))
        config = config.model_copy(
            update={
                "run": config.run.model_copy(
                    update={"float_dtype": float_dtype, "num_steps": 20}
                ),
                "md": config.md.model_copy(update={"integrator": "verlet"}),
                "inp_files": (*config.inp_files, _ar_cif(rattle=0.2)),
            }
        )
        states.append(run(config))
    state, state32 = states
    assert _float_dtypes(state32) == {jnp.dtype(jnp.float32)}
    # positions are wrapped every step: compare minimum images
    cells = state.systems.map_data(lambda s: s.cell)[state.particles.data.system]
    delta = state32.particles.data.positions - state.particles.data.positions
    assert np.abs(cells.wrap(delta)).max() < 1e-4
    np.testing.assert_allclose(
        state32.systems.data.potential_energy,
        state.systems.data.potential_energy,
        rtol=1e-4,
    )


def test_float32_int32_npt() -> None:
    """BAOAB NPT on a float32 state with int32 indices and float32 pair terms keeps
    every dtype at a cutoff that is not exact in float32 (5.1 Å), and the analysis
    reads its output."""
    config = _config(_tmp_h5(), _ar_cif(rattle=0.1))
    config = config.model_copy(
        update={
            "run": config.run.model_copy(
                update={"float_dtype": "float32", "index_dtype": "int32"}
            ),
            "md": config.md.model_copy(
                update={
                    "integrator": "baoab_npt_langevin",
                    "friction_coefficient": 0.01,
                    "target_pressure": 4.2e7,
                    "pressure_coupling_time": 1000.0,
                    "compressibility": 5.4e-10,
                }
            ),
            "potential": config.potential.model_copy(
                update={"cutoff": 5.1, "kernel_dtype": "float32"}
            ),
        }
    )
    state = run(config)
    assert _float_dtypes(state) == {jnp.dtype(jnp.float32)}
    assert state.particles.data.system.raw_indices.dtype == jnp.int32
    result = next(iter(analyze_md_file(config.run.out_file, n_blocks=2).values()))
    assert np.isfinite(result.pressure.mean)


def test_skin_does_not_change_results() -> None:
    """``md.skin`` changes how often neighbors are listed, not the trajectory."""
    states = []
    for skin in (0.05, 1.5):
        # a fresh input file per run: particles_from_ase caches by path and the
        # MD cycle donates the cached buffers
        config = _config(_tmp_h5(), _ar_cif(rattle=0.1, repeat=4))
        md = config.md.model_copy(update={"skin": skin, "integrator": "verlet"})
        states.append(run(config.model_copy(update={"md": md})))
    small, large = (np.asarray(s.skin_list.reference.positions) for s in states)
    assert not np.array_equal(small, large)  # the small skin was listed again
    np.testing.assert_allclose(
        states[1].particles.data.positions,
        states[0].particles.data.positions,
        atol=1e-8,
    )
