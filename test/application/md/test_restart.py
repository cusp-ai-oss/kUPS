# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""kups_md starts from the final state of an earlier run."""

from collections.abc import Sequence
from pathlib import Path

import ase
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.application.md.data import (
    MdParameters,
    MDParticles,
    MdRunConfig,
    MdState,
    MDSystems,
    md_state_from_ase,
    md_state_from_run,
)
from kups.application.md.logging import MDLoggedData
from kups.application.simulations.md import Config, run
from kups.core.constants import FEMTO_SECOND
from kups.core.data import Table
from kups.core.storage import HDF5StorageReader
from kups.core.typing import ParticleId, SystemId

from .._builders import ROWS, argon, md_parameters, write_cif

_AR_KR = ROWS.model_copy(
    update={"parameters": {"Ar": (3.405, 0.010326), "Kr": (3.636, 0.01437)}}
)


def _inputs(directory: Path, atoms: Sequence[ase.Atoms]) -> tuple[Path, ...]:
    """New input files, as kups_md donates the arrays it caches per path."""
    directory.mkdir()
    return tuple(write_cif(a, directory / f"{i}.cif") for i, a in enumerate(atoms))


def _run(
    out_file: Path,
    inp_files: Sequence[Path],
    md: MdParameters,
    num_steps: int,
    **run_options: str,
) -> MdState:
    run_config = MdRunConfig(
        out_file=out_file,
        num_steps=num_steps,
        num_warmup_steps=0,
        seed=1,
        **run_options,
    )
    return run(
        Config(run=run_config, md=md, potential=_AR_KR, inp_files=tuple(inp_files))
    )


def test_verlet_continues_exactly(tmp_path: Path):
    """10 Verlet steps and 10 more from the output file end where 20 steps end."""
    md, atoms = md_parameters("verlet"), [argon(2, 0), argon(3, 1)]
    full = _run(tmp_path / "full.h5", _inputs(tmp_path / "a", atoms), md, 20)
    _run(tmp_path / "first.h5", _inputs(tmp_path / "b", atoms), md, 10)
    kept_momenta = md.model_copy(update={"initialize_momenta": False})
    rest = _run(tmp_path / "rest.h5", [tmp_path / "first.h5"], kept_momenta, 10)
    cells = full.systems.map_data(lambda s: s.cell)[full.particles.data.system]
    delta = rest.particles.data.positions - full.particles.data.positions
    assert np.abs(cells.wrap(delta)).max() < 1e-10
    np.testing.assert_allclose(
        rest.particles.data.momenta, full.particles.data.momenta, atol=1e-10
    )


def test_restart_keeps_particle_order(tmp_path: Path):
    """Interleaved argon and krypton keep the input order, positions and momenta;
    a key draws new momenta without center-of-mass motion."""
    atoms = argon(2, 0)
    atoms.set_chemical_symbols(
        np.random.default_rng(0).choice(["Ar", "Kr"], len(atoms))
    )
    md = md_parameters("verlet")
    state = _run(tmp_path / "md.h5", _inputs(tmp_path / "a", [atoms]), md, 4)
    particles, _ = md_state_from_run(tmp_path / "md.h5", md)
    data = particles.data
    assert list(data.labels.value) == atoms.get_chemical_symbols()
    np.testing.assert_array_equal(data.atomic_numbers, atoms.get_atomic_numbers())
    np.testing.assert_array_equal(data.masses, atoms.get_masses())
    np.testing.assert_array_equal(data.positions, state.particles.data.positions)
    np.testing.assert_array_equal(data.momenta, state.particles.data.momenta)
    drawn, _ = md_state_from_run(tmp_path / "md.h5", md, key=jax.random.key(0))
    assert not np.allclose(drawn.data.momenta, data.momenta)
    np.testing.assert_allclose(drawn.data.momenta.sum(axis=0), 0, atol=1e-10)


def _fractional(
    particles: Table[ParticleId, MDParticles], systems: Table[SystemId, MDSystems]
) -> Array:
    cells = systems.map_data(lambda s: s.cell)[particles.data.system]
    return cells.frame.to_fractional(particles.data.positions)


@pytest.mark.parametrize("integrator", ["csvr_npt", "baoab_npt_langevin"])
def test_nvt_at_the_mean_npt_volume(tmp_path: Path, integrator: str):
    """Rescaled cells have the mean volume of the NPT rows and keep fractional
    coordinates and momenta; a CSVR run continues at that volume."""
    npt = tmp_path / "npt.h5"
    inputs = _inputs(tmp_path / "a", [argon(2, 0), argon(3, 1)])
    _run(npt, inputs, md_parameters(integrator), 30)
    with HDF5StorageReader[MDLoggedData](npt) as reader:
        volume = reader.focus_group(lambda s: s.step).read(select=lambda d: d.volume)
    mean = volume.mean(axis=0)
    nvt = md_parameters("csvr").model_copy(update={"initialize_momenta": False})
    rescaled = nvt.model_copy(update={"rescale_to_mean_volume": True})
    particles, systems = md_state_from_run(npt, nvt)
    scaled_particles, scaled_systems = md_state_from_run(npt, rescaled)
    np.testing.assert_allclose(scaled_systems.data.cell.volume, mean, rtol=1e-12)
    np.testing.assert_allclose(
        _fractional(scaled_particles, scaled_systems),
        _fractional(particles, systems),
        atol=1e-12,
    )
    np.testing.assert_array_equal(scaled_particles.data.momenta, particles.data.momenta)
    assert not np.any(scaled_systems.data.cell_momentum)
    _run(tmp_path / "nvt.h5", [npt], rescaled, 10)
    with HDF5StorageReader[MDLoggedData](tmp_path / "nvt.h5") as reader:
        rows = reader.focus_group(lambda s: s.step)[:]
    np.testing.assert_allclose(rows.volume, np.tile(mean, (10, 1)), rtol=1e-12)
    assert np.isfinite(rows.kinetic_energy).all()


def test_restart_in_other_dtypes(tmp_path: Path):
    """A float32 run with int32 indices continues in float64 with int64 indices, with
    the integrator parameters of the new config, not rounded to float32."""
    md = md_parameters("csvr")
    inputs = _inputs(tmp_path / "a", [argon(2, 0), argon(3, 1)])
    _run(tmp_path / "f32.h5", inputs, md, 4, float_dtype="float32", index_dtype="int32")
    kept_momenta = md.model_copy(update={"initialize_momenta": False})
    state = _run(tmp_path / "f64.h5", [tmp_path / "f32.h5"], kept_momenta, 4)
    leaves = jax.tree.leaves((state.particles, state.systems))
    floats = {x.dtype for x in leaves if jnp.issubdtype(x.dtype, jnp.floating)}
    assert floats == {jnp.dtype(jnp.float64)}
    assert state.particles.data.system.raw_indices.dtype == jnp.int64
    params = state.systems.data.integrator_params
    np.testing.assert_array_equal(params.time_step, [md.time_step * FEMTO_SECOND] * 2)
    np.testing.assert_array_equal(
        params.thermostat_time_constant,
        [md.thermostat_time_constant * FEMTO_SECOND] * 2,
    )


def test_unfinished_runs_raise(tmp_path: Path):
    """Runs that stopped early, were killed or ran no step have no final state, and
    structure files cannot be rescaled."""
    md = md_parameters("verlet")
    _run(tmp_path / "md.h5", _inputs(tmp_path / "a", [argon(2, 0)]), md, 4)
    with h5py.File(tmp_path / "md.h5", "a") as file:
        file.attrs["actual_steps"] = 3
    with pytest.raises(ValueError, match="no final state"):
        md_state_from_run(tmp_path / "md.h5", md)
    with h5py.File(tmp_path / "md.h5", "a") as file:
        del file.attrs["actual_steps"]
    with pytest.raises(ValueError, match="no final state"):
        md_state_from_run(tmp_path / "md.h5", md)
    _run(tmp_path / "empty.h5", _inputs(tmp_path / "b", [argon(2, 0)]), md, 0)
    with pytest.raises(ValueError, match="no final state"):
        md_state_from_run(tmp_path / "empty.h5", md)
    rescaled = md.model_copy(update={"rescale_to_mean_volume": True})
    with pytest.raises(ValueError, match="rescale_to_mean_volume"):
        md_state_from_ase(argon(2, 0), rescaled)
