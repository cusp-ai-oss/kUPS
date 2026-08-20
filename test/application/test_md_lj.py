# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones MD entry point."""

import jax.numpy as jnp
import pytest

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import MdParameters, MdRunConfig
from kups.application.simulations.md import Config, run
from kups.application.simulations.potentials import LjPotentialConfig

from ._builders import ar_cif, tmp_h5


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
        out = tmp_h5()
        run(_config(out, ar_cif()))
        return out

    def test_analyzer_reads_back_physical_outputs(self, out_file: str) -> None:
        results = analyze_md_file(out_file, n_blocks=2)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(result.total_energy.mean).all().item()
        assert jnp.isfinite(result.temperature.mean).all().item()
        assert (result.temperature.mean >= 0.0).all().item()
