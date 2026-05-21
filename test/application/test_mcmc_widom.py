# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Structural and end-to-end tests for the Widom simulation entry point."""

from __future__ import annotations

import tempfile

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.application.mcmc.analysis import analyze_widom_file
from kups.application.mcmc.data import (
    AdsorbateConfig,
    BlockingSphereConfig,
    HostConfig,
)
from kups.application.simulations.mcmc_rigid import EwaldConfig, LJConfig
from kups.application.simulations.mcmc_widom import (
    Config,
    WidomRunConfig,
    init_state,
    make_propagator,
    run,
)

from ..clear_cache import clear_cache  # noqa: F401

L = 10.0  # box side (Å)


# Hand-written CIF: ASE's writer auto-uniquifies `_atom_site_label`
# (`Ar1`, `Ar2`, ...) which fails to match the LJ parameter table.
_AR_CIF = f"""data_test
_cell_length_a  {L:.6f}
_cell_length_b  {L:.6f}
_cell_length_c  {L:.6f}
_cell_angle_alpha  90.0
_cell_angle_beta   90.0
_cell_angle_gamma  90.0
_symmetry_space_group_name_H-M  'P 1'
loop_
_symmetry_equiv_pos_as_xyz
 'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ar Ar 0.0 0.0 0.0
Ar Ar 0.5 0.0 0.0
Ar Ar 0.0 0.5 0.0
Ar Ar 0.0 0.0 0.5
"""


def _write_cubic_ar_cif() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.write(_AR_CIF)
    f.close()
    return f.name


def _ar_adsorbate() -> AdsorbateConfig:
    # A single-site neutral Ar pseudo-adsorbate: same LJ params as host atoms,
    # no charges (Ewald path is skipped via state.is_charged == False).
    return AdsorbateConfig(
        critical_temperature=150.7,
        critical_pressure=4.86e6,
        acentric_factor=-0.002,
        positions=((0.0, 0.0, 0.0),),
        symbols=("Ar",),
    )


def _host(cif_file: str, blocking_spheres=((),)) -> HostConfig:
    return HostConfig(
        cif_file=cif_file,
        pressure=1e5,
        temperature=300.0,
        init_adsorbates=(0,),
        adsorbate_composition=(1.0,),
        adsorbate_interaction=((0.0,),),
        blocking_spheres=blocking_spheres,
    )


def _lj() -> LJConfig:
    # σ, ε in (Å, eV); vanilla UFF Ar from trappe.yaml.
    return LJConfig(
        parameters={"Ar": (3.446, 0.008023)},
        cutoff=4.5,
        tail_correction=False,
        mixing_rule="lorentz_berthelot",
    )


def _ewald() -> EwaldConfig:
    return EwaldConfig(real_cutoff=4.5, precision=1.0e-4)


def _tmp_h5() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    f.close()
    return f.name


def _config(host: HostConfig, run_config: WidomRunConfig | None = None) -> Config:
    return Config(
        adsorbates=(_ar_adsorbate(),),
        hosts=(host,),
        run=run_config
        or WidomRunConfig(
            out_file=_tmp_h5(),
            num_cycles=2,
            num_warmup_cycles=1,
            num_displacements_per_cycle=1,
            num_widom_per_cycle=4,
            translation_prob=1.0,
            rotation_prob=0.0,
            reinsertion_prob=0.0,
            seed=0,
        ),
        lj=_lj(),
        ewald=_ewald(),
        max_num_adsorbates=4,
    )


class TestInitState:
    def test_no_blocking_spheres_by_default(self):
        cif = _write_cubic_ar_cif()
        state = init_state(jax.random.key(0), _config(_host(cif)))
        assert not state.has_blocking_spheres
        assert state.blocking_spheres_parameters.radii.shape == (0,)

    def test_blocking_spheres_flow_through(self):
        cif = _write_cubic_ar_cif()
        spheres = ((BlockingSphereConfig(center=(5.0, 5.0, 5.0), radius=2.0),),)
        state = init_state(
            jax.random.key(0), _config(_host(cif, blocking_spheres=spheres))
        )
        assert state.has_blocking_spheres
        assert state.blocking_spheres_parameters.radii.shape[0] == 1
        npt.assert_allclose(state.blocking_spheres_parameters.radii, jnp.array([2.0]))


class TestMakePropagator:
    def test_returns_init_and_production_pair(self):
        cif = _write_cubic_ar_cif()
        config = _config(_host(cif))
        state = init_state(jax.random.key(0), config)
        init_prop, production = make_propagator(state, config.run)
        # Both must be callable propagators.
        assert callable(init_prop)
        assert callable(production)

    def test_blocking_spheres_state_pathway(self):
        """A blocking-sphere-bearing host produces a state where the
        propagator builds the blocking-spheres potential branch."""
        cif = _write_cubic_ar_cif()
        spheres = ((BlockingSphereConfig(center=(5.0, 5.0, 5.0), radius=2.0),),)
        config = _config(_host(cif, blocking_spheres=spheres))
        state = init_state(jax.random.key(0), config)
        assert state.has_blocking_spheres
        # ``make_propagator`` reads ``state.has_blocking_spheres`` to decide
        # whether to add the blocking term; ensure it executes that branch
        # without error.
        init_prop, production = make_propagator(state, config.run)
        assert callable(init_prop)
        assert callable(production)


class TestRun:
    """End-to-end smoke: a short run writes an HDF5 file the analyzer can read."""

    def test_state_has_accumulated_widom_statistics(self):
        cif = _write_cubic_ar_cif()
        config = _config(_host(cif))
        state = run(config)
        stats = state.widom_statistics.data
        assert int(stats.n_samples[0]) == 4 * 2
        assert jnp.isfinite(stats.sum_boltzmann[0]).item()
        assert float(stats.sum_boltzmann[0]) > 0.0

    def test_analyzer_reads_back_physical_outputs(self):
        cif = _write_cubic_ar_cif()
        config = _config(_host(cif))
        run(config)
        results = analyze_widom_file(config.run.out_file, n_blocks=2)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(result.excess_chemical_potential.mean).item()
        assert jnp.isfinite(result.henry_coefficient.mean).item()
        assert jnp.isfinite(result.heat_of_adsorption.mean).item()
        assert float(result.henry_coefficient.mean) > 0.0
        assert float(result.henry_coefficient.sem) >= 0.0
