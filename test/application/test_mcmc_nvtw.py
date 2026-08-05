# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Structural and end-to-end tests for the TMMC (NVT+W) simulation entry point."""

from __future__ import annotations

import tempfile

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
)
from kups.application.simulations.mcmc_nvtw import (
    Config,
    NVTWidomRunConfig,
    NVTWidomState,
    init_state,
    make_propagator,
    run,
    summarize,
)
from kups.application.simulations.mcmc_rigid import EwaldConfig, LJConfig

L = 10.0  # box side (Å)
N_MAX = 3


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


def _host(cif_file: str) -> HostConfig:
    return HostConfig(
        cif_file=cif_file,
        pressure=1e5,
        temperature=300.0,
        init_adsorbates=(0,),
        adsorbate_composition=(1.0,),
        adsorbate_interaction=((0.0,),),
    )


def _lj() -> LJConfig:
    # σ, ε in (Å, eV); vanilla UFF Ar from trappe.yaml.
    return LJConfig(
        parameters={"Ar": (3.446, 0.008023)},
        cutoff=4.5,
        tail_correction=False,
        mixing_rule="lorentz_berthelot",
    )


def _tmp_h5() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    f.close()
    return f.name


def _config(host: HostConfig, run_config: NVTWidomRunConfig | None = None) -> Config:
    return Config(
        adsorbates=(_ar_adsorbate(),),
        host=host,
        run=run_config
        or NVTWidomRunConfig(
            out_file=_tmp_h5(),
            n_max=N_MAX,
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
        ewald=EwaldConfig(real_cutoff=4.5, precision=1.0e-4),
    )


class TestInitState:
    @pytest.fixture(scope="class")
    def state(self) -> NVTWidomState:
        return init_state(jax.random.key(0), _config(_host(_write_cubic_ar_cif())))

    def test_one_system_per_macrostate(self, state):
        assert len(state.systems) == N_MAX + 1
        npt.assert_array_equal(
            state.macrostate_n, jnp.arange(N_MAX + 1, dtype=jnp.int32)
        )

    def test_macrostate_populations_match_labels(self, state):
        # System n must hold exactly n adsorbate groups.
        npt.assert_array_equal(
            state.groups.data.system.counts.data, jnp.arange(N_MAX + 1)
        )

    def test_accumulators_start_at_zero(self, state):
        stats = state.transition_statistics.data
        npt.assert_array_equal(stats.acceptance_insertion, jnp.zeros(N_MAX + 1))
        npt.assert_array_equal(
            stats.n_trials_insertion, jnp.zeros(N_MAX + 1, dtype=jnp.int32)
        )
        npt.assert_array_equal(
            state.energy_moments.data.count, jnp.zeros(N_MAX + 1, dtype=jnp.int32)
        )

    def test_multi_species_rejected(self):
        config = _config(_host(_write_cubic_ar_cif()))
        config = config.model_copy(
            update={"adsorbates": (_ar_adsorbate(), _ar_adsorbate())}
        )
        with pytest.raises(AssertionError, match="single adsorbate species"):
            init_state(jax.random.key(0), config)


class TestMakePropagator:
    def test_returns_init_and_production_pair(self):
        config = _config(_host(_write_cubic_ar_cif()))
        state = init_state(jax.random.key(0), config)
        init_prop, production = make_propagator(state, config.run)
        assert callable(init_prop)
        assert callable(production)


class TestRun:
    """End-to-end smoke: a short deterministic run accumulates a valid C-matrix."""

    @pytest.fixture(scope="class")
    def run_result(self) -> tuple[NVTWidomState, Config]:
        config = _config(_host(_write_cubic_ar_cif()))
        return run(config), config

    def test_trial_counts_match_schedule(self, run_result):
        state, config = run_result
        n_trials = config.run.num_widom_per_cycle * config.run.num_cycles
        stats = state.transition_statistics.data
        npt.assert_array_equal(
            stats.n_trials_insertion,
            jnp.full(N_MAX + 1, n_trials, dtype=jnp.int32),
        )
        npt.assert_array_equal(
            stats.n_trials_deletion,
            jnp.full(N_MAX + 1, n_trials, dtype=jnp.int32),
        )
        npt.assert_array_equal(
            state.energy_moments.data.count,
            jnp.full(N_MAX + 1, n_trials, dtype=jnp.int32),
        )

    def test_no_deletion_acceptance_at_empty_macrostate(self, run_result):
        state, _ = run_result
        stats = state.transition_statistics.data
        assert float(stats.acceptance_deletion[0]) == 0.0

    def test_acceptances_are_valid_fractions(self, run_result):
        state, config = run_result
        n_trials = config.run.num_widom_per_cycle * config.run.num_cycles
        stats = state.transition_statistics.data
        assert bool(jnp.all(stats.acceptance_insertion >= 0.0))
        assert bool(jnp.all(stats.acceptance_insertion <= n_trials))
        assert bool(jnp.all(stats.acceptance_deletion >= 0.0))
        assert bool(jnp.all(stats.acceptance_deletion <= n_trials))

    def test_summary_reconstructs_finite_log_partition_fn(self, run_result):
        state, config = run_result
        summary = summarize(config, state)
        log_qc = summary.log_partition_fn_sim
        assert log_qc.shape == (N_MAX + 1,)
        assert float(log_qc[0]) == 0.0  # anchored at N = 0
        assert bool(jnp.all(jnp.isfinite(log_qc)))

    def test_isotherm_is_finite_and_nonnegative(self, run_result):
        state, config = run_result
        summary = summarize(config, state)
        pressures = jnp.array([1e4, 1e5, 1e6])
        loading = summary.isotherm(pressures, jnp.asarray(300.0))
        assert loading.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(loading)))
        assert bool(jnp.all(loading >= 0.0))
        assert bool(jnp.all(loading <= N_MAX))
