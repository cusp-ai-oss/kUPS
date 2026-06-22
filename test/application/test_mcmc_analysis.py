# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.mcmc.analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.application.mcmc.analysis import (
    _analyze_single_system,
    _analyze_single_widom_system,
    analyze_mcmc,
    analyze_mcmc_file,
)
from kups.application.mcmc.data import StressResult
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.lens import view
from kups.core.storage import EveryNStep, HDF5StorageWriter, Once, WriterGroupConfig
from kups.core.typing import MotifId, SystemId
from kups.core.utils.jax import dataclass as jax_dataclass
from kups.core.utils.jax import no_post_init


@jax_dataclass
class _Temps:
    temperature: Array


@dataclass
class _FixedData:
    systems: Table[SystemId, _Temps]


@dataclass
class _SystemStepData:
    potential_energy: Array
    guest_stress: StressResult


@dataclass
class _StepData:
    particle_count: Table[tuple[SystemId, MotifId], Array]
    systems: Table[SystemId, _SystemStepData]


class TestAnalyzeSingleSystem:
    """Tests for _analyze_single_system."""

    def test_constant_energy(self):
        """Constant energy and counts yield exact means and near-zero SEM."""
        n_steps = 200
        energy = jnp.full((n_steps,), 2.0)
        counts = jnp.tile(jnp.array([3.0, 5.0]), (n_steps, 1))

        result = _analyze_single_system(energy, counts, temperature=300.0, n_blocks=10)

        assert float(result.energy.mean) == pytest.approx(2.0)
        assert float(result.energy.sem) == pytest.approx(0.0, abs=1e-12)
        assert float(result.loading.mean[0]) == pytest.approx(3.0)
        assert float(result.loading.mean[1]) == pytest.approx(5.0)
        assert float(result.loading.sem[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(result.loading.sem[1]) == pytest.approx(0.0, abs=1e-12)

    def test_loading_average(self):
        """Alternating counts yield correct loading average."""
        n_steps = 200
        energy = jnp.ones(n_steps)
        counts_a = jnp.array([2.0, 6.0])
        counts_b = jnp.array([4.0, 8.0])
        counts = jnp.tile(jnp.stack([counts_a, counts_b]), (n_steps // 2, 1))

        result = _analyze_single_system(energy, counts, temperature=300.0, n_blocks=10)

        assert float(result.loading.mean[0]) == pytest.approx(3.0)
        assert float(result.loading.mean[1]) == pytest.approx(7.0)


class TestAnalyzeMCMC:
    """Tests for analyze_mcmc with multiple systems."""

    def test_multi_system(self):
        """Two systems at different temperatures produce per-system results."""
        n_steps = 200
        sys0, sys1 = SystemId(0), SystemId(1)

        # Per-step data: constant energy per system, one motif each.
        # HDF5 reader stacks steps on axis 0 → shape (n_steps, n_systems),
        # which mismatches len(keys); bypass Table validation.
        energy_data = jnp.stack(
            [jnp.full(n_steps, 1.0), jnp.full(n_steps, 2.0)], axis=1
        )
        count_keys = ((sys0, MotifId(0)), (sys1, MotifId(0)))
        count_data = jnp.stack([jnp.full(n_steps, 4.0), jnp.full(n_steps, 6.0)], axis=1)
        with no_post_init():
            systems = Table(
                (sys0, sys1),
                _Temps(temperature=jnp.array([300.0, 400.0])),
            )
            particle_count = Table(count_keys, count_data)
            z = jnp.zeros((n_steps, 2, 3, 3))
            step_systems = Table(
                (sys0, sys1),
                _SystemStepData(
                    potential_energy=energy_data,
                    guest_stress=StressResult(z, z, z),
                ),
            )
        fixed = _FixedData(systems=systems)

        per_step = _StepData(
            particle_count=particle_count,
            systems=step_systems,
        )

        results = analyze_mcmc(fixed, per_step, n_blocks=10)

        assert set(results.keys()) == {sys0, sys1}
        assert float(results[sys0].energy.mean) == pytest.approx(1.0)
        assert float(results[sys1].energy.mean) == pytest.approx(2.0)
        assert float(results[sys0].loading.mean[0]) == pytest.approx(4.0)
        assert float(results[sys1].loading.mean[0]) == pytest.approx(6.0)

    def test_stress_to_pressure(self):
        """Isotropic stress σ = -P·I yields correct scalar pressure."""
        n_steps = 200
        sys0 = SystemId(0)
        P = 2.0

        energy_data = jnp.ones((n_steps, 1))
        count_keys = ((sys0, MotifId(0)),)
        count_data = jnp.full((n_steps, 1), 3.0)
        # σ = P·I → Tr(σ)/3 = P
        stress_step = P * jnp.eye(3)
        stress_data = jnp.broadcast_to(stress_step, (n_steps, 1, 3, 3))
        with no_post_init():
            systems = Table((sys0,), _Temps(temperature=jnp.array([300.0])))
            particle_count = Table(count_keys, count_data)
            z = jnp.zeros_like(stress_data)
            step_systems = Table(
                (sys0,),
                _SystemStepData(
                    potential_energy=energy_data,
                    guest_stress=StressResult(stress_data, z, z),
                ),
            )
        fixed = _FixedData(systems=systems)

        per_step = _StepData(
            particle_count=particle_count,
            systems=step_systems,
        )

        results = analyze_mcmc(fixed, per_step, n_blocks=10)
        result = results[sys0]

        assert result.stress is not None
        assert result.pressure is not None
        assert float(result.pressure.mean) == pytest.approx(P)
        expected_stress = P * jnp.eye(3)
        assert jnp.allclose(result.stress.mean, expected_stress, atol=1e-12)


class TestAnalyzeSingleWidomSystem:
    def test_constant_input_recovers_analytic_mu_ex_kh_qst(self):
        """Constant ΔU stream: μ_ex = ΔU, K_H = V·exp(-βΔU)/kT, q_st = kT − ΔU."""
        n_cycles = 10
        delta_U = -0.05  # eV
        temperature = 300.0
        volume = 100.0
        kT = float(BOLTZMANN_CONSTANT * temperature)
        beta = 1.0 / kT
        W = float(jnp.exp(-beta * delta_U))

        mean_w_per_cycle = jnp.full((n_cycles,), W)
        mean_du_w_per_cycle = jnp.full((n_cycles,), delta_U * W)

        result = _analyze_single_widom_system(
            mean_w_per_cycle,
            mean_du_w_per_cycle,
            temperature=temperature,
            volume=volume,
            n_blocks=2,
        )
        npt.assert_allclose(
            float(result.excess_chemical_potential.mean), delta_U, rtol=1e-9
        )
        npt.assert_allclose(
            float(result.henry_coefficient.mean), volume * W / kT, rtol=1e-9
        )
        npt.assert_allclose(
            float(result.heat_of_adsorption.mean), kT - delta_U, rtol=1e-9
        )
        # Constant input -> zero variance -> zero SEM (within fp64 rounding).
        npt.assert_allclose(float(result.excess_chemical_potential.sem), 0.0, atol=1e-9)
        npt.assert_allclose(float(result.henry_coefficient.sem), 0.0, atol=1e-9)
        npt.assert_allclose(float(result.heat_of_adsorption.sem), 0.0, atol=1e-9)


jax.tree_util.register_dataclass(_FixedData)
jax.tree_util.register_dataclass(_SystemStepData)
jax.tree_util.register_dataclass(_StepData)


@jax_dataclass
class _MCMCFileConfig:
    """Two-group writer config mirroring the MCMC logging schema shape."""

    fixed: WriterGroupConfig[Any, Any]
    per_step: WriterGroupConfig[Any, Any]


@jax_dataclass
class _MCMCFileState:
    """Per-step state the writer extracts ``fixed``/``per_step`` groups from."""

    fixed: _FixedData
    per_step: _StepData


def test_analyze_mcmc_file_matches_in_memory(tmp_path: Path):
    """analyze_mcmc_file (selective reads) matches analyze_mcmc on identical data."""
    n_steps, sys0, sys1 = 60, SystemId(0), SystemId(1)
    count_keys = ((sys0, MotifId(0)), (sys1, MotifId(0)))
    base = jnp.array([1.0, 2.0])

    def step(t: int) -> _StepData:
        energy = base + 0.05 * jnp.sin(jnp.array([t, t + 1.0]))
        counts = jnp.array([4.0, 6.0]) + 0.1 * jnp.cos(jnp.array([t, t + 2.0]))
        stress = jnp.stack([jnp.eye(3) * (0.3 + 0.01 * t), jnp.eye(3) * 0.5])
        zero = jnp.zeros_like(stress)
        with no_post_init():
            return _StepData(
                particle_count=Table(count_keys, counts),
                systems=Table(
                    (sys0, sys1),
                    _SystemStepData(
                        potential_energy=energy,
                        guest_stress=StressResult(stress, zero, zero),
                    ),
                ),
            )

    with no_post_init():
        fixed = _FixedData(
            systems=Table((sys0, sys1), _Temps(temperature=jnp.array([300.0, 400.0])))
        )
    # Build the in-memory comparison object from the stacked per-step trajectory.
    steps = [step(t) for t in range(n_steps)]
    with no_post_init():
        per_step_full = _StepData(
            particle_count=Table(
                count_keys, jnp.stack([s.particle_count.data for s in steps], axis=0)
            ),
            systems=Table(
                (sys0, sys1),
                _SystemStepData(
                    potential_energy=jnp.stack(
                        [s.systems.data.potential_energy for s in steps], axis=0
                    ),
                    guest_stress=StressResult(
                        jnp.stack(
                            [s.systems.data.guest_stress.potential for s in steps],
                            axis=0,
                        ),
                        jnp.stack(
                            [
                                s.systems.data.guest_stress.tail_correction
                                for s in steps
                            ],
                            axis=0,
                        ),
                        jnp.stack(
                            [s.systems.data.guest_stress.ideal_gas for s in steps],
                            axis=0,
                        ),
                    ),
                ),
            ),
        )
    expected = analyze_mcmc(fixed, per_step_full, n_blocks=5)

    config = _MCMCFileConfig(
        fixed=WriterGroupConfig(view=view(lambda s: s.fixed), logging_frequency=Once()),
        per_step=WriterGroupConfig(
            view=view(lambda s: s.per_step), logging_frequency=EveryNStep(1)
        ),
    )
    path = tmp_path / "mcmc.h5"
    writer = HDF5StorageWriter(
        path, config, _MCMCFileState(fixed, steps[0]), total_steps=n_steps
    )
    with writer:
        for t in range(n_steps):
            writer.log(_MCMCFileState(fixed, steps[t]), t)

    result = analyze_mcmc_file(path, n_blocks=5)

    assert result.keys() == expected.keys()
    for sys_id in expected:
        e, r = expected[sys_id], result[sys_id]
        npt.assert_allclose(r.energy.mean, e.energy.mean, rtol=1e-6)
        npt.assert_allclose(r.loading.mean, e.loading.mean, rtol=1e-6)
        npt.assert_allclose(
            r.heat_of_adsorption.mean, e.heat_of_adsorption.mean, rtol=1e-6
        )
        assert (r.stress is None) == (e.stress is None)
        if e.stress is not None:
            assert r.stress is not None
            assert r.pressure is not None and e.pressure is not None
            npt.assert_allclose(r.stress.mean, e.stress.mean, rtol=1e-6)
            npt.assert_allclose(r.pressure.mean, e.pressure.mean, rtol=1e-6)
