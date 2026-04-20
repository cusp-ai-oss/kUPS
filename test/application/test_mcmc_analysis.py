# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.mcmc.analysis."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from jax import Array

from kups.application.mcmc.analysis import (
    _analyze_single_system,
    analyze_mcmc,
)
from kups.application.mcmc.data import StressResult
from kups.core.data import Table
from kups.core.typing import MotifId, SystemId
from kups.core.utils.jax import dataclass as jax_dataclass
from kups.core.utils.jax import no_post_init

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests for _analyze_single_system
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests for analyze_mcmc
# ---------------------------------------------------------------------------


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
