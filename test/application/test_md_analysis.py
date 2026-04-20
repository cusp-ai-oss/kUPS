# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.md.analysis.analyze_md."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from kups.application.md.analysis import analyze_md
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Index, Table
from kups.core.typing import ParticleId, SystemId


@dataclass
class MockAtomData:
    """Mock satisfying _IsMDAtoms (HasPositions + system index)."""

    positions: Array
    system: Index[SystemId]


jax.tree_util.register_dataclass(MockAtomData)


@dataclass
class MockInitData:
    """Mock satisfying IsMDInitData."""

    atoms: Table[ParticleId, MockAtomData]


@dataclass
class MockStepData:
    """Mock satisfying IsMDStepData."""

    potential_energy: Array
    kinetic_energy: Array
    stress_tensor: Array


def _make_init(n_atoms: int = 10, n_systems: int = 1) -> MockInitData:
    """Create mock init data distributing atoms evenly across systems."""
    keys = tuple(ParticleId(i) for i in range(n_atoms))
    system_labels = [SystemId(i % n_systems) for i in range(n_atoms)]
    data = MockAtomData(
        positions=jnp.zeros((n_atoms, 3)),
        system=Index.new(system_labels),
    )
    return MockInitData(atoms=Table(keys=keys, data=data))


def _make_step(
    pe: Array,
    ke: Array,
    stress: Array,
) -> MockStepData:
    """Create mock step data."""
    return MockStepData(
        potential_energy=pe,
        kinetic_energy=ke,
        stress_tensor=stress,
    )


class TestAnalyzeMD:
    """Tests for analyze_md."""

    def test_constant_energy(self):
        """Constant PE and KE yield exact means, near-zero SEM and drift."""
        n_steps = 100
        pe = jnp.full((n_steps, 1), 1.0)
        ke = jnp.full((n_steps, 1), 0.5)
        stress = jnp.zeros((n_steps, 1, 3, 3))

        results = analyze_md(_make_init(), _make_step(pe, ke, stress), n_blocks=10)
        r = results[SystemId(0)]

        assert r.potential_energy.mean == pytest.approx(1.0)
        assert r.kinetic_energy.mean == pytest.approx(0.5)
        assert r.total_energy.mean == pytest.approx(1.5)
        assert r.potential_energy.sem == pytest.approx(0.0, abs=1e-12)
        assert r.total_energy.sem == pytest.approx(0.0, abs=1e-12)
        assert r.energy_drift == pytest.approx(0.0, abs=1e-12)

    def test_temperature(self):
        """Temperature follows T = 2*KE / (k_B * DOF)."""
        n_atoms = 10
        n_steps = 100
        ke_val = 0.3
        dof = 3 * n_atoms - 3
        expected_temp = 2 * ke_val / (BOLTZMANN_CONSTANT * dof)

        pe = jnp.zeros((n_steps, 1))
        ke = jnp.full((n_steps, 1), ke_val)
        stress = jnp.zeros((n_steps, 1, 3, 3))

        results = analyze_md(
            _make_init(n_atoms), _make_step(pe, ke, stress), n_blocks=10
        )
        r = results[SystemId(0)]

        assert r.temperature.mean == pytest.approx(expected_temp, rel=1e-6)
        assert r.n_atoms == n_atoms

    def test_pressure(self):
        """Pressure equals trace of diagonal stress / 3."""
        n_steps = 100
        p_val = 2.0
        stress = jnp.tile(jnp.eye(3) * p_val, (n_steps, 1, 1, 1))

        pe = jnp.zeros((n_steps, 1))
        ke = jnp.zeros((n_steps, 1))

        results = analyze_md(_make_init(), _make_step(pe, ke, stress), n_blocks=10)
        r = results[SystemId(0)]

        assert r.pressure.mean == pytest.approx(p_val, rel=1e-6)
        assert r.pressure.sem == pytest.approx(0.0, abs=1e-12)

    def test_energy_drift(self):
        """Linearly increasing PE gives matching drift slope."""
        n_steps = 100
        slope = 0.01
        pe = (jnp.arange(n_steps, dtype=jnp.float64) * slope)[:, None]
        ke = jnp.zeros((n_steps, 1))
        stress = jnp.zeros((n_steps, 1, 3, 3))

        results = analyze_md(_make_init(), _make_step(pe, ke, stress), n_blocks=10)
        r = results[SystemId(0)]

        assert r.energy_drift == pytest.approx(slope, rel=1e-6)
        assert r.energy_drift_per_atom == pytest.approx(slope / 10, rel=1e-6)
        assert r.n_steps == n_steps

    def test_multi_system(self):
        """Per-system analysis with two independent systems."""
        n_steps = 100
        pe = jnp.stack([jnp.full(n_steps, 1.0), jnp.full(n_steps, 2.0)], axis=1)
        ke = jnp.stack([jnp.full(n_steps, 0.5), jnp.full(n_steps, 1.0)], axis=1)
        stress = jnp.zeros((n_steps, 2, 3, 3))

        results = analyze_md(
            _make_init(n_atoms=10, n_systems=2),
            _make_step(pe, ke, stress),
            n_blocks=10,
        )

        r0 = results[SystemId(0)]
        r1 = results[SystemId(1)]
        assert r0.potential_energy.mean == pytest.approx(1.0)
        assert r1.potential_energy.mean == pytest.approx(2.0)
        assert r0.kinetic_energy.mean == pytest.approx(0.5)
        assert r1.kinetic_energy.mean == pytest.approx(1.0)
        assert r0.n_atoms == 5
        assert r1.n_atoms == 5
