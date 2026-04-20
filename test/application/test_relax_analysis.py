# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.relaxation.analysis."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest
from jax import Array

from kups.application.relaxation.analysis import analyze_relax
from kups.core.data import Table
from kups.core.typing import SystemId


@dataclass
class MockRelaxStepData:
    """Mock satisfying IsRelaxStepData."""

    _potential_energy: Array
    _max_force: Array

    @property
    def potential_energy(self) -> Array:
        return self._potential_energy

    @property
    def max_force(self) -> Array:
        return self._max_force


@dataclass
class MockRelaxInitData:
    """Mock satisfying _IsRelaxInitData."""

    systems: Table[SystemId, object]


def _make_init(n_systems: int = 1) -> MockRelaxInitData:
    keys = tuple(SystemId(i) for i in range(n_systems))
    return MockRelaxInitData(systems=Table(keys=keys, data=None))


def test_final_values() -> None:
    """Verify final_energy and final_max_force match the input step data."""
    step = MockRelaxStepData(
        jnp.array([-3.5], dtype=jnp.float32), jnp.array([0.01], dtype=jnp.float32)
    )
    results = analyze_relax(_make_init(), step, n_steps=10)
    r = results[SystemId(0)]

    assert r.final_energy == -3.5
    assert r.final_max_force == float(jnp.float32(0.01))


def test_n_steps_passthrough() -> None:
    """Verify n_steps is forwarded unchanged into the result."""
    step = MockRelaxStepData(jnp.array([0.0]), jnp.array([0.0]))
    results = analyze_relax(_make_init(), step, n_steps=42)

    assert results[SystemId(0)].n_steps == 42


def test_multi_system() -> None:
    """Per-system results with two systems."""
    step = MockRelaxStepData(
        jnp.array([-1.0, -2.0]),
        jnp.array([0.05, 0.03]),
    )
    results = analyze_relax(_make_init(n_systems=2), step, n_steps=5)

    assert results[SystemId(0)].final_energy == -1.0
    assert results[SystemId(1)].final_energy == -2.0
    assert results[SystemId(0)].final_max_force == pytest.approx(0.05)
    assert results[SystemId(1)].final_max_force == pytest.approx(0.03)
