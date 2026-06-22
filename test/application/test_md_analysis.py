# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for kups.application.md.analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from kups.application.md.analysis import analyze_md, analyze_md_file
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Index, Table
from kups.core.lens import view
from kups.core.storage import EveryNStep, HDF5StorageWriter, Once, WriterGroupConfig
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass as jax_dataclass


@dataclass
class MockAtomData:
    """Mock init atoms providing the per-atom system index."""

    positions: Array
    system: Index[SystemId]


jax.tree_util.register_dataclass(MockAtomData)


@dataclass
class MockInitData:
    """Mock satisfying IsMDInitData."""

    atoms: Table[ParticleId, MockAtomData]


@dataclass
class MockStepData:
    """Mock satisfying IsMDStepData (per-step thermodynamic scalars)."""

    potential_energy: Array
    kinetic_energy: Array
    stress_tensor: Array
    volume: Array
    internal_kinetic_energy: Array


jax.tree_util.register_dataclass(MockInitData)
jax.tree_util.register_dataclass(MockStepData)


@jax_dataclass
class _MDFileConfig:
    """Two-group writer config mirroring the MD logging schema shape."""

    init: WriterGroupConfig[Any, Any]
    step: WriterGroupConfig[Any, Any]


@jax_dataclass
class _MDFileState:
    """Per-step state the writer extracts ``init``/``step`` groups from."""

    init: MockInitData
    step: MockStepData


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
    volume: Array | None = None,
    internal_ke: Array | None = None,
) -> MockStepData:
    """Create mock step data; internal KE defaults to the logged KE."""
    return MockStepData(
        potential_energy=pe,
        kinetic_energy=ke,
        stress_tensor=stress,
        volume=jnp.ones(pe.shape) if volume is None else volume,
        internal_kinetic_energy=ke if internal_ke is None else internal_ke,
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
        """Temperature follows T = 2*internal_KE / (k_B * DOF)."""
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

    def test_temperature_uses_internal_kinetic_energy(self):
        """Temperature is driven by internal KE, independent of the logged KE."""
        n_atoms, n_steps = 10, 100
        dof = 3 * n_atoms - 3
        pe = jnp.zeros((n_steps, 1))
        ke = jnp.full((n_steps, 1), 5.0)
        internal_ke = jnp.full((n_steps, 1), 0.3)
        stress = jnp.zeros((n_steps, 1, 3, 3))

        results = analyze_md(
            _make_init(n_atoms),
            _make_step(pe, ke, stress, internal_ke=internal_ke),
            n_blocks=10,
        )
        r = results[SystemId(0)]

        assert r.kinetic_energy.mean == pytest.approx(5.0)
        assert r.temperature.mean == pytest.approx(
            2 * 0.3 / (BOLTZMANN_CONSTANT * dof), rel=1e-6
        )

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


def _trajectory(
    n_steps: int, n_systems: int, n_atoms: int
) -> tuple[MockInitData, list[MockStepData]]:
    """Constant init data and a deterministic per-step trajectory."""
    init = _make_init(n_atoms, n_systems)
    base = jnp.arange(n_systems, dtype=jnp.float64)
    steps = [
        MockStepData(
            potential_energy=base + 0.1 * t,
            kinetic_energy=0.5 * base + 0.01 * t + 1.0,
            stress_tensor=jnp.eye(3) * (0.2 * t + base[:, None, None]),
            volume=base + 100.0 + t,
            internal_kinetic_energy=0.4 * base + 0.01 * t + 0.9,
        )
        for t in range(n_steps)
    ]
    return init, steps


def _stack_steps(steps: list[MockStepData]) -> MockStepData:
    """Stack per-step states into one trajectory object (leading time axis)."""
    return MockStepData(
        potential_energy=jnp.stack([s.potential_energy for s in steps]),
        kinetic_energy=jnp.stack([s.kinetic_energy for s in steps]),
        stress_tensor=jnp.stack([s.stress_tensor for s in steps]),
        volume=jnp.stack([s.volume for s in steps]),
        internal_kinetic_energy=jnp.stack([s.internal_kinetic_energy for s in steps]),
    )


def test_analyze_md_file_matches_in_memory(tmp_path: Path):
    """analyze_md_file (selective reads) matches analyze_md on identical data."""
    n_steps, n_systems, n_atoms = 20, 2, 8
    init, steps = _trajectory(n_steps, n_systems, n_atoms)
    expected = analyze_md(init, _stack_steps(steps), n_blocks=5)

    config = _MDFileConfig(
        init=WriterGroupConfig(view=view(lambda s: s.init), logging_frequency=Once()),
        step=WriterGroupConfig(
            view=view(lambda s: s.step), logging_frequency=EveryNStep(1)
        ),
    )
    path = tmp_path / "md.h5"
    writer = HDF5StorageWriter(
        path, config, _MDFileState(init, steps[0]), total_steps=n_steps
    )
    with writer:
        for t in range(n_steps):
            writer.log(_MDFileState(init, steps[t]), t)

    result = analyze_md_file(path, n_blocks=5)

    assert result.keys() == expected.keys()
    fields = (
        "potential_energy",
        "kinetic_energy",
        "total_energy",
        "temperature",
        "pressure",
        "volume",
    )
    for sys_id in expected:
        e, r = expected[sys_id], result[sys_id]
        assert (r.n_atoms, r.n_steps) == (e.n_atoms, e.n_steps)
        assert r.energy_drift == pytest.approx(e.energy_drift, rel=1e-6, abs=1e-9)
        for f in fields:
            assert getattr(r, f).mean == pytest.approx(getattr(e, f).mean, rel=1e-6)
            assert getattr(r, f).sem == pytest.approx(
                getattr(e, f).sem, rel=1e-6, abs=1e-9
            )
