# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for pressure computation via different methods."""

from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.typing import SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.observables.pressure import ideal_gas_pressure, pressure_from_stress


@dataclass
class _Systems:
    temperature: Array
    unitcell: UnitCell


def _make_systems(
    box_size: float, temperature: float | Array, n: int = 1
) -> Table[SystemId, _Systems]:
    lv = jnp.eye(3)[None] * box_size
    if n > 1:
        lv = jnp.tile(lv, (n, 1, 1))
    uc = TriclinicUnitCell.from_matrix(lv)
    t = jnp.broadcast_to(jnp.asarray(temperature), (n,))
    keys = tuple(SystemId(i) for i in range(n))
    return Table(keys, _Systems(temperature=t, unitcell=uc))


class TestPressureFromStress:
    """Test pressure calculation from stress tensor."""

    @pytest.mark.parametrize(
        "tensor,expected",
        [
            pytest.param(jnp.diag(jnp.full(3, 2.5))[None], 2.5, id="isotropic"),
            pytest.param(
                jnp.array([[[1.0, 0.5, 0.2], [0.5, 2.0, 0.3], [0.2, 0.3, 3.0]]]),
                2.0,
                id="anisotropic",
            ),
            pytest.param(jnp.zeros((1, 3, 3)), 0.0, id="zero"),
            pytest.param(
                jnp.array([[[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]]]),
                0.0,
                id="off_diagonal_only",
            ),
        ],
    )
    def test_pressure_from_stress_tensor(self, tensor: Array, expected: float):
        """Pressure = trace(stress)/3 for various stress tensors."""
        stress = Table((SystemId(0),), tensor)
        result = pressure_from_stress(stress)
        npt.assert_allclose(result.data[0], expected, rtol=1e-6, atol=1e-10)

    def test_pressure_from_stress_batch(self):
        """Batch of stress tensors produces per-system pressures."""
        stress_batch = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                [[3.0, 0.5, 0.0], [0.5, 4.0, 0.0], [0.0, 0.0, 5.0]],
            ]
        )
        keys = (SystemId(0), SystemId(1), SystemId(2))
        stress = Table(keys, stress_batch)
        result = pressure_from_stress(stress)
        npt.assert_allclose(result.data, jnp.array([1.0, 2.0, 4.0]), rtol=1e-6)


class TestIdealGasPressure:
    """Test ideal gas pressure calculation P = NkT/V."""

    def test_basic(self):
        """P = NkT/V for known values."""
        systems = _make_systems(box_size=2.0, temperature=300.0)
        counts = Table((SystemId(0),), jnp.array([2.0]))
        result = ideal_gas_pressure(counts, systems)
        expected = 2 * BOLTZMANN_CONSTANT * 300.0 / 8.0
        npt.assert_allclose(result.data[0], expected, rtol=1e-6)

    def test_cross_method(self):
        """Ideal gas pressure matches trace of diagonal stress."""
        systems = _make_systems(box_size=2.0, temperature=300.0)
        counts = Table((SystemId(0),), jnp.array([2.0]))
        p = ideal_gas_pressure(counts, systems)
        stress = Table((SystemId(0),), jnp.diag(jnp.full(3, p.data[0]))[None])
        p_from_stress = pressure_from_stress(stress)
        npt.assert_allclose(p_from_stress.data[0], p.data[0], rtol=1e-6)

    def test_volume_scaling(self):
        """Pressure scales as 1/V: doubling box size reduces pressure 8x."""
        counts = Table((SystemId(0),), jnp.array([5.0]))
        small = ideal_gas_pressure(counts, _make_systems(1.0, 400.0))
        large = ideal_gas_pressure(counts, _make_systems(2.0, 400.0))
        npt.assert_allclose(small.data[0] / large.data[0], 8.0, rtol=1e-6)

    def test_units_consistency(self):
        """At T=1K, N=1, V=1: P = k_B exactly."""
        systems = _make_systems(box_size=1.0, temperature=1.0)
        counts = Table((SystemId(0),), jnp.array([1.0]))
        result = ideal_gas_pressure(counts, systems)
        npt.assert_allclose(result.data[0], BOLTZMANN_CONSTANT, rtol=1e-6)
        assert result.data[0] > 0
