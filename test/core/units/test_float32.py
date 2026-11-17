# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Float32 tests (x64 off, jax's default): rescales must be right."""

import jax.numpy as jnp
import pytest

from kups.core import units
from kups.core.units import E_A, E_K, E_KG, E_M, E_S, Unit
from kups.core.units import catalog as u
from kups.core.units import testing as ut

from ._helpers import x64_mode


@pytest.fixture(autouse=True)
def float32():
    with x64_mode(False):
        yield


def _roundtrip(unit: Unit[E_M, E_KG, E_S, E_A, E_K]) -> float:
    return float((1.0 * unit).to(unit))


def test_to_rescales_once():
    # two-factor conversions can hit a subnormal intermediate that flushes to zero
    for got in (
        _roundtrip(u.fs**3),
        _roundtrip(u.angstrom**3 * u.fs),
        _roundtrip(u.amu * u.angstrom**3),
        _roundtrip(u.nm**6),
        _roundtrip(1 / u.fs**3),
    ):
        assert jnp.isfinite(got) and abs(got - 1.0) < 1e-6, got
    assert float((12.5 * u.amu * u.angstrom**2).to(u.amu * u.nm**2)) == pytest.approx(
        0.125, rel=1e-6
    )
    assert float((1.0 * u.angstrom**4).to(u.nm**4)) == pytest.approx(1e-4, rel=1e-6)


def test_allclose_float32_rtol():
    # a fixed 1e-7 rtol could not pass at float32 precision
    ut.assert_allclose(1.0 * u.hartree, 27.211386245988 * u.eV, u.eV)
    ut.assert_allclose(1.0 * u.eV, 23.060547830619 * u.kcal / u.mol, u.eV)
    with pytest.raises(AssertionError):
        ut.assert_allclose(1.0 * u.eV, 1.000002 * u.eV, u.eV)


def test_scalar_repr():
    # jax's formatter, not the float64 expansion of a float32
    assert repr(0.1 * u.angstrom) == "0.1 Å"
    assert str(0.3 * u.eV) == "0.3 eV"
    assert f"{0.1 * u.angstrom:.2e}" == "1.00e-01 Å"
    with units.base(units.SI):
        assert repr(3.0 * u.angstrom) == "3e-10 m"
        assert f"{3.0 * u.angstrom:.2e}" == "3.00e-10 m"
