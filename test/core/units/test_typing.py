# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Static typing tests; every pyrefly ignore pins an expected checker error."""

import importlib.util
import pathlib
import subprocess
import sys
from typing import Never, assert_type

import jax
import numpy as np
import pytest
from jax.typing import ArrayLike

from kups.core import units
from kups.core.units import (
    Area,
    AreaUnit,
    CompressibilityUnit,
    Dimensionless,
    Energy,
    EnergyUnit,
    ForceUnit,
    Frequency,
    FrequencyUnit,
    Length,
    LengthUnit,
    Momentum,
    NumberDensity,
    NumberDensityUnit,
    OddExponentRoot,
    Quantity,
    Unit,
    Velocity,
    Volume,
    VolumeUnit,
)
from kups.core.units import catalog as u
from kups.core.units import testing as ut

from ._helpers import arr

# aliases for dimensions no named type covers
type ForceSquaredUnit = Unit[2, 2, -4, 0, 0]
type UnknownUnit = Unit[int, int, int, int, int]
type Length12 = Quantity[12, 0, 0, 0, 0]


type _VelocitySquared = units.Quantity[2, 0, -2, 0, 0]


def test_unit_types():
    assert_type(u.angstrom, LengthUnit)
    assert_type(u.Angstrom, LengthUnit)
    assert_type(u.mol, Dimensionless)
    # 1 / u.s and u.s**-1 are the same type; any other numerator gives a Quantity
    assert_type(1 / u.s, FrequencyUnit)
    assert_type(u.s**-1, FrequencyUnit)
    assert_type(1.0 / u.s, Frequency)
    assert_type(2 / u.s, Frequency)


def test_misspelled_unit():
    with pytest.raises(AttributeError):
        _ = u.angstrum  # pyrefly: ignore[missing-attribute]


def test_unit_algebra():
    # multiplying units adds their exponents, and pyrefly does the addition
    assert_type(u.m * u.m, AreaUnit)
    assert_type(u.m * u.m * u.m, VolumeUnit)  # and a chain stays exact
    assert_type(u.m**3, VolumeUnit)
    assert_type(u.eV / u.angstrom, ForceUnit)  # energy over length is force
    assert_type(u.N * u.m, EnergyUnit)  # and force times length is energy again
    assert_type(u.kcal / u.mol, EnergyUnit)  # the mole is a pure count
    assert_type(u.mol * u.m, LengthUnit)
    assert_type(u.m / u.m, Dimensionless)  # the exponents cancel
    # a unit parsed at runtime has unknown exponents, and so do its products
    assert_type(u.unit("m") * u.s, UnknownUnit)
    assert_type(u.unit("m") / u.s, UnknownUnit)
    assert_type(u.unit("m") ** 2, UnknownUnit)
    # no named unit for this dimension, but still exact
    assert_type((u.eV / u.angstrom) ** 2, ForceSquaredUnit)


def test_quantity_algebra():
    r = 3.0 * u.angstrom
    assert_type(r, Length)
    assert_type(2.0 * u.mol, jax.Array)  # dimensionless decays to a plain array
    assert_type(1.0 / u.s, Frequency)  # reciprocal magnitudes, dimension negated
    assert_type(r + r, Length)
    assert_type(r + u.angstrom, Length)  # a bare unit means one of it
    assert_type(2.0 * r, Length)  # a plain factor rescales without changing dimension
    assert_type(r / 2.0, Length)
    assert_type(-r, Length)
    assert_type(abs(r), Length)
    assert_type(r * r, Area)
    assert_type(r**3, Volume)
    assert_type(r / (2.0 * u.fs), Velocity)
    assert_type(r / r, jax.Array)  # identical dimensions cancel to plain magnitudes
    assert_type(r.rebase(units.SI), Length)
    assert_type(r.to(u.nm), jax.Array)  # same dimension, another unit
    assert_type(r.value, jax.Array)
    assert_type(r < r, jax.Array)
    assert_type(float(r), float)
    assert_type(r.sqrt(), OddExponentRoot)  # m^1/2 is not an integer exponent
    _ = r.sqrt() * 2.0  # pyrefly: ignore[unsupported-operation]


def test_expression_chain():
    mass, speed = 1200.0 * u.kg, 90.0 * u.km / u.hour
    assert_type(speed, Velocity)
    assert_type(speed**2, _VelocitySquared)
    assert_type(mass * speed, Momentum)
    assert_type(0.5 * mass * speed**2, Energy)  # the whole of 1/2 m v^2
    assert float((0.5 * mass * speed**2).to(u.kJ)) == pytest.approx(375.0)
    # the Lennard-Jones exponents, and a dimensionless ratio in the middle
    r, sigma = 3.8 * u.angstrom, 3.4 * u.angstrom
    assert_type(sigma / r, jax.Array)
    assert_type(sigma**12, Length12)
    assert_type(sigma**12 / r**12, jax.Array)


def test_dimensionless_algebra():
    assert_type(u.mol**-1, Dimensionless)
    assert_type(u.mol * u.mol, Dimensionless)
    assert_type(u.mol / u.mol, Dimensionless)
    assert_type(2.0 * (u.mol**-1), jax.Array)  # so no .to() is offered on it
    assert_type(u.angstrom * 2.0, Length)  # unit times magnitudes, either order
    assert_type(u.angstrom / 2.0, Length)


def test_equality_raises():
    # == and != raise, and their declared return type is Never
    r = 3.0 * u.angstrom
    with pytest.raises(units.DimensionError):
        assert_type(r == r, Never)
    with pytest.raises(units.DimensionError):
        assert_type(r != r, Never)


def test_new_dimension_names():
    assert_type(u.Pa**-1, CompressibilityUnit)
    assert_type(u.angstrom**-3, NumberDensityUnit)
    assert_type(2.0 * u.angstrom**-3, NumberDensity)


def test_testing_helpers():
    # the helpers pin both sides to the unit's dimension, so a wrong unit in a
    # test is a checker error before it is a DimensionError
    r, t = arr([1.0, 2.0]) * u.angstrom, arr([1.0, 2.0]) * u.s
    ut.assert_allclose(r, arr([1.0, 2.0]) * u.angstrom, u.angstrom)
    ut.assert_allclose(r, arr([1.0, 2.0]), u.angstrom, rtol=1e-9)
    ut.assert_allclose(2.0 * u.eV, 2.0, u.eV)  # a bare number is eV
    ut.assert_array_equal(r, r, u.angstrom)
    with pytest.raises(units.DimensionError):
        ut.assert_allclose(r, r, u.eV)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(units.DimensionError):
        ut.assert_array_equal(r, t, u.angstrom)  # pyrefly: ignore[bad-argument-type]


def test_quantity_arraylike():
    # the raw Quantity dataclass takes any ArrayLike; every exit is a jax.Array
    dim = units.dim_of(u.angstrom)
    q = units.Quantity(3.0, dim, units.SI)
    n = units.Quantity(np.zeros(3), dim, units.SI)
    assert_type(q.v, ArrayLike)
    assert_type(q.value, jax.Array)
    assert_type(q.to(u.angstrom), jax.Array)
    assert_type(n.value.shape, tuple[int, ...])
    assert isinstance(q.value, jax.Array) and isinstance(n.to(u.nm), jax.Array)


def test_checker_accepts_this_file():
    # every assert_type above holds and every pyrefly ignore suppresses a real error
    if importlib.util.find_spec("pyrefly") is None:
        pytest.skip("pyrefly is not installed")
    root = pathlib.Path(__file__).parents[3]
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--search-path",
            str(root),
            __file__,
        ],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
