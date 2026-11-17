# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Static typing tests; every pyrefly ignore pins an expected checker error."""

import importlib.util
import pathlib
import subprocess
import sys
from typing import assert_type

import pytest

from kups.core.units import (
    AreaUnit,
    Dimensionless,
    EnergyUnit,
    ForceUnit,
    FrequencyUnit,
    LengthUnit,
    Unit,
    VolumeUnit,
)
from kups.core.units import catalog as u

# aliases for dimensions no named type covers
type ForceSquaredUnit = Unit[2, 2, -4, 0, 0]
type UnknownUnit = Unit[int, int, int, int, int]


def test_unit_types():
    assert_type(u.angstrom, LengthUnit)
    assert_type(u.Angstrom, LengthUnit)
    assert_type(u.mol, Dimensionless)
    # 1 / u.s and u.s**-1 are the same type
    assert_type(1 / u.s, FrequencyUnit)
    assert_type(u.s**-1, FrequencyUnit)


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
    assert_type(u.m**0.5, UnknownUnit)  # a non-integer power
    # no named unit for this dimension, but still exact
    assert_type((u.eV / u.angstrom) ** 2, ForceSquaredUnit)


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
