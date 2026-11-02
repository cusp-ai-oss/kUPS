# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unit catalogue."""

import pathlib

import jax
import jax.numpy as jnp
import pytest

from kups.core import units
from kups.core.units import (
    Unit,
)
from kups.core.units import catalog as u
from kups.core.units.catalog import angstrom, eV


def test_unit_catalogue():
    assert (
        u.__name__ == "kups.core.units.catalog"
    )  # a module, so a checker resolves attributes
    assert angstrom is u.angstrom and eV is u.eV  # the from-import spelling agrees
    assert u.Da is u.amu and u.Angstrom is u.angstrom
    # the display catalogue and the constants are the same objects, one per name
    assert len({x.name for x in u.CATALOG}) == len(u.CATALOG)
    spelled = {"Å": "angstrom"}  # the one display name that is not an identifier
    assert all(getattr(u, spelled.get(x.name, x.name)) is x for x in u.CATALOG)
    assert u.SI_BASE == (u.m, u.kg, u.s, u.A, u.K)
    # unit algebra stays Unit, and a cancelled dimension becomes Dimensionless
    assert isinstance(u.eV / u.angstrom, Unit)
    assert (u.eV / u.angstrom).dim == (1, 1, -2, 0, 0)
    # names canonicalize factor order, so how a unit was built does not show
    assert u.eV * u.eV == u.eV**2 and str(u.eV * u.eV) == "eV^2"
    assert str(u.angstrom * u.amu) == str(u.amu * u.angstrom) == "amu.Å"
    assert str((u.eV / u.angstrom) ** 2) == "eV^2/Å^2"  # and it reads back
    assert str(1 / (u.eV * u.angstrom)) == "1/(eV.Å)"  # one slash, one group
    assert str(u.eV / u.eV) == "1" and (u.eV / u.eV).dim == (0, 0, 0, 0, 0)
    # Dimensionless is a runtime subclass; the stubs spell it as an alias of Unit[0, ...]
    dimensionless = type(u.m / u.m)
    assert dimensionless.__name__ == "Dimensionless"
    assert isinstance(u.mol, dimensionless) and not isinstance(u.m, dimensionless)
    assert isinstance(u.m / u.m, dimensionless) and isinstance(u.m * u.mol, Unit)


def test_reciprocal_unit():
    assert 1 / u.eV == u.eV**-1  # the two spellings agree
    assert isinstance(1 / u.eV, units.Unit)


def test_si_prefixes():
    assert u.unit("km") is u.km and u.unit("kg") is u.kg and u.unit("fs") is u.fs
    assert u.unit("um") is u.unit("µm") is u.unit("\u03bcm") is u.unit("microm")
    assert u.unit("um").scale == 1e-6 and u.unit("mg").scale == 1e-6  # exactly 1e-6


def test_prefix_round_trip():
    # every prefixed spelling has one reading, and every printed name parses back
    for prefix, symbol in u._PREFIX_SPELLINGS.items():
        for spelling, root in u._ROOTS.items():
            if (
                symbol + root.name in u._REFUSED
            ):  # a unit kups.core.units refuses to have
                continue
            got = u.unit(prefix + spelling)  # raises if two readings fit
            want = root.scale * 10.0 ** u._PREFIXES[symbol]
            assert got.dim == root.dim and got.scale == pytest.approx(want, rel=1e-15)
            assert u.unit(got.name) is got
    for x in u.CATALOG:
        assert u.unit(x.name) is x  # the catalogue's own prefixed names included


def test_no_jax_patching():
    # structural check that kups.core.units does not reach into or modify jax
    src = pathlib.Path(units.__file__).parent
    assert not [
        f.name
        for f in src.glob("*.py")
        if any(
            line.lstrip().startswith(("from jax._src", "import jax._src"))
            for line in f.read_text().splitlines()
        )
    ]
    # every jax attribute is jax's own object, not a kups.core.units wrapper
    for mod in (jax, jnp, jax.lax, jax.tree, jax.tree_util):
        for name in dir(mod):
            got = getattr(getattr(mod, name, None), "__module__", "") or ""
            assert not got.startswith("kups.core.units"), (
                f"{mod.__name__}.{name} is kups.core.units'"
            )
    for cls in (jax.core.ShapedArray, jax.Array):
        for name in dir(cls):
            got = getattr(getattr(cls, name, None), "__module__", "") or ""
            assert not got.startswith("kups.core.units"), (
                f"{cls.__name__}.{name} is kups.core.units'"
            )
