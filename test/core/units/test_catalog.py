# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unit catalogue."""

import math
import re
from fractions import Fraction

import pytest

from kups.core import units
from kups.core.units import (
    Unit,
)
from kups.core.units import catalog as u
from kups.core.units._unit import Atom
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
    # unit algebra stays Unit, including a cancelled dimension
    assert isinstance(u.eV / u.angstrom, Unit)
    assert (u.eV / u.angstrom).dim == (1, 1, -2, 0, 0)
    # names canonicalize factor order, so how a unit was built does not show
    assert u.eV * u.eV == u.eV**2 and str(u.eV * u.eV) == "eV^2"
    assert str(u.angstrom * u.amu) == str(u.amu * u.angstrom) == "amu.Å"
    assert str((u.eV / u.angstrom) ** 2) == "eV^2/Å^2"
    assert str(1 / (u.eV * u.angstrom)) == "1/(eV.Å)"  # one slash, one group
    assert str(u.eV / u.eV) == "1" and (u.eV / u.eV).dim == (0, 0, 0, 0, 0)
    assert type(u.m / u.m) is Unit and type(u.mol) is Unit


def test_unit_identity():
    # units are equal when their reduced factors are, however they were built
    for unit in (
        (u.eV / u.angstrom) * u.angstrom,
        (u.eV / u.angstrom**2) * u.angstrom**2,
    ):
        assert unit == u.eV and hash(unit) == hash(u.eV) and str(unit) == "eV"
        assert unit.scale == u.eV.scale  # rounded once from the exact product
    assert u.J != u.N * u.m  # a different spelling is a different unit
    # a product of several atoms and an odd root are each rounded once from the
    # exact value
    exact = Fraction(u.eV.scale) / Fraction(u.angstrom.scale) ** 2
    assert (u.eV / u.angstrom**2).scale == float(exact)
    assert (u.km ** Fraction(1, 3)).scale == 10.0
    # an atom is its name, dimension and scale together
    assert Unit(((Atom("eV", u.eV.dim, u.eV.scale), Fraction(1)),)) == u.eV
    assert Unit(((Atom("eV", u.eV.dim, 2 * u.eV.scale), Fraction(1)),)) != u.eV


def test_reciprocal_unit():
    assert 1 / u.eV == u.eV**-1  # the two spellings agree
    assert isinstance(1 / u.eV, units.Unit)
    # a number is not a unit, so 1 / unit is the only mixed form
    for build in (
        lambda: 2 / u.eV,
        lambda: 1.0 / u.eV,
        lambda: u.eV * 2,
        lambda: u.eV / 2,
    ):
        with pytest.raises(TypeError):
            build()


def test_si_prefixes():
    assert u.unit("km") is u.km and u.unit("kg") is u.kg and u.unit("fs") is u.fs
    assert u.unit("um") is u.unit("µm") is u.unit("\u03bcm") is u.unit("microm")
    assert u.unit("um").scale == 1e-6 and u.unit("mg").scale == 1e-6  # exactly 1e-6


def test_unit_lookup_errors(monkeypatch: pytest.MonkeyPatch):
    for name, message in (
        ("furlong", "'furlong' is not a unit"),
        ("mkm", "'km' carries a prefix already"),
        ("hbar", "reads as it rather than as a hectobar"),
        ("hectobar", "reads as it rather than as a hectobar"),
    ):
        with pytest.raises(units.DimensionError, match=re.escape(message)):
            u.unit(name)
    # a root that would let a name split two ways makes the name ambiguous
    monkeypatch.setitem(u._ROOTS, "am", u.m)
    with pytest.raises(units.DimensionError, match="'dam' is ambiguous"):
        u.unit("dam")


def test_fractional_exponents():
    assert (u.m**0.5).dim == (Fraction(1, 2), 0, 0, 0, 0)
    assert (u.km ** Fraction(1, 2)).scale == math.sqrt(1000.0)
    assert (u.angstrom ** Fraction(1, 2)).scale == math.sqrt(u.angstrom.scale)
    # a fractional power is bracketed, so no name reads two ways
    assert str(u.m ** Fraction(1, 2)) == "m^(1/2)"
    assert str(1 / u.m ** Fraction(1, 2)) == "1/m^(1/2)"
    assert str(u.m ** Fraction(1, 2) / u.s) == "m^(1/2)/s"
    assert str(u.m ** Fraction(-3, 2)) == "1/m^(3/2)"
    for e in (math.pi, math.inf):
        with pytest.raises(units.DimensionError, match="is not a small rational"):
            _ = u.m**e


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


def test_overflowing_scales():
    # a scale past the float range is inf, never an error
    assert (u.unit("Qm") ** 11).scale == math.inf
    assert (u.unit("Qm") ** Fraction(23, 2)).scale == math.inf
    huge = Atom("huge", u.m.dim, math.inf)
    big = Atom("big", u.m.dim, 1e300)
    assert Unit(((huge, Fraction(1)),)).scale == math.inf
    assert Unit(((huge, Fraction(1)), (big, Fraction(2)))).scale == math.inf
