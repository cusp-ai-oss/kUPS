# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unit catalogue and assertions."""

import pathlib
import re
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core import units
from kups.core.units import (
    Energy,
    Force,
    Length,
    Unit,
)
from kups.core.units import catalog as u
from kups.core.units import testing as ut
from kups.core.units._unit import Atom
from kups.core.units.catalog import angstrom, eV

from ._helpers import (
    ASE_TIME,
    arr,
    jit,
    make_jaxpr,
    val,
)


def test_zero_overhead():
    # quantities lower to the same HLO as plain-array arithmetic
    def dot_q(a: Force, b: Length) -> Energy:
        return (a * b).sum()

    def dot_plain(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.sum(a * b)

    def ops(text: str) -> list[str]:
        return re.findall(r"stablehlo\.\w+", text)

    f, x = arr([1.0, 2.0, 3.0]) * u.N, arr([4.0, 5.0, 6.0]) * u.m
    with_units, plain = jit(dot_q).lower(f, x), jit(dot_plain).lower(f.value, x.value)
    text_u, text_p = with_units.as_text(), plain.as_text()
    # same operations, same count, same order
    assert len(ops(text_u)) == len(ops(text_p)) == 4
    assert ops(text_u) == ops(text_p)
    # the only difference is the result name
    assert 'jax.result_info = "result.v"' in text_u
    assert 'jax.result_info = "result"' in text_p
    cost_u, cost_p = (
        with_units.compile().cost_analysis(),
        plain.compile().cost_analysis(),
    )
    assert cost_u is not None and cost_p is not None
    assert cost_u["flops"] == cost_p["flops"]
    # the jaxprs match equation for equation
    eqns_u = make_jaxpr(dot_q)(f, x).eqns
    eqns_p = make_jaxpr(dot_plain)(f.value, x.value).eqns
    assert len(eqns_u) == len(eqns_p) == 2
    assert [e.primitive.name for e in eqns_u] == [e.primitive.name for e in eqns_p]


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
    # unit algebra stays Unit, including a cancelled dimension
    assert isinstance(u.eV / u.angstrom, Unit)
    assert (u.eV / u.angstrom).dim == (1, 1, -2, 0, 0)
    # names canonicalize factor order, so how a unit was built does not show
    assert u.eV * u.eV == u.eV**2 and str(u.eV * u.eV) == "eV^2"
    assert str(u.angstrom * u.amu) == str(u.amu * u.angstrom) == "amu.Å"
    assert str((u.eV / u.angstrom) ** 2) == "eV^2/Å^2"  # and it reads back
    assert float(u.parse("1 eV^2/Å^2").to((u.eV / u.angstrom) ** 2)) == 1.0
    assert str(1 / (u.eV * u.angstrom)) == "1/(eV.Å)"  # one slash, one group
    assert str(u.eV / u.eV) == "1" and (u.eV / u.eV).dim == (0, 0, 0, 0, 0)
    assert type(u.m / u.m) is Unit and type(u.mol) is Unit
    # unit times magnitudes attaches, in either order, and so does the quotient
    assert val(u.angstrom * 3.0) == 3.0 and val(3.0 * u.angstrom) == 3.0
    assert list((u.angstrom * arr([1.0, 2.0])).value) == [1.0, 2.0]
    assert val(u.angstrom / 4.0) == 0.25 and val(4.0 / u.angstrom) == 4.0
    q = 3.0 * u.eV
    assert val(u.eV * q) == val(q * u.eV) and float(u.eV / q) == pytest.approx(1 / 3)

    class Scaled:
        def __rmul__(self, other: object) -> str:
            return f"scaled by {other}"

    assert u.m * Scaled() == "scaled by m"  # an unrecognised operand defers

    seqs: list[Any] = [[1.0, 2.0], (1.0, 2.0), "x"]  # sequences defer
    for seq in seqs:
        for unit in (u.m, u.mol):  # dimensionless units too
            with pytest.raises(TypeError, match="can't multiply sequence"):
                _ = unit * seq
            with pytest.raises(TypeError, match="can't multiply sequence"):
                _ = seq * unit
            with pytest.raises(TypeError, match="unsupported operand"):
                _ = seq / unit


def test_unit_identity():
    # units are equal when their reduced factors are, however they were built
    for unit in (
        (u.eV / u.angstrom) * u.angstrom,
        (u.eV / u.angstrom**2) * u.angstrom**2,
    ):
        assert unit == u.eV and hash(unit) == hash(u.eV) and str(unit) == "eV"
        assert unit.scale == u.eV.scale  # rounded once from the exact product
    assert u.J != u.N * u.m  # a different spelling is a different unit
    # an atom is its name, dimension and scale together
    assert Unit(((Atom("eV", u.eV.dim, u.eV.scale), Fraction(1)),)) == u.eV
    assert Unit(((Atom("eV", u.eV.dim, 2 * u.eV.scale), Fraction(1)),)) != u.eV


def test_reciprocal_unit():
    # 1/u.eV is the reciprocal unit itself; any other numerator is a quantity
    inverse = 1.0 / (1.0 * u.eV)
    assert 1 / u.eV == u.eV**-1  # the two spellings agree
    assert isinstance(1 / u.eV, units.Unit)
    assert float(inverse.to(1 / u.eV)) == 1.0
    # only Literal[1] declares a unit
    assert float((2 / u.eV).to(u.eV**-1)) == 2.0
    assert float((1.0 / u.eV).to(u.eV**-1)) == 1.0
    assert not isinstance(1 / u.mol, units.Unit)  # an empty dimension stays a number
    with pytest.raises(TypeError, match=r"must be a Unit.*without magnitudes"):
        inverse.to(2 / u.eV)  # pyrefly: ignore[bad-argument-type]


def test_power_units():
    # thermal conductivity, W/(m K), reads and converts back unchanged
    k = u.parse("0.6 W/m/K")
    assert units.dim_of(k) == (1, 1, -3, 0, -1)
    assert units.dim_of(u.W) == units.dim_of(u.J / u.s)
    assert float(k.to(u.W / u.m / u.K)) == pytest.approx(0.6)
    assert float(u.parse("3 mW").to(u.W)) == pytest.approx(3e-3)
    assert float(u.parse("2 kW").to(u.J / u.s)) == pytest.approx(2e3)


def test_parse():
    # unit() resolves a spelling, parse() a value, require_dim() a dimension
    for name in ("angstrom", "Angstrom", "Ang", "Å", "\u212b"):
        assert u.unit(name) is u.angstrom
    assert u.unit("A") is u.A and u.unit("ampere") is u.A  # A is the ampere
    assert u.unit("e") is u.e and u.unit("Da") is u.amu
    assert u.unit("Ry") is u.rydberg and u.unit("D") is u.debye
    reads: list[tuple[str, object, float]] = [
        ("300 K", u.K, 300.0),
        ("2 fs", u.fs, 2.0),
        ("10 bar", u.bar, 10.0),
        ("4.57e-10 1/Pa", u.Pa**-1, 4.57e-10),
        ("0.001 eV/angstrom", u.eV / u.angstrom, 0.001),
        ("3.405 Å", u.angstrom, 3.405),
        ("1e6 Pa", u.Pa, 1e6),
        ("-2 angstrom**-3", u.angstrom**-3, -2.0),
        ("1 eV.angstrom", u.eV * u.angstrom, 1.0),
        # a bracket groups a product, so a symbol's single divisor reads back
        ("1 m^2.kg/(s^2.K)", u.J / u.K, 1.0),
        ("4 (eV/angstrom)^2", (u.eV / u.angstrom) ** 2, 4.0),
    ]
    for text, unit, want in reads:
        assert float(u.parse(text).to(unit)) == pytest.approx(want), text
    assert float(u.parse("300")) == 300.0  # a bare number is dimensionless
    assert not units.is_quantity(u.parse("300"))
    # a rational power reads back what a base's own symbol writes
    symbol = units.ATOMISTIC.symbol(units.dim_of(u.s))
    assert float(u.parse(f"1 {symbol}").to(u.s)) == pytest.approx(ASE_TIME, rel=1e-15)
    bad = [
        ("K", "must start with a number"),
        ("1 furlong", "'furlong' is not a unit"),
        ("3 eV/", "ends after an operator"),
        ("2 eV%m", "'%m' is not part"),
        ("5 eV^x", "'x' is not an exponent"),
        ("2 3 eV", "'3' is a number where"),
        ("1 (eV", "a '(' with no ')'"),
        ("1 eV)", "expected '*' or '/' before ')'"),
        ("1 (eV/)", "ends after an operator"),
    ]
    for text, message in bad:
        with pytest.raises(units.DimensionError, match=re.escape(message)):
            u.parse(text)
    assert u.require_dim(2 * u.fs, u.fs) is None
    with pytest.raises(units.DimensionError, match=r"in eV, got one in K"):
        u.require_dim(300 * u.K, u.eV)
    with pytest.raises(units.DimensionError, match=r"in K, got one in no unit at all"):
        u.require_dim(300.0, u.K)


def test_si_prefixes():
    assert u.unit("km") is u.km and u.unit("kg") is u.kg and u.unit("fs") is u.fs
    assert u.unit("um") is u.unit("µm") is u.unit("\u03bcm") is u.unit("microm")
    assert u.unit("um").scale == 1e-6 and u.unit("mg").scale == 1e-6  # exactly 1e-6
    reads: list[tuple[str, object, float]] = [
        ("1 Mm", u.m, 1e6),
        ("2 meV", u.eV, 2e-3),
        ("1 pm", u.angstrom, 1e-2),
        ("1 kbar", u.bar, 1e3),
        ("3 mÅ", u.angstrom, 3e-3),
        ("1 mA", u.A, 1e-3),
        ("1 kilom", u.m, 1e3),
        ("1 GHz", u.Hz, 1e9),
        ("1 nN.pm", u.eV, 6.241509074e-3),
    ]
    for text, unit, want in reads:
        assert float(u.parse(text).to(unit)) == pytest.approx(want), text
    for text, message in (
        ("1 mkm", "carries a prefix already"),
        ("1 kfurlong", "'kfurlong' is not a unit"),
        ("1 hbar", "reads as it rather than as a hectobar"),
        ("1 hectobar", "reads as it rather than as a hectobar"),
    ):
        with pytest.raises(units.DimensionError, match=re.escape(message)):
            u.parse(text)
    with units.base(length=u.unit("pm"), energy=u.unit("meV")):
        assert str(1.0 * u.angstrom) == "100.0 pm"
        assert float(u.parse(str(1.0 * u.angstrom)).to(u.angstrom)) == pytest.approx(
            1.0
        )


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


def test_testing_assertions():
    # tolerances are over magnitudes in a unit the caller names, never raw magnitudes,
    # which would mean a different distance in each base
    r = arr([1.0, 2.0]) * u.angstrom
    ut.assert_allclose(r, arr([0.1, 0.2]) * u.nm, u.angstrom)
    ut.assert_allclose(r, arr([1.0, 2.0]), u.angstrom)  # magnitudes
    ut.assert_allclose(2.0 * u.eV, 2.0 + 1e-9, u.eV)
    ut.assert_array_equal(r, r.rebase(units.SI), u.angstrom)
    ut.assert_array_equal(arr([1.0, 2.0]), r, u.angstrom)
    with pytest.raises(AssertionError, match="magnitudes in Å"):
        ut.assert_allclose(r, arr([1.0, 2.1]) * u.angstrom, u.angstrom)
    with pytest.raises(AssertionError, match=r"2\.1e-10"):
        # both sides are quoted in the unit the caller named, not in the base
        ut.assert_array_equal(r, arr([1.0, 2.1]) * u.angstrom, u.m)
    ut.assert_allclose(r, arr([1.0, 2.1]) * u.angstrom, u.angstrom, rtol=0.1)
    ut.assert_allclose(r, arr([1.0, 2.1]) * u.angstrom, u.angstrom, rtol=0.0, atol=0.2)
    # a dimension mismatch is caught on the way in, by .to(unit) itself
    with pytest.raises(units.DimensionError, match="cannot express"):
        ut.assert_allclose(r, r, u.eV)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(units.DimensionError, match="neither side carries"):
        ut.assert_allclose(arr([1.0]), arr([1.0]), u.angstrom)
    with pytest.raises(TypeError, match=r"must be a units\.Unit"):
        ut.assert_allclose(r, r, "angstrom")  # pyrefly: ignore[bad-argument-type]
    # numpy's own assertion refuses a quantity and points at units.testing
    with pytest.raises(units.DimensionError, match=r"units\.testing"):
        npt.assert_allclose(r, r)


def test_deferred_annotations():
    # the exponents in the signatures only exist under TYPE_CHECKING, so the
    # annotations must stay strings
    hints = ut.assert_allclose.__annotations__
    assert hints and all(isinstance(h, str) for h in hints.values())


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
