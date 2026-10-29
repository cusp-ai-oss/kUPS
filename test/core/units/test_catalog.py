# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unit catalogue, constants, assertions, and documented limits."""

import pathlib
import re
import warnings
from decimal import Decimal
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir

from kups.core import units
from kups.core.units import (
    Dimensionless,
    Unit,
    constants,
)
from kups.core.units import catalog as u
from kups.core.units import testing as ut
from kups.core.units.catalog import angstrom, eV

from ._helpers import (
    ASE_TIME,
    arr,
    jit,
    make_jaxpr,
    ty,
    val,
    x64_mode,
)


def test_zero_overhead():
    # quantities lower to the same HLO as plain-array arithmetic
    def dot_q(a: Any, b: Any) -> Any:
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
    # unit algebra stays Unit, and a cancelled dimension becomes Dimensionless
    assert isinstance(u.eV / u.angstrom, Unit)
    assert (u.eV / u.angstrom).dim == (1, 1, -2, 0, 0)
    # names canonicalize factor order, so how a unit was built does not show
    assert u.eV * u.eV == u.eV**2 and str(u.eV * u.eV) == "eV^2"
    assert str(u.angstrom * u.amu) == str(u.amu * u.angstrom) == "amu.Å"
    assert str((u.eV / u.angstrom) ** 2) == "eV^2/Å^2"  # and it reads back
    assert float(u.parse("1 eV^2/Å^2").to((u.eV / u.angstrom) ** 2)) == 1.0
    assert str(1 / (u.eV * u.angstrom)) == "1/(eV.Å)"  # one slash, one group
    assert str(u.eV / u.eV) == "1" and (u.eV / u.eV).dim == (0, 0, 0, 0, 0)
    # Dimensionless is a runtime subclass; the stubs spell it as an alias of Unit[0, ...]
    dimensionless: Any = Dimensionless
    assert isinstance(u.mol, dimensionless) and not isinstance(u.m, dimensionless)
    assert isinstance(u.m / u.m, dimensionless) and isinstance(u.m * u.mol, Unit)
    # unit times magnitudes attaches, in either order, and so does the quotient
    assert val(u.angstrom * 3.0) == 3.0 and val(3.0 * u.angstrom) == 3.0
    assert list((u.angstrom * arr([1.0, 2.0])).value) == [1.0, 2.0]
    assert val(u.angstrom / 4.0) == 0.25 and val(4.0 / u.angstrom) == 4.0
    q = 3.0 * u.eV
    assert val(u.eV * q) == val(q * u.eV) and float(u.eV / q) == pytest.approx(1 / 3)
    text: Any = "x"  # Any: a static checker rejects this too
    with pytest.raises(TypeError, match="not a valid JAX array type"):
        _ = u.m * text


def test_constant_read_in_trace():
    # a constant first read inside jit is built eagerly; caching a tracer instead
    # would make the next eager use recurse until the stack ran out
    name = "k_B"
    constants._CACHE.pop((name, jnp.result_type(float), units.active_base()), None)

    @jax.jit
    def thermal(t: Any) -> Any:
        return constants.k_B * t

    assert float(thermal(300.0 * u.K).to(u.eV)) == pytest.approx(0.025852, rel=1e-4)
    cached = constants.k_B
    assert isinstance(cached, units.Quantity)  # not a tracer
    assert float((cached * 21).to(u.eV / u.K)) == pytest.approx(
        21 * 8.617333e-5, rel=1e-5
    )


def test_constant_cache_distinguishes_bases_with_the_same_name(monkeypatch: Any):
    monkeypatch.setattr(constants, "_CACHE", {})
    metre = units.BaseSystem("custom", units.SI.pins, units.SI.labels)
    kilometre = units.BaseSystem(
        "custom", (u.km, *metre.pins[1:]), ("km", *metre.labels[1:])
    )
    for enabled in (False, True):
        with x64_mode(enabled):
            for bs, expected in ((metre, 299792458.0), (kilometre, 299792.458)):
                with units.base(bs):
                    c = constants.c
                    assert c.bs == bs
                    assert c.dtype == jnp.result_type(float)
                    assert float(c.value) == pytest.approx(expected)
                    assert constants.c is c
                    equivalent = units.BaseSystem("renamed", bs.pins, bs.labels)
                    with units.base(equivalent):
                        assert constants.c is c
    assert len(constants._CACHE) == 4


def test_constant_rounding():
    # the algebra is exact rationals, so the only rounding is the final one
    exact = {  # value in SI base units, to well past float64
        "c": "299792458",
        "hbar": "1.05457181764615639e-34",
        "eps_0": "8.85418781277334739e-12",
        "k_e": "8987551792.28822621535",
    }
    with units.base(units.SI):
        for name, text in exact.items():
            want = float(Decimal(text))
            assert float(getattr(constants, name).value) == want, name

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with units.base(units.ATOMISTIC):
            # 14.399645478469015956... in eV.angstrom/e^2, so float64 lands here
            assert float(constants.k_e.value) == 14.399645478469015
            assert float(constants.e.value) == 1.0  # the base pins the charge
        assert not [w for w in caught if "kups.core.units:" in str(w.message)]


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
    wrong: Any = 2 / u.eV
    with pytest.raises(TypeError, match=r"must be a Unit.*without magnitudes"):
        inverse.to(wrong)


def test_charge_and_dipole_units():
    # e, rydberg, invcm and debye, each against its defining expression
    assert u.e.scale == 1.602176634e-19  # SI-2019 exact, as eV is
    assert u.rydberg.scale == pytest.approx(u.hartree.scale / 2, rel=1e-14)
    assert u.invcm.scale == pytest.approx(
        float(constants.h.to(u.J * u.s)) * 299792458.0 / u.cm.scale, rel=1e-15
    )
    assert u.debye.scale == 1e-21 / 299792458.0
    assert units.dim_of(u.debye) == units.dim_of(u.e * u.angstrom)
    assert u.invcm.dim == u.rydberg.dim == u.eV.dim
    assert float((1.0 * u.invcm).to(u.eV)) == pytest.approx(1.2398419843320023e-4)
    assert float((1.0 * u.debye).to(u.e * u.angstrom)) == pytest.approx(
        0.20819433270935594
    )


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
    reads: list[tuple[str, Unit, float]] = [
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


def test_constants():
    # the constants agree with the unit catalogue
    assert float(constants.m_u.to(u.kg)) == u.amu.scale  # CODATA 2018, exactly
    assert float(constants.a_0.to(u.m)) == u.bohr.scale
    assert float(constants.E_h.to(u.J)) == u.hartree.scale
    ratio = constants.k_e / (1.0 * u.hartree * u.bohr / u.e**2)
    assert float(ratio) == pytest.approx(1.0, rel=1e-12)  # 8.9e-13: float64 noise
    assert float(constants.c.to(u.m / u.s)) == 299792458.0
    assert float(constants.h.to(u.J * u.s)) == 6.62607015e-34
    assert float(constants.hbar.to(u.J * u.s)) == 1.0545718176461565e-34
    assert float(constants.e.to(u.e)) == 1.0
    # stored in eV/K, so this is the nearest float64 to the exact ratio
    assert float(constants.k_B.to(u.eV / u.K)) == 8.617333262145177e-05
    assert (constants.N_A, constants.alpha) == (6.02214076e23, 7.2973525693e-3)
    # the derived two each equal the nearest float64 to the exact rational
    with units.base(units.SI):
        assert val(constants.eps_0) == 8.854187812773347e-12
        assert val(constants.k_e) == 8987551792.288227
    assert units.dim_of(constants.k_B * (300.0 * u.K)) == units.dim_of(u.eV)
    assert units.dim_of(constants.eps_0) == units.dim_of(u.C**2 / (u.J * u.m))
    assert [n for n in dir(constants) if not n.startswith("_")] == [
        "E_h", "N_A", "a_0", "alpha", "c", "e", "eps_0", "h", "hbar", "k_B", "k_e",
        "m_e", "m_u",
    ]  # fmt: skip


def test_si_prefixes():
    assert u.unit("km") is u.km and u.unit("kg") is u.kg and u.unit("fs") is u.fs
    assert u.unit("um") is u.unit("µm") is u.unit("\u03bcm") is u.unit("microm")
    assert u.unit("um").scale == 1e-6 and u.unit("mg").scale == 1e-6  # exactly 1e-6
    reads: list[tuple[str, Unit, float]] = [
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
    prefixed_count = u.parse("2 mmol")  # a prefixed count is still a count
    assert not units.is_quantity(prefixed_count)
    assert float(prefixed_count) == pytest.approx(2e-3 * constants.N_A)
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
    untyped: Any = r
    with pytest.raises(units.DimensionError, match="cannot express"):
        ut.assert_allclose(untyped, untyped, u.eV)
    with pytest.raises(units.DimensionError, match="neither side carries"):
        ut.assert_allclose(arr([1.0]), arr([1.0]), u.angstrom)
    with pytest.raises(TypeError, match=r"must be a units\.Unit"):
        not_a_unit: Any = "angstrom"
        ut.assert_allclose(untyped, untyped, not_a_unit)
    # numpy's own assertion refuses a quantity and points at units.testing
    with pytest.raises(units.DimensionError, match=r"units\.testing"):
        npt.assert_allclose(untyped, untyped)


def test_documented_limits():
    # the hard limits, each with the failure it names
    r = arr([1.0, 2.0]) * u.angstrom
    # the jaxpr carries no unit and matches the plain-array one exactly
    text = str(make_jaxpr(lambda x: (x * x).sum())(r))
    assert text == str(make_jaxpr(lambda x: jnp.sum(x * x))(r.value))
    assert "Quantity" not in text and "Å" not in text
    assert ty(r) == "float64[2]{Å|atomistic}"
    # a quantity never converts to an array implicitly, by either route
    with pytest.raises(units.DimensionError, match="does not convert to a plain array"):
        np.asarray(r)
    with pytest.raises((TypeError, units.DimensionError)):
        jnp.asarray(r)

    doubled = Primitive("units_test_double")
    doubled.def_abstract_eval(lambda x: ShapedArray(x.shape, x.dtype))
    doubled.def_impl(lambda x: 2.0 * x)
    mlir.register_lowering(
        doubled, mlir.lower_fun(lambda x: 2.0 * x, multiple_results=False)
    )

    def public_double(x: Any) -> Any:
        return doubled.bind(x)

    # a foreign primitive binds arrays; apply hands it the magnitudes
    with pytest.raises(TypeError, match=r"units\._core\.Quantity"):
        jax.jit(public_double)(r)
    npt.assert_allclose(units.apply(public_double, r).to(u.angstrom), [2.0, 4.0])
    traced = jax.jit(lambda q: units.apply(public_double, q))(r)
    npt.assert_allclose(traced.to(u.angstrom), [2.0, 4.0])
    assert units.dim_of(traced) == units.dim_of(u.angstrom)

    def stretch(x: Any) -> Any:
        return (x * x).sum().to(u.angstrom**2)

    # jax's reverse mode labels the cotangent with the primal's dimension,
    # not the derivative's, so use kups.core.units' spellings
    assert ty(jax.grad(stretch)(r)) == "float64[2]{Å|atomistic}"
    assert ty(units.grad(stretch)(r)) == "float64[2]{1/Å|atomistic}"
    assert ty(jax.vjp(stretch, r)[1](1.0)[0]) == "float64[2]{Å|atomistic}"
    pull = units.vjp(stretch, r)[1]
    assert ty(pull(1.0)[0]) == "float64[2]{1/Å|atomistic}"

    @jax.custom_vjp
    def measured(x: Any) -> Any:
        return x.sum().to(u.angstrom)

    def measured_fwd(x: Any) -> tuple[Any, None]:
        return measured(x), None

    def measured_bwd(_res: None, _ct: Any) -> tuple[Any]:
        # a cotangent whose dimension differs from the primal's
        return (r * (1.0 * u.eV),)

    measured.defvjp(measured_fwd, measured_bwd)
    # the treedef carries the dimension, so jax's own structure check catches this
    with pytest.raises(TypeError, match=r"same container \(pytree\) structure"):
        jax.grad(measured)(r)


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
