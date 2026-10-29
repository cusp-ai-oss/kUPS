# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for quantities, dimensions, and base systems."""

import asyncio
import itertools
import math
import pathlib
import pickle
import subprocess
import sys
import types
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from kups.core import units
from kups.core.units import (
    Quantity,
    Unit,
    constants,
)
from kups.core.units import catalog as u

from ._helpers import (
    ASE_TIME,
    arange,
    arr,
    eye,
    jit,
    make_jaxpr,
    ones,
    ty,
    ulps,
    val,
)


def test_arithmetic():
    assert val(1 * u.nm + 5 * u.angstrom) == 15.0
    assert val(2 * u.amu * (3 * u.eV / u.angstrom)) == 6.0
    assert val((6 * u.eV) / (2 * u.angstrom)) == 3.0
    assert val(-(3 * u.angstrom)) == -3.0
    assert val(2.0 * (3 * u.angstrom)) == 6.0
    assert val((2 * u.angstrom) ** 3) == 8.0
    assert list((arr([1.0, 2.0, 3.0]) * u.angstrom)[1:].value) == [2.0, 3.0]
    # dimensionless results decay to plain arrays
    assert not units.is_quantity((1 * u.m) / (2 * u.m))
    assert float(jnp.exp((1 * u.m) / (2 * u.m))) == pytest.approx(1.6487212707001282)
    assert float(2 * u.mol) == pytest.approx(1.204428152e24)
    # comparisons give plain bools
    assert bool(3 * u.m > 1 * u.m) and not bool(3 * u.m < 1 * u.m)
    mass: Any = 1 * u.kg  # Any: a checker rejects this
    with pytest.raises(units.DimensionError):
        _ = 3 * u.m > mass


def test_jit_label_round_trip():
    # the label keys jit's cache but never appears in the jaxpr
    traced: list[str] = []

    def ke(m: Any, v: Any) -> Any:
        traced.append(ty(m))
        return 0.5 * m * v**2

    speed = (u.eV / u.amu) ** Fraction(1, 2)  # the atomistic velocity unit
    f = jit(ke)
    out = f(2.0 * u.amu, 3.0 * speed)
    assert val(out) == 9.0
    assert units.dim_of(out) == units.dim_of(u.J)
    assert ty(out) == "float64[]{eV|atomistic}"  # base preserved too
    assert val(f(4.0 * u.amu, 5.0 * speed)) == 50.0  # same labels: cache hit
    f(2.0 * u.angstrom, 3.0 * speed)  # different dimension: different key
    assert traced == ["float64[]{amu|atomistic}", "float64[]{Å|atomistic}"]
    # a quantity is one pytree leaf, and the jaxpr matches the plain-array one
    labelled = make_jaxpr(ke)(2.0 * u.amu, 3.0 * speed)
    plain = make_jaxpr(ke)(arr(2.0), arr(3.0))
    assert str(labelled) == str(plain) and len(labelled.eqns) == len(plain.eqns) == 3


def test_dimension_errors():
    def bad(m: Any, v: Any) -> Any:
        return m + v

    args = (2.0 * u.amu, 3.0 * u.eV / u.angstrom)
    for fn in (bad, jit(bad)):  # same message eagerly and under jit
        with pytest.raises(units.DimensionError) as e:
            fn(*args)
        msg = str(e.value)
        assert "cannot add 'amu' and 'eV/Å'" in msg
        assert "(m=0, kg=1, s=0, A=0, K=0)" in msg
    metre: Any = u.m
    with pytest.raises(units.DimensionError, match="cannot express 'eV' in m"):
        (1.0 * u.J).to(metre)
    # Python rewrites this as __gt__, so the message must not name an operator
    plain: Any = arr([1.0, 2.0])
    with pytest.raises(units.DimensionError) as e:
        _ = plain < (3.0 * u.m)
    assert "cannot compare" in str(e.value) and "a dimensionless value" in str(e.value)


def test_fractional_exponents():
    assert val(units.sqrt(4.0 * u.angstrom**2)) == 2.0
    assert units.dim_of(units.sqrt(4.0 * u.m**2)) == (1, 0, 0, 0, 0)
    assert units.dim_of(units.sqrt(2.0 * u.m)) == (Fraction(1, 2), 0, 0, 0, 0)
    assert ty(units.sqrt(2.0 * u.angstrom)) == "float64[]{Å^1/2|atomistic}"
    assert units.dim_of(units.sqrt(2.0 * u.m) ** 2) == (1, 0, 0, 0, 0)
    assert units.dim_of((3.0 * u.m) ** Fraction(3, 2)) == (
        Fraction(3, 2),
        0,
        0,
        0,
        0,
    )
    assert units.dim_of(u.m) == (1, 0, 0, 0, 0)  # a Unit reports its own dimension
    assert units.dim_of(u.eV / u.angstrom) == (1, 1, -2, 0, 0)
    assert units.dim_of(3.0) == (0, 0, 0, 0, 0)

    def power(q: Any, p: Any) -> Any:
        return q**p

    with pytest.raises(units.DimensionError, match="must be a static int or Fraction"):
        jit(power)(2.0 * u.m, 2.0)  # traced exponent


def test_default_base_is_atomistic():
    assert units.active_base() is units.ATOMISTIC
    assert ty(1 * u.nm) == "float64[]{Å|atomistic}"
    assert val(1 * u.nm + 5 * u.angstrom) == 15.0
    # every pinned unit is exactly one
    for unit in (u.angstrom, u.amu, u.eV, u.e, u.K):
        assert val(1 * unit) == 1.0
    assert units.SI.base_units() == {
        "m": 1.0,
        "kg": 1.0,
        "s": 1.0,
        "A": 1.0,
        "K": 1.0,
    }
    with units.base(units.SI):
        assert ty(1 * u.km) == "float64[]{m|SI}"
        assert val(1 * u.km + 5 * u.m) == 1005.0


@pytest.mark.parametrize("field", ["pins", "labels", "name"])
def test_base_metadata_is_immutable(field: str):
    bs = units.BaseSystem("custom", u.SI_BASE, ("m", "kg", "s", "A", "K"))
    original_hash = hash(bs)
    replacements = {
        "pins": (u.km, *bs.pins[1:]),
        "labels": ("km", *bs.labels[1:]),
        "name": "changed",
    }
    with pytest.raises(AttributeError):
        setattr(bs, field, replacements[field])
    assert bs.scale(u.m.dim) ** 2 == bs.scale((u.m**2).dim)
    assert bs == units.SI and hash(bs) == original_hash


def test_base_owns_its_metadata():
    pins: Any = list(u.SI_BASE)
    labels: Any = ["m", "kg", "s", "A", "K"]
    bs = units.BaseSystem("custom", pins, labels)
    pins[0], labels[0] = u.km, "km"
    assert bs.pins == u.SI_BASE
    assert bs.labels[0] == "m"
    assert bs.scale(u.m.dim) ** 2 == bs.scale((u.m**2).dim)


def test_base_hash_after_unpickling_in_another_process():
    custom = units.BaseSystem(
        "custom", (u.km, *units.SI.pins[1:]), ("km", *units.SI.labels[1:])
    )
    probe = """
import pickle
import sys
from kups.core import units
from kups.core.units import catalog as u

custom = units.BaseSystem(
    "custom", (u.km, *units.SI.pins[1:]), ("km", *units.SI.labels[1:])
)
for restored, local in zip(
    pickle.loads(sys.stdin.buffer.read()), (units.SI, units.ATOMISTIC, custom),
    strict=True,
):
    assert restored == local
    assert hash(restored) == hash(local)
    assert {restored: "found"}[local] == "found"
    assert len({restored, local}) == 1
"""
    out = subprocess.run(
        [sys.executable, "-c", probe],
        input=pickle.dumps((units.SI, units.ATOMISTIC, custom)),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert out.returncode == 0, out.stderr.decode()


def test_reconfigured_base():
    e = 5 * u.kcal / u.mol + 1 * u.eV
    assert ty(e) == "float64[]{eV|atomistic}"
    assert val(e) == pytest.approx(1.2168205212090046, abs=0, rel=1e-15)
    with units.base(units.SI):
        e = 5 * u.kcal / u.mol + 1 * u.eV
        assert ty(e) == "float64[]{J|SI}"
        assert val(e) == pytest.approx(
            5 * 4184 / 6.02214076e23 + 1.602176634e-19, abs=0, rel=1e-15
        )
        assert val(1 * u.km + 5 * u.m) == 1005.0
        assert val(1 * u.J) == 1.0  # a pinned unit is exact
    assert units.active_base() is units.ATOMISTIC  # scope unwound
    # spelling the preset by hand gives the same base, sharing its jit cache
    with units.base(mass=u.amu, energy=u.eV, length=u.angstrom, charge=u.e) as bs:
        assert bs == units.ATOMISTIC
    # base identity is the induced scales, so these all resolve to SI itself
    for kw in ({"force": u.N}, {"energy": u.J}, {"len2": u.m**2}):
        with units.base(**kw) as bs:
            assert bs is units.SI
    with units.base(len2=u.angstrom**2, mass=u.amu, energy=u.eV, charge=u.e):
        alt = 1.0 * u.eV
    with units.base(units.ATOMISTIC):
        native = 1.0 * u.eV
    assert ty(alt) == ty(native)
    assert float((alt + native).to(u.eV)) == 2.0


def test_base_solver():
    # pinning length, mass and energy yields ASE's time unit in any pin order
    pins = {"length": u.angstrom, "mass": u.amu, "energy": u.eV}
    for perm in itertools.permutations(pins.items()):
        with units.base(**dict(perm)) as bs:
            assert bs.base_units()["s"] == ASE_TIME
            # scale is a homomorphism: two routes to force give identical bits
            force = bs.scale((1, 1, -2, 0, 0))
            assert force == bs.scale((2, 1, -2, 0, 0)) / bs.scale((1, 0, 0, 0, 0))
            assert force == 1.6021766339999998e-09
    assert float((1 * u.eV / u.angstrom).to(u.N)) == 1.6021766339999998e-09
    # any dimensionally independent units work, not just known kinds
    with units.base(len2=u.angstrom**2, mass=u.amu, energy=u.eV) as bs:
        assert bs.base_units()["s"] == ASE_TIME
    # each pin's scale is reproduced exactly
    for p in units.ATOMISTIC.pins:
        assert units.ATOMISTIC.scale(p.dim) == p.scale
    for e1 in itertools.product(range(-1, 2), repeat=3):
        for e2 in itertools.product(range(-1, 2), repeat=3):
            d1, d2 = (*e1, 0, 0), (*e2, 0, 0)
            ds = tuple(a + b for a, b in zip(d1, d2, strict=True))
            assert units.SI.scale(ds) == units.SI.scale(d1) * units.SI.scale(d2)
            assert (
                ulps(
                    units.ATOMISTIC.scale(ds),
                    units.ATOMISTIC.scale(d1) * units.ATOMISTIC.scale(d2),
                )
                <= 2
            )


def test_incoherent_base():
    for order in (
        ["length", "mass", "energy", "time"],
        ["time", "energy", "mass", "length"],
    ):
        kw = {"length": u.angstrom, "mass": u.amu, "energy": u.eV, "time": u.fs}
        with pytest.raises(units.DimensionError) as e:
            units.base(**{k: kw[k] for k in order}).__enter__()
        msg = str(e.value)
        assert all(f"={n}" in msg for n in ("Å", "amu", "eV", "fs"))
        assert "not 1" in msg and "is dimensionless" in msg
    with pytest.raises(units.DimensionError, match="energy must be"):
        units.base(energy=u.m).__enter__()  # a known kind must match its dimension


def test_dependent_base():
    with pytest.raises(
        units.DimensionError, match=r"\(length=m\)\^-1 \* \(length2=km\)\^1"
    ):
        units.base(length=u.m, length2=u.km).__enter__()
    with pytest.raises(units.DimensionError, match="drop one of them"):
        units.base(length=u.m, time=u.s, speed=u.m / u.s).__enter__()
    with pytest.raises(units.DimensionError):  # six units cannot be independent
        units.base(a=u.m, b=u.kg, c=u.s, d=u.A, e=u.K, f=u.J).__enter__()


def test_order_independence():
    # bitwise identical regardless of operand order
    assert val(1 * u.km + 5 * u.m) == val(5 * u.m + 1 * u.km)
    assert val(5 * u.kcal / u.mol + 1 * u.eV) == val(1 * u.eV + 5 * u.kcal / u.mol)
    with units.base(units.SI):
        assert val(1 * u.km + 5 * u.m) == val(5 * u.m + 1 * u.km)
        assert val(5 * u.kcal / u.mol + 1 * u.eV) == val(1 * u.eV + 5 * u.kcal / u.mol)


def test_jit_cache_key():
    # dimension and base key jit's cache; the unit, the value and the ambient base do not
    traces: list[str] = []

    @jit
    def square(q: Any) -> Any:
        traces.append(ty(q))
        return q * q

    at = 1.0 * u.eV
    with units.base(units.SI):
        si = 1.0 * u.eV
    assert val(square(2.0 * u.angstrom)) == 4.0
    assert val(square(3.0 * u.nm)) == 900.0  # the same label, so no second trace
    assert (
        val(square(1.0 * u.angstrom * (2.0 * u.angstrom) / (1.0 * u.angstrom))) == 4.0
    )  # built by arithmetic
    assert val(square(at)) == 1.0  # a different dimension retraces
    assert val(square(si)) == 1.602176634e-19**2  # and so does a different base
    with units.base(units.SI):
        assert val(square(2.0 * u.J)) == 4.0  # same label hits the cache
    assert traces == [
        "float64[]{Å|atomistic}",
        "float64[]{eV|atomistic}",
        "float64[]{J|SI}",
    ]
    # identical physics despite different magnitudes and types
    assert float(square(si).to(u.eV**2)) == pytest.approx(float(square(at).to(u.eV**2)))
    # different bases combine in the active base
    for lhs, rhs in ((si, at), (at, si)):
        s = lhs + rhs
        assert units.base_of(s) == units.active_base()
        assert float(s.to(u.eV)) == pytest.approx(2.0)
    assert val(si + at) == val(at + si)
    assert float(at.rebase(units.SI).to(u.eV)) == 1.0
    assert val(at) == 1.0  # the original value is unchanged

    ambient: list[str] = []

    @jit
    def build(x: Any) -> Any:
        ambient.append(units.active_base().name)
        return x * u.eV

    assert val(build(1.0)) == 1.0
    with units.base(units.SI):
        assert val(build(1.0)) == 1.0  # still the first trace's result
    assert ambient == ["atomistic"]


def test_iteration():
    # list(q) must not fall back to the sequence protocol (jax clamps indices)
    q = arr([1.0, 2.0, 3.0]) * u.angstrom
    assert len(q) == 3
    assert [val(x) for x in q] == [1.0, 2.0, 3.0]
    # length is static, so len works inside a trace
    assert val(jit(lambda x: (x * 0 + len(x) * u.angstrom).sum())(q)) == 9.0
    with pytest.raises(TypeError, match="0-d quantity"):
        len(1.0 * u.m)


def test_complex_rejected():
    with pytest.raises(units.DimensionError, match="complex"):
        _ = arr([1.0 + 2.0j]) * u.m
    # operators that would produce complex magnitudes are refused as well
    q = arr([3.0, 6.0]) * u.m
    for build in (
        lambda: q * 1j,
        lambda: q / 1j,
        lambda: arr([1.0j, 2.0j]) * q,
        lambda: jit(lambda z: z * 1j)(q),
    ):
        with pytest.raises(units.DimensionError, match="complex"):
            build()
    assert (q / q * 1j).dtype == jnp.complex128  # dimensionless is a plain array


def test_exact_exponents():
    # int/Fraction exponents are never snapped; bad floats raise
    assert units.dim_of((2.0 * u.m) ** Fraction(1, 128))[0] == Fraction(1, 128)
    assert units.dim_of((2.0 * u.m) ** Fraction(3, 100))[0] == Fraction(3, 100)
    with pytest.raises(units.DimensionError, match="not a small rational"):
        _ = (2.0 * u.m) ** 0.51
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(units.DimensionError, match="has to be finite"):
            _ = (2.0 * u.m) ** bad
        with pytest.raises(units.DimensionError, match="has to be finite"):
            _ = u.m**bad
    # both cube-root spellings give the same dimension, so they are addable
    root = 1 * (u.m ** Fraction(1, 3)) + (1 * u.m) ** Fraction(1, 3)
    assert float(root.to(u.m ** Fraction(1, 3))) == pytest.approx(2.0, rel=1e-15)


def test_plain_times_quantity():
    def left(z: Any) -> Any:
        return eye(2) * z

    def right(z: Any) -> Any:
        return z * eye(2)

    def mat(z: Any) -> Any:
        return eye(2) @ z

    q = ones((2, 2)) * u.m
    assert ty(jit(left)(q)) == ty(jit(right)(q))
    assert jit(mat)(q).shape == (2, 2)


def test_transpose():
    # transpose is registered on Quantity, so jit/vmap compositions work
    def times_s(v: Any) -> Any:
        return v * (2.0 * u.s)

    def tr(z: Any) -> Any:
        return z.transpose()

    xs = arange(6.0).reshape(2, 3) * u.m
    assert jit(jax.vmap(times_s, out_axes=1))(xs).shape == (3, 2)
    assert jit(tr)(xs).shape == (3, 2)


def test_eq_and_grad_rejected():
    def vecout(z: Any) -> Any:
        return (z / (1 * u.m)).sum() * ones(3)

    with pytest.raises(units.DimensionError, match="=="):
        # the declared return type of == is Never
        _ = (1.0 * u.m) == (1.0 * u.m)
    q = arr([1.0, 2.0, 3.0]) * u.m
    with pytest.raises(TypeError, match="scalar output"):
        units.grad(vecout)(q)


def test_cold_warm_agree():
    # a jit cache primed under another base must not leak into this one
    def mixed(x: Any) -> Any:
        return x + 1.0 * u.eV

    with units.base(units.SI):
        q = 1.0 * u.eV
    cold, warm = jit(mixed), jit(mixed)
    with units.base(units.SI):
        warm(q)
    assert ty(cold(q)) == ty(warm(q))
    assert val(cold(q)) == val(warm(q))


def test_async_scopes():
    # each task keeps its own base across an await
    async def inner(bs: Any) -> tuple[str, str]:
        with units.base(bs):
            before = units.active_base().name
            await asyncio.sleep(0)
            return before, units.active_base().name

    async def main() -> Any:
        return await asyncio.gather(inner(units.ATOMISTIC), inner(units.SI))

    assert asyncio.run(main()) == [("atomistic", "atomistic"), ("SI", "SI")]
    assert units.active_base() is units.ATOMISTIC  # the caller's base is untouched


def test_set_base():
    try:
        assert units.set_base(units.SI) is units.SI
        assert units.active_base() is units.SI
        assert val(1.0 * u.eV) == 1.602176634e-19
        with units.base(units.ATOMISTIC):  # a scope still wins inside its block
            assert val(1.0 * u.eV) == 1.0
        assert val(1.0 * u.eV) == 1.602176634e-19
    finally:
        units.set_base(units.ATOMISTIC)
    assert val(1.0 * u.eV) == 1.0


def test_base_validation():
    with pytest.raises(units.DimensionError, match="at least one pinned unit"):
        units.base().__enter__()
    with pytest.raises(units.DimensionError, match="at least one pinned unit"):
        units.set_base()
    with pytest.raises(units.DimensionError, match="fractional dimension"):
        units.base(rootlen=u.angstrom ** Fraction(1, 2)).__enter__()
    wrong: Any = "angstrom"  # Any: a checker rejects these spellings too
    not_a_base: Any = u.m
    with pytest.raises(
        TypeError, match="kups.core.units: length= must be a Unit, not str"
    ):
        units.base(length=wrong).__enter__()
    with pytest.raises(
        TypeError, match="kups.core.units: the base system must be a BaseSystem"
    ):
        units.base(wrong).__enter__()
    with pytest.raises(
        TypeError, match="kups.core.units: the unit to convert to must be a Unit"
    ):
        (1.0 * u.m).to(wrong)
    with pytest.raises(
        TypeError, match="kups.core.units: the base to rebase to must be"
    ):
        (1.0 * u.m).rebase(not_a_base)


def test_derived_symbols():
    # a derived unit with no catalogue name displays as a quotient
    at = units.ATOMISTIC
    assert at.symbol((1, 1, -2, 0, 0)) == "eV/Å"  # a force
    assert at.symbol((-1, 1, -2, 0, 0)) == "eV/Å^3"  # a stress
    assert at.symbol((0, 1, -2, 0, 0)) == "eV/Å^2"  # a hessian entry
    assert at.symbol((0, 0, 1, 0, 0)) == "Å.amu^1/2/eV^1/2"  # a time
    with units.base(at):  # repr uses the same symbol
        assert str(1 * u.fs) == "0.09822694750253276 Å.amu^1/2/eV^1/2"
    assert units.SI.symbol((1, 0, -1, 0, 0)) == "m/s"
    assert units.SI.symbol((0, 0, -1, 0, 0)) == "Hz"  # a catalogue name wins
    assert units.SI.symbol((1, 0, 0, 0, 0)) == "m"
    assert units.SI.symbol((0, 0, 0, 0, 0)) == ""  # dimensionless has no symbol


def test_is_quantity():
    seen: list[tuple[bool, bool]] = []

    def probe(z: Any) -> Any:
        seen.append((units.is_quantity(z), isinstance(z, Quantity)))
        return z * z

    q = 1.0 * u.m
    probe(q)
    jit(probe)(q)
    # under a trace the magnitudes are a tracer but the leaf is still a Quantity
    assert seen == [(True, True), (True, True)]
    seen.clear()
    jit(probe)(arr(2.0))  # a plain array is not a quantity
    assert seen == [(False, False)]
    assert not units.is_quantity(u.m) and not units.is_quantity(1.0)
    assert not units.is_quantity((1.0 * u.m) / (2.0 * u.m))  # dimensionless decays
    assert not hasattr(u, "one")


@pytest.mark.parametrize("inner_base", [units.SI, units.ATOMISTIC])
def test_scope_nesting(inner_base: units.BaseSystem):
    outer, inner = units.base(units.SI), units.base(inner_base)
    outer.__enter__()
    inner.__enter__()
    for cm in (outer, inner):  # exiting the outer scope first corrupts the stack
        with pytest.raises(units.DimensionError, match="not the innermost one"):
            cm.__exit__(None, None, None)
    assert units.active_base() is units.ATOMISTIC  # state was reset to the default
    assert val(1.0 * u.eV) == 1.0
    with units.base(units.SI):  # proper nesting still restores each level
        with units.base(inner_base):
            assert units.active_base() is inner_base
        assert units.active_base() is units.SI


def test_scope_restores_base_after_exception():
    with units.base(units.SI):
        with (
            pytest.raises(RuntimeError, match="inside the scope"),
            units.base(units.SI),
        ):
            raise RuntimeError("inside the scope")
        assert units.active_base() is units.SI
    assert units.active_base() is units.ATOMISTIC


def test_pin_validation():
    for scale in (0.0, -1e-10, math.inf, math.nan):
        bad = Unit("odd", units.dim_of(u.m), scale)
        with pytest.raises(units.DimensionError, match="positive finite size"):
            units.base(length=bad, mass=u.amu, energy=u.eV).__enter__()
    with pytest.raises(units.DimensionError, match="positive finite size"):
        units.base(size=u.angstrom**40).__enter__()  # underflows to 0.0


def test_saturated_scales():
    # a scale outside the float range raises instead of returning 0 or inf
    assert units.ATOMISTIC.scale((40, 0, 0, 0, 0)) == 0.0  # documented
    for unit in (u.J**18, u.m**40):
        with pytest.raises(units.DimensionError, match="no representable unit"):
            _ = 1.0 * unit


def test_iteration_under_jit():
    q = arr([1.0, 2.0]) * u.angstrom

    def unrolled(z: Any) -> Any:
        total = 0.0 * u.angstrom
        for x in z:
            total = total + x
        return total

    assert val(unrolled(q)) == 3.0 and val(jit(unrolled)(q)) == 3.0
    with pytest.raises(TypeError, match="0-d quantity"):
        jit(unrolled)(1.0 * u.angstrom)


def test_atomistic_charge():
    at = units.ATOMISTIC
    assert at.base_units()["s"] == ASE_TIME  # the solved time unit is still ASE's
    assert at.base_units()["A"] == 1.5737692011770263e-05  # pinning e moves it
    assert "charge=e" in at.describe()
    assert at.symbol(units.dim_of(u.e)) == "e"
    assert units.SI.symbol(units.dim_of(u.e)) == "C"
    with units.base(at):
        assert val(1 * u.e) == 1.0  # the charge unit is e
        assert val(1.0 * u.bar) == 6.241509074460763e-07
        assert float(constants.k_e.to(u.eV * u.angstrom / u.e**2)) == pytest.approx(
            14.399645478469017, rel=1e-15
        )
    # "charge" is a known kind, so pinning a length under that label is caught
    with pytest.raises(units.DimensionError, match="charge must be"):
        units.base(length=u.angstrom, charge=u.m).__enter__()


def test_pytree_leaf():
    q = arr([1.0, 2.0]) * u.m
    mags = q.value
    leaves, _ = jax.tree.flatten(q)
    assert len(leaves) == 1 and leaves[0] is mags  # the label is in the treedef
    flat = jax.tree.leaves({"a": q})
    assert len(flat) == 1 and flat[0] is mags  # jax's walk descends into the quantity
    walk = jax.tree.leaves({"a": q}, is_leaf=units.is_leaf)
    assert len(walk) == 1 and walk[0] is q  # is_leaf stops the walk at it


def test_treedef_equality():
    # treedefs agree only when dimension and base both agree
    def struct(q: Any) -> Any:
        return jax.tree.flatten(q)[1]

    assert struct(1.0 * u.m) == struct(arr([2.0, 3.0]) * u.km)  # shape is in the leaf
    assert hash(struct(1.0 * u.m)) == hash(struct(2.0 * u.km))
    assert struct(1.0 * u.m) != struct(1.0 * u.s)  # a different dimension
    with units.base(units.SI):
        elsewhere = struct(1.0 * u.m)
    assert struct(1.0 * u.m) != elsewhere  # a different base


def test_unflatten():
    q = arr([1.0, 2.0]) * u.angstrom
    leaves, td = jax.tree.flatten(q)
    back = jax.tree.unflatten(td, leaves)
    assert ty(back) == ty(q)
    assert float(back.sum().to(u.angstrom)) == 3.0
    relabelled = jax.tree.unflatten(td, [arr([3.0, 4.0])])
    assert ty(relabelled) == "float64[2]{Å|atomistic}"
    assert float(relabelled.sum().to(u.angstrom)) == 7.0


def test_no_global_registration(monkeypatch: pytest.MonkeyPatch):
    # Load a second copy of the base state and quantity classes in the same process.
    package = pathlib.Path(units.__file__).parent
    modules: dict[str, Any] = {}
    for name in ("_base", "_precision", "_core"):
        path = package / f"{name}.py"
        source = path.read_text()
        for dependency in ("_base", "_precision"):
            source = source.replace(
                f"from kups.core.units.{dependency} import",
                f"from vendored.{dependency} import",
            )
        mod = types.ModuleType(f"vendored.{name}")
        mod.__file__ = str(path)
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        exec(compile(source, str(path), "exec"), mod.__dict__)
        modules[name] = mod

    bases, core = modules["_base"], modules["_core"]
    assert bases.SI is not units.SI
    bases.set_base(bases.SI)
    q = core.from_unit(2.0, u.m)
    assert type(q) is core.Quantity and q.bs is bases.SI
    assert float(jit(lambda x: x + x)(q).to(u.m)) == 4.0
    assert (
        val(1.0 * u.angstrom + 2.0 * u.angstrom) == 3.0
    )  # the live module is untouched
    assert units.active_base() is units.ATOMISTIC


def test_tree_walks():
    # is_leaf walks hand a function whole quantities; jax's own reach the magnitudes
    tree = {"len": arr([4.0, 9.0]) * u.angstrom**2, "charge": 9.0 * u.e}
    assert units.is_leaf(1.0 * u.m)
    assert not units.is_leaf(arr([1.0])) and not units.is_leaf(u.m)
    # each quantity is one leaf, and the treedef puts both labels back
    assert [ty(x) for x in jax.tree.leaves(tree, is_leaf=units.is_leaf)] == [
        "float64[]{e|atomistic}",
        "float64[2]{Å^2|atomistic}",
    ]
    leaves, td = jax.tree.flatten(tree, is_leaf=units.is_leaf)
    back = jax.tree.unflatten(td, leaves)
    assert ty(back["len"]) == ty(tree["len"])
    assert ty(back["charge"]) == ty(tree["charge"])
    # a nonlinear function must see the whole quantity, or the label comes out wrong
    area = {"len": tree["len"]}
    assert ty(units.tree_map(units.sqrt, area)["len"]) == "float64[2]{Å|atomistic}"
    assert ty(jax.tree.map(jnp.sqrt, area)["len"]) == "float64[2]{Å^2|atomistic}"
    # with is_leaf, jnp is handed the whole quantity and refuses it
    with pytest.raises(TypeError, match="requires ndarray"):
        jax.tree.map(jnp.sqrt, tree, is_leaf=units.is_leaf)
    assert val(units.tree_map(lambda q: q * 2, 3.0 * u.angstrom)) == 6.0  # a bare leaf
