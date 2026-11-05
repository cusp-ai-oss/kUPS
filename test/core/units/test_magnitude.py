# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for conversions, cross-base arithmetic and float32 guards."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from kups.core import units
from kups.core.units import (
    Length,
    Quantity,
)
from kups.core.units import catalog as u

from ._helpers import (
    arr,
    make_jaxpr,
    ty,
    ulps,
    val,
)

type _Count = Callable[[jax.Array, jax.Array], jax.Array]


def test_conversion_and_mole():
    # kcal/mol == eV as a dimension (locked decision 1)
    with units.base(units.SI):
        assert float((1 * u.km).to(u.m)) == 1000.0
        assert float((1500 * u.m).to(u.km)) == 1.5
        assert float((90 * u.km / u.hour).to(u.m / u.s)) == 25.0
        e = 5 * u.kcal / u.mol + 1 * u.eV
        assert val(e) == 1.9495614068527687e-19
        assert float(e.to(u.kcal / u.mol)) == 28.06054783061903  # the SI decimal
    # built in the atomistic base, one rescale on exit lands on the same decimal
    at = 5 * u.kcal / u.mol + 1 * u.eV
    assert float(at.to(u.kcal / u.mol)) == 28.06054783061903
    assert ulps(float(at.to(u.kcal / u.mol)), 28.06054783061903) <= 2
    # these two are bit-identical in every base
    for bs in (units.SI, units.ATOMISTIC):
        with units.base(bs):
            assert float((1 * u.eV).to(u.kcal / u.mol)) == 23.06054783061903
            assert float((1 * u.eV / u.angstrom).to(u.N)) == 1.6021766339999998e-09
    with units.base(units.SI):  # the nearest float64 to the exact value
        assert float((1 * u.kcal / u.mol).to(u.J)) == 6.947695457055374e-21
    assert ulps(float((1 * u.kcal / u.mol).to(u.J)), 6.947695457055374e-21) <= 2


def test_identity_conversion():
    # .to() on a pinned/derived unit is a no-op, so f32 cannot underflow
    def as_eV2(q: Quantity[4, 2, -4, 0, 0]) -> jax.Array:
        return q.to(u.eV**2)

    sq = (arr([0.5, -0.5]) * u.eV) ** 2
    assert [float(x) for x in sq.to(u.eV**2)] == [0.25, 0.25]
    jaxpr = str(make_jaxpr(as_eV2)(sq))
    assert "mul" not in jaxpr and "div" not in jaxpr


def test_cross_base_arithmetic():
    # a foreign-base operand is rebased into the active base by one multiply
    with units.base(units.SI):
        cut = 5.0 * u.angstrom  # e.g. a module-level constant built under SI
    r = arr([1.0, 3.0, 7.0]) * u.angstrom
    assert list(map(bool, r < cut)) == [True, True, False]
    assert val(r[0] + cut) == 6.0 and val(cut + r[0]) == 6.0

    def shift(a: Length) -> Length:
        return a + cut

    jaxpr = str(make_jaxpr(shift)(r))
    assert jaxpr.count("mul") == 1  # the rebase, and nothing else
    assert "angstrom" not in jaxpr  # the jaxpr sees only magnitudes
    assert ty(shift(r)) == "float64[3]{Å|atomistic}"
    with units.base(units.SI):  # both operands in one base: never rebased
        a1, a2 = 2.0 * u.m, 3.0 * u.m

    def add() -> Length:
        return a1 + a2

    assert "mul" not in str(make_jaxpr(add)())
    assert val(a1 + a2) == 5.0 and ty(a1 + a2) == "float64[]{m|SI}"


def test_unit_property():
    # q.to(q.unit) is q.value, in whichever base the quantity was built
    r, f = 3.0 * u.angstrom, 1.0 * u.eV / u.angstrom
    assert r.unit is u.angstrom and float(r.to(r.unit)) == float(r.value)
    assert (5.0 * u.eV).unit is u.eV
    # no catalogue unit is that size, so it is named the way the base displays it
    assert f.unit.name == "eV/Å" and f.unit.scale == (u.eV / u.angstrom).scale
    assert float(f.to(f.unit)) == float(f.value)
    with units.base(units.SI):
        a = 3.0 * u.angstrom
        assert a.unit is u.m and float(a.to(a.unit)) == float(a.value)
        assert (5.0 * u.eV).unit is u.J
        assert (jnp.arange(3.0) * u.angstrom / u.fs).unit.name == "m/s"
    # the unit is a Unit, so it composes
    assert units.dim_of(r.unit * r.unit) == units.dim_of(u.m**2)


def test_ulps_saturates():
    # nan compares all-False, so a helper that only walked the gap would call a
    # nan bit-identical; ulps reports its cap instead
    assert ulps(1.0, 1.0) == 0
    assert ulps(1.0, math.nextafter(1.0, math.inf)) == 1
    assert ulps(math.nan, 1.0) == 9 and ulps(1.0, math.inf) == 9
    assert ulps(1.0, 1e300) == 9  # beyond the cap, not a near miss


def test_saturated_unit_refused():
    # u.nm**36 is 1e-324 in SI: its scale saturates to 0.0, and dividing by it
    # would raise a raw ZeroDivisionError; .to refuses it by name instead
    tiny = u.nm**36
    assert tiny.scale == 0.0
    with units.base(units.SI):  # where m^36 and m^108 attach for free
        with pytest.raises(units.DimensionError, match="no representable size"):
            (arr([1.0]) * u.m**36).to(tiny)
        with pytest.raises(units.DimensionError, match="no representable size"):
            (arr([1.0]) * u.m**108).to(u.km**108)  # saturated the other way


def test_integer_rescale_refused():
    # a rebase reaches jnp.finfo, so it must raise its own error rather than
    # leak numpy's "not compatible with finfo"
    n = units.Quantity(jnp.arange(3), units.dim_of(u.m), units.SI)
    with pytest.raises(units.DimensionError, match="cannot be re-expressed"):
        n.rebase(units.ATOMISTIC)
    assert n.rebase(units.SI).dtype == jnp.int64  # no rescale, so nothing to refuse
    with pytest.raises(units.DimensionError, match="cannot be measured in"):
        n.to(u.angstrom)
