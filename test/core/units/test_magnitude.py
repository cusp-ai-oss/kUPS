# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for conversions, cross-base arithmetic, float32 guards and serialisation."""

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core import units
from kups.core.units import (
    Length,
    Quantity,
)
from kups.core.units import catalog as u
from kups.core.units import testing as ut

from ._helpers import (
    arr,
    eval_shape,
    make_jaxpr,
    ty,
    ulps,
    val,
)


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


def test_unit_under_trace():
    # the aval carries the dimension and base, so q.unit resolves under a trace
    seen: list[object] = []
    with units.base(units.SI):
        q = arr(1e5) * u.Pa
        assert q.unit is u.Pa
        jax.jit(lambda x: (seen.append(x.unit), x.value)[1])(q)
        assert seen[0] is u.Pa
        npt.assert_allclose(jax.jit(lambda x: x.to(x.unit))(q), q.value)
    r = arr([1.0, 2.0]) * u.angstrom
    assert r.unit is u.angstrom
    jax.jit(lambda x: (seen.append(x.unit), x.value)[1])(r)
    assert seen[-1] is u.angstrom
    guarded = jax.jit(lambda x: units.maximum(x, 1e-10 * x.unit))(r)
    npt.assert_allclose(guarded.to(u.angstrom), [1.0, 2.0])


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


def test_serialisation():
    # the dimension travels as five base-independent exponents, not as a symbol
    r = arr([1.0, 2.5, 3.0]) * u.angstrom
    assert units.base_of(r) is units.ATOMISTIC
    assert units.dim_str(units.dim_of(r)) == "m=1 kg=0 s=0 A=0 K=0"
    assert units.dim_str(units.ONE) == "m=0 kg=0 s=0 A=0 K=0"
    # the writer's four attributes, and the reader that undoes them
    dim, bs = units.dim_of(r), units.base_of(r)
    attrs = {
        "units_unit": bs.symbol(dim),
        "units_dim": units.dim_str(dim),
        "units_base": bs.name,
        "units_si_scale": bs.scale(dim),
    }
    assert attrs["units_unit"] == "Å" and attrs["units_base"] == "atomistic"
    assert attrs["units_si_scale"] == 1e-10
    stored = np.asarray(r.value)  # what an HDF5 dataset holds
    back = units.from_dim(stored, dim, units.ATOMISTIC)
    ut.assert_array_equal(back, r, u.angstrom)  # bit-exact, zero ops
    assert units.base_of(back) is units.ATOMISTIC
    # a symbol is not the dimension: the atomistic time unit uses half powers
    tdim = units.dim_of(
        1.0 * u.angstrom * u.amu ** Fraction(1, 2) / u.eV ** Fraction(1, 2)
    )
    assert bs.symbol(tdim) == "amu^(1/2).Å/eV^(1/2)"
    assert units.dim_str(tdim) == "m=0 kg=0 s=1 A=0 K=0"
    # half exponents print as fractions
    half = units.dim_of((1.0 * u.eV).sqrt())
    assert units.dim_str(half) == "m=1 kg=1/2 s=-1 A=0 K=0"
    # a reader in another base rescales through SI rather than reading raw
    si = units.from_dim(stored * attrs["units_si_scale"], dim, units.SI)
    assert units.base_of(si) is units.SI
    ut.assert_allclose(si, r, u.angstrom)
    # a traced quantity answers both: the label is on the value, not the tracer
    seen: list[tuple[str, str]] = []

    def probe(x: Length) -> Length:
        seen.append((units.dim_str(units.dim_of(x)), units.base_of(x).name))
        return x * 2.0

    jax.jit(probe)(r)
    assert seen == [("m=1 kg=0 s=0 A=0 K=0", "atomistic")]
    # eval_shape's abstract result carries the label too
    shape_only = eval_shape(probe, r)
    assert units.is_quantity(shape_only)
    assert units.dim_of(shape_only) == units.dim_of(u.angstrom)
    assert units.base_of(shape_only) is units.ATOMISTIC
    # each refusal names what was expected
    with pytest.raises(TypeError, match="is not a quantity"):
        units.base_of(arr([1.0]))
    with pytest.raises(units.DimensionError, match="is not a dimension"):
        units.dim_str((1, 0, 0))  # three exponents are not a dimension
    with pytest.raises(units.DimensionError, match="from_dim reads plain numbers"):
        units.from_dim(r, units.dim_of(u.eV))  # already labelled
    assert not units.is_quantity(units.from_dim(stored, units.ONE))  # decays


def test_attached_magnitudes_are_not_weak():
    # an attached magnitude is never weakly typed, eagerly or traced
    def promote(a: Length, b: jax.Array) -> jax.Array:
        return a.value * b

    b16 = arr([1.0, 2.0], dtype=jnp.bfloat16)
    assert promote(3.0 * u.m, b16).dtype == jnp.float64
    assert eval_shape(promote, 3.0 * u.m, b16).dtype == jnp.float64


def test_arraylike_magnitudes():
    # a scalar-returning apply holds a jax literal rather than an array, and is read
    # through the same conversion as any ArrayLike
    dim = units.dim_of(u.angstrom)
    at = units.ATOMISTIC
    for q in (
        units.Quantity(3.0, dim, at),
        units.Quantity(np.float64(3.0), dim, at),
        units.Quantity(np.array([3.0]), dim, at),
        units.apply(lambda a: 3.0, 3.0 * u.angstrom, dim=dim),
    ):
        assert isinstance(q.value, jax.Array)
        assert isinstance(q.to(u.angstrom), jax.Array)
        assert isinstance(q / q, jax.Array)  # the dimensions cancel
        assert q.item() == 3.0
        first = q.at[()] if q.value.ndim == 0 else q.at[0]
        npt.assert_allclose(first.get().to(u.angstrom), 3.0)
    n = units.Quantity(np.zeros(3), dim, at)
    assert n.value.shape == (3,)
    npt.assert_allclose(n.at[0].set(1.0 * u.angstrom).to(u.angstrom), [1.0, 0.0, 0.0])
    plus_one = jax.jit(lambda x: (x + 1.0 * u.angstrom).sum())
    npt.assert_allclose(plus_one(n).to(u.angstrom), 3.0)


def test_unit_operand_dtype():
    # a Unit operand is checked against the dtype the operands promote to, not
    # jax's default float64, or the float32 band would go unguarded
    v32 = arr(1.0, dtype=jnp.float32) * u.angstrom**6  # 1 here, so the attach is free
    for build in (
        lambda: v32 * u.m**6,
        lambda: v32 / u.m**6,
        lambda: v32 < u.m**6,  # same dimension: only the size refuses
        lambda: jax.jit(lambda q: q * u.m**6)(v32),
        lambda: v32.at[()].set(u.m**6),
        lambda: v32.at[()].add(u.m**6),
        lambda: v32.reshape(1).take(arr([0, 5]), mode="fill", fill_value=u.m**6),
    ):
        with pytest.raises(units.DimensionError, match="not a normal float32"):
            build()
    with pytest.raises(units.DimensionError, match=r"wider than float32"):
        v32 * u.m**6  # with x64 already on, the advice is to widen the dtype
    # float64 magnitudes can hold the factor
    v64 = (arr(1.0) * u.angstrom**6) * u.m**6
    assert float(v64.to(u.angstrom**6 * u.m**6)) == pytest.approx(1.0)
    assert units.full_like(arr([1.0, 1.0]) * u.angstrom**6, u.m**6).value.shape == (2,)
