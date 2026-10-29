# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the array surface: reductions, movers, constructors, and apply."""

import inspect
import math
import re
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core import units
from kups.core.units import (
    Energy,
    Length,
)
from kups.core.units import catalog as u

from ._helpers import (
    arange,
    arr,
    eye,
    jit,
    linalg_norm,
    ones,
    segment_sum,
    ty,
    val,
    x64_mode,
    zeros_like,
)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, jnp.complex64])
@pytest.mark.parametrize("unit", [u.m, u.cm])
def test_linspace_requires_floating_dtype(unit: units.Unit, dtype: Any):
    def build(unit: units.Unit, dtype: Any) -> Any:
        return units.linspace(1.0 * unit, 3.0 * unit, 2, dtype=dtype)

    for call in (build, jit(build, static_argnums=(0, 1))):
        with pytest.raises(units.DimensionError, match="magnitudes cannot carry"):
            call(unit, dtype)


@pytest.mark.parametrize(
    "name, build",
    [
        ("zeros_like", lambda x, one: units.zeros_like(x, dtype=jnp.int32)),
        ("ones_like", lambda x, one: units.ones_like(x, dtype=jnp.int32)),
        ("full_like", lambda x, one: units.full_like(x, one, dtype=jnp.int32)),
    ],
)
def test_like_requires_floating_dtype(name: str, build: Callable[[Any, Any], Any]):
    with pytest.raises(units.DimensionError, match=name):
        build(arr([1.0, 2.0]) * u.m, u.nm)
    assert build(arr([1.0, 2.0]), 1.0).dtype == jnp.int32


def test_dimensionless_linspace_keeps_integer_dtype():
    out = units.linspace(1, 2, 2, dtype=jnp.int32)
    assert isinstance(out, jax.Array) and out.dtype == jnp.int32
    npt.assert_array_equal(out, [1, 2])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_astype_preserves_values_and_base(compiled: bool):
    q = arr([0.0, -1.25, 2.5, np.inf, -np.inf, np.nan]) * u.eV

    def cast(x: Any) -> Any:
        return x.astype(jnp.float32)

    out = (jit(cast) if compiled else cast)(q)
    assert out.bs is units.ATOMISTIC
    assert out.dim == q.dim
    assert out.dtype == jnp.float32
    npt.assert_array_equal(out.value, q.value.astype(jnp.float32))
    with (
        x64_mode(False),
        pytest.warns(UserWarning, match="not available"),
        pytest.raises(units.DimensionError, match="magnitudes are unavailable"),
    ):
        out.astype(jnp.float64)


@pytest.mark.parametrize("system", [units.SI, units.ATOMISTIC])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_dimension_changing_operations(system: units.BaseSystem, compiled: bool):
    def calculate(q: Any) -> Any:
        return q.sqrt(), q.prod(axis=1, keepdims=True), q.var(axis=0, ddof=1)

    with units.base(system):
        q = arr([[1.0, 4.0], [9.0, 16.0]]) * u.cm**2
        root, product, variance = (jit(calculate) if compiled else calculate)(q)
    npt.assert_allclose(root.to(u.cm), [[1.0, 2.0], [3.0, 4.0]])
    npt.assert_allclose(product.to(u.cm**4), [[4.0], [144.0]])
    npt.assert_allclose(variance.to(u.cm**4), [32.0, 72.0])


def test_array_surface():
    # eV is pinned, so magnitudes are exact
    f = arange(1.0, 7.0).reshape(3, 2) * u.eV
    assert val(f.mean()) == 3.5 and val(f.sum()) == 21.0
    assert f.sum(axis=0, keepdims=True).shape == (1, 2)
    assert val(f.min()) == 1.0 and val(f.max()) == 6.0
    assert val(abs(-f).sum()) == 21.0
    assert val(units.norm(arr([3.0, 4.0]) * u.eV)) == 5.0
    # every order is degree 1 in the magnitudes except ord=0, which counts nonzeros
    assert val(units.norm(arr([3.0, 4.0]) * u.eV, ord=jnp.inf)) == 4.0
    with pytest.raises(units.DimensionError, match=r"jnp\.count_nonzero"):
        units.norm(f, ord=0)
    plain: Any = arr([0.0, 3.0])  # untyped: the stub declares a quantity operand
    assert float(units.norm(plain, ord=0)) == 1.0  # plain arrays untouched
    assert f.reshape((2, 3)).shape == (2, 3) and f.T.shape == (2, 3)
    assert f.transpose().shape == (2, 3)
    # repeat is a method only; generic code calls it on a leaf
    assert list(f[0].repeat(2).value) == [1.0, 1.0, 2.0, 2.0]
    assert val(f.std()) == pytest.approx(1.7078251, rel=1e-6)
    assert val(f.trace()) == 5.0
    assert val(f.cumsum()[-1]) == 21.0 and f.ravel().shape == (6,)
    assert f.squeeze().shape == (3, 2) and f.swapaxes(0, 1).shape == (2, 3)
    assert list(f.diagonal().value) == [1.0, 4.0]
    assert list(f[0].sort().value) == [1.0, 2.0]
    assert list(f[0].take(arr([1, 0])).value) == [2.0, 1.0]
    assert f.copy().shape == (3, 2)
    assert val(f.var()) == pytest.approx(2.9166667, rel=1e-6)  # eV^2
    assert units.dim_of(f.var()) == units.dim_of(u.eV**2)
    assert val(f[0].prod()) == 2.0 and units.dim_of(f[0].prod()) == units.dim_of(
        u.eV**2
    )
    assert val(f[0].dot(f[0])) == 5.0 and units.dim_of(f[0].dot(f[0])) == units.dim_of(
        u.eV**2
    )
    for plain in (f.argmin(), f.argmax(), f[0].argsort(), f.all(), f.any()):
        assert not units.is_quantity(plain)
    assert len(f.nonzero()) == 2 and not units.is_quantity(f.nonzero()[0])
    assert val(jax.jit(lambda q: q.std())(f)) == pytest.approx(1.7078251, rel=1e-6)
    assert f.repeat(2, axis=0).shape == (6, 2)
    assert list(f[0].repeat(arr([1, 3])).value) == [1.0, 2.0, 2.0, 2.0]
    assert f[0].repeat(2, total_repeat_length=3).shape == (3,)
    assert jax.jit(lambda q: q.repeat(2).sum())(f[0]).to(u.eV) == 6.0
    assert not hasattr(units, "repeat")
    assert f.astype(jnp.float32).dtype == jnp.float32
    assert units.stack([f, f]).shape == (2, 3, 2)
    assert units.concatenate([f, f], axis=1).shape == (3, 4)
    assert list(f[0].clip(1.5 * u.eV, None).value) == [1.5, 2.0]
    assert list(units.where(arr([True, False]), f[0], 0.0).value) == [1.0, 0.0]
    assert list(units.zeros_like(f[0]).value) == [0.0, 0.0]
    assert list(units.ones_like(f[0]).value) == [1.0, 1.0]  # one base unit each
    assert list(units.full_like(f[0], 2.0 * u.eV).value) == [2.0, 2.0]
    assert float(f[0, 0]) == val(f[0, 0]) == 1.0  # float() is the own-base magnitude
    # results keep the operand's dimension and base
    for q in (f.mean(), f.min(), f.norm(), f.T, units.stack([f, f])):
        assert units.dim_of(q) == (2, 1, -2, 0, 0)
        assert units.base_of(q) == units.ATOMISTIC
    # apply is the escape hatch for anything not named above
    assert list(units.apply(jnp.flip, f[0]).value) == [2.0, 1.0]


def test_masking_and_scatter():
    r = arr([1.0, 3.0, 7.0]) * u.angstrom
    cut = 5.0 * u.angstrom

    def energy(z: Any) -> Any:
        # masked 1/r energy: the standard cutoff idiom
        return units.where(z < cut, (1.0 * u.eV * u.angstrom) / z, 0.0).sum()

    g = units.grad(energy)(r)
    assert bool(jnp.all(jnp.isfinite(g.value)))  # forces must not be NaN
    assert float(energy(r).to(u.eV)) == pytest.approx(1 + 1 / 3)
    assert val(jit(energy)(r)) == val(energy(r))
    ids = arr([0, 0, 1])
    per_atom = units.apply(segment_sum, arr([1.0, 2.0, 3.0]) * u.eV, ids, 2)
    assert [float(x) for x in per_atom.to(u.eV)] == [3.0, 3.0]
    acc = units.zeros_like(per_atom).at[ids[:2]].add(arr([1.0, 2.0]) * u.eV)
    assert [float(x) for x in acc.to(u.eV)] == [3.0, 0.0]
    assert float(acc.at[0].get().to(u.eV)) == 3.0
    assert [float(x) for x in acc.at[0].set(0.0 * u.eV).to(u.eV)] == [0.0, 0.0]
    assert [float(x) for x in acc.at[0].multiply(2.0).to(u.eV)] == [6.0, 0.0]

    def scatter(v: Any) -> Any:
        return units.apply(segment_sum, v, ids, 2).sum()

    assert list(units.grad(scatter)(arr([1.0, 2.0, 3.0]) * u.eV)) == [1.0] * 3
    assert jax.vmap(units.norm)(arange(6.0).reshape(3, 2) * u.eV).shape == (3,)


def test_dimension_checks():
    f = arr([1.0, 2.0]) * u.eV
    d: Any = arr([1.0, 2.0]) * u.angstrom  # untyped: the checker refuses these
    mask = arr([True, False])
    with pytest.raises(units.DimensionError, match="cannot select between"):
        units.where(mask, f, d)
    with pytest.raises(units.DimensionError, match="cannot clip"):
        f.clip(d)
    with pytest.raises(units.DimensionError, match="cannot stack"):
        units.stack([f, d])
    with pytest.raises(units.DimensionError, match="cannot concatenate"):
        units.concatenate([f, d])
    with pytest.raises(units.DimensionError, match="cannot fill"):
        units.full_like(f, d)
    with pytest.raises(units.DimensionError, match="cannot add"):
        f.at[0].add(d)
    with pytest.raises(units.DimensionError, match="dimensionless factor"):
        f.at[0].multiply(d)
    with pytest.raises(units.DimensionError, match="the condition must be a plain"):
        units.where(f, f, f)  # pyrefly: ignore[no-matching-overload]
    with units.base(units.SI):  # bases resolve exactly as they do for +
        si = 1.0 * u.eV
        assert units.base_of(units.stack([f, f])) == units.ATOMISTIC  # one base
        mixed = units.stack([f[0], si])  # two bases: resolved in the active one
        assert units.base_of(mixed) == units.SI
        assert list(mixed.value) == [1.602176634e-19, 1.602176634e-19]


def test_like_under_vmap():
    # vmap batches jnp.full_like, so results stay batched instead of becoming constants
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    rs = units.stack([r, r])
    assert jax.vmap(units.zeros_like)(rs).shape == (2, 3)
    assert list(jax.vmap(units.ones_like)(rs)[0].value) == [1.0, 1.0, 1.0]

    def filled(x: Any) -> Any:
        return units.full_like(x, 2.0 * u.angstrom)

    assert jax.vmap(filled)(rs).shape == (2, 3)
    nan = arr([float("nan"), 1.0]) * u.angstrom
    assert list(units.zeros_like(nan).value) == [0.0, 0.0]  # a constant, not 0*nan

    def constant(z: Any) -> Any:
        # ignores the batched argument, so vmap broadcasts it
        del z
        return 1.0 * u.eV

    broadcast = jax.vmap(constant)(rs)
    assert broadcast.shape == (2,)  # one value per row of the mapped axis
    assert units.dim_of(broadcast) == units.dim_of(u.eV)
    assert list(broadcast.to(u.eV)) == [1.0, 1.0]
    # out_axes=None returns the unbatched constant at its original shape
    assert jax.vmap(constant, out_axes=None)(rs).shape == ()


def test_where_needs_sanitized_operand():
    # where evaluates both branches, so mask the operand before the divide
    z = arr([1.0, 3.0, 0.0]) * u.angstrom  # a padded neighbour entry
    strength = 1.0 * u.eV * u.angstrom

    def naive(r: Length) -> Energy:
        # masks the result: finite energy, NaN force
        return units.where(r > (0.0 * u.angstrom), strength / r, 0.0).sum()

    def cured(r: Length) -> Energy:
        # masks the operand first, so the gradient is finite too
        good = r > (0.0 * u.angstrom)
        rr = units.where(good, r, 1.0 * u.angstrom)
        return units.where(good, strength / rr, 0.0).sum()

    assert float(naive(z).to(u.eV)) == pytest.approx(float(cured(z).to(u.eV)))
    assert not bool(jnp.all(jnp.isfinite(units.grad(naive)(z).value)))
    assert bool(jnp.all(jnp.isfinite(units.grad(cured)(z).value)))


def test_apply_override():
    # det/inv/cross carry the right dimension; apply needs dim= to change one
    cell = eye(3) * 5.43 * u.angstrom
    assert units.dim_of(units.det(cell)) == units.dim_of(u.angstrom**3)
    assert float(units.det(cell).to(u.angstrom**3)) == pytest.approx(5.43**3)
    assert units.dim_of(units.inv(cell)) == units.dim_of(u.angstrom**-1)
    assert units.dim_of(units.cross(cell[0], cell[1])) == units.dim_of(u.angstrom**2)
    assert units.dim_of(
        units.apply(jnp.linalg.det, cell, dim=units.dim_of(u.angstrom**3))
    ) == (3, 0, 0, 0, 0)
    with pytest.raises(units.DimensionError, match="index, a count or a mask"):
        units.apply(jnp.argsort, units.norm(cell, axis=-1))


def test_minimum_maximum():
    r = arr([1.0, 5.0]) * u.angstrom
    cut = 4.0 * u.angstrom
    assert [float(x) for x in units.minimum(r, cut).to(u.angstrom)] == [1.0, 4.0]
    assert [float(x) for x in units.maximum(r, cut).to(u.angstrom)] == [4.0, 5.0]
    with pytest.raises(units.DimensionError, match=r"units.minimum"):
        r.min(cut)  # min's argument is the axis, not a bound
    untyped: Any = r  # untyped: the checker refuses the mismatches below
    with pytest.raises(units.DimensionError, match="cannot take the elementwise"):
        units.minimum(untyped, 1.0 * u.eV)
    # jnp spellings fail on their own, since a quantity won't convert to an array
    slips: list[Callable[[], Any]] = [
        lambda: jnp.where(untyped > cut, untyped, 0.0),
        lambda: linalg_norm(untyped),
        lambda: jnp.stack([untyped, untyped]),
        lambda: jnp.minimum(untyped, untyped),
        lambda: segment_sum(untyped, arr([0, 1]), 2),
    ]
    for fn in slips:
        with pytest.raises((TypeError, ValueError, units.DimensionError)):
            fn()
    assert float(linalg_norm(arr([3.0, 4.0]))) == 5.0  # plain arrays untouched


def test_apply_keeps_dimension():
    # a single operand's dimension is apply's default label
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    cell = eye(3) * 5.0 * u.angstrom
    kept: list[Any] = [
        units.apply(jnp.tril, cell),
        units.apply(jnp.diagonal, cell),
        units.apply(jnp.ravel, cell),
        units.apply(jnp.moveaxis, cell, 0, 1),
        units.apply(jnp.pad, r, (1, 1)),
        units.apply(jnp.std, r, ddof=1),
        units.apply(jnp.trace, cell),
        units.apply(jnp.flip, r),
        units.apply(jnp.cumsum, r),
        units.apply(jnp.sort, r),
        units.apply(jax.ops.segment_max, r, arr([0, 0, 1]), 2),
        units.apply(jnp.hypot, r, r, dim=units.dim_of(u.angstrom)),
    ]
    for q in kept:
        assert units.dim_of(q) == units.dim_of(u.angstrom), q
        assert units.base_of(q) == units.ATOMISTIC
    assert float(units.apply(jnp.trace, cell).to(u.angstrom)) == 15.0
    # padded gather: fill_value=0 reads zero rather than clamping to the last row
    padded = arr([2, 9])
    got = units.apply(jnp.take, r, padded, mode="fill", fill_value=0)
    assert units.dim_of(got) == units.dim_of(u.angstrom)
    assert list(got.to(u.angstrom)) == [3.0, 0.0]
    along = units.apply(
        jnp.take_along_axis, r, padded, axis=0, mode="fill", fill_value=0
    )
    assert list(along.to(u.angstrom)) == [3.0, 0.0]
    assert list(r[jnp.clip(padded, 0, 2)].to(u.angstrom)) == [3.0, 3.0]  # clamping
    # without dim=, a second dimensioned operand multiplies into the result
    assert units.dim_of(units.apply(jnp.hypot, r, r)) == units.dim_of(u.angstrom**2)


def test_apply_dim_one():
    # dim=ONE marks the result as not a measurement, giving back a plain array
    r = arr([1.0, jnp.nan, 3.0]) * u.angstrom
    plain: list[Any] = [
        units.apply(jnp.argsort, r, dim=units.ONE),
        units.apply(jnp.argmin, r, dim=units.ONE),
        units.apply(jnp.sign, r, dim=units.ONE),
        units.apply(jnp.isfinite, r, dim=units.ONE),
        units.apply(jnp.isnan, r, dim=units.ONE),
    ]
    for x in plain:
        assert not units.is_quantity(x), x
    assert list(units.apply(jnp.isnan, r, dim=units.ONE)) == [False, True, False]
    # an integer or boolean result is refused rather than labelled
    for fn in (jnp.argsort, jnp.argmin, jnp.isfinite, jnp.isnan):
        with pytest.raises(units.DimensionError, match="index, a count or a mask"):
            units.apply(fn, r)
    # a dimensionless float result needs an explicit dim=ONE; the default labels it
    assert units.dim_of(units.apply(jnp.sign, r)) == units.dim_of(u.angstrom)


def test_contractions():
    # these compute their own dimensions, since apply's product default is wrong here
    # ewald.py:390's k_squared, with the real subscripts
    kv = ones((2, 4, 3)) / u.angstrom
    k2 = units.einsum("bkd,bkd->bk", kv, kv)
    assert units.dim_of(k2) == units.dim_of(u.angstrom**-2)
    assert float(k2.to(u.angstrom**-2)[0, 0]) == 3.0
    # ewald.py:419's k . r, which must be dimensionless
    assert not units.is_quantity(
        units.einsum(
            "psd,pd->ps", ones((5, 4, 3)) / u.angstrom, ones((5, 3)) * u.angstrom
        )
    )
    cell = eye(3) * 2.0 * u.angstrom
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    assert units.dim_of(cell @ r) == units.dim_of(u.angstrom**2)
    assert units.dim_of(units.vdot(r, r)) == units.dim_of(u.angstrom**2)
    assert float(units.vdot(r, r).to(u.angstrom**2)) == 14.0
    vol = (arr([2.0, 3.0, 4.0]) * u.angstrom).prod()
    assert units.dim_of(vol) == units.dim_of(u.angstrom**3)
    assert float(vol.to(u.angstrom**3)) == 24.0
    rows = ones((2, 3)) * u.angstrom
    assert units.dim_of(rows.prod(axis=-1)) == units.dim_of(u.angstrom**3)
    assert units.dim_of(r.var(ddof=1)) == units.dim_of(u.angstrom**2)
    assert float(r.var(ddof=1).to(u.angstrom**2)) == 1.0


def test_decompositions():
    # named functions, since lift has no rule for a tuple
    cell = eye(3) * 5.43 * u.angstrom
    force = arr([1.0, 2.0, 3.0]) * u.eV / u.angstrom
    stiffness = units.dim_of(u.eV / u.angstrom**2)
    x = units.solve(cell, force)
    assert units.dim_of(x) == stiffness
    npt.assert_allclose(x.to(u.eV / u.angstrom**2), [1 / 5.43, 2 / 5.43, 3 / 5.43])
    npt.assert_allclose((cell @ x).to(u.eV / u.angstrom), [1.0, 2.0, 3.0])
    # cell.py:1317's chain: Q is dimensionless and R a length, so sign(diag(R)) is plain
    q, upper = units.qr(cell.T)
    assert not units.is_quantity(q)
    assert units.dim_of(upper) == units.dim_of(u.angstrom)
    diag = units.apply(jnp.diagonal, upper)
    assert list(units.apply(jnp.sign, diag, dim=units.ONE)) == [1.0, 1.0, 1.0]
    assert units.dim_of(units.qr(cell, mode="r")) == units.dim_of(u.angstrom)
    hess = eye(3) * 2.0 * u.eV / u.angstrom**2
    w, vecs = units.eigh(hess)
    assert units.dim_of(w) == stiffness and not units.is_quantity(vecs)
    npt.assert_allclose(w.to(u.eV / u.angstrom**2), [2.0, 2.0, 2.0])
    npt.assert_allclose(abs(jnp.linalg.det(vecs)), 1.0)
    # eigenvalues keep the operand's base, whatever base is active
    with units.base(units.SI):
        w_si, _ = units.eigh(hess)
    assert units.base_of(w_si) == units.ATOMISTIC
    # slogdet refuses a quantity; take the magnitudes explicitly first
    sign, logabs = jnp.linalg.slogdet(eye(3) * 2.0)
    assert float(sign) == 1.0 and float(logabs) == pytest.approx(3 * math.log(2))
    with pytest.raises(TypeError, match="requires ndarray"):
        jnp.linalg.slogdet(cell)
    _, log_of_magnitudes = jnp.linalg.slogdet(cell.to(u.angstrom))
    assert float(log_of_magnitudes) == pytest.approx(3 * math.log(5.43))


def test_constructors():
    # shape= exists because a padding buffer may not have an operand to read it from
    edges = units.linspace(0.0 * u.angstrom, 10.0 * u.angstrom, 5)
    assert list(edges.to(u.angstrom)) == [0.0, 2.5, 5.0, 7.5, 10.0]
    centres = 0.5 * (edges[1:] + edges[:-1])
    assert units.dim_of(centres) == units.dim_of(u.angstrom)
    r = arr([1.0, 2.0]) * u.angstrom
    for like in (
        units.zeros_like(r, shape=(4, 2)),
        units.ones_like(r, shape=(4, 2)),
        units.full_like(r, 1.0 * u.angstrom, shape=(4, 2)),
    ):
        assert like.shape == (4, 2)
        assert units.dim_of(like) == units.dim_of(u.angstrom)
    untyped: Any = 1.0 * u.eV  # untyped: the checker refuses the mismatch
    with pytest.raises(units.DimensionError, match="cannot space"):
        units.linspace(0.0 * u.angstrom, untyped)


def test_at_fill_and_drop():
    # mode="fill" reads the fill value instead of clamping to the last real row
    shifts = arr([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]) * u.angstrom
    padded = arr([1, 7])  # the second row is padding
    clamped = shifts[jnp.clip(padded, 0, 1)]
    got = shifts.at[padded].get(mode="fill", fill_value=0)
    assert list(got.to(u.angstrom)[1]) == [0.0, 0.0, 0.0]
    assert list(clamped.to(u.angstrom)[1]) == [1.0, 2.0, 3.0]  # clamping instead
    # a dimensioned fill lowers to a static scalar, since jax hashes it
    far = shifts.at[padded].get(mode="fill", fill_value=9.0 * u.angstrom)
    assert list(far.to(u.angstrom)[1]) == [9.0, 9.0, 9.0]
    assert list(far.to(u.angstrom)[0]) == [1.0, 2.0, 3.0]
    # a scatter drops the padded row instead of accumulating it into row zero
    hist = zeros_like(arr([0.0, 0.0])) * u.eV
    assert list(hist.at[padded].add(1.0 * u.eV, mode="drop").to(u.eV)) == [0.0, 1.0]
    assert list(hist.at[padded].set(2.0 * u.eV, mode="drop").to(u.eV)) == [0.0, 2.0]
    assert list(shifts.at[padded].multiply(2.0, mode="drop").to(u.angstrom)[1]) == [
        2.0,
        4.0,
        6.0,
    ]
    assert list(
        shifts.at[arr([0])]
        .get(
            indices_are_sorted=True,
            unique_indices=True,
            wrap_negative_indices=False,
        )
        .to(u.angstrom)[0]
    ) == [0.0, 0.0, 0.0]
    # a fill in the wrong unit is refused; Any, since the stub rejects it too
    wrong_fills: list[Any] = [1.0 * u.eV, 1.0, u.eV]
    for fill in wrong_fills:
        with pytest.raises(units.DimensionError, match="cannot fill"):
            shifts.at[padded].get(mode="fill", fill_value=fill)

    @jax.jit
    def traced_fill(x: Any, fill: Any) -> Any:
        return x.at[padded].get(mode="fill", fill_value=fill)

    with pytest.raises(units.DimensionError, match="jax can hash"):
        traced_fill(shifts, 1.0 * u.angstrom)
    # a closed-over constant is concrete, so it lowers to a number
    fill = 4.0 * u.angstrom

    @jax.jit
    def constant_fill(x: Any) -> Any:
        return x.at[padded].get(mode="fill", fill_value=fill)

    assert list(constant_fill(shifts).to(u.angstrom)[1]) == [4.0, 4.0, 4.0]
    # the fill's magnitude is hashed into the gather, so it is rebased first
    with units.base(units.SI):
        inside = arr([1.0, 2.0]) * u.eV
    outside = 7.0 * u.eV  # built in the default atomistic base, active again
    for gather in (
        lambda q: q.at[padded].get(mode="fill", fill_value=outside),
        jit(lambda q: q.at[padded].get(mode="fill", fill_value=outside)),
    ):
        npt.assert_allclose(gather(inside).to(u.eV), [2.0, 7.0], rtol=1e-15)


def test_apply_lowers_operands():
    # positional and keyword quantities alike; the default dim is their product
    r = arr([1.0, 2.0]) * u.angstrom
    f = arr([3.0, 4.0]) * u.eV / u.angstrom
    work = units.apply(jnp.dot, r, f)
    assert units.dim_of(work) == units.dim_of(u.eV)
    assert float(work.to(u.eV)) == 11.0
    # a keyword quantity is lowered too, and counts towards the product
    assert units.dim_of(units.apply(jnp.clip, r, min=1.5 * u.angstrom)) == (
        units.dim_of(u.angstrom**2)
    )
    clipped = units.apply(
        jnp.clip, r, min=1.5 * u.angstrom, dim=units.dim_of(u.angstrom)
    )
    assert list(clipped.to(u.angstrom)) == [1.5, 2.0]
    # dim= still overrides, and a foreign base is rebased rather than read raw
    with units.base(units.SI):
        same = units.apply(jnp.dot, r.rebase(), f)
        assert float(same.to(u.eV)) == pytest.approx(11.0)
    assert units.dim_of(
        units.apply(jnp.dot, r, r, dim=units.dim_of(u.eV))
    ) == units.dim_of(u.eV)


def test_mirror_surface():
    # kups.core.units names an op only when it computes a dimension or apply can't
    # express it; everything else is absent and its jnp spelling refuses a quantity
    q: Any = arr([1.0, 2.0]) * u.angstrom
    cases: list[tuple[Callable[..., Any], str]] = [
        (jnp.isfinite, "isfinite"),
        (jnp.sign, "sign"),
        (jnp.argsort, "argsort"),
        (jnp.sin, "sin"),
        (jax.scipy.special.erfc, "erfc"),
        (jnp.linalg.slogdet, "slogdet"),
    ]
    for fn, name in cases:
        assert name not in units.__all__ and not hasattr(units, name), name
        with pytest.raises((TypeError, ValueError, units.DimensionError)):
            fn(q)
    assert units.dim_of(1.0) == units.ONE  # the guards' spelling is public
    # the jnp spellings of % and // refuse a quantity; the operators check dimensions
    agree: list[tuple[Callable[..., Any], Callable[..., Any], str]] = [
        (jnp.mod, lambda x, y: x % y, "cannot take the remainder of"),
        (jnp.floor_divide, lambda x, y: x // y, "cannot floor-divide"),
        (jnp.matmul, lambda x, y: x @ y, ""),
    ]
    for jfn, op, verb in agree:
        with pytest.raises((TypeError, ValueError, units.DimensionError)):
            jfn(q, q)
        if verb:
            with pytest.raises(units.DimensionError, match=verb):
                op(q, 1.0 * u.eV)
    # the whole public surface, so an accidental addition or removal fails
    dimensions = {
        "Acceleration", "Action", "Area", "Charge", "Compressibility",
        "CoulombConstant", "Current", "Density", "DipoleMoment", "Energy", "Entropy",
        "Force", "Frequency", "Length", "Mass", "Momentum", "NumberDensity",
        "Permittivity", "Power", "Pressure", "Stiffness", "Temperature", "Time", "Velocity",
        "Voltage", "Volume", "Wavenumber",
    }  # fmt: skip
    surface = {
        "ATOMISTIC", "BaseSystem", "DimensionError", "Dimensionless", "ONE",
        "Quantity", "SI", "Unit", "active_base", "apply", "base", "base_of",
        "catalog", "concatenate", "constants", "cross", "det", "dim_from_str",
        "dim_of", "dim_str", "eigh", "einsum", "from_dim", "full_like", "grad",
        "hessian", "inv", "is_leaf", "is_quantity", "jacfwd", "jacrev", "linspace",
        "maximum", "minimum", "norm", "ones_like", "qr", "relabel", "set_base",
        "solve", "sqrt", "stack", "tree_map", "value_and_grad", "vdot", "vectorize",
        "vjp", "where", "zeros_like", "Log", "absolute", "clip", "full", "max",
        "mean", "min", "ones", "prod", "select_n", "stop_gradient", "sum", "svd",
        "tree_flatten", "tree_leaves", "zeros",
    }  # fmt: skip
    assert set(units.__all__) == surface | dimensions | {
        f"{name}Unit" for name in dimensions
    }
    # a body that is one apply() call only forwards; the *_like constructors and the
    # free spellings of Quantity methods are the exception
    forwards = {"zeros_like", "ones_like", "sum", "mean", "min", "max", "absolute"}
    found = set()
    for name in units.__all__:
        fn = getattr(units, name)
        # the filter is over the package, so a forwarder counts wherever it is written
        if not callable(fn):
            continue
        home = getattr(fn, "__module__", "")
        assert home.startswith("kups.core.units"), f"units.{name} is defined in {home}"
        body = inspect.getsource(fn).split('"""')[-1].strip()
        m = re.search(r"return apply\((.*?)\)\s*$", body, re.DOTALL)
        if m and "dim=" not in m.group(1):
            found.add(name)
    assert found == forwards, f"units.{sorted(found ^ forwards)} only forwards to jnp"
    batched = arange(6.0).reshape(2, 3) * u.angstrom

    def apply_zeros(x: Any) -> Any:
        return units.apply(zeros_like, x)

    for out in (jax.vmap(units.zeros_like)(batched), jax.vmap(apply_zeros)(batched)):
        npt.assert_array_equal(out.value, np.zeros((2, 3)))
        assert units.dim_of(out) == units.dim_of(u.angstrom)
    # a constant fill is unbatched, so out_axes=None returns one row
    assert jax.vmap(units.zeros_like, out_axes=None)(batched).shape == (3,)
    assert jax.vmap(apply_zeros, out_axes=None)(batched).shape == (3,)


def test_operands_must_agree():
    # % and // relate both operands' dimensions
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    box = 2.5 * u.angstrom
    assert list((r % box).to(u.angstrom)) == [1.0, 2.0, 0.5]
    assert units.dim_of(r % box) == units.dim_of(u.angstrom)
    assert list(r // box) == [0.0, 0.0, 1.0]
    assert not units.is_quantity(r // box)  # a count
    # a mismatch raises, and the message names both units
    untyped: Any = r  # untyped: the checker refuses the mismatches below
    mismatched: list[tuple[Callable[[], Any], str]] = [
        (lambda: untyped % (1.0 * u.eV), "cannot take the remainder of"),
        (lambda: untyped // (1.0 * u.eV), "cannot floor-divide"),
    ]
    for fn, verb in mismatched:
        with pytest.raises(units.DimensionError, match=verb) as caught:
            fn()
        assert "Å" in str(caught.value) and "eV" in str(caught.value)
    # a plain zero means the same in every unit
    zero: Any = 0.0  # untyped: the stubs put the exemption on the second operand
    assert float((zero % box).to(u.angstrom)) == 0.0
    with units.base(units.SI):
        r_si = arr([1.0, 3.0]) * u.angstrom
    # an operand from another base is rebased, not compared raw
    box = 2.5 * u.angstrom
    assert list((r_si % box).to(u.angstrom)) == [1.0, 0.5]
    assert list(r_si // box) == [0.0, 1.0]


def test_array_protocol():
    # generic pytree code reads these attributes and dunders without seeing units
    m = arr([[1.0, 2.0], [3.0, 4.0]]) * u.angstrom
    assert m.size == 4
    assert list(m.ravel().to(u.angstrom)) == [1.0, 2.0, 3.0, 4.0]
    assert units.dim_of(m.ravel()) == units.dim_of(u.angstrom)
    assert m.copy().to(u.angstrom).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert m.tolist() == [[1.0, 2.0], [3.0, 4.0]]  # magnitudes, in m's own base
    assert (2.5 * u.angstrom).item() == 2.5
    # bool() reads the magnitudes, not mere non-emptiness
    assert bool(2.5 * u.angstrom) and not bool(0.0 * u.angstrom)
    with pytest.raises(ValueError, match="ambiguous"):
        bool(m)
    assert int(3.7 * u.angstrom) == 3
    assert f"{2.5 * u.angstrom:.3f}" == "2.500 Å"
    assert f"{2.5 * u.angstrom}" == "2.5 Å"
    # % keeps the shared dimension and // cancels it
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    assert list((r % (2.5 * u.angstrom)).to(u.angstrom)) == [1.0, 2.0, 0.5]
    assert list(r // (2.5 * u.angstrom)) == [0.0, 0.0, 1.0]
    untyped: Any = r
    with pytest.raises(units.DimensionError, match="cannot take the remainder of"):
        untyped % (1.0 * u.eV)
    # under a trace only the magnitudes change; the leaf is still a Quantity
    assert (r.shape, r.ndim) == ((3,), 1)
    assert ty(r) == "float64[3]{Å|atomistic}"

    def sizes(x: Length) -> Any:
        assert x.size == 3
        assert type(x).__name__ == "Quantity" and ty(x) == ty(r)
        return x.ravel().copy() * 2.0

    assert list(jax.jit(sizes)(r).to(u.angstrom)) == [2.0, 4.0, 6.0]

    def as_int(x: Length) -> int:
        return int(x)

    with pytest.raises(units.DimensionError, match="needs a concrete magnitude"):
        jit(as_int)(3.0 * u.angstrom)

    # an unrecognised operand defers to the other type's reflected dunder
    class Rotation:
        def __rmatmul__(self, other: Any) -> str:
            return f"rotated {type(other).__name__}"

    assert r @ Rotation() == "rotated Quantity"
    seen: list[str] = []

    def rotate(x: Length) -> Length:
        seen.append(x @ Rotation())
        seen.append(type(x.value).__name__)
        return x

    jax.jit(rotate)(r)
    # the deferral works the same under a trace
    assert seen == ["rotated Quantity", "DynamicJaxprTracer"]
    with pytest.raises(TypeError, match="unsupported operand"):
        untyped + Rotation()
    with pytest.raises(TypeError, match="not supported between"):
        _ = untyped < Rotation()
    # a value's sharding is the magnitudes'
    assert untyped.sharding == r.value.sharding
    assert isinstance(untyped.sharding, jax.sharding.Sharding)


def test_method_lowering():
    # methods lower operands first, so a quantity never reaches jnp through one
    q = arr([[1.0, 2.0], [3.0, 4.0]]) * u.eV
    keeps = (
        "squeeze",
        "swapaxes",
        "take",
        "sort",
        "diagonal",
        "trace",
        "cumsum",
        "std",
    )
    plains = ("argmin", "argmax", "argsort", "all", "any")
    extra: dict[str, tuple[Any, ...]] = {"swapaxes": (0, 1), "take": (arr([0, 1]),)}
    for name in keeps:
        out = getattr(q, name)(*extra.get(name, ()))
        assert units.is_quantity(out), name
        assert units.dim_of(out) == units.dim_of(u.eV), name
    for name in plains:
        assert not units.is_quantity(getattr(q, name)()), name
    # an index is a count, so a quantity there is refused, statically and at runtime
    with pytest.raises((ValueError, TypeError, units.DimensionError)):
        q.take(arr([0, 1]) * u.eV)  # pyrefly: ignore[bad-argument-type]


def test_tree_walk_labels():
    # the surface survives a pytree walk, however the walk sees the leaf
    tree = {"r": arange(6.0).reshape(3, 2) * u.angstrom, "n": arr([2.0, 3.0])}
    # is_leaf hands the walker the quantity itself
    flat = jax.tree.map(lambda x: x.ravel(), tree, is_leaf=units.is_quantity)
    assert flat["r"].shape == (6,)
    npt.assert_array_equal(flat["r"].to(u.angstrom), np.arange(6.0))
    assert units.dim_of(flat["r"]) == units.dim_of(u.angstrom)
    assert not units.is_quantity(flat["n"])
    # without is_leaf the walker gets magnitudes; the treedef relabels the result
    halved = jax.tree.map(lambda x: x / 2.0, tree)
    want = np.arange(6.0).reshape(3, 2) / 2
    npt.assert_array_equal(halved["r"].to(u.angstrom), want)
    assert ty(halved["r"]) == ty(tree["r"])
    leaves, treedef = jax.tree.flatten(tree)
    assert len(leaves) == 2 and not any(units.is_quantity(x) for x in leaves)
    back = jax.tree.unflatten(treedef, leaves)
    assert ty(back["r"]) == ty(tree["r"])
    assert float((back["r"] - tree["r"]).sum().to(u.angstrom)) == 0.0
    # the label is in the treedef and the shape in the leaf
    reshaped = jax.tree.unflatten(treedef, [x.reshape(-1) for x in leaves])
    assert reshaped["r"].shape == (6,)
    assert units.dim_of(reshaped["r"]) == units.dim_of(u.angstrom)


def test_apply_under_transforms():
    # apply labels a result under vmap and jit the same way it does eagerly
    cubed = units.dim_of(u.angstrom**3)
    cell = eye(3) * 5.43 * u.angstrom
    cells = units.stack([cell, 2.0 * cell])

    def volume(c: Any) -> Any:
        return units.apply(jnp.linalg.det, c, dim=cubed)

    def unnamed(c: Any) -> Any:
        return units.apply(jnp.linalg.det, c)

    # magnitudes are cubed either way; the label follows the operand, not fn
    assert units.dim_of(unnamed(cell)) == units.dim_of(u.angstrom)
    assert units.dim_of(jax.jit(unnamed)(cell)) == units.dim_of(u.angstrom)
    assert units.dim_of(jax.vmap(unnamed)(cells)) == units.dim_of(u.angstrom)
    assert units.dim_of(jax.jit(volume)(cell)) == cubed
    volumes = jax.vmap(volume)(cells).to(u.angstrom**3)
    npt.assert_allclose(volumes, [5.43**3, (2 * 5.43) ** 3])
    # a plain operand stays plain under a transform
    rows = jax.vmap(lambda x: units.apply(jnp.cumsum, x))(arange(6.0).reshape(2, 3))
    assert not units.is_quantity(rows)
    # an index measures nothing, so labelling one is refused
    with pytest.raises(units.DimensionError, match="index, a count or a mask"):
        jax.vmap(lambda c: units.apply(jnp.argmax, units.norm(c, axis=-1)))(cells)


def test_free_functions():
    q: Any = arr([[1.0, -2.0], [3.0, -4.0]]) * u.angstrom
    for name in ("sum", "mean", "min", "max"):
        out = getattr(units, name)(q, axis=0)
        npt.assert_allclose(out.to(u.angstrom), getattr(jnp, name)(q.value, axis=0))
    npt.assert_allclose(units.absolute(q).to(u.angstrom), jnp.abs(q.value))
    assert units.dim_of(units.prod(q, axis=1)) == units.dim_of(u.angstrom**2)
    npt.assert_allclose(
        units.clip(q, 0.0, 2.0 * u.angstrom).to(u.angstrom), [[1, 0], [2, 0]]
    )
    picked = units.select_n(arr([[0, 1], [1, 0]]), q, -q)
    npt.assert_allclose(picked.to(u.angstrom), [[1, 2], [-3, -4]])
    npt.assert_allclose(units.zeros((2,), u.eV).to(u.eV), [0, 0])
    npt.assert_allclose(units.ones((2,), u.nm).to(u.angstrom), [10, 10])
    npt.assert_allclose(units.full((2,), 3.0 * u.K).to(u.K), [3, 3])
    s = units.svd(q, compute_uv=False)
    npt.assert_allclose(s.to(u.angstrom), jnp.linalg.svd(q.value, compute_uv=False))
    assert units.dim_of(units.stop_gradient(q)) == units.dim_of(q)
    assert units.tree_leaves({"a": q, "b": 1.0})[0] is q
    with pytest.raises(units.DimensionError):
        units.clip(q, 1.0 * u.eV)


def test_log():
    p: Any = arr([1.0, 10.0]) * u.bar
    log = units.Log.of(p, u.Pa)
    npt.assert_allclose(log.log, jnp.log(arr([1e5, 1e6])))
    npt.assert_allclose(log.exp().to(u.bar), [1.0, 10.0], rtol=1e-12)
    assert jax.tree.leaves(log) == [log.log]
    assert jit(lambda x: x)(log).ref == u.Pa
    assert log[1:].shape == (1,)
    with pytest.raises(units.DimensionError):
        units.Log.of(p, u.eV)
