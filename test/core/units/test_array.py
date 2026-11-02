# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the array surface: reductions, movers, and apply."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core import units
from kups.core.units import (
    Area,
    Length,
    Quantity,
)
from kups.core.units import catalog as u

from ._helpers import (
    arange,
    arr,
    eye,
    jit,
    ty,
    zeros_like,
)


@pytest.mark.parametrize("system", [units.SI, units.ATOMISTIC])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_dimension_changing_operations(system: units.BaseSystem, compiled: bool):
    def calculate(
        q: Area,
    ) -> tuple[Length, Quantity[4, 0, 0, 0, 0], Quantity[4, 0, 0, 0, 0]]:
        return q.sqrt(), q.prod(axis=1, keepdims=True), q.var(axis=0, ddof=1)

    with units.base(system):
        q = arr([[1.0, 4.0], [9.0, 16.0]]) * u.cm**2
        root, product, variance = (jit(calculate) if compiled else calculate)(q)
    npt.assert_allclose(root.to(u.cm), [[1.0, 2.0], [3.0, 4.0]])
    npt.assert_allclose(product.to(u.cm**4), [[4.0], [144.0]])
    npt.assert_allclose(variance.to(u.cm**4), [32.0, 72.0])


def test_apply_keeps_dimension():
    # a single operand's dimension is apply's default label
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    cell = eye(3) * 5.0 * u.angstrom
    kept: list[Length] = [
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
    plain: list[jax.Array] = [
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


def test_at_fill_and_drop():
    # mode="fill" reads the fill value instead of clamping to the last real row
    shifts = arr([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]) * u.angstrom
    padded = arr([1, 7])  # the second row is padding
    clamped = shifts[jnp.clip(padded, 0, 1)]
    got = shifts.at[padded].get(mode="fill", fill_value=0.0 * u.angstrom)
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
    # a fill in the wrong unit is refused, statically and at runtime
    for fill in (1.0 * u.eV, 1.0, u.eV):
        with pytest.raises(units.DimensionError, match="cannot fill"):
            shifts.at[padded].get(
                mode="fill",
                fill_value=fill,  # pyrefly: ignore[bad-argument-type]
            )

    @jax.jit
    def traced_fill(x: Length, fill: Length) -> Length:
        return x.at[padded].get(mode="fill", fill_value=fill)

    with pytest.raises(units.DimensionError, match="jax can hash"):
        traced_fill(shifts, 1.0 * u.angstrom)
    # a closed-over constant is concrete, so it lowers to a number
    fill = 4.0 * u.angstrom

    @jax.jit
    def constant_fill(x: Length) -> Length:
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


def test_operands_must_agree():
    # % and // relate both operands' dimensions
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    box = 2.5 * u.angstrom
    assert list((r % box).to(u.angstrom)) == [1.0, 2.0, 0.5]
    assert units.dim_of(r % box) == units.dim_of(u.angstrom)
    assert list(r // box) == [0.0, 0.0, 1.0]
    assert not units.is_quantity(r // box)  # a count
    # a mismatch raises, and the message names both units
    mismatched: list[tuple[Callable[[], object], str]] = [
        (
            lambda: r % (1.0 * u.eV),  # pyrefly: ignore[unsupported-operation]
            "cannot take the remainder of",
        ),
        (
            lambda: r // (1.0 * u.eV),  # pyrefly: ignore[unsupported-operation]
            "cannot floor-divide",
        ),
    ]
    for fn, verb in mismatched:
        with pytest.raises(units.DimensionError, match=verb) as caught:
            fn()
        assert "Å" in str(caught.value) and "eV" in str(caught.value)
    # a plain zero is a mismatch too
    with pytest.raises(units.DimensionError, match="cannot take the remainder of"):
        0.0 % box  # pyrefly: ignore[unsupported-operation]
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
    with pytest.raises(units.DimensionError, match="cannot take the remainder of"):
        r % (1.0 * u.eV)  # pyrefly: ignore[unsupported-operation]
    # under a trace only the magnitudes change; the leaf is still a Quantity
    assert (r.shape, r.ndim) == ((3,), 1)
    assert ty(r) == "float64[3]{Å|atomistic}"

    def sizes(x: Length) -> Length:
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
        def __rmatmul__(self, other: object) -> str:
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
        r + Rotation()  # pyrefly: ignore[unsupported-operation]
    with pytest.raises(TypeError, match="not supported between"):
        _ = r < Rotation()  # pyrefly: ignore[unsupported-operation]
    # a value's sharding is the magnitudes'
    assert r.sharding == r.value.sharding
    assert isinstance(r.sharding, jax.sharding.Sharding)


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
    extra: dict[str, tuple[int | jax.Array, ...]] = {
        "swapaxes": (0, 1),
        "take": (arr([0, 1]),),
    }
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
