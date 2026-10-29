# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for mixed plain/quantity operators and blocked implicit conversions."""

import functools
import math
import operator as op
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core import units
from kups.core.units import catalog as u
from kups.core.units._core import Quantity

from ._helpers import (
    arr,
    asarray,
    debug_callback,
    debug_print,
    empty_like,
    eval_shape,
    full_like,
    np_array,
    np_asarray,
    ones_like,
    ty,
    val,
    zeros_like,
)

# plain operands to mix with quantities, in both orders
_ARRAYS: list[Any] = [np.array([[2.0, 0.0], [0.0, 3.0]]), arr([[2.0, 0.0], [0.0, 3.0]])]
_PLAINS: list[Any] = [*_ARRAYS, 2.0]


def _both(f: Any, p: Any, x: Any) -> list[Any]:
    """Thunks for f(p, x) and f(x, p), eagerly and under eval_shape."""
    return [
        functools.partial(f, p, x),
        functools.partial(f, x, p),
        functools.partial(eval_shape, f, p, x),
        functools.partial(eval_shape, f, x, p),
    ]


def _types(f: Any, p: Any, x: Any) -> list[str]:
    """Result type of each spelling in _both, as text."""
    return [ty(call()) for call in _both(f, p, x)]


def _refused(f: Any, p: Any, x: Any, msg: str) -> None:
    """Assert every spelling in _both raises kups.core.units' own message."""
    for call in _both(f, p, x):
        with pytest.raises(units.DimensionError, match=msg):
            call()


def _magnitudes(q: Any) -> Any:
    return q.v


def test_jnp_interop():
    # jnp never silently drops units; the escape hatch round-trips
    q = arr([1.0, 2.0, 3.0]) * u.angstrom
    untyped: Any = q  # a checker already rejects a quantity as a jnp argument
    with pytest.raises(TypeError, match="requires ndarray"):
        jnp.sum(untyped)  # jax's own refusal: __array__ is what stops it
    placed = jax.device_put(q)  # placement moves the magnitudes and keeps the label
    assert units.dim_of(placed) == units.dim_of(u.angstrom) and val(placed[0]) == 1.0
    assert val(q.sum()) == 6.0
    assert list((jnp.cumsum(q.to(u.angstrom)) * u.angstrom).value) == [1.0, 3.0, 6.0]
    # a quantity is one leaf, so a walk keeps its dimension
    assert jax.tree.leaves({"q": q}, is_leaf=units.is_quantity) == [q]
    (leaf,) = jax.tree.leaves({"q": q})
    assert not units.is_quantity(leaf) and list(leaf) == [1.0, 2.0, 3.0]


def test_jnp_like_drops_units():
    # jnp's *_like duck-type shape and dtype, so kups.core.units never sees the call
    q = arr([1.0, 2.0]) * u.m
    untyped: Any = q  # a checker rejects a quantity as a jnp argument
    for fn in (zeros_like, ones_like, empty_like):
        out = fn(untyped)
        assert not units.is_quantity(out) and out.shape == (2,)
    assert not units.is_quantity(full_like(untyped, 1.0))
    assert list(zeros_like(q.value)) == [0.0, 0.0]  # plain arrays untouched
    assert ty(units.zeros_like(q)) == ty(q)
    assert ty(units.full_like(q, 1.0 * u.m)) == ty(q)


def test_asarray_refused():
    # __array__ raises, so no asarray spelling reads base-dependent magnitudes
    q = arr([1.0, 2.0]) * u.angstrom
    left: Any = np.array([1.0, 2.0])  # a checker rejects this spelling as well
    assert list((left * q).value) == [1.0, 4.0]  # a Quantity, not an object array
    assert list((arr([2.0, 3.0]) * q).value) == [2.0, 6.0]  # jnp still defers
    untyped: Any = q
    for fn in (np_asarray, np_array):
        with pytest.raises(units.DimensionError, match=r"q\.to\(unit\)"):
            fn(untyped)
        with pytest.raises(units.DimensionError, match=r"q\.to\(unit\)"):
            fn(3.0 * u.angstrom)  # a 0-d quantity has __float__, which numpy duck-types
    # jnp's constructors refuse it too, with either message
    for fn in (asarray, arr):
        with pytest.raises(TypeError, match=r"q\.to\(unit\)|Unexpected input type"):
            fn(untyped)
        with pytest.raises(TypeError, match=r"q\.to\(unit\)|Unexpected input type"):
            fn(3.0 * u.angstrom)
    # the named exits, each explicit about the unit or base
    assert float(3.0 * u.angstrom) == 3.0 and int(3.0 * u.angstrom) == 3
    assert (3.0 * u.angstrom).item() == 3.0 and (3.0 * u.angstrom).tolist() == 3.0
    assert float((3.0 * u.nm).to(u.angstrom)) == 30.0
    with pytest.raises(units.DimensionError, match=r"float\(q\) needs a concrete"):
        jax.jit(float)(3.0 * u.angstrom)  # under a trace there is no number to give


def test_dimensionless_is_plain():
    # a dimensionless value is a plain jax.Array, so jnp takes it directly
    alpha, r = 0.3 / u.angstrom, arr([1.0, 2.0]) * u.angstrom
    split = alpha * r  # erfc(alpha * r) is the Ewald real-space split
    assert not units.is_quantity(split)
    assert float(jax.scipy.special.erfc(split)[0]) == pytest.approx(0.6713732051035724)
    assert float(jax.scipy.special.erf(split)[0]) == pytest.approx(0.3286267948964276)
    assert float(jnp.sin(90.0 * u.deg)) == pytest.approx(1.0)  # deg is a ratio
    assert float(jnp.arctan(1.0)) == pytest.approx(math.pi / 4)
    untyped: Any = r  # untyped: the checker refuses these outright
    for fn in (
        jnp.sin,
        jnp.cos,
        jnp.tan,
        jnp.arcsin,
        jnp.arccos,
        jnp.arctan,
        jax.scipy.special.erf,
        jax.scipy.special.erfc,
    ):
        with pytest.raises(TypeError, match="requires ndarray"):
            fn(untyped)


def test_debug_print(capsys: Any):
    # printed lines name the unit, and a callback's argument is a quantity again
    r = arr([1.0, 2.5]) * u.angstrom
    seen: list[Any] = []
    debug_print("r = {r}", r=r)
    jax.jit(lambda x: debug_print("jit {x}", x=x))(r)
    debug_callback(lambda q: seen.append(units.dim_of(q)), r)
    debug_print("plain {n}", n=arr([1, 2]))
    assert seen == [units.dim_of(u.angstrom)]
    assert sorted(capsys.readouterr().out.splitlines()) == sorted(
        ["r = [1.  2.5] Å", "jit [1.  2.5] Å", "plain [1 2]"]
    )


def test_custom_jvp_nondiff():
    # jax binds stop_gradient on traced nondiff arguments; a quantity travels
    # through that bind as a pytree and keeps its dimension

    @functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
    def scaled(x, const):
        return x + const

    @scaled.defjvp
    def scaled_jvp(const, primals, tangents):
        (x,), (dx,) = primals, tangents
        return scaled(x, const), dx

    def loop(carry, const):
        # two steps, so the jaxpr is lowered rather than evaluated
        return jax.lax.scan(
            lambda c, _: (scaled(c, const), None), carry, None, length=2
        )[0]

    q = arr([0.0, 0.0]) * u.e
    out = jax.jit(loop)(q, arr([1.0, 1.0]) * u.e)
    assert units.dim_of(out) == units.dim_of(u.e)
    assert list(out.to(u.e)) == [2.0, 2.0]


def test_mixed_operators():
    # every binary operator between a plain value and a quantity reaches kups.core.units
    q = arr([[1.0, 2.0], [3.0, 4.0]]) * u.angstrom
    m, per_m = "float64[2,2]{Å|atomistic}", "float64[2,2]{1/Å|atomistic}"
    for p in _PLAINS:
        assert _types(op.mul, p, q) == [m, m, m, m]
        assert _types(op.truediv, p, q) == [per_m, m, per_m, m]
        _refused(op.add, p, q, "cannot add")
        _refused(op.sub, p, q, "cannot subtract")
        for f in (op.lt, op.le, op.gt, op.ge):
            _refused(f, p, q, "cannot compare")
    for p in _ARRAYS:  # a matmul operand needs an axis to contract
        assert _types(op.matmul, p, q) == [m, m, m, m]


def test_deferral_attributes(monkeypatch: pytest.MonkeyPatch):
    # jax reads __jax_array__ before a reflected operator, and numpy takes an
    # operator over unless __array_ufunc__ is None; removing either breaks deferral

    # a fresh shape per probe, so jax traces each one instead of using a cached trace
    def traced(n: int) -> Any:
        return eval_shape(op.mul, np.ones(n, "float32"), np.ones(n) * u.m)

    assert units.is_quantity(3.0 * u.m * np.float32(1.0))
    assert units.is_quantity(traced(1))
    monkeypatch.setattr(Quantity, "__jax_array__", _magnitudes, raising=False)
    assert not units.is_quantity(traced(2))  # converted instead of deferring
    monkeypatch.undo()
    assert units.is_quantity(traced(3))
    monkeypatch.delattr(Quantity, "__array_ufunc__")
    with pytest.raises(units.DimensionError, match="does not convert"):
        np.ones(()) * (3.0 * u.m)  # numpy reaches __array__ instead of __rmul__
    monkeypatch.undo()
    assert units.is_quantity(np.ones(()) * (3.0 * u.m))
