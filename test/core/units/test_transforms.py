# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for jax transforms over quantities."""

from fractions import Fraction
from typing import TypedDict

import jax
import numpy.testing as npt
import pytest

from kups.core import units
from kups.core.units import Energy, Length, Mass, Momentum, Velocity, VelocityUnit
from kups.core.units import catalog as u

from ._helpers import (
    arange,
    arr,
    checkpoint,
    jit,
    lax_cond,
    lax_switch,
    ty,
    val,
)


class _State(TypedDict):
    r: Length
    e: Energy
    n: jax.Array


SPEED: VelocityUnit = (u.eV / u.amu) ** Fraction(1, 2)  # the atomistic velocity unit


def test_vmap():
    vs = arr([1.0, 2.0, 3.0]) * SPEED
    m = 2.0 * u.amu

    def ke(v: Velocity) -> Energy:
        return 0.5 * m * v**2

    def momentum(v: Velocity) -> Momentum:
        return v * m

    out = jax.vmap(ke)(vs)
    assert out.shape == (3,) and list(out.value) == [1.0, 4.0, 9.0]
    assert list(jit(jax.vmap(momentum))(vs).value) == [2.0, 4.0, 6.0]
    xs = arange(6.0).reshape(2, 3) * u.angstrom
    assert list(jax.vmap(lambda x: x.sum(), in_axes=1)(xs).value) == [3.0, 5.0, 7.0]
    assert jax.vmap(momentum, out_axes=0)(vs).shape == (3,)


def test_aot_recheck():
    # jit's .lower() records the argument types and the kernel re-checks them
    def ke(m: Mass, v: Velocity) -> Energy:
        return 0.5 * m * v**2

    args = (2.0 * u.amu, 3.0 * SPEED)
    j = jax.jit(ke)
    lowered = j.lower(*args)
    comp = lowered.compile()
    assert val(comp(*args)) == 9.0
    assert "@jit_ke" in lowered.as_text()  # plain jax lowering underneath
    assert j.eval_shape(*args).dim == (2, 1, -2, 0, 0)
    # labels are pytree metadata, so jax's structure check rejects these
    plain = 2.0
    with units.base(units.SI):
        foreign = 2.0 * u.amu
    for bad in ((2.0 * u.angstrom, args[1]), (plain, plain), (foreign, args[1])):
        with pytest.raises(TypeError, match="does not match the input pytree"):
            comp(*bad)
    with pytest.raises(TypeError, match="does not match the input pytree"):
        comp(args[0])
    # jit kwargs pass through, and the AOT surface is present
    assert val(jax.jit(ke, static_argnums=1)(2.0 * u.amu, 3.0)) == 9.0
    assert val(jax.jit(ke, donate_argnums=0)(*args)) == 9.0
    assert jit(ke).__name__ == "ke"  # pyrefly: ignore[missing-attribute]
    assert hasattr(j, "trace")


def test_aot_trace_path():
    # both AOT spellings record the argument types: .lower() and .trace().lower()
    def ke(m: Mass, v: Velocity) -> Energy:
        return 0.5 * m * v**2

    args = (2.0 * u.amu, 3.0 * SPEED)
    with units.base(units.SI):
        foreign = 2.0 * u.amu
    for comp in (
        jax.jit(ke).lower(*args).compile(),
        jax.jit(ke).trace(*args).lower().compile(),
    ):
        assert val(comp(*args)) == 9.0
        with pytest.raises(TypeError, match="does not match the input pytree"):
            comp(args[1], args[1])  # a speed where a mass belongs
        with pytest.raises(TypeError, match="does not match the input pytree"):
            comp(foreign, args[1])  # the same dimension in another base


def test_jit_methods_and_constants():
    class Model:
        # the energy method is jitted with self static

        def __hash__(self) -> int:
            return 0

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Model)

        def _energy(self, m: Mass, v: Velocity) -> Energy:
            return 0.5 * m * v**2

        energy = jax.jit(_energy, static_argnums=0)

    assert val(Model().energy(2.0 * u.amu, 3.0 * SPEED)) == 9.0
    cutoff = 4.0 * u.angstrom  # eager, hence concrete inside a trace

    def scaled(x: jax.Array) -> jax.Array:
        return x * float(cutoff.value)

    assert float(jit(scaled)(arr(1.0))) == 4.0

    def as_float(z: Length) -> float:
        return float(z)

    with pytest.raises(units.DimensionError, match="needs a concrete magnitude"):
        jit(as_float)(3.0 * u.angstrom)


def test_scan_carry():
    def step(c: Length, x: Length) -> tuple[Length, Length]:
        return c + x, c

    def run(c: Length, x: Length) -> tuple[Length, Length]:
        return jax.lax.scan(step, c, x, length=3)

    xs = arr([1.0, 2.0, 3.0]) * u.angstrom
    final, ys = jax.lax.scan(step, 0.0 * u.angstrom, xs, length=3)
    assert val(final) == 6.0 and list(ys.value) == [0.0, 1.0, 3.0]
    jfinal, _ = jit(run)(0.0 * u.angstrom, xs)
    assert val(jfinal) == 6.0
    # length inferred from xs's leading axis
    inferred, _ = jax.lax.scan(step, 0.0 * u.angstrom, xs)
    assert val(inferred) == 6.0
    # jax's carry check catches a dimension change between iterations
    with pytest.raises(TypeError, match="carry input and carry output"):
        jax.lax.scan(lambda c, x: (c * (1.0 * u.s), c), 0.0 * u.angstrom, xs, length=3)


def test_cond_and_switch():
    # quantities are pytrees: jax hands cond and switch the magnitudes and compares
    # dimensions and bases as part of the tree structure
    state: _State = {
        "r": arr([1.0, 2.0, 3.0]) * u.angstrom,
        "e": 5.0 * u.eV,
        "n": arr([1, 2, 3]),
    }
    stay = jax.lax.cond(True, lambda s: s, lambda s: {**s, "r": s["r"] * 2}, state)
    assert list(stay["r"].to(u.angstrom)) == [1.0, 2.0, 3.0]
    assert float(stay["e"].to(u.eV)) == 5.0 and list(stay["n"]) == [1, 2, 3]
    moved = jax.lax.switch(1, [lambda s: s, lambda s: {**s, "r": s["r"] * 2}], state)
    assert list(moved["r"].to(u.angstrom)) == [2.0, 4.0, 6.0]
    # no operands at all, every value closed over
    assert (
        float(
            jax.lax.switch(arr(1), [lambda: 300.0 * u.K, lambda: 400.0 * u.K]).to(u.K)
        )
        == 400.0
    )
    # only the chosen branch runs, so the other branch's 1/0 never evaluates
    assert list(
        jax.lax.cond(True, lambda s: s["r"], lambda s: s["r"] / 0.0, state).to(
            u.angstrom
        )
    ) == [1.0, 2.0, 3.0]
    # a batched predicate selects between the branch results
    batched = jax.vmap(lambda p, q: jax.lax.cond(p, lambda x: x, lambda x: x * 2, q))(
        arr([True, False, True]), state["r"]
    )
    assert list(batched.to(u.angstrom)) == [1.0, 4.0, 3.0]
    chosen = jit(
        jax.vmap(lambda a, b: jax.lax.switch(a, [lambda x: x, lambda x: x * 2], b))
    )(arr([0, 1, 1]), state["r"])
    assert list(chosen.to(u.angstrom)) == [1.0, 4.0, 6.0]
    # a branch may close over a batched quantity: select_n sees the magnitudes
    closed = jax.vmap(lambda p, q: jax.lax.cond(p, lambda: q, lambda: q * 2))(
        arr([True, False, True]), state["r"]
    )
    assert list(closed.to(u.angstrom)) == [1.0, 4.0, 3.0]
    # branches must agree on dimension, plainness and base
    for other in (
        lambda q: q * (1.0 * u.eV),
        lambda q: q.value,
        lambda q: q.rebase(units.SI),
    ):
        with pytest.raises(TypeError, match="same pytree structure"):
            lax_cond(True, lambda q: q, other, state["r"])
    with pytest.raises(TypeError, match="same pytree structure"):
        lax_switch(0, [lambda q: q, lambda q: q * (1.0 * u.eV)], state["r"])
    # lax.select_n takes arrays only; apply lowers the quantities for it
    with pytest.raises(TypeError, match=r"JAX (value|type)"):
        # pyrefly: ignore[bad-argument-type]
        jax.lax.select_n(arr(1), state["r"], state["r"] * 2)
    picked = units.apply(
        jax.lax.select_n,
        arr([0, 1, 0]),
        state["r"],
        state["r"] * 2,
        dim=units.dim_of(u.angstrom),
    )
    assert list(picked.to(u.angstrom)) == [1.0, 4.0, 3.0]


def test_stop_gradient():
    r = arr([1.0, 2.0]) * u.angstrom

    @jax.jit
    def held(x: Length) -> Length:
        return x + jax.lax.stop_gradient(x)

    npt.assert_allclose(held(r).to(u.angstrom), [2.0, 4.0])

    tree = jax.lax.stop_gradient({"r": r, "n": arr([3.0])})
    assert units.dim_of(tree["r"]) == units.dim_of(u.angstrom)
    assert not units.is_quantity(tree["n"])


def test_checkpoint():
    def energy(r: Length) -> Energy:
        return (r * r).sum() / (1.0 * u.angstrom**2) * (1.0 * u.eV)

    ck = checkpoint(energy)
    r = arr([1.0, 2.0, 3.0]) * u.angstrom
    assert float(energy(r).to(u.eV)) == 14.0
    assert float(jax.jit(ck)(r).to(u.eV)) == 14.0
    _, ys = jax.lax.scan(lambda c, _: (c, ck(c)), r, None, length=2)
    npt.assert_allclose(ys.to(u.eV), [14.0, 14.0])
    total = jax.lax.fori_loop(0, 3, lambda i, acc: acc + ck(r), 0.0 * u.eV)
    assert float(total.to(u.eV)) == 42.0
    rows = arr([[1.0, 2.0], [3.0, 4.0]]) * u.angstrom
    npt.assert_allclose(jax.jit(jax.vmap(ck))(rows).to(u.eV), [5.0, 25.0])
    # the checkpointed results are quantities again
    assert float(jax.jit(lambda x: ck(x) / (1.0 * u.eV))(x=r)) == 14.0
    # untraced, remat returns the same type the function itself returns
    assert ty(ck(r)) == ty(energy(r))
    assert float(ck(r).to(u.eV)) == 14.0
