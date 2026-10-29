# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for jax transforms over quantities."""

import functools
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core import units
from kups.core.units import catalog as u

from ._helpers import (
    arange,
    arr,
    checkpoint,
    eye,
    jax_grad,
    jit,
    jnp_vectorize,
    lax_cond,
    lax_switch,
    np_array,
    ones,
    ty,
    val,
)

SPEED = (u.eV / u.amu) ** Fraction(1, 2)  # the atomistic velocity unit


def test_grad_units():
    # grad relabels cotangents to dim(out)/dim(arg); jax.grad cannot
    def spring(k: Any, x: Any) -> Any:
        return 0.5 * k * x**2

    k, x = 250.0 * u.N / u.m, 3.0 * u.cm
    g = units.grad(spring, 1)(k, x)
    assert float(g.to(u.N)) == pytest.approx(7.5)  # a force, not a length
    assert units.dim_of(g) == (1, 1, -2, 0, 0)
    v, g2 = units.value_and_grad(spring, 1)(k, x)
    assert float(v.to(u.J)) == pytest.approx(0.1125) and val(g2) == val(g)
    assert units.dim_of(units.grad(spring, 0)(k, x)) == (2, 0, 0, 0, 0)  # m^2
    assert float(jit(units.grad(spring, 1))(k, x).to(u.N)) == pytest.approx(7.5)
    # second order, and grad through a broadcasting reduction
    assert units.dim_of(units.grad(units.grad(spring, 1), 1)(k, x)) == (
        0,
        1,
        -2,
        0,
        0,
    )
    xs = arr([1.0, 2.0, 3.0]) * u.m

    def stretch(kk: Any) -> Any:
        return (0.5 * kk * xs**2).sum()

    gk = units.grad(stretch, 0)(1.0 * u.N / u.m)
    assert float(gk.to(u.m**2)) == pytest.approx(7.0)
    with pytest.raises(TypeError, match="Gradient only defined for scalar"):
        jax_grad(spring, 1)(k, x)  # jax.grad itself rejects the quantity output


def test_vmap():
    vs = arr([1.0, 2.0, 3.0]) * SPEED
    m = 2.0 * u.amu

    def ke(v: Any) -> Any:
        return 0.5 * m * v**2

    def momentum(v: Any) -> Any:
        return v * m

    out = jax.vmap(ke)(vs)
    assert out.shape == (3,) and list(out.value) == [1.0, 4.0, 9.0]
    assert list(jit(jax.vmap(momentum))(vs).value) == [2.0, 4.0, 6.0]
    xs = arange(6.0).reshape(2, 3) * u.angstrom
    assert list(jax.vmap(lambda x: x.sum(), in_axes=1)(xs).value) == [3.0, 5.0, 7.0]
    assert jax.vmap(momentum, out_axes=0)(vs).shape == (3,)


def test_scan_carry():
    def step(c: Any, x: Any) -> Any:
        return c + x, c

    def run(c: Any, x: Any) -> Any:
        return jax.lax.scan(step, c, x, length=3)

    def total(x: Any) -> Any:
        return run(0.0 * u.angstrom, x)[0]

    xs = arr([1.0, 2.0, 3.0]) * u.angstrom
    final, ys = jax.lax.scan(step, 0.0 * u.angstrom, xs, length=3)
    assert val(final) == 6.0 and list(ys.value) == [0.0, 1.0, 3.0]
    jfinal, _ = jit(run)(0.0 * u.angstrom, xs)
    assert val(jfinal) == 6.0
    g = units.grad(total)(xs)
    assert list(g) == [1.0, 1.0, 1.0]  # d(m)/d(m) is dimensionless -> plain array
    # length inferred from xs's leading axis
    inferred, _ = jax.lax.scan(step, 0.0 * u.angstrom, xs)
    assert val(inferred) == 6.0
    # jax's carry check catches a dimension change between iterations
    with pytest.raises(TypeError, match="carry input and carry output"):
        jax.lax.scan(lambda c, x: (c * (1.0 * u.s), c), 0.0 * u.angstrom, xs, length=3)


def test_grad_has_aux():
    def loss(p: Any) -> Any:
        return (p * p).sum(), {"n": p.shape[0]}

    p = arr([1.0, 2.0]) * u.eV
    (v, aux), g = units.value_and_grad(loss, has_aux=True)(p)
    assert float(v.to(u.eV**2)) == 5.0 and aux == {"n": 2}
    assert units.dim_of(g) == (2, 1, -2, 0, 0)  # d(eV^2)/d(eV) is an energy
    g2, aux2 = units.grad(loss, has_aux=True)(p)
    assert list(g2.value) == list(g.value) and aux2 == aux


def test_aot_recheck():
    # jit's .lower() records the argument types and the kernel re-checks them
    def ke(m: Any, v: Any) -> Any:
        return 0.5 * m * v**2

    args = (2.0 * u.amu, 3.0 * SPEED)
    j = jax.jit(ke)
    lowered = j.lower(*args)
    comp = lowered.compile()
    assert val(comp(*args)) == 9.0
    assert "@jit_ke" in lowered.as_text()  # plain jax lowering underneath
    assert j.eval_shape(*args).dim == (2, 1, -2, 0, 0)
    # labels are pytree metadata, so jax's structure check rejects these
    plain: Any = 2.0
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
    assert jit(ke).__name__ == "ke" and hasattr(j, "trace")


def test_aot_trace_path():
    # both AOT spellings record the argument types: .lower() and .trace().lower()
    def ke(m: Any, v: Any) -> Any:
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


def test_grad_labels_leaves():
    def energy(p: dict[str, Any], n: float) -> Any:
        return p["e0"] * n

    for bs in (units.SI, units.ATOMISTIC):
        with units.base(bs):
            g = units.grad(energy)({"e0": 1.0 * u.eV}, 3.0)
            assert not units.is_quantity(g["e0"])  # d(eV)/d(eV) is a number
            assert float(g["e0"]) == 3.0

    def strained(r: Any, eps: Any) -> Any:
        return ((r + r * eps) ** 2).sum() * (1.0 * u.eV / u.angstrom**2)

    r0 = arr([1.0, 2.0]) * u.angstrom
    for bs in (units.SI, units.ATOMISTIC):
        with units.base(bs):
            g = units.grad(strained, 1)(r0.rebase(), 0.0)
            assert float(g.to(u.eV)) == pytest.approx(10.0)  # an energy, labelled

    def kwloss(a: Any, scale: float = 1.0) -> Any:
        return (a * a).sum() * scale

    gk = units.grad(kwloss)(arr([1.0, 2.0]) * u.angstrom, scale=2.0)
    assert [float(x) for x in gk.to(u.angstrom)] == [4.0, 8.0]


def test_jacobians():
    # units.jacrev/jacfwd/hessian label dim(out)/dim(arg); jax's label each level
    def pair_energy(x: Any) -> Any:
        return (x * x).sum() * (1.0 * u.eV / u.angstrom**2)

    x = arr([0.5, 1.1]) * u.angstrom
    h = units.hessian(pair_energy)(x)
    assert units.dim_of(h) == units.dim_of(u.eV / u.angstrom**2)
    assert float(h[1, 1].to(u.eV / u.angstrom**2)) == pytest.approx(2.0)
    j = units.jacrev(pair_energy)(x)
    assert units.dim_of(j) == units.dim_of(u.eV / u.angstrom)
    assert ty(units.jacfwd(pair_energy)(x)) == ty(j)

    # a dimensionless argument keeps dim(out), and a pytree is per leaf
    def doubled(p: dict[str, Any]) -> dict[str, Any]:
        return {"e": p["s"] * 2.0}

    d = units.jacrev(doubled)({"s": 3.0 * u.eV})
    assert not units.is_quantity(d["e"]["s"]) and float(d["e"]["s"]) == 2.0
    # jax's jacobians label every level of the tree; kups.core.units' result is one leaf
    for name in ("jacrev", "jacfwd"):
        theirs = getattr(jax, name)(pair_energy)(x)
        assert units.dim_of(theirs) == units.dim_of(u.eV)  # the output's
        assert units.dim_of(theirs.v) == units.dim_of(u.angstrom)  # the argument's
        npt.assert_allclose(theirs.v.value, j.value)
    assert units.dim_of(jax.hessian(pair_energy)(x).v.v) == units.dim_of(u.angstrom)

    def spread(p: Any) -> Any:
        # the same energy over 1 eV, so the output is dimensionless
        return pair_energy(p) / (1.0 * u.eV)

    theirs, ours = jax.grad(spread)(x), units.grad(spread)(x)
    assert units.dim_of(theirs) == units.dim_of(u.angstrom)  # the primal's own
    assert units.dim_of(ours) == units.dim_of(1.0 / u.angstrom)  # the derivative's
    npt.assert_allclose(theirs.value, ours.value)


def test_jit_methods_and_constants():
    class Model:
        # the energy method is jitted with self static

        def __hash__(self) -> int:
            return 0

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Model)

        def _energy(self, m: Any, v: Any) -> Any:
            return 0.5 * m * v**2

        energy = jax.jit(_energy, static_argnums=0)

    assert val(Model().energy(2.0 * u.amu, 3.0 * SPEED)) == 9.0
    cutoff = 4.0 * u.angstrom  # eager, hence concrete inside a trace

    def scaled(x: Any) -> Any:
        return x * float(cutoff.value)

    assert float(jit(scaled)(arr(1.0))) == 4.0

    def as_float(z: Any) -> float:
        return float(z)

    with pytest.raises(units.DimensionError, match="needs a concrete magnitude"):
        jit(as_float)(3.0 * u.angstrom)


def test_cond_and_switch():
    # kups.core.units lowers the operands to magnitudes (lax.select_n needs arrays once a
    # batched predicate selects) and pre-checks dimensions, bases and the predicate
    state: dict[str, Any] = {
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
    # the derivative flows through the branch that ran
    pull = units.grad(
        lambda q: (
            jax.lax.cond(False, lambda x: x, lambda x: x * 2, q)
            * (1.0 * u.eV / u.angstrom)
        ).sum()
    )(state["r"])
    assert list(pull.to(u.eV / u.angstrom)) == [2.0, 2.0, 2.0]
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
        jax.lax.select_n(arr(1), state["r"], state["r"] * 2)
    picked = units.apply(
        jax.lax.select_n,
        arr([0, 1, 0]),
        state["r"],
        state["r"] * 2,
        dim=units.dim_of(u.angstrom),
    )
    assert list(picked.to(u.angstrom)) == [1.0, 4.0, 3.0]


def test_vjp_and_linearize():
    # cotangents are relabelled dim(out)/dim(primal) per leaf, matching grad;
    # jax's own typing would give them the primal's dimension
    k = 2.0 * u.eV / u.angstrom**2
    r = arr([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.0, 3.0]]) * u.angstrom

    def energies(pos: Any) -> Any:
        return 0.5 * k * (pos * pos).sum(axis=1)

    out, vjp_fn = units.vjp(energies, r)
    assert list(out.to(u.eV)) == [1.0, 4.0, 9.25]
    (force,) = vjp_fn(units.ones_like(out))
    assert units.dim_of(force) == units.dim_of(u.eV / u.angstrom)
    assert force.to(u.eV / u.angstrom)[2].tolist() == [1.0, 0.0, 6.0]
    # a plain seed is read as ones of the output's own dimension, same numbers
    (plain,) = vjp_fn(ones(3))
    assert plain.to(u.eV / u.angstrom)[2].tolist() == [1.0, 0.0, 6.0]
    # the seed is read in the output's own unit and base
    (twice,) = vjp_fn(2.0 * units.ones_like(out))
    assert twice.to(u.eV / u.angstrom)[2].tolist() == [2.0, 0.0, 12.0]

    # one cotangent per primal leaf: d(eV)/d(eV) is a number, d(eV)/d(Å) a force
    def mixed(d: dict[str, Any]) -> Any:
        return (d["r"] * (1.0 * u.eV / u.angstrom)).sum() + 2.0 * d["e"]

    (ct,) = units.vjp(mixed, {"r": r, "e": 3.0 * u.eV})[1](1.0 * u.eV)
    assert float(ct["e"]) == 2.0
    assert ct["r"].to(u.eV / u.angstrom)[0].tolist() == [1.0, 1.0, 1.0]
    value, _, aux = units.vjp(lambda p: (energies(p), "aux"), r, has_aux=True)
    assert aux == "aux" and list(value.to(u.eV)) == [1.0, 4.0, 9.25]
    # the output tangent carries the output's dimension
    _, tangent_out = jax.jvp(energies, (r,), (r,))
    assert list(tangent_out.to(u.eV)) == [2.0, 8.0, 18.5]
    # jax requires tangents with the primal's structure
    step = units.ones_like(r)
    assert list(jax.jvp(energies, (r,), (step,))[1].to(u.eV)) == [2.0, 4.0, 7.0]
    with pytest.raises(TypeError, match="same tree structure"):
        jax.jvp(energies, (r,), (ones((3, 3)),))

    def energy_and_force(pos: Any) -> tuple[Any, Any]:
        e, pull = units.vjp(energies, pos)
        return pull(units.ones_like(e))[0], e

    gradient, row_fn, energy = jax.linearize(energy_and_force, r, has_aux=True)
    assert gradient.to(u.eV / u.angstrom)[1].tolist() == [0.0, 4.0, 0.0]
    assert list(energy.to(u.eV)) == [1.0, 4.0, 9.25]
    # one Hessian row times a step of one angstrom: eV/angstrom
    flat = units.zeros_like(r).reshape((-1,)).at[4].set(1.0 * u.angstrom)
    basis = flat.reshape(r.shape)
    row = row_fn(basis)
    assert units.dim_of(row) == units.dim_of(u.eV / u.angstrom)
    assert row.to(u.eV / u.angstrom)[1].tolist() == [0.0, 2.0, 0.0]
    full = units.hessian(lambda p: energies(p).sum())(r)
    assert full.to(u.eV / u.angstrom**2)[1, 1][1].tolist() == pytest.approx(
        [0.0, 2.0, 0.0]
    )

    # a custom_vjp backward rule returns the primal's dimension unless it relabels
    @jax.custom_vjp
    def energy(pos: Any) -> Any:
        return energies(pos).sum()

    def fwd(pos: Any) -> tuple[Any, Any]:
        return energy(pos), pos

    energy.defvjp(fwd, lambda res, g: (k * res * g.to(u.eV),))
    structure = re.escape("same container (pytree) structure")
    with pytest.raises(TypeError, match=structure):
        units.grad(energy)(r)

    @jax.custom_vjp
    def relabelled(pos: Any) -> Any:
        return energies(pos).sum()

    relabelled.defvjp(
        lambda pos: (relabelled(pos), pos),
        lambda res, g: (
            units.relabel(k * res * g.to(u.eV), units.dim_of(1.0 * u.angstrom)),
        ),
    )
    assert units.grad(relabelled)(r).to(u.eV / u.angstrom)[2].tolist() == [
        1.0,
        0.0,
        6.0,
    ]
    assert list(units.relabel(ones(2), units.dim_of(u.eV)).to(u.eV)) == [1.0, 1.0]
    assert list(units.relabel(r, units.ONE)[1]) == [0.0, 2.0, 0.0]
    # an output of plain leaves measures nothing, so its cotangent is 1/dim(primal)
    counts = units.vjp(lambda p: (p.sum() / (1.0 * u.angstrom),), r)
    assert units.dim_of(counts[1]((1.0,))[0]) == (-1, 0, 0, 0, 0)
    # wrong cotangent dimensions and mixed-dimension outputs are refused
    refused: list[tuple[Callable[[], Any], str]] = [
        (
            lambda: vjp_fn(units.ones_like(out) * (1.0 * u.angstrom)),
            "this cotangent is 'Å.eV' where the value it belongs to is 'eV'",
        ),
        (
            lambda: units.vjp(lambda p: (energies(p), p), r),
            "units.vjp: the output measures both 'eV' and 'Å'",
        ),
        (
            lambda: units.vjp(lambda p: (energies(p), p.sum() / r), r),
            "units.vjp: the output measures both 'eV' and a plain array",
        ),
    ]
    for fn, message in refused:
        with pytest.raises(units.DimensionError, match=re.escape(message)):
            fn()


def test_vectorize():
    # jnp.vectorize calls jnp.asarray on every operand, which a quantity refuses;
    # units.vectorize hands it magnitudes and relabels them around each call
    @functools.partial(
        units.vectorize, signature="(),(n),(n)->(n)", excluded=frozenset({0})
    )
    def flow(state: Any, dt: Any, primal: Any, tangent: Any) -> Any:
        del state  # the excluded argument, passed through untouched
        return primal + tangent * dt

    pos = arr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * u.angstrom
    vel = ones((2, 3)) * u.angstrom / u.fs
    moved = flow(None, 0.5 * u.fs, pos, vel)
    assert units.dim_of(moved) == units.dim_of(u.angstrom)
    assert moved.to(u.angstrom)[1].tolist() == [4.5, 5.5, 6.5]
    # a scalar broadcasts, a per-system value maps, and one system needs no batch axis
    per_system = flow(None, arr([0.5, 1.0]) * u.fs, pos, vel)
    assert per_system.to(u.angstrom)[1].tolist() == [5.0, 6.0, 7.0]
    assert flow(None, 0.5 * u.fs, pos[0], vel[0]).to(u.angstrom).tolist() == [
        1.5,
        2.5,
        3.5,
    ]
    # dimension checks still run inside the call
    with pytest.raises(units.DimensionError, match="cannot add"):
        flow(None, 0.5 * u.fs, pos, pos)

    # determinant and inverse of a 3x3 cell, by the adjugate
    @functools.partial(units.vectorize, signature="(3,3)->(),(3,3)")
    def det_and_inverse(a: Any) -> tuple[Any, Any]:
        cof = [
            [
                a[(i + 1) % 3, (j + 1) % 3] * a[(i + 2) % 3, (j + 2) % 3]
                - a[(i + 1) % 3, (j + 2) % 3] * a[(i + 2) % 3, (j + 1) % 3]
                for i in range(3)
            ]
            for j in range(3)
        ]
        det = a[0, 0] * cof[0][0] + a[0, 1] * cof[1][0] + a[0, 2] * cof[2][0]
        return det, units.stack([units.stack(row) for row in cof]) / det

    cells = jnp.stack([jnp.diag(arr([2.0, 3.0, 4.0])), eye(3)]) * u.angstrom
    volume, inverse = det_and_inverse(cells)
    assert list(volume.to(u.angstrom**3)) == [24.0, 1.0]
    assert inverse.to(u.angstrom**-1)[0, 0].tolist() == [0.5, 0.0, 0.0]

    # a plain call stays plain, and relabel asserts what the function computed
    @functools.partial(units.vectorize, signature="(3,3),(3)->(3)")
    def matvec(m: Any, v: Any) -> Any:
        return units.einsum("ij,j->i", m, v)

    assert matvec(cells, ones((2, 3))).to(u.angstrom)[0].tolist() == [2.0, 3.0, 4.0]
    assert matvec(ones((2, 3, 3)), ones((2, 3)))[0].tolist() == [3.0, 3.0, 3.0]
    counted = units.vectorize(lambda a, b: a * b, signature="(),()->()")(
        ones(3), arange(3.0)
    )
    counted = units.relabel(counted, units.dim_of(u.eV))
    assert list(counted.to(u.eV)) == [0.0, 1.0, 2.0]
    # size-1 batch axes broadcast against each other and against longer ones
    grid = flow(None, arr([[0.5], [1.0]]) * u.fs, pos[None, :1], vel[None, :1])
    assert grid.shape == (2, 1, 3)
    assert grid.to(u.angstrom)[:, 0, 0].tolist() == [1.5, 2.0]
    lone = flow(None, arr([[0.5]]) * u.fs, pos[:1], vel[:1])
    assert lone.shape == (1, 1, 3)
    assert lone.to(u.angstrom)[0, 0].tolist() == [1.5, 2.5, 3.5]

    # the decorator-factory form, with an excluded scale and a keyword argument
    @units.vectorize(signature="(n)->()", excluded=frozenset({1}))
    def total(x: Any, scale: float, *, offset: Any) -> Any:
        return x.sum() * scale + offset

    summed = total(pos, 2.0, offset=1.0 * u.angstrom)
    assert summed.to(u.angstrom).tolist() == [13.0, 31.0]
    # jnp.vectorize checks the signature, the exclusions and the core sizes
    with pytest.raises(ValueError, match="not a valid gufunc signature"):
        units.vectorize(lambda a: a, signature="(n)")(pos)
    with pytest.raises(TypeError, match="can only exclude integer or string"):
        bad: Any = frozenset({1.5})
        units.vectorize(lambda a: a, excluded=bad)(pos)
    with pytest.raises(ValueError, match="inconsistent size for core dimension '3'"):
        matvec(cells, ones((2, 4)))
    with pytest.raises(TypeError, match="wrong number of output arguments"):
        units.vectorize(lambda a: (a, a), signature="(n)->(n)")(pos)
    # jnp.vectorize itself rejects quantities when it converts the operands
    with pytest.raises(TypeError, match=r"does not convert|Unexpected input type"):
        jnp_vectorize(lambda a, b: a + b, signature="(n),(n)->(n)")(pos, pos)
    assert jnp_vectorize(lambda a, b: a + b, signature="(n),(n)->(n)")(
        ones((2, 3)), ones((2, 3))
    )[0].tolist() == [2.0, 2.0, 2.0]


def test_checkpoint():
    def energy(r: Any) -> Any:
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
    # grad through the checkpoint is labelled as usual
    g = units.grad(ck)(r)
    assert units.dim_of(g) == units.dim_of(u.eV / u.angstrom)
    npt.assert_allclose(g.to(u.eV / u.angstrom), [2.0, 4.0, 6.0])


def test_label_validation():
    # wrong labels are caught early: branch mismatches, vjp outputs mixing bases,
    # and dim= arguments that are not dimension vectors (a Unit is the likely slip)
    r = arr([1.0, 2.0]) * u.angstrom
    with pytest.raises(TypeError, match="same pytree structure"):
        jax.lax.cond(True, lambda x: (x, x * 2.0), lambda x: [x, x * 2.0], r)
    with pytest.raises(TypeError, match="same pytree structure"):
        jax.lax.cond(True, lambda x: {"a": x}, lambda x: {"b": x}, r)
    with pytest.raises(TypeError, match="same pytree structure"):
        jax.lax.switch(1, [lambda x: (x, x), lambda x: (x, (x,))], r)
    # matching structures work
    got = jax.lax.cond(False, lambda x: {"a": x}, lambda x: {"a": x * 3.0}, r)
    npt.assert_allclose(got["a"].to(u.angstrom), [3.0, 6.0])

    def two_bases(x: Any) -> tuple[Any, Any]:
        out = x * (1.0 * u.eV / u.angstrom)
        with units.base(units.SI):
            return out, out.rebase()

    with pytest.raises(units.DimensionError, match="in base atomistic and in base SI"):
        units.vjp(two_bases, r)
    out, pull = units.vjp(lambda x: (x * (1.0 * u.eV / u.angstrom),) * 2, r)
    (grad,) = pull(tuple(units.ones_like(o) for o in out))
    npt.assert_allclose(grad.to(u.eV / u.angstrom), [2.0, 2.0])

    for name, call in (
        ("apply", lambda d: units.apply(jnp.mod, r, r, dim=d)),
        ("apply", lambda d: units.einsum("i,i->", r, r, dim=d)),
        ("relabel", lambda d: units.relabel(r, d)),
    ):
        for bad in (u.angstrom, (0.5, 0, 0, 0, 0), "m=1", 1.0 * u.angstrom):
            with pytest.raises(
                units.DimensionError, match=rf"units\.{name}: .* is not a dimension"
            ):
                call(bad)
    # dim= is optional everywhere except relabel
    with pytest.raises(units.DimensionError, match=r"units\.relabel: None is not"):
        units.relabel(r, None)  # pyrefly: ignore[bad-argument-type]
    assert units.dim_of(
        units.apply(jnp.mod, r, r, dim=units.dim_of(u.angstrom))
    ) == units.dim_of(u.angstrom)
    with pytest.raises(units.DimensionError, match="not a rational exponent"):
        units.dim_from_str("m=1/0")

    # magnitudes have to be floating: an integer dtype rounds the rescale factor away
    counts = np_array([1, 2])
    for what, call in (
        ("from_dim", lambda d: units.from_dim(counts, d)),
        ("relabel", lambda d: units.relabel(arr([1, 2]), d)),
    ):
        with pytest.raises(
            units.DimensionError, match=rf"units\.{what}: int64 magnitudes cannot carry"
        ):
            call(units.dim_of(u.angstrom))
        assert units.dim_of(call(units.ONE)) == units.ONE  # a count reads back as one
    floats = units.from_dim(counts.astype(float), units.dim_of(u.angstrom))
    npt.assert_allclose(floats.to(u.nm), [0.1, 0.2])  # the same numbers, converted


def test_stop_gradient():
    r = arr([1.0, 2.0]) * u.angstrom

    @jax.jit
    def held(x: Any) -> Any:
        return x + jax.lax.stop_gradient(x)

    npt.assert_allclose(held(r).to(u.angstrom), [2.0, 4.0])

    tree = jax.lax.stop_gradient({"r": r, "n": arr([3.0])})
    assert units.dim_of(tree["r"]) == units.dim_of(u.angstrom)
    assert not units.is_quantity(tree["n"])

    def energy(x: Any) -> Any:
        # only the live half of the sum carries a derivative
        return ((x + jax.lax.stop_gradient(x)) ** 2).sum() * (
            1.0 * u.eV / u.angstrom**2
        )

    g = jax.jit(units.grad(energy))(r)
    npt.assert_allclose(g.to(u.eV / u.angstrom), [4.0, 8.0])


def test_argnums_validation():
    # a repeated argnum would overwrite itself and hand back a zero gradient
    def area(a: Any, b: Any) -> Any:
        return a * b

    args = (2.0 * u.angstrom, 3.0 * u.angstrom)
    for build in (units.grad, units.value_and_grad, units.jacrev, units.jacfwd):
        for argnums in ((0, 0), (0, -2)):  # the second names argument 0 again
            with pytest.raises(ValueError, match="names the same argument twice"):
                build(area, argnums=argnums)(*args)
        with pytest.raises(TypeError, match="requires at least 4 positional"):
            build(area, argnums=(0, 3))(*args)
    assert val(units.grad(area, argnums=-1)(*args)) == 2.0
