# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Static typing tests; every pyrefly ignore pins an expected checker error."""

import importlib.util
import pathlib
import subprocess
import sys
from typing import Any, Never, assert_type

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from kups.core import units
from kups.core.units import (
    Action,
    Area,
    AreaUnit,
    Charge,
    CompressibilityUnit,
    CoulombConstant,
    Dimensionless,
    Energy,
    EnergyUnit,
    Entropy,
    Force,
    ForceUnit,
    Frequency,
    FrequencyUnit,
    Length,
    LengthUnit,
    Momentum,
    NumberDensity,
    NumberDensityUnit,
    OddExponentRoot,
    Permittivity,
    Pressure,
    Quantity,
    Stiffness,
    Time,
    Unit,
    Velocity,
    Volume,
    VolumeUnit,
    Wavenumber,
    constants,
)
from kups.core.units import catalog as u
from kups.core.units import testing as ut

from ._helpers import arr

# aliases for dimensions no named type covers
type ForceSquaredUnit = Unit[2, 2, -4, 0, 0]
type UnknownUnit = Unit[int, int, int, int, int]
type Length12 = Quantity[12, 0, 0, 0, 0]


# more unnamed dimensions; lazy because Quantity is not subscriptable at runtime
type _EnergyLength = units.Quantity[3, 1, -2, 0, 0]
type _M4 = units.Quantity[4, 0, 0, 0, 0]
type _VelocitySquared = units.Quantity[2, 0, -2, 0, 0]


def test_unit_types():
    assert_type(u.angstrom, LengthUnit)
    assert_type(u.Angstrom, LengthUnit)
    assert_type(u.mol, Dimensionless)
    # 1 / u.s and u.s**-1 are the same type; any other numerator gives a Quantity
    assert_type(1 / u.s, FrequencyUnit)
    assert_type(u.s**-1, FrequencyUnit)
    assert_type(1.0 / u.s, Frequency)
    assert_type(2 / u.s, Frequency)


def test_misspelled_unit():
    with pytest.raises(AttributeError):
        _ = u.angstrum  # pyrefly: ignore[missing-attribute]


def test_unit_algebra():
    # multiplying units adds their exponents, and pyrefly does the addition
    assert_type(u.m * u.m, AreaUnit)
    assert_type(u.m * u.m * u.m, VolumeUnit)  # and a chain stays exact
    assert_type(u.m**3, VolumeUnit)
    assert_type(u.eV / u.angstrom, ForceUnit)  # energy over length is force
    assert_type(u.N * u.m, EnergyUnit)  # and force times length is energy again
    assert_type(u.kcal / u.mol, EnergyUnit)  # the mole is a pure count
    assert_type(u.mol * u.m, LengthUnit)
    assert_type(u.m / u.m, Dimensionless)  # the exponents cancel
    # a unit parsed at runtime has unknown exponents, and so do its products
    assert_type(u.unit("m") * u.s, UnknownUnit)
    assert_type(u.unit("m") / u.s, UnknownUnit)
    assert_type(u.unit("m") ** 2, UnknownUnit)
    assert_type(u.m**0.5, UnknownUnit)  # a non-integer power
    # no named unit for this dimension, but still exact
    assert_type((u.eV / u.angstrom) ** 2, ForceSquaredUnit)


def test_quantity_algebra():
    r = 3.0 * u.angstrom
    assert_type(r, Length)
    assert_type(2.0 * u.mol, jax.Array)  # dimensionless decays to a plain array
    assert_type(1.0 / u.s, Frequency)  # reciprocal magnitudes, dimension negated
    assert_type(r + r, Length)
    assert_type(r + u.angstrom, Length)  # a bare unit means one of it
    assert_type(2.0 * r, Length)  # a plain factor rescales without changing dimension
    assert_type(r / 2.0, Length)
    assert_type(-r, Length)
    assert_type(abs(r), Length)
    assert_type(r * r, Area)
    assert_type(r**3, Volume)
    assert_type(r / (2.0 * u.fs), Velocity)
    assert_type(r / r, jax.Array)  # identical dimensions cancel to plain magnitudes
    assert_type(r.rebase(units.SI), Length)
    assert_type(r.mean(), Length)
    assert_type(r.T, Length)
    assert_type(r.sum(), Length)
    assert_type(r.to(u.nm), jax.Array)  # same dimension, another unit
    assert_type(r.value, jax.Array)
    assert_type(r < r, jax.Array)
    assert_type(float(r), float)
    rs = arr([1.0, 2.0]) * u.angstrom
    assert_type(rs.at[0].add(r), Length)
    assert_type(rs.at[0].get(), Length)
    assert_type(r.sqrt(), OddExponentRoot)  # m^(1/2) is not an integer exponent
    _ = r.sqrt() * 2.0  # pyrefly: ignore[unsupported-operation]


def test_expression_chain():
    mass, speed = 1200.0 * u.kg, 90.0 * u.km / u.hour
    assert_type(speed, Velocity)
    assert_type(speed**2, _VelocitySquared)
    assert_type(mass * speed, Momentum)
    assert_type(0.5 * mass * speed**2, Energy)  # the whole of 1/2 m v^2
    assert float((0.5 * mass * speed**2).to(u.kJ)) == pytest.approx(375.0)
    # the Lennard-Jones exponents, and a dimensionless ratio in the middle
    r, sigma = 3.8 * u.angstrom, 3.4 * u.angstrom
    assert_type(sigma / r, jax.Array)
    assert_type(sigma**12, Length12)
    assert_type(sigma**12 / r**12, jax.Array)


def test_derivative_types():
    # a derivative's dimension is the output's over the input's, per order
    def energy(r: Length) -> Energy:
        return (1.0 * u.eV) * (u.angstrom / r)

    r = 3.0 * u.angstrom
    assert_type(units.grad(energy)(r), Force)  # dE/dr is a force
    assert_type(units.value_and_grad(energy)(r), tuple[Energy, Force])
    assert_type(units.hessian(energy)(r), Stiffness)  # d2E/dr2 is a stiffness
    assert_type(jax.vmap(energy)(arr([1.0, 2.0]) * u.angstrom), Energy)  # as is vmap
    assert float(units.grad(energy)(r).to(u.eV / u.angstrom)) == pytest.approx(
        -1.0 / 9.0
    )
    # an output of the input's dimension (or its square, for hessian) differentiates
    # to a plain array

    def double(x: Length) -> Length:
        return 2.0 * x

    def area(x: Length) -> Area:
        return x * x

    def mix(x: Length, t: Time) -> Length:
        return x + t * (1.0 * u.angstrom / u.s)

    assert_type(units.grad(double)(r), jax.Array)
    assert_type(units.value_and_grad(double)(r), tuple[Length, jax.Array])
    assert_type(units.hessian(area)(r), jax.Array)
    assert_type(units.vjp(double, r)[1](1.0), tuple[jax.Array])
    assert_type(units.vjp(double, r)[1](r), tuple[Length])
    pull = units.vjp(mix, r, 2.0 * u.s)[1]
    assert_type(pull(1.0), tuple[jax.Array, Velocity])
    assert not units.is_quantity(units.vjp(double, r)[1](1.0)[0])
    assert not units.is_quantity(pull(1.0)[0]) and units.is_quantity(pull(1.0)[1])
    # a seed of the inverse dimension cancels it
    assert_type(units.vjp(area, r)[1](1.0 / r), tuple[jax.Array])
    assert not units.is_quantity(units.vjp(area, r)[1](1.0 / r)[0])


def test_vjp_types():
    # the pullback labels each cotangent dim(ct) * dim(out) / dim(primal)
    def energy(r: Length) -> Energy:
        return (1.0 * u.eV) * (u.angstrom / r)

    def with_aux(r: Length) -> tuple[Energy, str]:
        return energy(r), "aux"

    r = 3.0 * u.angstrom
    assert_type(units.vectorize(energy)(r), Energy)  # signature-preserving
    assert_type(units.vectorize(energy, signature="()->()")(r), Energy)
    out, vjp_fn = units.vjp(energy, r)
    assert_type(out, Energy)
    assert_type(vjp_fn(1.0), tuple[Force])  # a plain cotangent is dimensionless
    assert_type(vjp_fn(r), tuple[Energy])
    _, aux_fn, aux = units.vjp(with_aux, r, has_aux=True)
    assert_type(aux_fn(1.0), tuple[Force])
    assert_type(aux, str)

    # a dimensionless output pulls back to one over the primal's dimension
    def ratio(x: Length) -> jax.Array:
        return x / (1.0 * u.angstrom)

    assert_type(units.vjp(ratio, r)[1](1.0), tuple[Wavenumber])

    def spring(k: Stiffness, x: Length) -> Energy:
        return 0.5 * k * x**2

    _, pull2 = units.vjp(spring, 250.0 * u.N / u.m, r)
    assert_type(pull2(1.0), tuple[Area, Force])
    assert_type(pull2(1.0 / r), tuple[Length, Stiffness])
    assert_type(pull2(1.0 / (1.0 * u.angstrom**2)), tuple[jax.Array, Pressure])
    # jax annotates these Any, so a checker cannot follow a dimension through one
    assert_type(jax.jvp(energy, (r,), (r,)), tuple[Any, ...])
    assert_type(jax.lax.cond(True, energy, energy, r), Any)
    # relabel's dimension is a runtime value, so its result is Any too
    assert_type(units.relabel(r, units.ONE), Any)
    assert float(vjp_fn(1.0)[0].to(u.eV / u.angstrom)) == pytest.approx(-1.0 / 9.0)
    assert float(vjp_fn(r)[0].to(u.eV)) == pytest.approx(-1.0 / 3.0)
    assert not units.is_quantity(pull2(1.0 / (1.0 * u.angstrom**2))[0])


def test_dimensionless_algebra():
    assert_type(u.mol**-1, Dimensionless)
    assert_type(u.mol * u.mol, Dimensionless)
    assert_type(u.mol / u.mol, Dimensionless)
    assert_type(2.0 * (u.mol**-1), jax.Array)  # so no .to() is offered on it
    assert_type(u.angstrom * 2.0, Length)  # unit times magnitudes, either order
    assert_type(u.angstrom / 2.0, Length)


def test_equality_raises():
    # == and != raise, and their declared return type is Never
    r = 3.0 * u.angstrom
    with pytest.raises(units.DimensionError):
        assert_type(r == r, Never)
    with pytest.raises(units.DimensionError):
        assert_type(r != r, Never)


def test_plain_operand_types():
    # dimension-agnostic names return plain arrays for plain operands
    a, cond = arr([1.0, -2.0]), arr([True, False])
    r = arr([3.0, 4.0]) * u.angstrom
    assert_type(units.minimum(a, a), jax.Array)
    assert_type(units.maximum(a, 0.0), jax.Array)
    assert_type(units.zeros_like(a), jax.Array)
    assert_type(units.ones_like(a), jax.Array)
    assert_type(units.full((2,), u.deg), jax.Array)
    assert_type(units.where(cond, a, a), jax.Array)
    assert_type(units.stack([a, a]), jax.Array)
    assert_type(units.concatenate([a, a]), jax.Array)
    # a quantity operand still carries the dimension out
    assert_type(units.minimum(r, r), Length)
    assert_type(units.maximum(r, 0.0 * u.angstrom), Length)
    assert_type(units.zeros_like(r), Length)
    assert_type(units.ones_like(r), Length)
    assert_type(units.where(cond, r, r), Length)
    assert_type(r.clip(0.0 * u.angstrom), Length)
    assert_type(r.norm(ord=jnp.inf), Length)  # ord takes a float, as in jnp
    assert_type(units.stack([r, r]), Length)
    assert_type(units.concatenate([r, r]), Length)


def test_plain_operand_runtime():
    # the runtime half of the overload contract above
    a = arr([1.0, -2.0])
    for value in (
        units.minimum(a, a),
        units.maximum(a, 0.0),
        units.zeros_like(a),
        units.ones_like(a),
        units.where(arr([True, False]), a, a),
        units.stack([a, a]),
        units.concatenate([a, a]),
    ):
        assert not units.is_quantity(value)
        assert isinstance(value, jax.Array)


def test_constant_types():
    assert_type(constants.c, Velocity)
    assert_type(constants.hbar, Action)
    assert_type(constants.k_B, Entropy)
    assert_type(constants.eps_0, Permittivity)
    assert_type(constants.k_e, CoulombConstant)
    assert_type(constants.k_B * (300.0 * u.K), Energy)  # k_B T is an energy
    q, r = 1.0 * u.e, 3.0 * u.angstrom
    assert_type(q, Charge)
    assert_type(constants.k_e * q * q / r, Energy)  # and so is a Coulomb pair term
    assert float((constants.k_e * q * q / r).to(u.eV)) == pytest.approx(
        4.799881826156339
    )


def test_new_dimension_names():
    assert_type(u.Pa**-1, CompressibilityUnit)
    assert_type(u.angstrom**-3, NumberDensityUnit)
    assert_type(2.0 * u.angstrom**-3, NumberDensity)


def test_linalg_types():
    # a wrong result dimension should be a checker error rather than a wrong
    # number at runtime
    cell = arr([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * u.angstrom
    assert_type(units.qr(cell), tuple[jax.Array, Length])
    assert_type(units.qr(cell, mode="r"), Length)
    assert_type(units.eigh(cell), tuple[Length, jax.Array])
    assert_type(units.inv(cell), Wavenumber)
    assert_type(units.cross(cell[0], cell[1]), Area)
    assert_type(units.cross(cell[0], cell[1] / (1.0 * u.angstrom**2)), jax.Array)


def test_testing_helpers():
    # the helpers pin both sides to the unit's dimension, so a wrong unit in a
    # test is a checker error before it is a DimensionError
    r, t = arr([1.0, 2.0]) * u.angstrom, arr([1.0, 2.0]) * u.s
    ut.assert_allclose(r, arr([1.0, 2.0]) * u.angstrom, u.angstrom)
    ut.assert_allclose(r, arr([1.0, 2.0]), u.angstrom, rtol=1e-9)
    ut.assert_allclose(2.0 * u.eV, 2.0, u.eV)  # a bare number is eV
    ut.assert_array_equal(r, r, u.angstrom)
    with pytest.raises(units.DimensionError):
        ut.assert_allclose(r, r, u.eV)  # pyrefly: ignore[bad-argument-type]
    with pytest.raises(units.DimensionError):
        ut.assert_array_equal(r, t, u.angstrom)  # pyrefly: ignore[bad-argument-type]


def test_quantity_arraylike():
    # the raw Quantity dataclass takes any ArrayLike; every exit is a jax.Array
    dim = units.dim_of(u.angstrom)
    q = units.Quantity(3.0, dim, units.SI)
    n = units.Quantity(np.zeros(3), dim, units.SI)
    assert_type(q.v, ArrayLike)
    assert_type(q.value, jax.Array)
    assert_type(q.to(u.angstrom), jax.Array)
    assert_type(n.value.shape, tuple[int, ...])
    assert isinstance(q.value, jax.Array) and isinstance(n.to(u.nm), jax.Array)


def test_einsum_types():
    # the result dimension is the product of the operands', whatever the subscripts do
    A = arr([[1.0, 0.0], [0.0, 1.0]]) * u.angstrom
    B = arr([[1.0, 0.0], [0.0, 1.0]]) * u.eV
    p = arr([[1.0, 0.0], [0.0, 1.0]])
    inv = p / (1.0 * u.angstrom)
    assert_type(units.einsum("ii->i", A), Length)
    assert_type(units.einsum("ij,jk->ik", A, A), Area)
    assert_type(units.einsum("ij,jk->ik", A, B), _EnergyLength)
    assert_type(units.einsum("ij,jk->ik", A, p), Length)
    assert_type(units.einsum("ij,jk->ik", p, B), Energy)
    assert_type(units.einsum("ij,jk->ik", p, p), jax.Array)
    assert_type(units.einsum("ij,jk->ik", A, inv), jax.Array)  # the product cancels
    assert_type(units.einsum("ij,jk,kl->il", A, B, inv), Energy)
    assert_type(units.einsum("ij,jk,kl->il", p, A, p), Length)
    # two quantities that cancel next to a plain array give a plain array
    assert_type(units.einsum("ij,jk,kl->il", A, p, inv), jax.Array)
    assert_type(units.einsum("ij,jk,kl->il", A, inv, p), jax.Array)
    assert_type(units.einsum("ij,jk,kl->il", p, A, inv), jax.Array)
    assert not units.is_quantity(units.einsum("ij,jk,kl->il", A, p, inv))
    # four operands, a runtime dim= and the interleaved form are beyond the overloads
    assert_type(units.einsum("ij,jk->ik", A, B, dim=units.ONE), Any)
    assert_type(units.einsum(A, [0, 1], B, [1, 2]), Any)
    assert_type(units.einsum("ij,jk,kl,lm->im", A, A, A, A), Any)


def test_checker_accepts_this_file():
    # every assert_type above holds and every pyrefly ignore suppresses a real error
    if importlib.util.find_spec("pyrefly") is None:
        pytest.skip("pyrefly is not installed")
    root = pathlib.Path(__file__).parents[3]
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--search-path",
            str(root),
            __file__,
        ],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_quantity_surface_types():
    # each line mirrors a kUPS call site; a wrong result dimension should be a
    # checker error rather than a wrong number at runtime
    r = arr([1.0, 2.0]) * u.angstrom
    cell = arr([[1.0, 0.0], [0.0, 1.0]]) * u.angstrom
    assert_type(cell @ r, Area)  # a matrix of lengths times a length
    assert_type(r % r, Length)  # a remainder keeps the shared dimension
    assert_type(r // r, jax.Array)  # and a count cancels it
    assert_type(r.var(ddof=1), Area)  # a variance squares the dimension
    assert_type(r.at[0].get(mode="fill", fill_value=0.0 * u.angstrom), Length)
    assert_type(units.full_like(r, 1.0 * u.angstrom), Length)
    assert_type(units.linspace(0.0 * u.angstrom, 10.0 * u.angstrom, 5), Length)
    assert_type(units.zeros_like(r, shape=(4,)), Length)
    # apply keeps the dimension for one operand; dim= or a second operand gives Any
    assert_type(units.apply(jnp.tril, cell), Length)
    assert_type(units.apply(jnp.cumsum, r), Length)
    assert_type(units.apply(jnp.isfinite, r, dim=units.ONE), Any)
    assert_type(units.apply(jnp.matmul, cell, cell), Any)
    # .unit carries the quantity's exponents
    assert_type(r.unit, LengthUnit)
    assert_type((5.0 * u.eV).unit, EnergyUnit)
    assert_type(r.to(r.unit), jax.Array)


def test_dimension_mismatch():
    # mixing dimensions is a checker error before it is a runtime error
    r, t = 3.0 * u.angstrom, 2.0 * u.s
    with pytest.raises(units.DimensionError):
        r + t  # pyrefly: ignore[unsupported-operation]
    with pytest.raises(units.DimensionError):
        r < t  # pyrefly: ignore[unsupported-operation]
    with pytest.raises(units.DimensionError):
        r.to(u.s)  # pyrefly: ignore[bad-argument-type]
    # the must-agree names are overloaded, so a mismatch matches no overload
    with pytest.raises(units.DimensionError):
        units.minimum(r, t)  # pyrefly: ignore[no-matching-overload]
    with pytest.raises(units.DimensionError):
        r % t  # pyrefly: ignore[unsupported-operation]
    with pytest.raises(units.DimensionError):
        r // t  # pyrefly: ignore[unsupported-operation]
    # a dimensionless unit is a mismatch too
    with pytest.raises(units.DimensionError):
        r + u.mol  # pyrefly: ignore[unsupported-operation]
    with pytest.raises(units.DimensionError):
        r.to(u.mol)  # pyrefly: ignore[bad-argument-type]
    # so is a plain zero
    with pytest.raises(units.DimensionError):
        r + 0.0  # pyrefly: ignore[unsupported-operation]
    with pytest.raises(units.DimensionError):
        units.maximum(r, 0.0)  # pyrefly: ignore[no-matching-overload]
    with pytest.raises(units.DimensionError):
        r.at[0].set(0.0)  # pyrefly: ignore[bad-argument-type]


def test_narrowing_and_methods():
    # consumers hold leaves as Quantity | jax.Array unions, so is_quantity must
    # narrow the negative branch too (TypeIs)
    leaf: Length | jax.Array = 3.0 * u.angstrom
    if units.is_quantity(leaf):
        assert_type(leaf.to(u.angstrom), jax.Array)
        assert_type(leaf.value, jax.Array)
        assert_type(units.base_of(leaf), units.BaseSystem)
    else:
        assert_type(leaf, jax.Array)
    r = arr([1.0, 2.0]) * u.angstrom
    assert_type(r.ravel(), Length)  # the method keeps the dimension
    assert_type(r.repeat(2), Length)
    assert_type(r.sort(), Length)
    assert_type(r.std(), Length)
    assert_type(r.take(jnp.array([0])), Length)
    assert_type(r.var(), Area)  # the variance of a length is an area
    # sqrt halves the exponents, written out for the square of every named dimension
    assert_type(units.sqrt(r * r), Length)
    assert_type(units.sqrt(r.var()), Length)
    assert_type(units.sqrt((r * r).mean()), Length)
    assert_type((r * r).sqrt(), Length)  # and the method agrees
    assert_type(units.sqrt(arr([4.0, 9.0])), jax.Array)  # plain in, plain out
    assert_type(
        units.sqrt(5.0 * u.eV), OddExponentRoot
    )  # eV^(1/2): no integer exponent
    # even exponents are exact for any dimension
    eVA = 5.0 * u.eV * r  # eV.angstrom, which no name covers
    assert_type(units.sqrt(eVA * eVA), _EnergyLength)
    assert_type(units.sqrt((eVA * eVA).mean()), _EnergyLength)
    assert_type(units.sqrt((r * r) * (r * r)), Area)  # m^4 -> m^2
    assert_type(units.sqrt(((r * r) * (r * r)) ** 2), _M4)  # m^8 -> m^4
    assert_type((r * r).sqrt(), Length)  # the method says the same
    # known gap: an unnamed odd exponent floors, so this is m/s for m^(3/2).kg^(1/2)/s
    assert_type(units.sqrt(eVA), Velocity)

    assert_type(r.dot(r), Area)
    assert_type(r.dot(arr([1.0, 2.0])), Length)  # a plain operand keeps it
    assert_type(r.argmin(), jax.Array)  # an index measures nothing
    assert_type(r.any(), jax.Array)
    assert_type(r.nonzero(), tuple[jax.Array, ...])
    assert_type(r.repeat(jnp.array([1, 3]), axis=0), Length)
    assert_type(r.copy(), Length)
    assert_type(r.size, int)
    assert_type(r % (2.0 * u.angstrom), Length)  # a remainder keeps it
    assert_type(r // (2.0 * u.angstrom), jax.Array)  # a count cancels it
    assert_type(f"{r[0]:.3f}", str)  # a numeric spec applies to one magnitude
    assert_type(units.dim_str(units.dim_of(u.angstrom)), str)
