# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for jax transforms over quantities."""

from fractions import Fraction

import jax
import pytest

from kups.core import units
from kups.core.units import Energy, Length, Mass, Velocity, VelocityUnit
from kups.core.units import catalog as u

from ._helpers import (
    arr,
    jit,
    val,
)

SPEED: VelocityUnit = (u.eV / u.amu) ** Fraction(1, 2)  # the atomistic velocity unit


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
