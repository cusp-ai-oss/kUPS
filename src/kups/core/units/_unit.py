# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dimension vectors and the ``Unit`` class.

A dimension is a 5-tuple of rational exponents over the SI base axes
``(m, kg, s, A, K)``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterable
from fractions import Fraction as Frac
from typing import TYPE_CHECKING, Generic, Literal, NamedTuple, overload

from kups.core.units._typed import E_A, E_K, E_KG, E_M, E_S

Dim = tuple[int | Frac, ...]


class Atom(NamedTuple):
    """A unit defined directly rather than as a product, e.g. ``eV`` or ``keV``."""

    name: str
    dim: Dim
    scale: float


Factors = tuple[tuple[Atom, Frac], ...]
ONE: Dim = (0, 0, 0, 0, 0)

LENGTH: Dim = (1, 0, 0, 0, 0)
MASS: Dim = (0, 1, 0, 0, 0)
TIME: Dim = (0, 0, 1, 0, 0)
CURRENT: Dim = (0, 0, 0, 1, 0)
TEMPERATURE: Dim = (0, 0, 0, 0, 1)
ENERGY: Dim = (2, 1, -2, 0, 0)
FORCE: Dim = (1, 1, -2, 0, 0)
PRESSURE: Dim = (-1, 1, -2, 0, 0)
CHARGE: Dim = (0, 0, 1, 1, 0)
FREQUENCY: Dim = (0, 0, -1, 0, 0)
VOLTAGE: Dim = (2, 1, -3, -1, 0)
POWER: Dim = (2, 1, -3, 0, 0)


class DimensionError(TypeError):
    """Raised for a dimension mismatch."""


def _mk(vals: Iterable[Frac]) -> Dim:
    """Normalize rational exponents, demoting integral ones to int."""
    return tuple(int(v) if Frac(v).denominator == 1 else Frac(v) for v in vals)


def dmul(a: Dim, b: Dim) -> Dim:
    """Dimension of a product.

    Args:
      a: left dimension.
      b: right dimension.

    Returns:
      The exponentwise sum.
    """
    return _mk(Frac(x) + Frac(y) for x, y in zip(a, b, strict=True))


def dpow(a: Dim, e: Frac) -> Dim:
    """Dimension of a power.

    Args:
      a: dimension of the base.
      e: exponent.

    Returns:
      The exponentwise product with ``e``.
    """
    return _mk(Frac(x) * e for x in a)


def exponent(e: float | Frac) -> Frac:
    """Convert a dimension exponent to a Fraction.

    Args:
      e: an int, a Fraction, or a float that is a small rational.

    Returns:
      The exponent as a Fraction.

    Raises:
      DimensionError: if ``e`` is a float that is not a small rational.
    """
    if isinstance(e, int | Frac):
        return Frac(e)
    if not math.isfinite(e):
        raise DimensionError(
            f"kups.core.units: exponent {e!r} is not a small rational; a dimension "
            "exponent "
            "has to be finite. Pass an exact Fraction, or take the power of a "
            "dimensionless ratio."
        )
    ef = Frac(e).limit_denominator(64)
    if float(ef) != float(e):
        raise DimensionError(
            f"kups.core.units: exponent {e!r} is not a small rational; the nearest is "
            f"{ef} ({float(ef)!r}), which would change the result's dimension. "
            "Pass an exact Fraction, or take the power of a dimensionless ratio."
        )
    return ef


if TYPE_CHECKING:
    from shape_extensions import Int

    from kups.core.units._typed import F_A, F_K, F_KG, F_M, F_S, K


@dataclasses.dataclass(frozen=True)
class Unit(Generic[E_M, E_KG, E_S, E_A, E_K]):
    """A product of powers of atoms, with the dimension exponents as type parameters.

    The factors are reduced on construction, so units are equal when they reduce
    alike: ``eV == (eV / Å) * Å`` while ``J != N * m``.

    Attributes:
      factors: sorted (atom, exponent) pairs; an atom alone is ``((atom, 1),)``.
    """

    factors: Factors

    def __post_init__(self) -> None:
        acc: dict[Atom, Frac] = {}
        for atom, e in self.factors:
            acc[atom] = acc.get(atom, Frac(0)) + e
        reduced = tuple(sorted((x, e) for x, e in acc.items() if e))
        object.__setattr__(self, "factors", reduced)

    @functools.cached_property
    def name(self) -> str:
        """Display name; a derived unit is named after its factors."""
        return _spell(self.factors)

    @functools.cached_property
    def dim(self) -> Dim:
        """Dimension exponents."""
        return _dim(self.factors)

    @functools.cached_property
    def scale(self) -> float:
        """Size in SI base units; a derived unit rounds the exact product once."""
        return _magnitude(self.factors)

    def __repr__(self) -> str:
        return self.name

    def __mul__(
        self, other: Unit[F_M, F_KG, F_S, F_A, F_K]
    ) -> Unit[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]:
        """Multiply by a unit."""
        return _product(self, other)

    def __truediv__(
        self, other: Unit[F_M, F_KG, F_S, F_A, F_K]
    ) -> Unit[
        E_M - F_M,
        E_KG - F_KG,
        E_S - F_S,
        E_A - F_A,
        E_K - F_K,
    ]:
        """Divide by a unit."""
        return _quotient(self, other)

    @overload
    def __pow__(
        self, e: Int[K]
    ) -> Unit[E_M * K, E_KG * K, E_S * K, E_A * K, E_K * K]: ...
    @overload
    def __pow__(self, e: float | Frac) -> AnyUnit: ...  # non-integer exponent
    def __pow__(self, e: float | Frac) -> AnyUnit:
        """Raise to a static rational power; the dimension is raised with it."""
        ef = exponent(e)
        return Unit(tuple((a, x * ef) for a, x in self.factors))

    def __rtruediv__(self, x: Literal[1]) -> Unit[-E_M, -E_KG, -E_S, -E_A, -E_K]:
        """``1 / unit`` is the reciprocal unit."""
        return self**-1 if type(x) is int and x == 1 else NotImplemented


type AnyUnit = Unit[int, int, int, int, int]

if TYPE_CHECKING:
    # An alias, since a subclass would be assignable to every Unit[...].
    type Dimensionless = Unit[0, 0, 0, 0, 0]
else:
    Dimensionless = Unit


if TYPE_CHECKING:
    # Dimension aliases used by the catalogue.
    type AreaUnit = Unit[2, 0, 0, 0, 0]
    type VolumeUnit = Unit[3, 0, 0, 0, 0]
    type VelocityUnit = Unit[1, 0, -1, 0, 0]
    type AccelerationUnit = Unit[1, 0, -2, 0, 0]
    type MomentumUnit = Unit[1, 1, -1, 0, 0]
    type DensityUnit = Unit[-3, 1, 0, 0, 0]
    type WavenumberUnit = Unit[-1, 0, 0, 0, 0]
    type StiffnessUnit = Unit[0, 1, -2, 0, 0]
    type ActionUnit = Unit[2, 1, -1, 0, 0]
    type EntropyUnit = Unit[2, 1, -2, 0, -1]
    type PermittivityUnit = Unit[-3, -1, 4, 2, 0]
    type CoulombConstantUnit = Unit[3, 1, -4, -2, 0]
    type DipoleMomentUnit = Unit[1, 0, 1, 1, 0]
    type NumberDensityUnit = Unit[-3, 0, 0, 0, 0]
    type CompressibilityUnit = Unit[1, -1, 2, 0, 0]
    type ChargeUnit = Unit[0, 0, 1, 1, 0]
    type CurrentUnit = Unit[0, 0, 0, 1, 0]
    type EnergyUnit = Unit[2, 1, -2, 0, 0]
    type ForceUnit = Unit[1, 1, -2, 0, 0]
    type LengthUnit = Unit[1, 0, 0, 0, 0]
    type MassUnit = Unit[0, 1, 0, 0, 0]
    type PowerUnit = Unit[2, 1, -3, 0, 0]
    type PressureUnit = Unit[-1, 1, -2, 0, 0]
    type TemperatureUnit = Unit[0, 0, 0, 0, 1]
    type TimeUnit = Unit[0, 0, 1, 0, 0]
    type FrequencyUnit = Unit[0, 0, -1, 0, 0]
    type VoltageUnit = Unit[2, 1, -3, -1, 0]

else:
    # Erased to plain Unit so runtime annotation readers (e.g. pydantic) see the class.
    type AccelerationUnit = Unit
    type ActionUnit = Unit
    type AreaUnit = Unit
    type ChargeUnit = Unit
    type CompressibilityUnit = Unit
    type CoulombConstantUnit = Unit
    type CurrentUnit = Unit
    type DensityUnit = Unit
    type DipoleMomentUnit = Unit
    type EnergyUnit = Unit
    type EntropyUnit = Unit
    type ForceUnit = Unit
    type FrequencyUnit = Unit
    type LengthUnit = Unit
    type MassUnit = Unit
    type MomentumUnit = Unit
    type NumberDensityUnit = Unit
    type PermittivityUnit = Unit
    type PowerUnit = Unit
    type PressureUnit = Unit
    type StiffnessUnit = Unit
    type TemperatureUnit = Unit
    type TimeUnit = Unit
    type VelocityUnit = Unit
    type VoltageUnit = Unit
    type VolumeUnit = Unit
    type WavenumberUnit = Unit


def _scaled(scale: float, e: Frac) -> float:
    """Raise a unit's scale to a power, returning inf on overflow."""
    try:
        return scale ** float(e)
    except OverflowError:
        return math.inf


def root(x: Frac, d: int) -> float:
    """Closest float to ``x ** (1/d)`` for positive rational ``x``; inf on overflow."""
    if x == 1:
        return 1.0
    if d == 1:
        try:
            return float(x)
        except OverflowError:
            return math.inf
    e, step = 0, 1 << d  # normalize into [1, 2**d)
    while x >= 2:
        x /= step
        e += 1
    while x < 1:
        x *= step
        e -= 1
    k = 1 << 200  # ~200 bits, rounded once
    n = (x.numerator * k**d) // x.denominator
    lo, hi = 1, 1 << (n.bit_length() // d + 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if mid**d < n:
            lo = mid + 1
        else:
            hi = mid
    try:
        return math.ldexp(float(Frac(lo, k)), e)
    except OverflowError:
        return math.inf


def _magnitude(factors: Factors) -> float:
    """Size of a product of atoms in SI base units, rounded once."""
    if not all(0 < s < math.inf for (_, _, s), _ in factors):
        return math.prod(_scaled(s, e) for (_, _, s), e in factors)
    d = math.lcm(*(e.denominator for _, e in factors))
    exact = math.prod(
        (Frac(s) ** int(e * d) for (_, _, s), e in factors), start=Frac(1)
    )
    return root(exact, d)


def _power(name: str, e: Frac) -> str:
    """Spell one factor, bracketing a fractional power: ``m^(1/2)``."""
    if e == 1:
        return name
    return f"{name}^{e}" if e.denominator == 1 else f"{name}^({e})"


def _spell(factors: Factors) -> str:
    """Spell factors as a unit name, e.g. ``eV^2/Å^2`` or ``m^(1/2)/s``."""
    num = ".".join(_power(n, e) for (n, _, _), e in factors if e > 0)
    den = [_power(n, -e) for (n, _, _), e in factors if e < 0]
    if not den:
        return num or "1"
    return f"{num or '1'}/{den[0] if len(den) == 1 else '(' + '.'.join(den) + ')'}"


def _dim(factors: Factors) -> Dim:
    """Dimension of a product of atoms."""
    dim = ONE
    for (_, d, _), e in factors:
        dim = dmul(dim, dpow(d, e))
    return dim


def _product(a: AnyUnit, b: object) -> AnyUnit:
    """Multiply a unit by a unit."""
    if not isinstance(b, Unit):
        return NotImplemented
    return Unit(a.factors + b.factors)


def _quotient(a: AnyUnit, b: object) -> AnyUnit:
    """Divide a unit by a unit."""
    if not isinstance(b, Unit):
        return NotImplemented
    return Unit(a.factors + tuple((x, -e) for x, e in b.factors))
