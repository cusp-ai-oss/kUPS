# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dimension vectors and the ``Unit`` class.

A dimension is a 5-tuple of rational exponents over the SI base axes
``(m, kg, s, A, K)``. Higher layers are imported inside functions to avoid cycles.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from fractions import Fraction as Frac
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, overload

import jax
from jax.typing import ArrayLike

from kups.core.units._typed import _E_A, _E_K, _E_KG, _E_M, _E_S

AXES = ("m", "kg", "s", "A", "K")
Dim = tuple[int | Frac, ...]
Factors = tuple[tuple[str, Frac], ...]
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
    """Raised for a dimension mismatch, a base mismatch, or a bad base request."""


def _mk(vals: Iterable[Any]) -> Dim:
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


def ddiv(a: Dim, b: Dim) -> Dim:
    """Dimension of a quotient.

    Args:
      a: numerator dimension.
      b: denominator dimension.

    Returns:
      The exponentwise difference.
    """
    return _mk(Frac(x) - Frac(y) for x, y in zip(a, b, strict=True))


def dpow(a: Dim, e: Frac) -> Dim:
    """Dimension of a power.

    Args:
      a: dimension of the base.
      e: exponent.

    Returns:
      The exponentwise product with ``e``.
    """
    return _mk(Frac(x) * e for x in a)


def exps(d: Dim) -> str:
    """Format a dimension as its raw exponent list, for error messages.

    Args:
      d: dimension vector.

    Returns:
      A string like ``"(m=1, kg=1, s=-2, A=0, K=0)"``.
    """
    return "(" + ", ".join(f"{n}={v}" for n, v in zip(AXES, d, strict=True)) + ")"


def eliminate(rows: list[list[Frac]], ncols: int) -> int:
    """Gauss-Jordan eliminate ``rows`` in place over exact rationals.

    Args:
      rows: matrix rows; extra trailing columns are carried along.
      ncols: number of leading columns to pivot on.

    Returns:
      The rank; pivot rows come first, so rows from the rank on are zero in the
      first ``ncols`` columns.
    """
    r = 0
    for c in range(ncols):
        p = next((i for i in range(r, len(rows)) if rows[i][c]), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        inv = rows[r][c]
        rows[r] = [v / inv for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r], strict=True)]
        r += 1
    return r


def solve(cols: tuple[Dim, ...], target: Dim) -> tuple[Frac, ...] | None:
    """Write ``target`` as an exact rational combination of ``cols``.

    Args:
      cols: dimension vectors, assumed independent.
      target: dimension vector to decompose.

    Returns:
      Exponents ``x`` with ``sum_j x_j * cols[j] == target``, or None if
      ``target`` is outside their span.
    """
    k = len(cols)
    mat = [[Frac(c[i]) for c in cols] + [Frac(t)] for i, t in enumerate(target)]
    r = eliminate(mat, k)
    if any(row[k] for row in mat[r:]):
        return None
    x = [Frac(0)] * k
    for row in mat[:r]:  # a pivot row leads with a 1 in its pivot column
        x[next(j for j in range(k) if row[j])] = row[k]
    return tuple(x)


def exponent(e: Any) -> Frac:
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

    from kups.core.units._core import Quantity
    from kups.core.units._typed import _F_A, _F_K, _F_KG, _F_M, _F_S, _K


@dataclasses.dataclass(frozen=True)
class Unit(Generic[_E_M, _E_KG, _E_S, _E_A, _E_K]):
    """A named multiple of a dimension, with the exponents as type parameters.

    Attributes:
      name: display name.
      dim: dimension exponents.
      scale: size in SI base units.
      factors: component units when this is a derived product, otherwise empty.
    """

    name: str
    dim: Dim
    scale: float
    factors: Factors = ()

    __array_ufunc__: ClassVar[None] = None  # numpy defers to __rmul__

    def __repr__(self) -> str:
        return self.name

    # Overloaded on self: a subclass narrowing the return would break substitutability.
    @overload
    def __mul__(self: Dimensionless, other: Dimensionless) -> Dimensionless: ...
    @overload
    def __mul__(
        self: Dimensionless, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]: ...
    @overload
    def __mul__(self, other: Dimensionless) -> Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    @overload
    def __mul__(
        self, other: Unit[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]
    ) -> Dimensionless: ...
    @overload
    def __mul__(self, other: Unit[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]) -> jax.Array: ...
    @overload
    def __mul__(
        self, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Unit[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def __mul__(self: Dimensionless, other: ArrayLike) -> jax.Array: ...
    @overload
    def __mul__(self, other: ArrayLike) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __mul__(self, other: Any) -> Any:
        """Multiply by a unit, or attach this unit to magnitudes."""
        return _product(self, other)

    @overload
    def __truediv__(self: Dimensionless, other: Dimensionless) -> Dimensionless: ...
    @overload
    def __truediv__(
        self: Dimensionless, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Unit[-_F_M, -_F_KG, -_F_S, -_F_A, -_F_K]: ...
    @overload
    def __truediv__(
        self, other: Dimensionless
    ) -> Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    @overload
    def __truediv__(
        self, other: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]
    ) -> Dimensionless: ...
    @overload
    def __truediv__(self, other: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]) -> jax.Array: ...
    @overload
    def __truediv__(
        self, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Unit[
        _E_M - _F_M,
        _E_KG - _F_KG,
        _E_S - _F_S,
        _E_A - _F_A,
        _E_K - _F_K,
    ]: ...
    @overload
    def __truediv__(self: Dimensionless, other: ArrayLike) -> jax.Array: ...
    @overload
    def __truediv__(
        self, other: ArrayLike
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __truediv__(self, other: Any) -> Any:
        """Divide by a unit, or attach the reciprocal to magnitudes."""
        return _quotient(self, other)

    @overload
    def __pow__(self: Dimensionless, e: Int[_K]) -> Dimensionless: ...
    @overload
    def __pow__(
        self, e: Int[_K]
    ) -> Unit[_E_M * _K, _E_KG * _K, _E_S * _K, _E_A * _K, _E_K * _K]: ...
    @overload
    def __pow__(self, e: float | Frac) -> Any: ...  # non-integer exponent
    def __pow__(self, e: Any) -> Any:
        """Raise to a static rational power; the dimension is raised with it."""
        ef = exponent(e)
        f = tuple((n, x * ef) for n, x in _atoms(self) if x * ef)
        return _derive(f, dpow(self.dim, ef), _scaled(self.scale, ef))

    @overload
    def __rmul__(self: Dimensionless, x: ArrayLike) -> jax.Array: ...
    @overload
    def __rmul__(self, x: ArrayLike) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __rmul__(self, x: Any) -> Any:
        """Attach this unit to plain magnitudes."""
        return _attach(x, self)

    @overload
    def __rtruediv__(self: Dimensionless, x: ArrayLike) -> jax.Array: ...
    @overload
    def __rtruediv__(
        self, x: Literal[1]
    ) -> Unit[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]: ...
    @overload
    def __rtruediv__(
        self, x: ArrayLike
    ) -> Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]: ...
    def __rtruediv__(self, x: Any) -> Any:
        """``1 / unit`` is the reciprocal unit; other ``x`` give ``x * self**-1``."""
        inv = self**-1
        return inv if type(x) is int and x == 1 else _attach(x, inv)


type AnyUnit = Unit[Any, Any, Any, Any, Any]

if TYPE_CHECKING:
    # An alias, since a subclass would be assignable to every Unit[...]. The
    # self-typed overloads above type its decay; the runtime subclass implements it.
    type Dimensionless = Unit[0, 0, 0, 0, 0]

    # The alias is not callable, so this stands in as its constructor.
    def _dimensionless(
        name: str, dim: Dim, scale: float, factors: Factors = ()
    ) -> Dimensionless: ...

else:

    class Dimensionless(Unit):
        """A unit with zero dimension; magnitudes times it are plain arrays.

        Only the reflected operators differ from ``Unit``.
        """

        def __rmul__(self, x):
            return _attach(x, self)

        # ``1 / u.mol`` is just the number 1/N_A, unlike ``1 / u.fs`` which is a unit.
        def __rtruediv__(self, x):
            return _attach(x, self**-1)

    _dimensionless = Dimensionless


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


def _require_unit(unit: object, what: str) -> None:
    if not isinstance(unit, Unit):
        raise TypeError(
            f"kups.core.units: {what} must be a Unit, not {type(unit).__name__} "
            f"({unit!r}). "
            "Pass a unit such as u.m or u.eV**-1, without magnitudes. "
            "Use units.catalog.unit(name) to look up a string."
        )


def _scaled(scale: float, e: Frac) -> float:
    """Raise a unit's scale to a power, returning inf on overflow."""
    try:
        return scale ** float(e)
    except OverflowError:
        return math.inf


def _atoms(x: AnyUnit) -> Factors:
    """Return the factors of ``x``, or ``x`` itself as a single factor."""
    return x.factors or ((x.name, Frac(1)),)


def _combine(a: Factors, b: Factors, sign: int) -> Factors:
    """Merge the factors of a product (sign=1) or quotient (sign=-1)."""
    acc: dict[str, Frac] = {}
    for name, e in a + tuple((n, sign * e) for n, e in b):
        acc[name] = acc.get(name, Frac(0)) + e
    return tuple(sorted((n, e) for n, e in acc.items() if e))


def _spell(factors: Factors) -> str:
    """Spell factors as a name ``units.catalog.parse`` can read back."""
    num = ".".join(n if e == 1 else f"{n}^{e}" for n, e in factors if e > 0)
    den = [n if -e == 1 else f"{n}^{-e}" for n, e in factors if e < 0]
    if not den:
        return num or "1"
    return f"{num or '1'}/{den[0] if len(den) == 1 else '(' + '.'.join(den) + ')'}"


def _derive(factors: Factors, dim: Dim, scale: float) -> AnyUnit:
    """Build a derived unit, named after its factors."""
    cls = _dimensionless if dim == ONE else Unit
    return cls(_spell(factors), dim, scale, factors)


def _forward(x: Any, unit: AnyUnit, invert: bool = False) -> Any:
    """Attach ``unit`` to plain magnitudes; return NotImplemented for quantities."""
    from kups.core.units._core import is_quantity  # avoids a cycle

    if is_quantity(x):
        return NotImplemented
    return _attach(1 / x if invert else x, unit)


def _product(a: AnyUnit, b: Any) -> Any:
    """Multiply a unit by a unit, or attach it to magnitudes."""
    if not isinstance(b, Unit):
        return _forward(b, a)
    f = _combine(_atoms(a), _atoms(b), 1)
    return _derive(f, dmul(a.dim, b.dim), a.scale * b.scale)


def _quotient(a: AnyUnit, b: Any) -> Any:
    """Divide a unit by a unit, or attach it to the reciprocal of magnitudes."""
    if not isinstance(b, Unit):
        return _forward(b, a, invert=True)
    f = _combine(_atoms(a), _atoms(b), -1)
    return _derive(f, ddiv(a.dim, b.dim), a.scale / b.scale)


def _attach(x: Any, unit: AnyUnit) -> Any:
    """Convert magnitudes given in ``unit`` into the active base."""
    if isinstance(x, Unit):
        return NotImplemented
    from kups.core.units._core import from_unit  # avoids a cycle

    return from_unit(x, unit)
