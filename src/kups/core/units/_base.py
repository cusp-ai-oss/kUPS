# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Coherent unit bases, their scale factors, and the active base context."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import math
from collections.abc import Generator
from fractions import Fraction as Frac

from kups.core.units._unit import (
    AXES,
    CHARGE,
    CURRENT,
    ENERGY,
    FORCE,
    LENGTH,
    MASS,
    ONE,
    PRESSURE,
    TEMPERATURE,
    TIME,
    AnyUnit,
    Dim,
    DimensionError,
    Factors,
    _require_unit,
    _spell,
    eliminate,
    exps,
)
from kups.core.units._unit import (
    solve as _dim_solve,
)
from kups.core.units.catalog import CATALOG, SI_BASE, amu, angstrom, e, eV

# Pin labels with a checked dimension, so `base(energy=u.m)` fails.
KINDS: dict[str, Dim] = {
    "length": LENGTH,
    "mass": MASS,
    "time": TIME,
    "current": CURRENT,
    "temperature": TEMPERATURE,
    "energy": ENERGY,
    "force": FORCE,
    "pressure": PRESSURE,
    "charge": CHARGE,
}


def _root(x: Frac, d: int) -> float:
    """Closest float to ``x ** (1/d)`` for positive rational ``x``; inf on overflow."""
    if x <= 0:
        raise DimensionError(
            f"kups.core.units: cannot take the {d}-th root of {x} while solving a base "
            "system; every base unit must be a positive finite size"
        )
    if x == 1:
        return 1.0
    if d == 1:
        try:
            return float(x)
        except OverflowError:
            return math.inf
    e, step = 0, 1 << d  # normalize into [1, 2)
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


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class BaseSystem:
    """An immutable coherent basis with one scale factor per dimension.

    Attributes:
      name: display name, shown in a quantity's type.
      pins: the pinned units, a full basis of the five dimensions.
      labels: the pins' labels, in the same order.
    """

    name: str
    pins: tuple[AnyUnit, ...]
    labels: tuple[str, ...]
    _cache: dict[Dim, tuple[tuple[Frac, ...], float]] = dataclasses.field(
        init=False, default_factory=dict, repr=False
    )
    _key: tuple[float, ...] = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Record the pins and solve the five base-dimension scales."""
        if len(self.pins) != len(self.labels):
            raise TypeError(
                "kups.core.units: BaseSystem needs one label per pin, got "
                f"{len(self.labels)} "
                f"labels for {len(self.pins)} pins."
            )
        object.__setattr__(self, "pins", tuple(self.pins))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "_key", tuple(self.base_units().values()))

    # compare by the solved scales, so equivalent bases share jit caches
    def __hash__(self) -> int:
        return hash((BaseSystem, self._key))

    def __eq__(self, other: object) -> bool:
        return self is other or (
            isinstance(other, BaseSystem) and self._key == other._key
        )

    def __repr__(self) -> str:
        return self.name

    def _decompose(self, dim: Dim) -> tuple[tuple[Frac, ...], float]:
        hit = self._cache.get(dim)
        if hit is None:
            x, acc, d = self._exact_scale(dim)
            hit = self._cache[dim] = (x, _root(acc, d))
        return hit

    def _exact_scale(self, dim: Dim) -> tuple[tuple[Frac, ...], Frac, int]:
        """Return powers, radicand and degree, with scale = radicand ** (1/degree)."""
        x = _dim_solve(tuple(p.dim for p in self.pins), dim)
        assert x is not None, "base pins are a full basis"
        d = math.lcm(*(c.denominator for c in x))
        acc = Frac(1)
        for p, c in zip(self.pins, x, strict=True):
            if c:
                acc *= Frac(p.scale) ** int(c * d)
        return x, acc, d

    def scale(self, dim: Dim) -> float:
        """Size of this base's unit of ``dim``, in SI base units.

        Args:
          dim: dimension vector.

        Returns:
          The correctly rounded scale; ``0.0`` or ``inf`` outside the float range.
        """
        return self._decompose(dim)[1]

    def factors(self, dim: Dim) -> Factors:
        """Write ``dim`` as a product of this base's pins.

        Args:
          dim: dimension vector.

        Returns:
          (name, exponent) pairs for pins with a nonzero exponent, in pin order.
        """
        x, _ = self._decompose(dim)
        return tuple((p.name, c) for p, c in zip(self.pins, x, strict=True) if c)

    def symbol(self, dim: Dim) -> str:
        """Display symbol for ``dim`` in this base, e.g. ``J`` in SI.

        Args:
          dim: dimension vector.

        Returns:
          A matching catalogue name, else a parseable product of pin names
          (``eV/Å``); "" for a dimensionless ``dim``.
        """
        if dim == ONE:
            return ""
        s = self.scale(dim)
        for u in CATALOG:
            if u.dim == dim and math.isclose(u.scale, s, rel_tol=1e-12):
                return u.name
        return _spell(self.factors(dim))

    def base_units(self) -> dict[str, float]:
        """Solved scales of the five base dimensions.

        Returns:
          Mapping from base-dimension symbol to its size in SI base units.
        """
        return {
            n: self.scale(tuple(1 if i == j else 0 for j in range(len(AXES))))
            for i, n in enumerate(AXES)
        }

    def describe(self) -> str:
        """Summarize the pinned units and the base sizes they imply.

        Returns:
          Two lines: the pins as ``label=unit``, then the five solved SI sizes.
        """
        pins = ", ".join(
            f"{lb}={p}" for lb, p in zip(self.labels, self.pins, strict=True)
        )
        solved = ", ".join(f"{k}={v!r}" for k, v in self.base_units().items())
        return f"base {self.name}: pinned [{pins}]\n    solved SI sizes: {solved}"


def _reject(labels: list[str], us: list[AnyUnit], row: tuple[Frac, ...]) -> None:
    """Raise DimensionError for pins that are not dimensionally independent."""
    rel = " * ".join(
        f"({lb}={u})^{c}" for lb, u, c in zip(labels, us, row, strict=True) if c
    )
    factor = math.prod(u.scale ** float(c) for u, c in zip(us, row, strict=True) if c)
    detail = (
        "so they cannot all be pinned independently; drop one of them"
        if math.isclose(factor, 1.0, rel_tol=1e-9)
        else f"and their scales are inconsistent too -- that product is {factor!r}, "
        "not 1, so no single scale factor can serve every dimension"
    )
    raise DimensionError(
        f"kups.core.units: not a coherent base system. {rel} is dimensionless, "
        f"{detail}."
    )


_BUILT: dict[BaseSystem, BaseSystem] = {}  # interned by solved scales


def _build(name: str, pinned: list[tuple[str, AnyUnit]]) -> BaseSystem:
    """Build a base system from (label, unit) pins, rejecting incoherent ones."""
    labels = [lb for lb, _ in pinned]
    us = [u for _, u in pinned]
    for lb, u in pinned:
        if not (math.isfinite(u.scale) and u.scale > 0):
            raise DimensionError(
                f"kups.core.units: {lb}={u} has scale {u.scale!r}; a base unit must "
                "have a "
                "positive finite size. If a power overflowed or underflowed, "
                "pin a smaller power."
            )
        if any(Frac(e).denominator != 1 for e in u.dim):
            raise DimensionError(
                f"kups.core.units: {lb}={u} has a fractional dimension {exps(u.dim)}. "
                "Fractional pins can introduce rounding error in the solved "
                "base scales; pin an integral power instead."
            )
        want = KINDS.get(lb)
        if want is not None and u.dim != want:
            raise DimensionError(
                f"kups.core.units: {lb}={u} has dimension {exps(u.dim)}, but {lb} must "
                "be "
                f"{exps(want)}. Rename the keyword if you meant a different kind."
            )
    # each pin's dimension, augmented with identity to track row provenance
    n = len(us)
    rows = [
        [Frac(x) for x in u.dim] + [Frac(int(i == j)) for j in range(n)]
        for i, u in enumerate(us)
    ]
    rank = eliminate(rows, len(AXES))
    for row in rows[rank:]:
        _reject(labels, us, tuple(row[len(AXES) :]))
    pins, labs = list(us), list(labels)
    for i, cand in enumerate(SI_BASE):  # pad with SI base units, axis order
        if len(pins) == len(AXES):
            break
        if _dim_solve(tuple(p.dim for p in pins), cand.dim) is None:
            pins.append(cand)
            labs.append(AXES[i])
    bs = BaseSystem(name, tuple(pins), tuple(labs))
    return _BUILT.setdefault(bs, bs)


SI = _build("SI", [])
# charge pinned to e: `1 * u.e` is 1 and the Coulomb constant is 14.4
ATOMISTIC = _build(
    "atomistic", [("length", angstrom), ("mass", amu), ("energy", eV), ("charge", e)]
)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class _Scope:
    """A context entry; identity tells apart nested scopes of the same system."""

    system: BaseSystem


# base used outside any scope; scopes are per task and per thread
_DEFAULT: BaseSystem = ATOMISTIC
_SCOPE: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "kups_units_base", default=None
)
_INTERLEAVED = (
    "kups.core.units: a 'with units.base(...)' scope was not the innermost one "
    "when it closed, so the base was reset to the units.set_base default. Use "
    "nested 'with' blocks to enter and leave scopes in order."
)


def active_base() -> BaseSystem:
    """Return the base system new quantities are stored in.

    Returns:
      The innermost ``units.base`` scope's system, else the ``units.set_base``
      default, which starts as ``ATOMISTIC``.
    """
    scope = _SCOPE.get()
    return _DEFAULT if scope is None else scope.system


def _require_base(system: object, what: str) -> None:
    if not isinstance(system, BaseSystem):
        raise TypeError(
            f"kups.core.units: {what} must be a BaseSystem, not "
            f"{type(system).__name__} "
            f"({system!r}). Use units.SI, units.ATOMISTIC, or a base created "
            "with units.base(...)."
        )


def _request(
    what: str, system: BaseSystem | None, pins: dict[str, AnyUnit]
) -> BaseSystem:
    """Turn a public base request into a BaseSystem."""
    if system is not None:
        if pins:
            raise TypeError(f"{what}: pass a BaseSystem or keyword pins, not both")
        _require_base(system, "the base system")
        return system
    if not pins:
        raise DimensionError(
            f"kups.core.units: {what} needs at least one pinned unit, e.g. "
            f"{what}(length=u.angstrom, mass=u.amu, energy=u.eV); pass units.SI "
            f"explicitly for the SI base"
        )
    for lb, unit in pins.items():
        _require_unit(unit, f"{lb}=")
    return _build("+".join(str(v) for v in pins.values()), list(pins.items()))


@contextlib.contextmanager
def base(
    system: BaseSystem | None = None, /, **pins: AnyUnit
) -> Generator[BaseSystem, None, None]:
    """Make a base system active inside a ``with`` block.

    Args:
      system: a ready-made BaseSystem, or None to build one from ``pins``.
      **pins: label=unit pairs to pin; the other dimensions are solved for.

    Yields:
      The active BaseSystem, reused if an equivalent one exists.

    Raises:
      TypeError: if both a system and pins are given, or either has the wrong type.
      DimensionError: if nothing is pinned, the request is not a coherent basis, or
        the scope was not the innermost one when it closed.
    """
    bs = _request("units.base", system, pins)
    scope = _Scope(bs)
    token = _SCOPE.set(scope)
    try:
        yield bs
    finally:
        if _SCOPE.get() is not scope:
            _SCOPE.set(None)  # scopes interleaved; fall back to the set_base default
            raise DimensionError(_INTERLEAVED)
        _SCOPE.reset(token)


def set_base(system: BaseSystem | None = None, /, **pins: AnyUnit) -> BaseSystem:
    """Set the default base system for the whole program; call it at startup.

    The default starts as ``ATOMISTIC``; ``units.set_base(units.SI)`` switches to SI.

    Args:
      system: a ready-made BaseSystem, or None to build one from ``pins``.
      **pins: label=unit pairs to pin.

    Returns:
      The active BaseSystem, reused if an equivalent one exists.

    Raises:
      TypeError: if both a system and pins are given, or either has the wrong type.
      DimensionError: if nothing is pinned, or the request is not a coherent basis.
    """
    global _DEFAULT  # the one writer of the default
    _DEFAULT = _request("units.set_base", system, pins)
    return _DEFAULT
