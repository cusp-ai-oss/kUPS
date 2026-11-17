# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Coherent unit bases, their scale factors, and the active base context."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
from collections.abc import Generator, Sequence
from fractions import Fraction as Frac

import jax

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
    Unit,
    eliminate,
    exps,
    require_unit,
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


def _check_pins(labels: Sequence[str], pins: Sequence[AnyUnit]) -> None:
    """Require each pin to be a positive finite integral power of its kind."""
    for lb, u in zip(labels, pins, strict=True):
        if not (math.isfinite(u.scale) and u.scale > 0):
            raise DimensionError(
                f"kups.core.units: {lb}={u} has scale {u.scale!r}; a base unit "
                "must have a positive finite size. If a power overflowed or "
                "underflowed, pin a smaller power."
            )
        if any(Frac(e).denominator != 1 for e in u.dim):
            raise DimensionError(
                f"kups.core.units: {lb}={u} has a fractional dimension "
                f"{exps(u.dim)}. Fractional pins can introduce rounding error in "
                "the solved base scales; pin an integral power instead."
            )
        want = KINDS.get(lb)
        if want is not None and u.dim != want:
            raise DimensionError(
                f"kups.core.units: {lb}={u} has dimension {exps(u.dim)}, but {lb} "
                f"must be {exps(want)}. Rename the keyword if you meant a "
                "different kind."
            )


@dataclasses.dataclass(frozen=True, eq=False)
class BaseSystem:
    """An immutable coherent basis with one scale factor per dimension.

    Attributes:
      name: display name.
      pins: the pinned units, a full basis of the five dimensions.
      labels: the pins' labels, in the same order.
    """

    name: str
    pins: tuple[AnyUnit, ...]
    labels: tuple[str, ...]
    _cache: dict[Dim, AnyUnit] = dataclasses.field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        """Store pins and labels as tuples, and check that the pins are a basis."""
        if len(self.pins) != len(self.labels):
            raise TypeError(
                "kups.core.units: BaseSystem needs one label per pin, got "
                f"{len(self.labels)} "
                f"labels for {len(self.pins)} pins."
            )
        object.__setattr__(self, "pins", tuple(self.pins))
        object.__setattr__(self, "labels", tuple(self.labels))
        _check_pins(self.labels, self.pins)
        dims = tuple(p.dim for p in self.pins)
        if len(dims) != len(AXES) or any(
            _dim_solve(dims, x.dim) is None for x in SI_BASE
        ):
            raise DimensionError(
                f"kups.core.units: the pins {', '.join(map(str, self.pins))} are not "
                "five independent units, so they do not fix every dimension's scale. "
                "Build a base with units.base(...), which pads the pins with SI base "
                "units."
            )

    @functools.cached_property
    def _key(self) -> tuple[float, ...]:
        """The solved base-dimension scales, which identify the base."""
        return tuple(self.base_units().values())

    # compare by the solved scales, so equivalent bases share jit caches
    def __hash__(self) -> int:
        return hash((BaseSystem, self._key))

    def __eq__(self, other: object) -> bool:
        return self is other or (
            isinstance(other, BaseSystem) and self._key == other._key
        )

    def __repr__(self) -> str:
        return self.name

    def _product(self, dim: Dim) -> AnyUnit:
        """The product of pins with dimension ``dim``, multiplied out into atoms."""
        hit = self._cache.get(dim)
        if hit is None:
            x = _dim_solve(tuple(p.dim for p in self.pins), dim)
            assert x is not None, "base pins are a full basis"
            f = tuple(
                (a, e * c)
                for p, c in zip(self.pins, x, strict=True)
                for a, e in p.factors
            )
            hit = self._cache[dim] = Unit(f)
        return hit

    def scale(self, dim: Dim) -> float:
        """Size of this base's unit of ``dim``, in SI base units.

        Args:
          dim: dimension vector.

        Returns:
          The correctly rounded scale; ``0.0`` or ``inf`` outside the float range.
        """
        return self._product(dim).scale

    def unit(self, dim: Dim) -> AnyUnit:
        """This base's unit of ``dim``, e.g. ``J`` in SI.

        Args:
          dim: dimension vector.

        Returns:
          A catalogue unit of exactly this size, else the product of pins
          (``eV/Å``).
        """
        out = self._product(dim)
        if dim == ONE:
            return out
        return next((u for u in CATALOG if u.dim == dim and u.scale == out.scale), out)

    def symbol(self, dim: Dim) -> str:
        """Display symbol for ``dim`` in this base, e.g. ``J`` in SI.

        Args:
          dim: dimension vector.

        Returns:
          The name of ``unit(dim)``; "" for a dimensionless
          ``dim``.
        """
        return "" if dim == ONE else self.unit(dim).name

    def base_units(self) -> dict[str, float]:
        """Solved scales of the five base dimensions.

        Returns:
          Mapping from base-dimension symbol to its size in SI base units.
        """
        return {n: self.scale(u.dim) for n, u in zip(AXES, SI_BASE, strict=True)}

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
    _check_pins(labels, us)
    # each pin's dimension, augmented with identity to track row provenance
    n = len(us)
    rows = [
        [Frac(x) for x in u.dim] + [Frac(int(i == j)) for j in range(n)]
        for i, u in enumerate(us)
    ]
    rank = eliminate(rows, len(AXES))
    for row in rows[rank:]:
        _reject(labels, us, tuple(row[len(AXES) :]))
    for ax, cand in zip(AXES, SI_BASE, strict=True):  # pad with SI base units
        if _dim_solve(tuple(p.dim for p in us), cand.dim) is None:
            us.append(cand)
            labels.append(ax)
    bs = BaseSystem(name, tuple(us), tuple(labels))
    return _BUILT.setdefault(bs, bs)


SI = _build("SI", [])
# charge pinned to e, so charges display in e
ATOMISTIC = _build(
    "atomistic", [("length", angstrom), ("mass", amu), ("energy", eV), ("charge", e)]
)


# jax keys its trace caches on the active base, so switching bases retraces
_ACTIVE = jax.make_user_context(ATOMISTIC)


def active_base() -> BaseSystem:
    """Return the innermost ``units.base`` scope's system, else ``ATOMISTIC``."""
    return _ACTIVE.value


def require_base(system: object, what: str) -> None:
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
        require_base(system, "the base system")
        return system
    if not pins:
        raise DimensionError(
            f"kups.core.units: {what} needs at least one pinned unit, e.g. "
            f"{what}(length=u.angstrom, mass=u.amu, energy=u.eV); pass units.SI "
            f"explicitly for the SI base"
        )
    for lb, unit in pins.items():
        require_unit(unit, f"{lb}=")
    return _build("+".join(str(v) for v in pins.values()), list(pins.items()))


@contextlib.contextmanager
def base(
    system: BaseSystem | None = None, /, **pins: AnyUnit
) -> Generator[BaseSystem, None, None]:
    """Make a base system active inside a ``with`` block.

    Scopes are per thread, like JAX config contexts.

    Args:
      system: a ready-made BaseSystem, or None to build one from ``pins``.
      **pins: label=unit pairs to pin; the other dimensions are solved for.

    Yields:
      The active BaseSystem, reused if an equivalent one exists.

    Raises:
      TypeError: if both a system and pins are given, or either has the wrong type.
      DimensionError: if nothing is pinned, or the request is not a coherent basis.
    """
    bs = _request("units.base", system, pins)
    with _ACTIVE(bs):
        yield bs
