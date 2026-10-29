# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The unit catalogue: one module-level constant per unit.

    >>> from kups.core.units import catalog as u
    >>> u.eV / u.angstrom
    eV/Å

Each constant is a typed ``Unit``; ``mol``, ``rad`` and ``deg`` are
``Dimensionless``. ``unit`` looks up a name, ``parse`` reads a number and a unit
expression, and ``require_dim`` checks a value's dimension. Names accept SI
prefixes, and every printed unit name parses back.
"""

import math
import re
from fractions import Fraction
from typing import Any, Final

from kups.core.units import _unit as u
from kups.core.units._unit import (
    AnyUnit,
    ChargeUnit,
    CurrentUnit,
    Dimensionless,
    DipoleMomentUnit,
    EnergyUnit,
    ForceUnit,
    FrequencyUnit,
    LengthUnit,
    MassUnit,
    PowerUnit,
    PressureUnit,
    TemperatureUnit,
    TimeUnit,
    Unit,
    VoltageUnit,
)

CATALOG: Final[list[AnyUnit]] = []

_NA = 6.02214076e23  # entities per mole, exact under the 2019 SI


def _u(name: str, dim: u.Dim, scale: float = 1.0) -> AnyUnit:
    """Define a unit and register it for display."""
    unit = Unit(name, dim, scale)
    CATALOG.append(unit)
    return unit


def _d(name: str, scale: float = 1.0) -> Dimensionless:
    """Define a dimensionless unit and register it for display."""
    unit = u._dimensionless(name, u.ONE, scale)
    CATALOG.append(unit)
    return unit


# SI base units
m: Final[LengthUnit] = _u("m", u.LENGTH)
kg: Final[MassUnit] = _u("kg", u.MASS)
s: Final[TimeUnit] = _u("s", u.TIME)
A: Final[CurrentUnit] = _u("A", u.CURRENT)
K: Final[TemperatureUnit] = _u("K", u.TEMPERATURE)

# Named SI derived units; `e` is listed so bases pinning it display charges in e.
N: Final[ForceUnit] = _u("N", u.FORCE)
J: Final[EnergyUnit] = _u("J", u.ENERGY)
Pa: Final[PressureUnit] = _u("Pa", u.PRESSURE)
Hz: Final[FrequencyUnit] = _u("Hz", u.FREQUENCY)
C: Final[ChargeUnit] = _u("C", u.CHARGE)
e: Final[ChargeUnit] = _u("e", u.CHARGE, 1.602176634e-19)
V: Final[VoltageUnit] = _u("V", u.VOLTAGE)
W: Final[PowerUnit] = _u("W", u.POWER)

# length
km: Final[LengthUnit] = _u("km", u.LENGTH, 1e3)
cm: Final[LengthUnit] = _u("cm", u.LENGTH, 1e-2)
mm: Final[LengthUnit] = _u("mm", u.LENGTH, 1e-3)
nm: Final[LengthUnit] = _u("nm", u.LENGTH, 1e-9)
angstrom: Final[LengthUnit] = _u("Å", u.LENGTH, 1e-10)
bohr: Final[LengthUnit] = _u("bohr", u.LENGTH, 5.29177210903e-11)

# mass
g: Final[MassUnit] = _u("g", u.MASS, 1e-3)
amu: Final[MassUnit] = _u("amu", u.MASS, 1.66053906660e-27)

# time
ms: Final[TimeUnit] = _u("ms", u.TIME, 1e-3)
ns: Final[TimeUnit] = _u("ns", u.TIME, 1e-9)
ps: Final[TimeUnit] = _u("ps", u.TIME, 1e-12)
fs: Final[TimeUnit] = _u("fs", u.TIME, 1e-15)
minute: Final[TimeUnit] = _u("minute", u.TIME, 60.0)
hour: Final[TimeUnit] = _u("hour", u.TIME, 3600.0)

# energy
eV: Final[EnergyUnit] = _u("eV", u.ENERGY, 1.602176634e-19)
kJ: Final[EnergyUnit] = _u("kJ", u.ENERGY, 1e3)
kcal: Final[EnergyUnit] = _u("kcal", u.ENERGY, 4184.0)
hartree: Final[EnergyUnit] = _u("hartree", u.ENERGY, 4.3597447222071e-18)
rydberg: Final[EnergyUnit] = _u("rydberg", u.ENERGY, 2.1798723611035e-18)
invcm: Final[EnergyUnit] = _u("invcm", u.ENERGY, 1.986445857148928e-23)

# pressure
GPa: Final[PressureUnit] = _u("GPa", u.PRESSURE, 1e9)
bar: Final[PressureUnit] = _u("bar", u.PRESSURE, 1e5)

# dipole moment
debye: Final[DipoleMomentUnit] = _u("debye", (1, 0, 1, 1, 0), 3.33564095198152e-30)

# dimensionless: the mole is a pure count, angles are in radians
mol: Final[Dimensionless] = _d("mol", _NA)
rad: Final[Dimensionless] = _d("rad")
deg: Final[Dimensionless] = _d("deg", math.pi / 180.0)

# aliases
Da: Final[MassUnit] = amu
Angstrom: Final[LengthUnit] = angstrom

SI_BASE: Final[tuple[AnyUnit, ...]] = (m, kg, s, A, K)

# Catalogue names plus common alternate spellings.
_SPELLINGS: Final[dict[str, AnyUnit]] = {x.name: x for x in CATALOG} | {
    "angstrom": angstrom,
    "Angstrom": angstrom,
    "Ang": angstrom,
    "\u212b": angstrom,  # ANGSTROM SIGN
    "ampere": A,
    "Da": amu,
    "dalton": amu,
    "D": debye,
    "Ry": rydberg,
}

# SI prefixes as powers of ten.
_PREFIXES: Final[dict[str, int]] = {
    "Q": 30,
    "R": 27,
    "Y": 24,
    "Z": 21,
    "E": 18,
    "P": 15,
    "T": 12,
    "G": 9,
    "M": 6,
    "k": 3,
    "h": 2,
    "da": 1,
    "d": -1,
    "c": -2,
    "m": -3,
    "µ": -6,
    "n": -9,
    "p": -12,
    "f": -15,
    "a": -18,
    "z": -21,
    "y": -24,
    "r": -27,
    "q": -30,
}
# Alternate prefix spellings mapped to canonical symbols.
_PREFIX_SPELLINGS: Final[dict[str, str]] = {p: p for p in _PREFIXES} | {
    "u": "µ",
    "\u03bc": "µ",
    "quetta": "Q",
    "ronna": "R",
    "yotta": "Y",
    "zetta": "Z",
    "exa": "E",
    "peta": "P",
    "tera": "T",
    "giga": "G",
    "mega": "M",
    "kilo": "k",
    "hecto": "h",
    "deca": "da",
    "deka": "da",
    "deci": "d",
    "centi": "c",
    "milli": "m",
    "micro": "µ",
    "nano": "n",
    "pico": "p",
    "femto": "f",
    "atto": "a",
    "zepto": "z",
    "yocto": "y",
    "ronto": "r",
    "quecto": "q",
}
# Names that already carry a prefix; they take no second one.
_PREFIXED: Final[frozenset[str]] = frozenset(
    {"kg", "km", "cm", "mm", "nm", "ms", "ns", "ps", "fs", "kJ", "kcal", "GPa"}
)
_ROOTS: Final[dict[str, AnyUnit]] = {
    k: v for k, v in _SPELLINGS.items() if k not in _PREFIXED
}
# Prefixed units, built on demand.
_DERIVED: dict[str, AnyUnit] = {}
# Rejected prefixed spellings, keyed by the name they would form, with the reason.
_REFUSED: Final[dict[str, str]] = {
    "hbar": "ħ is a constant, units.constants.hbar, and `hbar` reads as it rather "
    "than as a hectobar, which kups.core.units has no spelling for; write the pressure "
    "in bar or Pa",
}
# spellings listed in error messages
_LISTED: Final[str] = ", ".join(sorted(k for k in _SPELLINGS if k != "\u212b"))

_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
# '.' is the product operator inside expressions, so a number token cannot end in one
_TOKEN_NUMBER = r"[+-]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?"
# alternation order matters: `**` must come before `*`, and a number before `\w+`
_TOKEN = re.compile(rf"\*\*|{_TOKEN_NUMBER}|\w+|[*/.^()]")
_NUMBER_RE = re.compile(_NUMBER)
_INTEGER = re.compile(r"[+-]?\d+")


def _splits(name: str) -> list[tuple[str, AnyUnit]]:
    """Find every split of a name into an SI prefix and an unprefixed unit."""
    return [
        (_PREFIX_SPELLINGS[name[:i]], _ROOTS[name[i:]])
        for i in range(1, len(name))
        if name[:i] in _PREFIX_SPELLINGS and name[i:] in _ROOTS
    ]


def _prefixed(name: str) -> AnyUnit | None:
    """Resolve a name as prefix + unit, or None; raise if ambiguous or refused."""
    reads = _splits(name)
    if not reads:
        return None
    refused = {f"{sym}{root.name}" for sym, root in reads} & _REFUSED.keys()
    reads = [(sym, root) for sym, root in reads if f"{sym}{root.name}" not in _REFUSED]
    if not reads:
        why = _REFUSED[refused.pop()]
        raise u.DimensionError(f"kups.core.units: {name!r} is not a unit: {why}.")
    spelled = {f"{sym}{root.name}" for sym, root in reads}
    if len(spelled) > 1:
        raise u.DimensionError(
            f"kups.core.units: {name!r} is ambiguous: it reads as "
            f"{' and as '.join(sorted(spelled))}. Write the one you mean."
        )
    sym, root = reads[0]
    canonical = f"{sym}{root.name}"
    # `kilom` resolves to the catalogue's own `km`
    out = _SPELLINGS.get(canonical) or _DERIVED.get(canonical)
    if out is None:
        scale = float(Fraction(root.scale) * Fraction(10) ** _PREFIXES[sym])
        # a prefixed unit counts as a single factor, not as prefix times unit
        out = _DERIVED[canonical] = u._derive(
            ((canonical, Fraction(1)),), root.dim, scale
        )
    return out


def unit(name: str) -> AnyUnit:
    """Look up a unit by name.

    Args:
      name: a catalogue name, optionally with an SI prefix as a symbol or a word
        (``keV``, ``µm``, ``kilom``).

    Returns:
      The Unit; prefixed units are cached, so ``unit("um") is unit("µm")``.

    Raises:
      DimensionError: if the name is unknown, ambiguous, or a rejected spelling.
    """
    out = _SPELLINGS.get(name) or _prefixed(name)
    if out is None:
        doubled = next(
            (
                name[i:]
                for i in range(1, len(name))
                if name[:i] in _PREFIX_SPELLINGS and name[i:] in _PREFIXED
            ),
            None,
        )
        raise u.DimensionError(
            f"kups.core.units: {name!r} is not a unit. "
            + (
                f"{doubled!r} carries a prefix already, and a unit takes one. "
                if doubled
                else ""
            )
            + f"The spellings units.catalog answers to are {_LISTED}, each with an SI "
            f"prefix on it if it carries none, as a symbol or as a word: "
            f"{', '.join(_PREFIXES)}."
        )
    return out


def parse(text: str) -> Any:
    """Parse a number and a unit expression into a quantity.

    Args:
      text: the value to read, e.g. ``"300 K"`` or ``"0.001 eV/angstrom"``.

    Returns:
      The quantity in the active base, or a plain ``jax.Array`` when dimensionless.

    Raises:
      DimensionError: if the text has no leading number or a bad unit expression.
    """
    body = text.strip()
    lead = _NUMBER_RE.match(body)
    if lead is None:
        raise u.DimensionError(
            f"kups.core.units: cannot read {text!r} as a value: it must start with a "
            "number, "
            f"as in '300 K', '2 fs' or '0.001 eV/angstrom'."
        )
    rest = body[lead.end() :].strip()
    # a bare number is dimensionless; rad is the dimensionless unit with scale 1
    return float(lead.group()) * (_expression(rest, text) if rest else rad)


def require_dim(q: Any, unit: AnyUnit) -> None:
    """Check that a value has the same dimension as ``unit``.

    Args:
      q: a quantity, or any other value (treated as dimensionless).
      unit: the unit whose dimension the value must match.

    Raises:
      DimensionError: if the dimensions differ.
    """
    from kups.core.units._base import active_base  # avoids a cycle
    from kups.core.units._core import dim_of  # avoids a cycle

    got = dim_of(q)
    if got == unit.dim:
        return
    raise u.DimensionError(
        f"kups.core.units: expected a value in {unit.name}, got one in "
        f"{active_base().symbol(got) or 'no unit at all'}\n"
        f"  expected {u.exps(unit.dim)}\n  received {u.exps(got)}"
    )


def _expression(text: str, whole: str) -> AnyUnit:
    """Parse a whole unit expression; ``whole`` is the original text, for errors."""
    toks = _tokens(text, whole)
    out, i = _product(toks, 0, whole)
    if i < len(toks):
        raise u.DimensionError(
            f"kups.core.units: cannot read the unit of {whole!r}: expected '*' or '/' "
            "before "
            f"{toks[i]!r}."
        )
    return out


def _product(toks: list[str], i: int, whole: str) -> tuple[AnyUnit, int]:
    """Parse units joined by ``*``, ``.`` and ``/``, up to a ``)`` or the end."""
    out, i = _factor(toks, i, whole)
    while i < len(toks) and toks[i] in ("*", "/", "."):
        op, (rhs, i) = toks[i], _factor(toks, i + 1, whole)
        out = out / rhs if op == "/" else out * rhs
    return out, i


def _tokens(text: str, whole: str) -> list[str]:
    """Split a unit expression into unit, number and operator tokens."""
    out: list[str] = []
    pos = 0
    for m in _TOKEN.finditer(text):
        if text[pos : m.start()].strip():
            break
        out.append(m.group())
        pos = m.end()
    if text[pos:].strip():
        raise u.DimensionError(
            f"kups.core.units: cannot read the unit of {whole!r}: "
            f"{text[pos:].strip()!r} is "
            f"not part of a unit expression."
        )
    return out


def _factor(toks: list[str], i: int, whole: str) -> tuple[AnyUnit, int]:
    """Parse one unit or parenthesised group, with its power if any."""
    if i >= len(toks) or toks[i] == ")":
        raise u.DimensionError(
            f"kups.core.units: cannot read the unit of {whole!r}: it ends after an "
            "operator."
        )
    tok = toks[i]
    if tok == "(":
        out, i = _product(toks, i + 1, whole)
        if i >= len(toks) or toks[i] != ")":
            raise u.DimensionError(
                f"kups.core.units: cannot read the unit of {whole!r}: a '(' with no "
                "')'."
            )
        return _powered(out, toks, i + 1, whole)
    if _NUMBER_RE.fullmatch(tok):
        if float(tok) != 1.0:
            raise u.DimensionError(
                f"kups.core.units: cannot read the unit of {whole!r}: {tok!r} is a "
                "number "
                f"where a unit belongs; only a leading '1' stands for the "
                f"dimensionless unit, as in '4.57e-10 1/Pa'."
            )
        return rad, i + 1
    return _powered(unit(tok), toks, i + 1, whole)


def _powered(out: AnyUnit, toks: list[str], i: int, whole: str) -> tuple[AnyUnit, int]:
    """Apply a following ``^`` or ``**`` exponent, if any."""
    if i < len(toks) and toks[i] in ("**", "^"):
        power, i = _power(toks, i + 1, whole)
        out = out**power
    return out, i


def _power(toks: list[str], i: int, whole: str) -> tuple[Fraction, int]:
    """Parse an exponent: an integer, or a ratio of two integers."""
    tok = toks[i] if i < len(toks) else ""
    if not _INTEGER.fullmatch(tok):
        raise u.DimensionError(
            f"kups.core.units: cannot read the unit of {whole!r}: {tok!r} is not an "
            "exponent; "
            f"an exponent is an integer or a ratio of two, as in 'angstrom^-3' or "
            f"'amu^1/2'."
        )
    num, i = int(tok), i + 1
    if i + 1 < len(toks) and toks[i] == "/" and _INTEGER.fullmatch(toks[i + 1]):
        den = int(toks[i + 1])
        if den == 0:
            ratio = f"{tok}/{toks[i + 1]}"
            raise u.DimensionError(
                f"kups.core.units: cannot read the unit of {whole!r}: {ratio!r} is not "
                "an "
                f"exponent; a ratio needs a denominator that is not zero, as in "
                f"'amu^1/2'."
            )
        return Fraction(num, den), i + 2
    return Fraction(num), i


__all__ = [
    "CATALOG",
    "SI_BASE",
    "A",
    "Angstrom",
    "C",
    "Da",
    "Dimensionless",
    "GPa",
    "Hz",
    "J",
    "K",
    "N",
    "Pa",
    "Unit",
    "V",
    "W",
    "amu",
    "angstrom",
    "bar",
    "bohr",
    "cm",
    "debye",
    "deg",
    "e",
    "eV",
    "fs",
    "g",
    "hartree",
    "hour",
    "invcm",
    "kJ",
    "kcal",
    "kg",
    "km",
    "m",
    "minute",
    "mm",
    "mol",
    "ms",
    "nm",
    "ns",
    "parse",
    "ps",
    "rad",
    "require_dim",
    "rydberg",
    "s",
    "unit",
]
