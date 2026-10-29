# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Physical constants as dimension-checked quantities.

    >>> from kups.core.units import constants, catalog as u
    >>> round(float((constants.k_B * (300.0 * u.K)).to(u.eV)), 6)
    0.025852

The 2019 SI defining constants are exact; measured ones are CODATA 2018, like the
unit catalogue. Values are exact rationals, rounded once when cast to the float
dtype. Each constant is built on first access in the active base and dtype, and
cached per base and dtype.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from kups.core.units import catalog as u
from kups.core.units._base import BaseSystem, active_base
from kups.core.units._core import dim_of, from_dim
from kups.core.units._unit import Dim

if TYPE_CHECKING:
    from kups.core.units._core import (
        Action,
        Charge,
        CoulombConstant,
        Energy,
        Entropy,
        Length,
        Mass,
        Permittivity,
        Velocity,
    )

    c: Velocity
    """Speed of light in vacuum."""
    h: Action
    """Planck constant."""
    hbar: Action
    """Reduced Planck constant."""
    e: Charge
    """Elementary charge."""
    k_B: Entropy
    """Boltzmann constant, energy per Kelvin."""
    N_A: float
    """Avogadro constant, a pure number since the mole is a count."""
    alpha: float
    """Fine-structure constant."""
    m_e: Mass
    """Electron mass."""
    m_u: Mass
    """Atomic mass constant, the size of ``u.amu``."""
    a_0: Length
    """Bohr radius, the size of ``u.bohr``."""
    E_h: Energy
    """Hartree energy, the size of ``u.hartree``."""
    eps_0: Permittivity
    """Vacuum electric permittivity."""
    k_e: CoulombConstant
    """Coulomb constant; 14.399645478469 in the atomistic base (e, eV, Å)."""


# Published decimals as exact rationals.
_LIT: dict[str, Fraction] = {
    name: Fraction(Decimal(text))
    for name, text in (
        ("c", "299792458"),  # c through N_A are 2019 SI defining constants, exact
        ("h", "6.62607015e-34"),
        ("e", "1.602176634e-19"),
        ("k_B", "1.380649e-23"),
        ("N_A", "6.02214076e23"),
        ("alpha", "7.2973525693e-3"),  # the rest are CODATA 2018
        ("m_e", "9.1093837015e-31"),
        ("m_u", "1.66053906660e-27"),
        ("a_0", "5.29177210903e-11"),
        ("E_h", "4.3597447222071e-18"),
    )
}

# pi to 50 digits, far beyond float64
_PI_EXACT = Fraction(Decimal("3.14159265358979323846264338327950288419716939937511"))

# name -> (exact SI value, dimension); eps_0 and k_e are derived below.
_EXACT: dict[str, tuple[Fraction, Any]] = {
    name: (_LIT[name], dim_of(unit))
    for name, unit in (
        ("c", u.m / u.s),
        ("h", u.J * u.s),
        ("e", u.C),
        ("k_B", u.J / u.K),
        ("m_e", u.kg),
        ("m_u", u.kg),
        ("a_0", u.m),
        ("E_h", u.J),
    )
}
_EXACT["hbar"] = (_LIT["h"] / (2 * _PI_EXACT), dim_of(u.J * u.s))
_EXACT["eps_0"] = (
    _LIT["e"] ** 2 / (2 * _LIT["alpha"] * _LIT["h"] * _LIT["c"]),
    dim_of(u.C**2 / (u.J * u.m)),
)
_EXACT["k_e"] = (
    1 / (4 * _PI_EXACT * _EXACT["eps_0"][0]),
    dim_of(u.J * u.m / u.C**2),
)

# dimensionless constants, the same in every base
_NUMBERS: dict[str, float] = {
    "N_A": float(_LIT["N_A"]),
    "alpha": float(_LIT["alpha"]),
}

_CACHE: dict[tuple[str, Any, BaseSystem], Any] = {}


def _magnitude(value: Fraction, dim: Dim, bs: BaseSystem) -> float:
    """Compute a constant's magnitude in ``bs``, rounding only once.

    Uses ``value / acc**(1/d) == (value**d / acc)**(1/d)`` to stay exact until
    the final root.
    """
    _, acc, d = bs._exact_scale(dim)
    ratio = value**d / acc
    with localcontext() as ctx:
        ctx.prec = 60  # far beyond float64's 17 digits
        root = Decimal(ratio.numerator) / Decimal(ratio.denominator)
        return float(root if d == 1 else root ** (Decimal(1) / Decimal(d)))


def __getattr__(name: str) -> Any:
    """Return a constant in the active base and dtype, cached per both.

    Args:
      name: the constant's name, e.g. ``"k_B"``.

    Returns:
      The constant as a quantity, or a float for dimensionless ones.

    Raises:
      AttributeError: if ``name`` is not a constant.
    """
    if name not in _EXACT and name not in _NUMBERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    bs = active_base()
    key = (name, jnp.result_type(float), bs)
    if key not in _CACHE:
        # build eagerly even under a trace; a cached tracer would outlive it
        with jax.ensure_compile_time_eval():
            if name in _NUMBERS:
                _CACHE[key] = _NUMBERS[name]
            else:
                value, dim = _EXACT[name]
                _CACHE[key] = from_dim(jnp.asarray(_magnitude(value, dim, bs)), dim, bs)
    return _CACHE[key]


def __dir__() -> list[str]:
    """List the constant names for ``dir()`` and tooling.

    Returns:
      The sorted constant names.
    """
    return sorted(_EXACT | _NUMBERS)
