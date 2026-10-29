# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit helpers shared by the test suite.

kups runs in the atomistic base of kups.core.units, so a test's plain numbers are
magnitudes in Angstrom, amu, eV, e and their combinations. The aliases below name
the units tests share, builders promote plain magnitudes into a quantity, and the
comparisons read a plain side in the unit they are given, so a value of the wrong
dimension fails instead of being compared as it stands. ``units.testing`` holds
the asserting counterparts.
"""

from typing import overload

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from kups.core import units
from kups.core.units import E_A, E_K, E_KG, E_M, E_S
from kups.core.units import catalog as u

A = u.angstrom
"""The length unit of kups's base."""

EV = u.eV
"""The energy unit of kups's base."""

FORCE = u.eV / u.angstrom
"""The unit a position gradient is reported in."""

PRESSURE = u.eV / u.angstrom**3
"""The pressure unit of kups's base."""

AMU = u.amu
"""The mass unit of kups's base."""

CHARGE = u.e
"""The charge unit of kups's base."""

T0 = A * (AMU / EV) ** 0.5
"""The time unit of kups's base, Å·√(amu/eV) ≈ 10.18 fs."""

MOMENTUM = AMU * A / T0
"""The momentum unit of kups's base, √(amu·eV)."""

AMU_A_PER_FS = AMU * A / u.fs
"""The momentum unit amu·Å/fs."""


@overload
def as_unit(
    values: units.Quantity[E_M, E_KG, E_S, E_A, E_K],
    unit: units.Unit[E_M, E_KG, E_S, E_A, E_K],
) -> units.Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
@overload
def as_unit(
    values: ArrayLike, unit: units.Unit[E_M, E_KG, E_S, E_A, E_K]
) -> units.Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
def as_unit(
    values: units.Measured | ArrayLike, unit: units.Unit[E_M, E_KG, E_S, E_A, E_K]
) -> units.Quantity[E_M, E_KG, E_S, E_A, E_K]:
    """``values`` as a quantity in ``unit``, passing a quantity through.

    Args:
        values: A quantity, or magnitudes in ``unit``.
        unit: The unit plain magnitudes are read in.

    Returns:
        The values as a quantity.
    """
    return values if units.is_quantity(values) else jnp.asarray(values) * unit


def as_length(values: units.Length | ArrayLike) -> units.Length:
    """``values`` as a length, reading plain magnitudes in Angstrom.

    Args:
        values: A length quantity, or magnitudes in Angstrom.

    Returns:
        The values as a length.
    """
    return as_unit(values, A)


def allclose(
    actual: units.Measured | ArrayLike,
    expected: units.Measured | ArrayLike,
    unit: units.Unit[E_M, E_KG, E_S, E_A, E_K],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """``jnp.allclose`` on magnitudes in ``unit``.

    Args:
        actual: A quantity, or magnitudes in ``unit``.
        expected: A quantity, or magnitudes in ``unit``.
        unit: The unit both sides are compared in.
        rtol: Relative tolerance.
        atol: Absolute tolerance, in ``unit``.

    Returns:
        Whether every element agrees.
    """
    return bool(
        jnp.allclose(*_magnitudes(actual, expected, unit), rtol=rtol, atol=atol)
    )


def _magnitudes(
    actual: units.Measured | ArrayLike,
    expected: units.Measured | ArrayLike,
    unit: units.Unit[E_M, E_KG, E_S, E_A, E_K],
) -> tuple[Array, Array]:
    """Both sides as magnitudes in ``unit``, reading a plain side in ``unit``."""
    return as_unit(actual, unit).to(unit), as_unit(expected, unit).to(unit)
