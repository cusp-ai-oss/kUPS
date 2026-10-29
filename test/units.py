# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit helpers shared by the test suite.

kups runs in the atomistic base of kups.core.units, so a test's plain numbers are
magnitudes in Angstrom, eV and their combinations. These helpers name that unit
once: builders promote plain magnitudes into a quantity, and ``assert_close``
compares a value against an expectation in the unit the value's own dimension
names.
"""

from typing import Any

import jax.numpy as jnp
import numpy.testing as npt

from kups.core import units
from kups.core.units import catalog as u
from kups.core.units import testing

A = u.angstrom
"""The length unit of kups's base."""

EV = u.eV
"""The energy unit of kups's base."""

FORCE = u.eV / u.angstrom
"""The unit a position gradient is reported in."""

PRESSURE = u.eV / u.angstrom**3
"""The pressure unit of kups's base."""


def as_unit(values: Any, unit: Any) -> Any:
    """``values`` as a quantity in ``unit``, passing a quantity through.

    Args:
        values: A quantity, or magnitudes in ``unit``.
        unit: The unit plain magnitudes are read in.

    Returns:
        The values as a quantity.
    """
    return values if units.is_quantity(values) else jnp.asarray(values) * unit


def as_length(values: Any) -> units.Length:
    """``values`` as a length, reading plain magnitudes in Angstrom.

    Args:
        values: A length quantity, or magnitudes in Angstrom.

    Returns:
        The values as a length.
    """
    return as_unit(values, A)


def assert_close(actual: Any, expected: Any, **kwargs: Any) -> None:
    """Compare a value to an expectation, in the unit the value's dimension names.

    Args:
        actual: The value under test; a quantity names the unit the comparison is
            made in, a plain array is compared as it stands.
        expected: The expectation, either a quantity or magnitudes in that unit.
        **kwargs: Tolerances forwarded to the comparison.
    """
    if not units.is_quantity(actual):
        npt.assert_allclose(actual, expected, **kwargs)
        return
    unit = actual.unit
    testing.assert_allclose(actual, as_unit(expected, unit), unit, **kwargs)


def allclose(actual: Any, expected: Any, **kwargs: Any) -> bool:
    """``jnp.allclose`` in the unit ``actual``'s own dimension names.

    Args:
        actual: A quantity, whose unit the comparison is made in, or an array.
        expected: A quantity, or magnitudes in that unit.
        **kwargs: Tolerances forwarded to ``jnp.allclose``.

    Returns:
        Whether every element agrees.
    """
    if not units.is_quantity(actual):
        return bool(jnp.allclose(actual, expected, **kwargs))
    unit = actual.unit
    other = expected.to(unit) if units.is_quantity(expected) else expected
    return bool(jnp.allclose(actual.to(unit), other, **kwargs))


def isclose(actual: Any, expected: Any, **kwargs: Any) -> Any:
    """``jnp.isclose`` in the unit ``actual``'s own dimension names.

    Args:
        actual: A quantity, whose unit the comparison is made in, or an array.
        expected: A quantity, or magnitudes in that unit.
        **kwargs: Tolerances forwarded to ``jnp.isclose``.

    Returns:
        The elementwise closeness mask.
    """
    if not units.is_quantity(actual):
        return jnp.isclose(actual, expected, **kwargs)
    unit = actual.unit
    other = expected.to(unit) if units.is_quantity(expected) else expected
    return jnp.isclose(actual.to(unit), other, **kwargs)
