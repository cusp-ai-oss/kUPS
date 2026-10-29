# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Numeric assertions for quantities, compared in an explicitly named unit.

    >>> from kups.core.units import catalog as u
    >>> from kups.core.units.testing import assert_allclose
    >>> assert_allclose(2.0 * u.angstrom, 0.2 * u.nm, u.angstrom)

Both sides are converted to the named unit, which catches dimension mismatches,
and the magnitudes are compared with ``numpy.testing``. A plain array counts as
magnitudes already in ``unit``, but at least one side must carry a dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy.testing as npt
from jax.typing import ArrayLike

from kups.core.units._core import Quantity, is_quantity
from kups.core.units._unit import ONE, DimensionError, Unit

if TYPE_CHECKING:
    from kups.core.units._typed import _E_A, _E_K, _E_KG, _E_M, _E_S

__all__ = ["assert_allclose", "assert_array_equal"]


def _magnitudes(actual: Any, expected: Any, unit: Any, name: str) -> tuple[Any, Any]:
    """Check both sides are comparable and convert them to magnitudes in ``unit``."""
    if not isinstance(unit, Unit):
        raise TypeError(
            f"units.testing.{name}: the unit to compare in must be a units.Unit, not "
            f"{type(unit).__name__} ({unit!r}); kups.core.units has no string units -- "
            "use "
            "units.catalog, e.g. u.angstrom or u.eV/u.angstrom."
        )
    if unit.dim != ONE and not any(map(is_quantity, (actual, expected))):
        raise DimensionError(
            f"units.testing.{name}: neither side carries a dimension, so there is "
            f"nothing to measure in {unit!r} and a lost unit would pass unnoticed. "
            "Compare plain arrays with numpy.testing directly."
        )
    return (
        actual.to(unit) if is_quantity(actual) else actual,
        expected.to(unit) if is_quantity(expected) else expected,
    )


def assert_allclose(
    actual: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | ArrayLike,
    expected: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | ArrayLike,
    unit: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> None:
    """Assert two quantities are close, comparing magnitudes in ``unit``.

    Args:
      actual: a quantity, or magnitudes already in ``unit``.
      expected: the expected value, in the same forms.
      unit: the unit both sides are converted to.
      rtol: relative tolerance; None means four times the dtype epsilon, at
        least 1e-7.
      atol: absolute tolerance, in ``unit``.

    Raises:
      AssertionError: if the magnitudes differ by more than the tolerance.
      TypeError: if ``unit`` is not a kups.core.units Unit.
      DimensionError: if a side's dimension differs from ``unit``'s, or neither
        side carries one while ``unit`` does.
    """
    a, b = _magnitudes(actual, expected, unit, "assert_allclose")
    if rtol is None:
        rtol = max(1e-7, 4.0 * float(jnp.finfo(jnp.result_type(a, b)).eps))
    npt.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"magnitudes in {unit}")


def assert_array_equal(
    actual: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | ArrayLike,
    expected: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | ArrayLike,
    unit: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
) -> None:
    """Assert two quantities are exactly equal, comparing magnitudes in ``unit``.

    Args:
      actual: a quantity, or magnitudes already in ``unit``.
      expected: the expected value, in the same forms.
      unit: the unit both sides are converted to.

    Raises:
      AssertionError: if any pair of magnitudes differs.
      TypeError: if ``unit`` is not a kups.core.units Unit.
      DimensionError: if a side's dimension differs from ``unit``'s, or neither
        side carries one while ``unit`` does.
    """
    a, b = _magnitudes(actual, expected, unit, "assert_array_equal")
    npt.assert_array_equal(a, b, err_msg=f"magnitudes in {unit}")
