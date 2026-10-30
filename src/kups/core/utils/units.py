# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Reading dimensioned values as the plain magnitudes a jax primitive takes."""

from __future__ import annotations

from typing import Any

from jax import Array

from kups.core import units

type Measured = Array | units.Quantity[Any, Any, Any, Any, Any]
"""Data that keeps whatever it measures, a plain array included.

A quantity of any dimension, or an array that carries none. Code written over
this reads magnitudes only where a primitive needs them.
"""


def magnitudes(x: Measured) -> Array:
    """Magnitudes of ``x`` in its own base.

    A plain array is already its own magnitudes, in the active base, which is how
    ``kups.core.units`` reads one that meets a quantity.

    Args:
        x: A quantity, or the magnitudes of one.

    Returns:
        The raw array.
    """
    return x.value if units.is_quantity(x) else x


def magnitudes_of[T](tree: T) -> T:
    """Magnitudes of every quantity leaf of ``tree``, in its own base.

    Args:
        tree: Any pytree, whose leaves may carry dimensions.

    Returns:
        The same structure with plain magnitudes.
    """
    return units.tree_map(magnitudes, tree)
