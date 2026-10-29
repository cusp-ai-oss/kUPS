# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared unit helpers in test/units.py."""

import jax.numpy as jnp
import pytest

from kups.core import units
from test.units import MOMENTUM, allclose


def test_allclose_needs_a_dimension_on_one_side():
    # two plain sides would let a lost unit pass unnoticed
    with pytest.raises(units.DimensionError, match="neither side carries"):
        allclose(jnp.zeros(3), 0.0, MOMENTUM)
    assert allclose(jnp.zeros(3) * MOMENTUM, 0.0, MOMENTUM)
