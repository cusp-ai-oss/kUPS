# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The NVE physics checks of the statistical CI."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "ci" / "statistical"))

from validate_nve_physics import validate_pressure_virial  # noqa: E402


def test_pressure_adds_the_ideal_gas_term():
    # the logged pressure is configurational, so the ideal gas term adds to it
    conf = np.full(10, 2.0e-4)
    kinetic = np.full(10, 3.0)
    passed, info = validate_pressure_virial(conf, kinetic, volume=1000.0)
    assert passed
    assert info["ideal_gas_pressure"] == pytest.approx(2.0e-3)
    assert info["total_pressure"] == pytest.approx(2.2e-3)
