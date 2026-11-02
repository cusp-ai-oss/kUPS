# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the kups.core.units tests (x64 comes from pytest-env)."""

import math

# ASE time unit in seconds, derived from ASE's length/mass/energy base
ASE_TIME = 1.0180505710759415e-14


def ulps(a: float, b: float) -> int:
    """Distance between a and b in ulps, capped at 9.

    Returns:
        Representable floats between the two; 9 means beyond the cap (or nan/inf).
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return 9
    n, lo, hi = 0, min(a, b), max(a, b)
    while lo < hi and n < 9:
        lo, n = math.nextafter(lo, math.inf), n + 1
    return n
