# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Ewald summation for long-range electrostatics in periodic systems.

Splits the Coulomb potential into short-range (real-space), long-range
(reciprocal-space), and self-interaction terms. Supports incremental
updates via cached structure factors for efficient Monte Carlo.
"""

from .parameters import (
    EwaldParameterEstimates,
    EwaldParameters,
    IsEwaldPointData,
    estimate_ewald_parameters,
    kvecs_from_kmax,
)
from .potential import (
    TO_STANDARD_UNITS,
    EwaldCache,
    EwaldCachePatch,
    EwaldLongRangeComposer,
    EwaldLongRangeInput,
    EwaldPotential,
    EwaldSelfInput,
    EwaldShortRangeInput,
    ewald_long_range_energy,
    ewald_net_charge_energy,
    ewald_self_interaction_energy,
    ewald_short_range_energy,
    long_range,
    make_ewald_long_range_potential,
    make_ewald_potential,
    make_ewald_self_interaction_potential,
    make_ewald_short_range_potential,
    pointcloud_geometry,
    prefactor,
    structure_factor,
)
from .potential import (
    _structure_factor_update as _structure_factor_update,
)
from .potential import (
    _structure_factor_update_jvp as _structure_factor_update_jvp,
)
from .reciprocal import (
    _frequency_response as _frequency_response,
)
from .reciprocal import (
    _structure_factor_full as _structure_factor_full,
)

__all__ = [
    "EwaldParameters",
    "IsEwaldPointData",
    "EwaldParameterEstimates",
    "estimate_ewald_parameters",
    "kvecs_from_kmax",
    "TO_STANDARD_UNITS",
    "pointcloud_geometry",
    "EwaldCache",
    "EwaldCachePatch",
    "EwaldShortRangeInput",
    "EwaldSelfInput",
    "EwaldLongRangeInput",
    "ewald_self_interaction_energy",
    "ewald_short_range_energy",
    "long_range",
    "prefactor",
    "structure_factor",
    "ewald_net_charge_energy",
    "ewald_long_range_energy",
    "EwaldLongRangeComposer",
    "EwaldPotential",
    "make_ewald_short_range_potential",
    "make_ewald_long_range_potential",
    "make_ewald_self_interaction_potential",
    "make_ewald_potential",
]
