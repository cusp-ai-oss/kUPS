# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Ewald electrostatics with cached, incremental reciprocal-space updates.

``parameters`` prepares cutoffs and lattice shifts, ``reciprocal`` implements
the numerical kernels, and ``potential`` composes energies and cache updates.
"""

from .parameters import (
    RECIPROCAL_GRID_BACKENDS,
    EwaldParameterEstimates,
    EwaldParameters,
    IsEwaldPointData,
    ReciprocalGridBound,
    estimate_ewald_parameters,
    kvecs_from_kmax,
    reciprocal_grid_bound,
    reciprocal_grid_shifts,
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
    IsChargedTemplate,
    ewald_long_range_energy,
    ewald_net_charge_energy,
    ewald_self_interaction_energy,
    ewald_short_range_energy,
    ewald_short_range_pair,
    ewald_short_range_pair_kernel,
    long_range,
    make_ewald_exclusion_correction_potential,
    make_ewald_long_range_potential,
    make_ewald_potential,
    make_ewald_self_interaction_potential,
    make_ewald_short_range_potential,
    prefactor,
    structure_factor,
)

__all__ = [
    "RECIPROCAL_GRID_BACKENDS",
    "ReciprocalGridBound",
    "EwaldParameterEstimates",
    "EwaldParameters",
    "IsEwaldPointData",
    "estimate_ewald_parameters",
    "kvecs_from_kmax",
    "reciprocal_grid_bound",
    "reciprocal_grid_shifts",
    "TO_STANDARD_UNITS",
    "EwaldCache",
    "EwaldCachePatch",
    "EwaldShortRangeInput",
    "IsChargedTemplate",
    "EwaldSelfInput",
    "EwaldLongRangeInput",
    "ewald_self_interaction_energy",
    "ewald_short_range_energy",
    "ewald_short_range_pair",
    "ewald_short_range_pair_kernel",
    "long_range",
    "prefactor",
    "structure_factor",
    "ewald_net_charge_energy",
    "ewald_long_range_energy",
    "EwaldLongRangeComposer",
    "EwaldPotential",
    "make_ewald_exclusion_correction_potential",
    "make_ewald_short_range_potential",
    "make_ewald_long_range_potential",
    "make_ewald_self_interaction_potential",
    "make_ewald_potential",
]
