# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Physical units, checked at runtime and in the type system.

    >>> from kups.core.units import catalog as u
    >>> u.eV / u.angstrom
    eV/Å

Model:
  A ``Unit`` is a named multiple of a dimension, five rational exponents over
  ``(m, kg, s, A, K)``; ``mol`` is dimensionless with scale ``N_A``. ``Unit`` is
  generic over symbolic integer exponents, so pyrefly tracks dimensions through
  unit algebra. ``E_M``, ``E_KG``, ``E_S``, ``E_A`` and ``E_K`` are those
  exponents as type variables, for signatures generic over the dimension, such as
  ``Unit[E_M, E_KG, E_S, E_A, E_K]``; ``F_M`` to ``F_K`` name a second operand's.
"""

from kups.core.units import catalog as catalog
from kups.core.units._typed import E_A, E_K, E_KG, E_M, E_S, F_A, F_K, F_KG, F_M, F_S
from kups.core.units._unit import (
    ONE,
    AccelerationUnit,
    ActionUnit,
    AreaUnit,
    ChargeUnit,
    CompressibilityUnit,
    CoulombConstantUnit,
    CurrentUnit,
    DensityUnit,
    Dim,
    DimensionError,
    Dimensionless,
    DipoleMomentUnit,
    EnergyUnit,
    EntropyUnit,
    ForceUnit,
    FrequencyUnit,
    LengthUnit,
    MassUnit,
    MomentumUnit,
    NumberDensityUnit,
    PermittivityUnit,
    PowerUnit,
    PressureUnit,
    StiffnessUnit,
    TemperatureUnit,
    TimeUnit,
    Unit,
    VelocityUnit,
    VoltageUnit,
    VolumeUnit,
    WavenumberUnit,
    dmul,
)

__all__ = [
    # submodules
    "catalog",
    # core types
    "Unit",
    "Dimensionless",
    "DimensionError",
    # unit aliases
    "AccelerationUnit",
    "ActionUnit",
    "AreaUnit",
    "ChargeUnit",
    "CompressibilityUnit",
    "CoulombConstantUnit",
    "CurrentUnit",
    "DensityUnit",
    "DipoleMomentUnit",
    "EnergyUnit",
    "EntropyUnit",
    "ForceUnit",
    "FrequencyUnit",
    "LengthUnit",
    "MassUnit",
    "MomentumUnit",
    "NumberDensityUnit",
    "PermittivityUnit",
    "PowerUnit",
    "PressureUnit",
    "StiffnessUnit",
    "TemperatureUnit",
    "TimeUnit",
    "VelocityUnit",
    "VoltageUnit",
    "VolumeUnit",
    "WavenumberUnit",
    # dimensions
    "ONE",
    "Dim",
    "dmul",
    # exponent type variables
    "E_M",
    "E_KG",
    "E_S",
    "E_A",
    "E_K",
    "F_M",
    "F_KG",
    "F_S",
    "F_A",
    "F_K",
]
