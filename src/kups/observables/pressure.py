# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pressure calculations from stress tensors and ideal gas law."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.typing import (
    HasTemperature,
    HasUnitCell,
    SystemId,
)


@runtime_checkable
class IsIdealGasSystems(HasTemperature, HasUnitCell, Protocol):
    """Systems with temperature and unit cell for ideal gas pressure."""


def _pressure_from_stress(stress: Array) -> Array:
    """P = Tr(σ) / 3."""
    return jnp.trace(stress, axis1=-2, axis2=-1) / 3


def _ideal_gas_pressure(
    particles_per_system: Array, temperature: Array, volume: Array
) -> Array:
    """P_ideal = N·k_B·T / V."""
    return particles_per_system * BOLTZMANN_CONSTANT * temperature / volume


pressure_from_stress = Table.transform(_pressure_from_stress)


def ideal_gas_pressure(
    counts: Table[SystemId, Array],
    systems: Table[SystemId, IsIdealGasSystems],
) -> Table[SystemId, Array]:
    """Compute ideal gas pressure P = N·k_B·T / V.

    Args:
        counts: Number of independent bodies per system (e.g. molecules
            for rigid-body MCMC, atoms for MD).
        systems: Per-system temperature and unit cell.

    Returns:
        Ideal gas pressure per system, shape ``(n_systems,)``.
    """
    return Table.transform(_ideal_gas_pressure)(
        counts,
        systems.map_data(lambda s: s.temperature),
        systems.map_data(lambda s: s.unitcell.volume),
    )
