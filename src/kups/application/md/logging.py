# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""HDF5 logging for molecular dynamics simulations."""

from __future__ import annotations

from typing import Protocol

import jax
from jax import Array

from kups.application.md.data import MDParticles, MDSystems
from kups.core.data import Table
from kups.core.storage import EveryNStep, Once, WriterGroupConfig
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


class HasMDData(Protocol):
    """Protocol for states containing MD particle and system data."""

    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]


@dataclass
class InitData:
    """Snapshot of initial MD state logged once at step 0.

    Attributes:
        atoms: Indexed particle data (positions, momenta, etc.).
        systems: Indexed system data (unit cell, temperature, etc.).
    """

    atoms: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]

    @staticmethod
    def from_state(state: HasMDData) -> InitData:
        return InitData(atoms=state.particles, systems=state.systems)


@dataclass
class MDStepData:
    """Per-step MD data logged at each production step.

    Attributes:
        atoms: Indexed particle data.
        potential_energy: Potential energy per system.
        kinetic_energy: Kinetic energy per system.
        stress_tensor: Virial stress tensor per system.
    """

    atoms: Table[ParticleId, MDParticles]
    potential_energy: Array
    kinetic_energy: Array
    stress_tensor: Array

    @staticmethod
    def from_state(state: HasMDData) -> MDStepData:
        ke = jax.ops.segment_sum(
            state.particles.data.kinetic_energy,
            state.particles.data.system.indices,
            state.particles.data.system.num_labels,
        )
        return MDStepData(
            atoms=state.particles,
            potential_energy=state.systems.data.potential_energy,
            kinetic_energy=ke,
            stress_tensor=state.systems.data.stress_tensor,
        )


@dataclass
class MDLoggedData:
    """Configuration for MD simulation logging groups.

    Attributes:
        init: Logs initial state once at step 0.
        step: Logs thermodynamic data every step.
    """

    init: WriterGroupConfig[HasMDData, InitData] = WriterGroupConfig(
        InitData.from_state, Once()
    )
    step: WriterGroupConfig[HasMDData, MDStepData] = WriterGroupConfig(
        MDStepData.from_state, EveryNStep(1)
    )
