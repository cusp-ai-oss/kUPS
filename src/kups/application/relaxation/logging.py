# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""HDF5 logging for structure relaxation."""

from __future__ import annotations

from jax import Array

from kups.application.relaxation.data import (
    IsRelaxData,
    RelaxParticles,
    RelaxSystems,
    relax_gradients,
    relax_index_prefix,
)
from kups.core.data import Table
from kups.core.storage import EveryNStep, Once, WriterGroupConfig
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.observables.stress import stress_via_virial_theorem
from kups.relaxation.convergence import max_dof_per_system


@dataclass
class RelaxInitData:
    """Initial snapshot for the HDF5 log.

    Attributes:
        atoms: Initial particle data.
        systems: Initial system data.
    """

    atoms: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]

    @staticmethod
    def from_state(state: IsRelaxData) -> RelaxInitData:
        """Extract initial snapshot from a relaxation state."""
        return RelaxInitData(atoms=state.particles, systems=state.systems)


@dataclass
class RelaxStepData:
    """Per-step snapshot for the HDF5 log.

    Attributes:
        atoms: Particle data at this step.
        potential_energy: Potential energy per system.
        max_force: Maximum atomic force magnitude per system (eV/Å).
        stress_tensor: Stress tensor per system, shape (..., 3, 3).
    """

    atoms: Table[ParticleId, RelaxParticles]
    potential_energy: Array
    max_force: Array
    stress_tensor: Array

    @staticmethod
    def from_state(state: IsRelaxData) -> RelaxStepData:
        """Extract per-step logging data from a relaxation state."""
        max_force = max_dof_per_system(
            relax_gradients(state),
            relax_index_prefix(state.particles, state.systems),
            include_cell=False,
        )
        return RelaxStepData(
            atoms=state.particles,
            potential_energy=state.systems.data.potential_energy,
            max_force=max_force.data,
            stress_tensor=stress_via_virial_theorem(
                state.particles, state.systems
            ).data,
        )


@dataclass
class RelaxLoggedData:
    """HDF5 writer configuration for relaxation simulations."""

    init: WriterGroupConfig[IsRelaxData, RelaxInitData] = WriterGroupConfig(
        RelaxInitData.from_state, Once()
    )
    step: WriterGroupConfig[IsRelaxData, RelaxStepData] = WriterGroupConfig(
        RelaxStepData.from_state, EveryNStep(1)
    )
