# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""HDF5 logging for structure relaxation."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.application.relaxation.data import RelaxParticles, RelaxSystems
from kups.core.data import Table
from kups.core.storage import EveryNStep, Once, WriterGroupConfig
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


class HasRelaxData(Protocol):
    """Protocol for states that can provide relaxation logging data."""

    @property
    def particles(self) -> Table[ParticleId, RelaxParticles]: ...
    @property
    def systems(self) -> Table[SystemId, RelaxSystems]: ...


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
    def from_state(state: HasRelaxData) -> RelaxInitData:
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
    def from_state(state: HasRelaxData) -> RelaxStepData:
        """Extract per-step logging data from a relaxation state."""
        forces = state.particles.data.forces
        force_norms = jnp.linalg.norm(forces, axis=-1)
        max_force = jax.ops.segment_max(
            force_norms,
            state.particles.data.system.indices,
            state.particles.data.system.num_labels,
        )
        return RelaxStepData(
            atoms=state.particles,
            potential_energy=state.systems.data.potential_energy,
            max_force=max_force,
            stress_tensor=state.systems.data.stress_tensor,
        )


@dataclass
class RelaxLoggedData:
    """HDF5 writer configuration for relaxation simulations."""

    init: WriterGroupConfig[HasRelaxData, RelaxInitData] = WriterGroupConfig(
        RelaxInitData.from_state, Once()
    )
    step: WriterGroupConfig[HasRelaxData, RelaxStepData] = WriterGroupConfig(
        RelaxStepData.from_state, EveryNStep(1)
    )
