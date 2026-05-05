# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""HDF5 logging for rigid-body MD simulations.

Mirrors :mod:`kups.application.md.logging`. Adds a ``groups`` table to
both init and step snapshots, and pulls kinetic energy from the per-group
rigid-body table.
"""

from __future__ import annotations

from typing import Protocol

import jax
from jax import Array

from kups.application.mcmc.data import MotifParticles
from kups.application.md.data import MDRigidGroup, MDRigidParticles, MDSystems
from kups.core.data import Table
from kups.core.storage import EveryNStep, Once, WriterGroupConfig
from kups.core.typing import GroupId, MotifParticleId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.core.utils.rigid_body import per_group_kinetic_energy


class HasRigidMDData(Protocol):
    """Protocol for states containing rigid-body MD particle, group, and system data."""

    particles: Table[ParticleId, MDRigidParticles]
    groups: Table[GroupId, MDRigidGroup]
    motifs: Table[MotifParticleId, MotifParticles]
    systems: Table[SystemId, MDSystems]


def _per_group_kinetic_energy(groups: Table[GroupId, MDRigidGroup]) -> Array:
    """Translational + rotational KE per group, shape ``(n_groups,)``."""
    g = groups.data
    return per_group_kinetic_energy(
        g.momenta, g.masses, g.angular_momentum, g.quaternion, g.inertia_diag
    )


@dataclass
class RigidInitData:
    """Snapshot of the initial rigid-MD state logged once at step 0."""

    atoms: Table[ParticleId, MDRigidParticles]
    groups: Table[GroupId, MDRigidGroup]
    systems: Table[SystemId, MDSystems]

    @staticmethod
    def from_state(state: HasRigidMDData) -> RigidInitData:
        return RigidInitData(
            atoms=state.particles, groups=state.groups, systems=state.systems
        )


@dataclass
class RigidMDStepData:
    """Per-step rigid MD data."""

    atoms: Table[ParticleId, MDRigidParticles]
    groups: Table[GroupId, MDRigidGroup]
    potential_energy: Array
    kinetic_energy: Array
    stress_tensor: Array

    @staticmethod
    def from_state(state: HasRigidMDData) -> RigidMDStepData:
        ke = jax.ops.segment_sum(
            _per_group_kinetic_energy(state.groups),
            state.groups.data.system.indices,
            state.groups.data.system.num_labels,
        )
        return RigidMDStepData(
            atoms=state.particles,
            groups=state.groups,
            potential_energy=state.systems.data.potential_energy,
            kinetic_energy=ke,
            stress_tensor=state.systems.data.stress_tensor,
        )


@dataclass
class RigidMDLoggedData:
    """Logging configuration for rigid-body MD."""

    init: WriterGroupConfig[HasRigidMDData, RigidInitData] = WriterGroupConfig(
        RigidInitData.from_state, Once()
    )
    step: WriterGroupConfig[HasRigidMDData, RigidMDStepData] = WriterGroupConfig(
        RigidMDStepData.from_state, EveryNStep(1)
    )
