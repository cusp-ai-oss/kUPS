# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""HDF5 logging for rigid-body MD simulations.

Mirrors :mod:`kups.application.md.logging` but pulls kinetic energy from
the per-group rigid-body table rather than the atom table (atoms in a
rigid molecule do not carry their own momenta).
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.application.mcmc.data import MotifParticles
from kups.application.md.data import MDRigidGroup, MDRigidParticles, MDSystems
from kups.core.data import Table
from kups.core.storage import EveryNStep, Once, WriterGroupConfig
from kups.core.typing import GroupId, MotifParticleId, ParticleId, SystemId
from kups.core.utils.jax import dataclass


class HasRigidMDData(Protocol):
    """Protocol for states containing rigid-body MD particle, group, and system data."""

    particles: Table[ParticleId, MDRigidParticles]
    groups: Table[GroupId, MDRigidGroup]
    motifs: Table[MotifParticleId, MotifParticles]
    systems: Table[SystemId, MDSystems]


def _per_group_kinetic_energy(groups: Table[GroupId, MDRigidGroup]) -> Array:
    """Translational + rotational KE per group, shape ``(n_groups,)``."""
    g = groups.data
    ke_trans = 0.5 * jnp.sum(g.momenta**2, axis=-1) / g.masses
    l_body = g.angular_momentum @ g.quaternion.inv()
    per_axis = jnp.where(
        jnp.isfinite(g.inertia_diag),
        l_body**2 / (2.0 * g.inertia_diag),
        0.0,
    )
    return ke_trans + jnp.sum(per_axis, axis=-1)


@dataclass
class RigidInitData:
    """Snapshot of the initial state, logged once at step 0."""

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

    atom_positions: Array
    com_positions: Array
    quaternion_components: Array
    com_momenta: Array
    angular_momentum: Array
    potential_energy: Array
    kinetic_energy: Array
    stress_tensor: Array
    volume: Array

    @staticmethod
    def from_state(state: HasRigidMDData) -> RigidMDStepData:
        ke_per_group = _per_group_kinetic_energy(state.groups)
        ke = jax.ops.segment_sum(
            ke_per_group,
            state.groups.data.system.indices,
            state.groups.data.system.num_labels,
        )
        return RigidMDStepData(
            atom_positions=state.particles.data.positions,
            com_positions=state.groups.data.positions,
            quaternion_components=state.groups.data.quaternion.components,
            com_momenta=state.groups.data.momenta,
            angular_momentum=state.groups.data.angular_momentum,
            potential_energy=state.systems.data.potential_energy,
            kinetic_energy=ke,
            stress_tensor=state.systems.data.stress_tensor,
            volume=state.systems.data.unitcell.volume,
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
