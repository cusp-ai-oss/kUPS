# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Logging for rigid-body MCMC simulations."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.application.mcmc.data import (
    MCMCGroup,
    MCMCParticles,
    MCMCSystems,
    MotifParticles,
    StressResult,
)
from kups.core.data import Buffered, Index, Table
from kups.core.data.wrappers import WithCache
from kups.core.parameter_scheduler import ParameterSchedulerState
from kups.core.potential import EmptyType, PotentialOut
from kups.core.propagator import StateProperty
from kups.core.storage import EveryNStep, Once, WriterGroupConfig
from kups.core.typing import GroupId, MotifId, MotifParticleId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.ewald import EwaldCache, EwaldParameters
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
)


class IsMCMCState(Protocol):
    """Protocol for states compatible with MCMC logging."""

    @property
    def particles(self) -> Buffered[ParticleId, MCMCParticles]: ...
    @property
    def groups(self) -> Buffered[GroupId, MCMCGroup]: ...
    @property
    def systems(self) -> Table[SystemId, MCMCSystems]: ...
    @property
    def motifs(self) -> Table[MotifParticleId, MotifParticles]: ...
    @property
    def translation_params(self) -> Table[SystemId, ParameterSchedulerState]: ...
    @property
    def rotation_params(self) -> Table[SystemId, ParameterSchedulerState]: ...
    @property
    def reinsertion_params(self) -> Table[SystemId, ParameterSchedulerState]: ...
    @property
    def exchange_params(self) -> Table[SystemId, ParameterSchedulerState]: ...
    @property
    def lj_parameters(
        self,
    ) -> WithCache[
        GlobalTailCorrectedLennardJonesParameters, PotentialOut[EmptyType, EmptyType]
    ]: ...
    @property
    def ewald_parameters(
        self,
    ) -> WithCache[EwaldParameters, EwaldCache[EmptyType, EmptyType]]: ...


@dataclass
class MCMCFixedData:
    """Immutable data logged once at the start of the simulation.

    Contains the host framework particles (which never change) and
    system-level thermodynamic state.
    """

    systems: Table[SystemId, MCMCSystems]
    motifs: Table[MotifParticleId, MotifParticles]
    host_particles: Table[ParticleId, MCMCParticles]


@dataclass
class MCMCSystemStepData:
    """Per-system, per-cycle data: energies, move statistics, and stress."""

    potential_energy: Array
    lj_energy: Array
    ewald_short_range_energy: Array
    ewald_long_range_energy: Array
    ewald_self_energy: Array
    ewald_exclusion_energy: Array
    translation_step_width: Array
    rotation_step_width: Array
    translation_acceptance: Array
    rotation_acceptance: Array
    reinsertion_acceptance: Array
    exchange_acceptance: Array
    guest_stress: StressResult


@dataclass
class MCMCStepData:
    """Per-cycle MCMC data: adsorbate positions, energies, and move statistics."""

    particles: Table[ParticleId, MCMCParticles]
    groups: Table[GroupId, MCMCGroup]
    particle_count: Table[tuple[SystemId, MotifId], Array]
    systems: Table[SystemId, MCMCSystemStepData]


@dataclass
class MCMCLoggedData[S]:
    """HDF5 writer group layout: one-shot host data + per-cycle adsorbate data."""

    fixed: WriterGroupConfig[S, MCMCFixedData]
    per_step: WriterGroupConfig[S, MCMCStepData]


def make_mcmc_logged_data[S: IsMCMCState](
    state: S,
    stress_fn: StateProperty[S, Table[SystemId, StressResult]] | None = None,
) -> MCMCLoggedData:
    """Create MCMC logging config that logs host particles once and adsorbate buffer per step.

    Host particles are identified by having out-of-bounds group indices
    (they belong to no molecular group). This works for any number of
    host systems regardless of how particles are interleaved in the buffer.

    Args:
        state: The initial MCMC state (after padding).
        stress_fn: Optional stress tensor function. When provided, the
            guest stress tensor is logged each cycle.

    Returns:
        Configured logging data with host/adsorbate split.
    """

    particles = state.particles
    is_occupied = np.asarray(particles.occupation)
    is_host = ~particles.data.group.valid_mask & is_occupied

    host_positions = np.where(is_host)[0]
    ads_positions = np.where(~is_host)[0]

    host_idx = Index(particles.keys, jnp.asarray(host_positions))
    ads_idx = Index(particles.keys, jnp.asarray(ads_positions))
    host_keys = tuple(particles.keys[int(i)] for i in host_positions)
    ads_keys = tuple(particles.keys[int(i)] for i in ads_positions)

    def fixed_view(state: S) -> MCMCFixedData:
        return MCMCFixedData(
            systems=state.systems,
            motifs=state.motifs,
            host_particles=Table(host_keys, state.particles[host_idx]),
        )

    def step_view(state: S) -> MCMCStepData:
        counts = Index.combine(state.groups.data.system, state.groups.data.motif).counts
        ewald = state.ewald_parameters.cache
        if stress_fn is not None:
            guest_stress = stress_fn(jax.random.key(0), state)
        else:
            z = jnp.zeros((len(state.systems), 3, 3))
            guest_stress = Table(state.systems.keys, StressResult(z, z, z))
        sys_keys = state.systems.keys
        system_step = MCMCSystemStepData(
            potential_energy=state.systems.data.potential_energy,
            lj_energy=state.lj_parameters.cache.total_energies.data,
            ewald_short_range_energy=ewald.short_range.total_energies.data,
            ewald_long_range_energy=ewald.long_range.total_energies.data,
            ewald_self_energy=ewald.self_interaction.total_energies.data,
            ewald_exclusion_energy=ewald.exclusion.total_energies.data,
            translation_step_width=state.translation_params.map_data(
                lambda p: p.value
            ).data,
            rotation_step_width=state.rotation_params.map_data(lambda p: p.value).data,
            translation_acceptance=state.translation_params.map_data(
                lambda p: p.history.values.mean(axis=-1)
            ).data,
            rotation_acceptance=state.rotation_params.map_data(
                lambda p: p.history.values.mean(axis=-1)
            ).data,
            reinsertion_acceptance=state.reinsertion_params.map_data(
                lambda p: p.history.values.mean(axis=-1)
            ).data,
            exchange_acceptance=state.exchange_params.map_data(
                lambda p: p.history.values.mean(axis=-1)
            ).data,
            guest_stress=guest_stress.data,
        )
        return MCMCStepData(
            particles=Table(ads_keys, state.particles[ads_idx]),
            groups=state.groups,
            particle_count=counts,
            systems=Table(sys_keys, system_step),
        )

    return MCMCLoggedData(
        fixed=WriterGroupConfig(fixed_view, Once()),
        per_step=WriterGroupConfig(step_view, EveryNStep(1)),
    )
