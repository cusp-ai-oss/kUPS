# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""HDF5 logging for molecular dynamics simulations."""

from __future__ import annotations

from typing import Protocol

from jax import Array

from kups.application.md.data import MDParticles, MdRunConfig, MDSystems
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.storage import EveryNStep, Once, WriterGroupConfig
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.md.observables import system_kinetic_energy
from kups.observables.stress import stress_via_virial_theorem


class HasMDData(Protocol):
    """Protocol for states containing MD particle and system data."""

    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    step: Array


@dataclass
class MDSnapshot:
    """Particle and system tables of one state.

    Attributes:
        atoms: Particle data (positions, momenta, masses, ...).
        systems: System data (cell, integrator parameters, ...).
    """

    atoms: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]

    @staticmethod
    def from_state(state: HasMDData) -> MDSnapshot:
        """Snapshot of a state.

        Args:
            state: MD state.

        Returns:
            Its particle and system tables.
        """
        return MDSnapshot(state.particles, state.systems)


@dataclass
class MDThermoData:
    """Energies, stress and volume of one logged step.

    Attributes:
        step: MD steps done, warmup included, shape ``(1,)``.
        potential_energy: Potential energy per system.
        kinetic_energy: Total kinetic energy per system.
        internal_kinetic_energy: Kinetic energy per system with center-of-mass
            motion removed (used for temperature).
        stress_tensor: Virial stress tensor per system.
        volume: Cell volume per system (A^3).
    """

    step: Array
    potential_energy: Array
    kinetic_energy: Array
    internal_kinetic_energy: Array
    stress_tensor: Array
    volume: Array

    @staticmethod
    def from_state(state: HasMDData) -> MDThermoData:
        """Energies, stress and volumes of a state.

        Args:
            state: MD state.

        Returns:
            Its row.
        """
        particles = state.particles
        system = particles.data.system
        momenta, masses = particles.data.momenta, particles.data.masses
        return MDThermoData(
            step=state.step,
            potential_energy=state.systems.data.potential_energy,
            kinetic_energy=system_kinetic_energy(momenta, masses, system),
            internal_kinetic_energy=system_kinetic_energy(
                momenta, masses, system, remove_com=True
            ),
            stress_tensor=stress_via_virial_theorem(particles, state.systems).data,
            volume=state.systems.data.cell.volume,
        )


@dataclass
class MDFrameData:
    """Positions, momenta and cells of one logged step.

    Attributes:
        step: MD steps done, warmup included, shape ``(1,)``.
        positions: Positions (A) wrapped into the cell, shape ``(n_atoms, 3)``.
        momenta: Momenta, shape ``(n_atoms, 3)``.
        cell: Cell of each system.
    """

    step: Array
    positions: Array
    momenta: Array
    cell: Cell[AnyPeriodicity]

    @staticmethod
    def from_state(state: HasMDData) -> MDFrameData:
        """Frame of a state.

        Args:
            state: MD state.

        Returns:
            Its positions, momenta and cells.
        """
        particles = state.particles.data
        return MDFrameData(
            state.step, particles.positions, particles.momenta, state.systems.data.cell
        )


@dataclass
class MDLoggedData:
    """Output groups of an MD run.

    Attributes:
        init: Tables after the first block.
        step: Energies, stress and volume every ``run.thermo_every`` steps.
        frames: Positions, momenta and cells every ``run.frames_every`` steps.
        final: Tables after the last block.
    """

    init: WriterGroupConfig[HasMDData, MDSnapshot]
    step: WriterGroupConfig[HasMDData, MDThermoData]
    frames: WriterGroupConfig[HasMDData, MDFrameData]
    final: WriterGroupConfig[HasMDData, MDSnapshot]

    @staticmethod
    def create(config: MdRunConfig) -> MDLoggedData:
        """Output groups of a run that logs once per block.

        Args:
            config: Run configuration with the block size and output intervals.

        Returns:
            The output groups, with rows and frames at the end of each interval.
        """
        blocks = config.num_steps // config.block_size
        thermo = config.thermo_every // config.block_size
        frames = config.frames_every // config.block_size
        return MDLoggedData(
            init=WriterGroupConfig(MDSnapshot.from_state, Once()),
            step=WriterGroupConfig(
                MDThermoData.from_state, EveryNStep(thermo, start=thermo - 1)
            ),
            frames=WriterGroupConfig(
                MDFrameData.from_state, EveryNStep(frames, start=frames - 1)
            ),
            final=WriterGroupConfig(MDSnapshot.from_state, Once(blocks - 1)),
        )
