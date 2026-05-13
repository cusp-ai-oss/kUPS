# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and ASE initialisation for molecular dynamics simulations."""

from __future__ import annotations

from pathlib import Path

import ase
import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel

from kups.application.utils.particles import (
    Particles,
    default_exclusion,
    particles_from_ase,
)
from kups.core.cell import Cell
from kups.core.constants import BOLTZMANN_CONSTANT, FEMTO_SECOND, PASCAL
from kups.core.data import Index, Table
from kups.core.typing import ExclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, tree_zeros_like
from kups.md.integrators import Integrator
from kups.md.observables import particle_kinetic_energy


@dataclass
class VerletParams:
    """Control parameters for the NVE Velocity Verlet integrator.

    Attributes:
        time_step: Integration timestep ``Δt`` (internal time units), shape ``(n_systems,)``.
    """

    time_step: Array


@dataclass
class BAOABLangevinParams:
    """Control parameters for the BAOAB Langevin (NVT) integrator.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
        temperature: Target temperature ``T`` (K), shape ``(n_systems,)``.
        friction_coefficient: Langevin friction ``γ`` (1/time), shape ``(n_systems,)``.
    """

    time_step: Array
    temperature: Array
    friction_coefficient: Array


@dataclass
class CSVRParams:
    """Control parameters for the CSVR (NVT) integrator.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
        temperature: Target temperature ``T`` (K), shape ``(n_systems,)``.
        thermostat_time_constant: CSVR coupling time ``τ`` (time), shape ``(n_systems,)``.
    """

    time_step: Array
    temperature: Array
    thermostat_time_constant: Array


@dataclass
class CSVRNPTParams:
    """Control parameters for the CSVR-NPT integrator.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
        temperature: Target temperature ``T`` (K), shape ``(n_systems,)``.
        thermostat_time_constant: CSVR coupling time ``τ`` (time), shape ``(n_systems,)``.
        target_pressure: Target pressure ``P₀`` (energy/length³), shape ``(n_systems,)``.
        pressure_coupling_time: Barostat coupling time ``τ_P`` (time), shape ``(n_systems,)``.
        compressibility: Isothermal compressibility ``β`` (length³/energy), shape ``(n_systems,)``.
        minimum_scale_factor: Minimum barostat scale factor, shape ``(n_systems,)``.
    """

    time_step: Array
    temperature: Array
    thermostat_time_constant: Array
    target_pressure: Array
    pressure_coupling_time: Array
    compressibility: Array
    minimum_scale_factor: Array


type IntegratorParams = VerletParams | BAOABLangevinParams | CSVRParams | CSVRNPTParams


@dataclass
class MDParticles(Particles):
    """Particle state for molecular dynamics simulations.

    Extends :class:`Particles` with gradient, momenta, and derived
    kinematic quantities needed by MD integrators.

    Attributes:
        position_gradients: Energy gradient w.r.t. positions, shape ``(n_atoms, 3)``.
        momenta: Particle momenta, shape ``(n_atoms, 3)``.
        exclusion: Per-particle exclusion index (defaults to one group per
            atom via :func:`default_exclusion` if not supplied).
    """

    position_gradients: Array
    momenta: Array
    exclusion: Index[ExclusionId] = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self):
        if self.exclusion is None:
            object.__setattr__(self, "exclusion", default_exclusion(len(self.charges)))

    @property
    def forces(self) -> Array:
        """Negative position gradient, shape ``(n_atoms, 3)``."""
        return -self.position_gradients

    @property
    def velocities(self) -> Array:
        """Velocities derived from momenta and masses, shape ``(n_atoms, 3)``."""
        return self.momenta / self.masses[..., None]

    @property
    def kinetic_energy(self) -> Array:
        """Per-particle kinetic energy, shape ``(n_atoms,)``."""
        return particle_kinetic_energy(self.momenta, self.masses)


@dataclass
class MDSystems:
    """Per-system state for molecular dynamics simulations.

    Attributes:
        cell: Cell geometry for each system.
        integrator_params: Bundled integrator control parameters; concrete shape
            (e.g. :class:`VerletParams`, :class:`BAOABLangevinParams`,
            :class:`CSVRParams`, :class:`CSVRNPTParams`) is chosen to match the
            selected integrator.
        cell_gradients: Energy gradient w.r.t. the cell, stored as a
            :class:`Cell` (the ``vectors`` leaf holds the
            shape-``(n_systems, 3, 3)`` gradient used by
            :attr:`stress_tensor`).
        potential_energy: Total potential energy per system (eV), shape ``(n_systems,)``.
    """

    cell: Cell
    integrator_params: IntegratorParams
    cell_gradients: Cell
    potential_energy: Array


class MdRunConfig(BaseModel):
    """Run configuration for an MD simulation."""

    out_file: str | Path
    """Path to the output HDF5 file."""
    num_steps: int
    """Number of production steps."""
    num_warmup_steps: int
    """Number of warmup steps before production."""
    seed: int | None
    """Random seed for reproducibility. None for time-based."""


class MdParameters(BaseModel):
    """Physical and numerical parameters for an MD simulation."""

    temperature: float
    """Target temperature (K)."""
    time_step: float
    """Integration timestep (fs)."""
    friction_coefficient: float
    """Langevin friction coefficient (1/fs)."""
    thermostat_time_constant: float
    """CSVR thermostat coupling time (fs)."""
    target_pressure: float
    """Target pressure for NPT barostat (Pa)."""
    pressure_coupling_time: float
    """Barostat coupling time (fs)."""
    compressibility: float
    """Isothermal compressibility (1/Pa)."""
    minimum_scale_factor: float
    """Minimum allowed box scaling factor per barostat step (dimensionless)."""
    integrator: Integrator
    """Integration algorithm to use."""
    initialize_momenta: bool = False
    """If True, initialize momenta from Maxwell-Boltzmann distribution."""


def md_state_from_ase(
    atoms: ase.Atoms | str | Path,
    config: MdParameters,
    *,
    key: Array | None = None,
) -> tuple[Table[ParticleId, MDParticles], Table[SystemId, MDSystems]]:
    """Build MD particles and system data from an ASE Atoms object or file.

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.
        config: MD configuration with temperature, timestep, and thermostat/barostat
            parameters.
        key: JAX PRNG key for Maxwell-Boltzmann momenta initialisation. If None,
            momenta are set to zero.

    Returns:
        Tuple of (particles, systems) ready for use with MD integrators.
    """
    base, cell, _ = particles_from_ase(atoms)
    p = base.data
    n_atoms = p.positions.shape[0]

    if key is not None:
        # Sample momenta from Maxwell-Boltzmann: p_i ~ N(0, sqrt(m_i * kT))
        std = jnp.sqrt(p.masses * config.temperature * BOLTZMANN_CONSTANT)
        momenta = jax.random.normal(key, (n_atoms, 3)) * std[:, None]
        # Remove centre-of-mass drift
        momenta -= momenta.sum(axis=0) / n_atoms
    else:
        momenta = jnp.zeros((n_atoms, 3))

    particles = Table.arange(
        MDParticles(
            positions=p.positions,
            masses=p.masses,
            atomic_numbers=p.atomic_numbers,
            charges=p.charges,
            labels=p.labels,
            system=p.system,
            position_gradients=jnp.zeros_like(p.positions),
            momenta=momenta,
        ),
        label=ParticleId,
    )

    cell = cell[None]  # Add system dimension
    systems = Table.arange(
        MDSystems(
            cell=cell,
            integrator_params=_build_integrator_params(config),
            cell_gradients=tree_zeros_like(cell),
            potential_energy=jnp.array([0.0]),
        ),
        label=SystemId,
    )

    return particles, systems


def _build_integrator_params(config: MdParameters) -> IntegratorParams:
    """Construct the concrete integrator-params dataclass matching ``config.integrator``."""
    time_step = jnp.array([config.time_step * FEMTO_SECOND])
    temperature = jnp.array([config.temperature])
    match config.integrator:
        case "verlet":
            return VerletParams(time_step=time_step)
        case "baoab_langevin":
            return BAOABLangevinParams(
                time_step=time_step,
                temperature=temperature,
                friction_coefficient=jnp.array(
                    [config.friction_coefficient / FEMTO_SECOND]
                ),
            )
        case "csvr":
            return CSVRParams(
                time_step=time_step,
                temperature=temperature,
                thermostat_time_constant=jnp.array(
                    [config.thermostat_time_constant * FEMTO_SECOND]
                ),
            )
        case "csvr_npt":
            return CSVRNPTParams(
                time_step=time_step,
                temperature=temperature,
                thermostat_time_constant=jnp.array(
                    [config.thermostat_time_constant * FEMTO_SECOND]
                ),
                target_pressure=jnp.array([config.target_pressure * PASCAL]),
                pressure_coupling_time=jnp.array(
                    [config.pressure_coupling_time * FEMTO_SECOND]
                ),
                compressibility=jnp.array([config.compressibility / PASCAL]),
                minimum_scale_factor=jnp.array([config.minimum_scale_factor]),
            )


__all__ = [
    "MDParticles",
    "MDSystems",
    "MdRunConfig",
    "MdParameters",
    "VerletParams",
    "BAOABLangevinParams",
    "CSVRParams",
    "CSVRNPTParams",
    "IntegratorParams",
    "md_state_from_ase",
]
