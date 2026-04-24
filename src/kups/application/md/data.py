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
from kups.core.constants import BOLTZMANN_CONSTANT, FEMTO_SECOND, PASCAL
from kups.core.data import Index, Table
from kups.core.typing import ExclusionId, ParticleId, SystemId
from kups.core.unitcell import UnitCell
from kups.core.utils.jax import dataclass, field, tree_zeros_like
from kups.md.integrators import Integrator
from kups.md.observables import particle_kinetic_energy


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
        unitcell: Unit cell geometry for each system.
        temperature: Target temperature (K), shape ``(n_systems,)``.
        time_step: Integration timestep (internal time units), shape ``(n_systems,)``.
        friction_coefficient: Langevin friction (1/time), shape ``(n_systems,)``.
        thermostat_time_constant: CSVR coupling time (time), shape ``(n_systems,)``.
        target_pressure: Target pressure (energy/length^3), shape ``(n_systems,)``.
        pressure_coupling_time: Barostat coupling time (time), shape ``(n_systems,)``.
        compressibility: Isothermal compressibility (length^3/energy), shape ``(n_systems,)``.
        minimum_scale_factor: Minimum barostat scale factor, shape ``(n_systems,)``.
        unitcell_gradients: Energy gradient w.r.t. the unit cell, stored as a
            :class:`UnitCell` (the ``lattice_vectors`` leaf holds the
            shape-``(n_systems, 3, 3)`` gradient used by
            :attr:`stress_tensor`).
        potential_energy: Total potential energy per system (eV), shape ``(n_systems,)``.
    """

    unitcell: UnitCell
    temperature: Array
    time_step: Array
    friction_coefficient: Array
    thermostat_time_constant: Array
    target_pressure: Array
    pressure_coupling_time: Array
    compressibility: Array
    minimum_scale_factor: Array
    unitcell_gradients: UnitCell
    potential_energy: Array

    @property
    def stress_tensor(self) -> Array:
        """Virial stress tensor, shape ``(n_systems, 3, 3)``."""
        return (
            -self.unitcell_gradients.lattice_vectors
            / self.unitcell.volume[..., None, None]
        )


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
    base, unitcell, _ = particles_from_ase(atoms)
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

    unitcell = unitcell[None]  # Add system dimension
    systems = Table.arange(
        MDSystems(
            unitcell=unitcell,
            temperature=jnp.array([config.temperature]),
            time_step=jnp.array([config.time_step * FEMTO_SECOND]),
            friction_coefficient=jnp.array(
                [config.friction_coefficient / FEMTO_SECOND]
            ),
            thermostat_time_constant=jnp.array(
                [config.thermostat_time_constant * FEMTO_SECOND]
            ),
            target_pressure=jnp.array([config.target_pressure * PASCAL]),
            pressure_coupling_time=jnp.array(
                [config.pressure_coupling_time * FEMTO_SECOND]
            ),
            compressibility=jnp.array([config.compressibility / PASCAL]),
            minimum_scale_factor=jnp.array([config.minimum_scale_factor]),
            unitcell_gradients=tree_zeros_like(unitcell),
            potential_energy=jnp.array([0.0]),
        ),
        label=SystemId,
    )

    return particles, systems


__all__ = [
    "MDParticles",
    "MDSystems",
    "MdRunConfig",
    "MdParameters",
    "md_state_from_ase",
]
