# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and builders for molecular dynamics simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import ase
import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel

from kups.application.utils.data import Scalar
from kups.application.utils.particles import (
    Particles,
    default_exclusion,
    particles_from_ase,
)
from kups.core import units
from kups.core.cell import AnyPeriodicity, Cell, zero_gradient_cell
from kups.core.data import Index, Table
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.typing import ExclusionId, ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.units.pydantic import UnitsModel
from kups.core.utils.jax import dataclass, field
from kups.md.integrators import Integrator
from kups.md.observables import particle_kinetic_energy, remove_center_of_mass_momentum


@dataclass
class VerletParams:
    """Control parameters for the NVE Velocity Verlet integrator.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
    """

    time_step: units.Time


@dataclass
class BAOABLangevinParams:
    """Control parameters for the BAOAB Langevin (NVT) integrator.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
        temperature: Target temperature ``T``, shape ``(n_systems,)``.
        friction_coefficient: Langevin friction ``γ``, shape ``(n_systems,)``.
    """

    time_step: units.Time
    temperature: units.Temperature
    friction_coefficient: units.Frequency


@dataclass
class CSVRParams:
    """Control parameters for the CSVR (NVT) integrator.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
        temperature: Target temperature ``T``, shape ``(n_systems,)``.
        thermostat_time_constant: CSVR coupling time ``τ``, shape ``(n_systems,)``.
    """

    time_step: units.Time
    temperature: units.Temperature
    thermostat_time_constant: units.Time


@dataclass
class CSVRNPTParams:
    """Control parameters for the CSVR-NPT integrator.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
        temperature: Target temperature ``T``, shape ``(n_systems,)``.
        thermostat_time_constant: CSVR coupling time ``τ``, shape ``(n_systems,)``.
        target_pressure: Target pressure ``P₀``, shape ``(n_systems,)``.
        pressure_coupling_time: Barostat coupling time ``τ_P``, shape ``(n_systems,)``.
        compressibility: Isothermal compressibility ``β``, shape ``(n_systems,)``.
        minimum_scale_factor: Minimum barostat scale factor, shape ``(n_systems,)``.
    """

    time_step: units.Time
    temperature: units.Temperature
    thermostat_time_constant: units.Time
    target_pressure: units.Pressure
    pressure_coupling_time: units.Time
    compressibility: units.Compressibility
    minimum_scale_factor: Array


@dataclass
class BAOABNPTLangevinParams:
    r"""Control parameters for the BAOAB NPT Langevin integrator.

    Implements the fully-flexible-cell extended-variable NPT Langevin
    formulation of Gao, Fang & Wang, *Sampling the isothermal-isobaric
    ensemble by Langevin dynamics*, JCP 2016 (arxiv 1601.01044). The atom
    side reuses the existing Langevin friction ``γ``; the cell side adds a
    fictitious mass tensor and a separate Langevin friction.

    Attributes:
        time_step: Integration timestep ``Δt``, shape ``(n_systems,)``.
        temperature: Target temperature ``T``, shape ``(n_systems,)``.
        friction_coefficient: Atom Langevin friction ``γ``,
            shape ``(n_systems,)``. Reused from BAOAB-Langevin.
        target_pressure: Target pressure ``P₀``,
            shape ``(n_systems,)``.
        pressure_coupling_time: Barostat coupling time ``τ_P``,
            shape ``(n_systems,)``. Drives the auto-derived
            :attr:`barostat_mass` per Gao Eq. (14)/(15).
        compressibility: Isothermal compressibility ``β``,
            shape ``(n_systems,)``. Drives :attr:`barostat_mass`.
        barostat_mass: Fictitious cell mass tensor $M_{\alpha\beta}$,
            lower-triangular, shape ``(n_systems, 3, 3)``.
            Auto-derived from ``compressibility``, ``pressure_coupling_time``
            and the initial cell at construction time.
        barostat_friction: Langevin friction on cell DOFs $\gamma_{\alpha\beta}$,
            lower-triangular, shape ``(n_systems, 3, 3)``. Defaults
            to ``friction_coefficient`` broadcast across the 6 cell DOFs per
            Gao Remark 3.
    """

    time_step: units.Time
    temperature: units.Temperature
    friction_coefficient: units.Frequency
    target_pressure: units.Pressure
    pressure_coupling_time: units.Time
    compressibility: units.Compressibility
    barostat_mass: units.Mass
    barostat_friction: units.Frequency


type IntegratorParams = (
    VerletParams
    | BAOABLangevinParams
    | CSVRParams
    | CSVRNPTParams
    | BAOABNPTLangevinParams
)


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

    position_gradients: units.Force
    momenta: units.Momentum
    exclusion: Index[ExclusionId] = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.exclusion is None:
            object.__setattr__(self, "exclusion", default_exclusion(len(self.charges)))

    @property
    def forces(self) -> units.Force:
        """Negative position gradient, shape ``(n_atoms, 3)``."""
        return -self.position_gradients

    @property
    def velocities(self) -> units.Velocity:
        """Velocities derived from momenta and masses, shape ``(n_atoms, 3)``."""
        return self.momenta / self.masses[..., None]

    @property
    def kinetic_energy(self) -> units.Energy:
        """Per-particle kinetic energy, shape ``(n_atoms,)``."""
        return particle_kinetic_energy(self.momenta, self.masses)


@dataclass
class MDSystems:
    r"""Per-system state for molecular dynamics simulations.

    Attributes:
        cell: Cell geometry for each system.
        integrator_params: Bundled integrator control parameters; concrete shape
            (e.g. :class:`VerletParams`, :class:`BAOABLangevinParams`,
            :class:`CSVRParams`, :class:`CSVRNPTParams`,
            :class:`BAOABNPTLangevinParams`) is chosen to match the selected
            integrator.
        cell_gradients: Energy gradient w.r.t. the cell, stored as a
            :class:`Cell` (the ``vectors`` leaf holds the
            shape-``(n_systems, 3, 3)`` gradient used by
            :attr:`stress_tensor`).
        cell_momentum: Extended-variable cell-momentum tensor $p^h$,
            lower-triangular ``(n_systems, 3, 3)``. Only meaningful for
            integrators that drive cell dynamics via an explicit conjugate
            momentum (e.g. BAOAB NPT Langevin); zero otherwise.
        potential_energy: Total potential energy per system, shape ``(n_systems,)``.
    """

    cell: Cell[Any]
    integrator_params: IntegratorParams
    cell_gradients: Cell[Any]
    cell_momentum: units.Momentum
    potential_energy: units.Energy


@dataclass
class MdState:
    """Force-field-agnostic molecular-dynamics state.

    The potential is built with its parameters at construction time (via the
    adapters' ``parameters=``), so no force-field field lives on the state.
    """

    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    step: Array


class MdRunConfig(BaseModel):
    """Run configuration for an MD simulation."""

    out_file: str | Path
    """Path to the output HDF5 file."""
    num_steps: int
    """Number of production steps."""
    num_warmup_steps: int
    """Number of warmup steps before production."""
    block_size: int = 1
    """Production steps fused per on-device dispatch (1 = per-step). Only the last frame of
    each block is saved, so the trajectory has num_steps // block_size frames."""
    seed: int | None
    """Random seed for reproducibility. None for time-based."""


class MdParameters(UnitsModel):
    """Physical and numerical parameters for an MD simulation."""

    temperature: Annotated[units.Temperature, Scalar]
    """Target temperature."""
    time_step: Annotated[units.Time, Scalar]
    """Integration timestep."""
    friction_coefficient: Annotated[units.Frequency, Scalar]
    """Langevin friction coefficient."""
    thermostat_time_constant: Annotated[units.Time, Scalar]
    """CSVR thermostat coupling time."""
    target_pressure: Annotated[units.Pressure, Scalar]
    """Target pressure for NPT barostat."""
    pressure_coupling_time: Annotated[units.Time, Scalar]
    """Barostat coupling time."""
    compressibility: Annotated[units.Compressibility, Scalar]
    """Isothermal compressibility."""
    minimum_scale_factor: float
    """Minimum allowed box scaling factor per barostat step (dimensionless)."""
    integrator: Integrator
    """Integration algorithm to use."""
    initialize_momenta: bool = False
    """If True, initialize momenta from Maxwell-Boltzmann distribution."""


def md_state_from_particles_and_cell(
    particles: Table[ParticleId, Particles],
    cell: Cell[AnyPeriodicity],
    config: MdParameters,
    *,
    key: Array | None = None,
) -> tuple[Table[ParticleId, MDParticles], Table[SystemId, MDSystems]]:
    """Build one-system MD particle and system tables from source-neutral data.

    The supplied particles and cell must already describe the same system in
    the same kUPS coordinate frame. This function does not transform geometry;
    it only adds the system dimension to the cell.

    Args:
        particles: Non-empty particle table for exactly one complete system.
        cell: Unbatched cell with vectors of shape ``(3, 3)``.
        config: MD configuration with temperature, timestep, and
            thermostat/barostat parameters.
        key: JAX PRNG key for Maxwell-Boltzmann momenta initialisation. If None,
            momenta are set to zero.

    Returns:
        MD particle and system tables preserving the input particle keys and
        referenced ``SystemId``.

    Raises:
        ValueError: If ``particles`` is empty, does not contain one valid
            system reference per particle, or if ``cell.vectors`` does not
            have shape ``(3, 3)``.
    """
    p = particles.data
    n_particles = len(particles)
    if n_particles == 0:
        raise ValueError("particles must contain at least one particle.")
    if len(p.system.keys) != 1:
        raise ValueError(
            "particles must reference exactly one system; "
            f"got {len(p.system.keys)} system keys."
        )
    if p.system.shape != (n_particles,):
        raise ValueError(
            "particles must contain one system reference per particle; "
            f"got shape {p.system.shape} for {n_particles} particles."
        )
    if not bool(p.system.valid_mask.all()):
        raise ValueError(
            "particles system references must all select the sole SystemId."
        )
    if cell.vectors.shape != (3, 3):
        raise ValueError(
            f"cell.vectors must have shape (3, 3); got {cell.vectors.shape}."
        )

    if key is not None:
        # Sample momenta from Maxwell-Boltzmann: p_i ~ N(0, sqrt(m_i * kT))
        std = units.sqrt(p.masses * units.constants.k_B * config.temperature)
        momenta = jax.random.normal(key, (n_particles, 3)) * std[:, None]
        momenta = remove_center_of_mass_momentum(momenta, p.masses, p.system)
    else:
        momenta = units.zeros((n_particles, 3), u.amu * u.angstrom / u.fs)

    md_particles = particles.set_data(
        MDParticles(
            positions=p.positions,
            masses=p.masses,
            atomic_numbers=p.atomic_numbers,
            charges=p.charges,
            labels=p.labels,
            system=p.system,
            position_gradients=units.zeros(
                p.positions.shape, u.eV / u.angstrom, dtype=p.positions.dtype
            ),
            momenta=momenta,
        )
    )

    batched_cell = cell[None]
    systems = Table(
        (p.system.keys[0],),
        MDSystems(
            cell=batched_cell,
            integrator_params=_build_integrator_params(config, batched_cell),
            cell_gradients=zero_gradient_cell(batched_cell),
            cell_momentum=units.zeros(
                batched_cell.vectors.shape, u.amu * u.angstrom / u.fs
            ),
            potential_energy=units.zeros((1,), u.eV),
        ),
    )

    return md_particles, systems


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
    particles, cell, _ = particles_from_ase(atoms)
    return md_state_from_particles_and_cell(particles, cell, config, key=key)


def _gao_barostat_mass(
    cell_vectors: units.Length,
    compressibility: units.Compressibility,
    tau_P: units.Time,
) -> units.Mass:
    r"""Gao Eq. (14)/(15) fictitious cell mass tensor.

    .. math::

        M_{\alpha\beta} = \frac{3\,\det(h_0)}{\kappa\,(h_0)_{\alpha\alpha}^2}
                          \left(\frac{\tau_{\alpha\beta}}{2\pi}\right)^2

    For simplicity all $\tau_{\alpha\beta}$ default to a single barostat
    time-scale ``tau_P``. Gao writes the formula for the paper's upper-triangular
    cell matrix ``h``. kUPS stores ``V = h.T`` and ``p_h = p_h_paper.T``, so the
    paper row index becomes the lower-triangular column index here.
    """
    V0 = units.absolute(units.det(cell_vectors))  # (n_sys,)
    h_diag = cell_vectors.diagonal(axis1=-2, axis2=-1)  # (n_sys, 3)
    # Per-paper-row scale. In kUPS' transposed lower-triangular convention this
    # is a per-column scale for active cell-momentum DOFs.
    per_column = (
        3.0
        * V0[..., None]
        * (tau_P[..., None] / (2.0 * jnp.pi)) ** 2
        / (compressibility[..., None] * h_diag**2)
    )  # (n_sys, 3)
    M = units.apply(
        jnp.broadcast_to, per_column[..., None, :], (*per_column.shape, 3)
    )  # (n_sys, 3, 3)
    return units.apply(jnp.tril, M)


def _build_integrator_params(
    config: MdParameters, cell: Cell[AnyPeriodicity]
) -> IntegratorParams:
    """Construct the concrete integrator-params dataclass matching ``config.integrator``."""
    time_step = config.time_step[None]
    temperature = config.temperature[None]
    match config.integrator:
        case "verlet":
            return VerletParams(time_step=time_step)
        case "baoab_langevin":
            return BAOABLangevinParams(
                time_step=time_step,
                temperature=temperature,
                friction_coefficient=config.friction_coefficient[None],
            )
        case "csvr":
            return CSVRParams(
                time_step=time_step,
                temperature=temperature,
                thermostat_time_constant=config.thermostat_time_constant[None],
            )
        case "csvr_npt":
            return CSVRNPTParams(
                time_step=time_step,
                temperature=temperature,
                thermostat_time_constant=config.thermostat_time_constant[None],
                target_pressure=config.target_pressure[None],
                pressure_coupling_time=config.pressure_coupling_time[None],
                compressibility=config.compressibility[None],
                minimum_scale_factor=jnp.array([config.minimum_scale_factor]),
            )
        case "baoab_npt_langevin":
            tau_P = config.pressure_coupling_time[None]
            kappa = config.compressibility[None]
            gamma = config.friction_coefficient[None]
            barostat_mass = _gao_barostat_mass(cell.vectors, kappa, tau_P)
            barostat_friction = units.apply(
                lambda g: jnp.tril(jnp.broadcast_to(g, barostat_mass.shape)),
                gamma[..., None, None],
            )
            return BAOABNPTLangevinParams(
                time_step=time_step,
                temperature=temperature,
                friction_coefficient=gamma,
                target_pressure=config.target_pressure[None],
                pressure_coupling_time=tau_P,
                compressibility=kappa,
                barostat_mass=barostat_mass,
                barostat_friction=barostat_friction,
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
    "BAOABNPTLangevinParams",
    "IntegratorParams",
    "md_state_from_particles_and_cell",
    "md_state_from_ase",
]
