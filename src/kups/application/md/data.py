# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and builders for molecular dynamics simulations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import ase
import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, Field, model_validator

from kups.application.utils.particles import (
    Particles,
    default_exclusion,
    particles_from_ase,
)
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.constants import BOLTZMANN_CONSTANT, FEMTO_SECOND, PASCAL
from kups.core.data import Index, Table, cast_indices
from kups.core.lens import bind
from kups.core.neighborlist import SkinList, UniversalNeighborlistParameters
from kups.core.storage import HDF5StorageReader
from kups.core.typing import DTypeLike, ExclusionId, ParticleId, SystemId
from kups.core.utils.jax import astype_floating, dataclass, field, tree_zeros_like
from kups.md.integrators import Integrator
from kups.md.observables import particle_kinetic_energy, remove_center_of_mass_momentum

if TYPE_CHECKING:
    # logging imports this module
    from kups.application.md.logging import MDLoggedData


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
        temperature: Target temperature ``T`` (K), shape ``(n_systems,)``.
        friction_coefficient: Atom Langevin friction ``γ`` (1/time),
            shape ``(n_systems,)``. Reused from BAOAB-Langevin.
        target_pressure: Target pressure ``P₀`` (energy/length³),
            shape ``(n_systems,)``.
        pressure_coupling_time: Barostat coupling time ``τ_P`` (time),
            shape ``(n_systems,)``. Drives the auto-derived
            :attr:`barostat_mass` per Gao Eq. (14)/(15).
        compressibility: Isothermal compressibility ``β`` (length³/energy),
            shape ``(n_systems,)``. Drives :attr:`barostat_mass`.
        barostat_mass: Fictitious cell mass tensor $M_{\alpha\beta}$
            (mass·length²), lower-triangular, shape ``(n_systems, 3, 3)``.
            Auto-derived from ``compressibility``, ``pressure_coupling_time``
            and the initial cell at construction time.
        barostat_friction: Langevin friction on cell DOFs $\gamma_{\alpha\beta}$
            (1/time), lower-triangular, shape ``(n_systems, 3, 3)``. Defaults
            to ``friction_coefficient`` broadcast across the 6 cell DOFs per
            Gao Remark 3.
    """

    time_step: Array
    temperature: Array
    friction_coefficient: Array
    target_pressure: Array
    pressure_coupling_time: Array
    compressibility: Array
    barostat_mass: Array
    barostat_friction: Array


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
            atom via :func:`default_exclusion`, in the dtype of ``system``).
    """

    position_gradients: Array
    momenta: Array
    exclusion: Index[ExclusionId] = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.exclusion is None:
            dtype = self.system.raw_indices.dtype
            exclusion = default_exclusion(len(self.charges), dtype)
            object.__setattr__(self, "exclusion", exclusion)

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
        potential_energy: Total potential energy per system (eV), shape ``(n_systems,)``.
    """

    cell: Cell[Any]
    integrator_params: IntegratorParams
    cell_gradients: Cell[Any]
    cell_momentum: Array
    potential_energy: Array


@dataclass
class MdState:
    """Force-field-agnostic molecular-dynamics state.

    The potential is built with its parameters at construction time (via the
    adapters' ``parameters=``), so no force-field field lives on the state.
    """

    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    skin_list: SkinList
    step: Array

    @staticmethod
    def create(
        particles: Table[ParticleId, MDParticles],
        systems: Table[SystemId, MDSystems],
        cutoffs: Table[SystemId, Array],
        skin: float,
        *,
        base: float = 2,
        multiplier: float = 1.0,
    ) -> MdState:
        """State at step 0 with capacities and the skin list seeded from the tables.

        Args:
            particles: Particles of all systems.
            systems: Their systems.
            cutoffs: Cutoffs (Å) of the potential.
            skin: Skin width (Å) of every system.
            base: Rounding base of the capacity estimates.
            multiplier: Headroom factor of the capacity estimates.

        Returns:
            The MD state.
        """
        params = UniversalNeighborlistParameters.estimate(
            particles.data.system.counts,
            systems,
            cutoffs,
            base=base,
            multiplier=multiplier,
        )
        skins = Table.arange(jnp.array([skin]), label=SystemId)
        skin_list = SkinList.create(particles, systems, cutoffs, skins)
        return MdState(particles, systems, params, skin_list, jnp.array([0]))


class MdRunConfig(BaseModel):
    """Run configuration for an MD simulation."""

    out_file: str | Path
    """Path to the output HDF5 file."""
    num_steps: int
    """Number of production steps, a multiple of ``block_size``."""
    num_warmup_steps: int
    """Number of warmup steps before production."""
    block_size: int = Field(default=1, ge=1)
    """Production steps fused per on-device dispatch (1 = per-step). Output is written
    after whole blocks."""
    thermo_every: int = Field(default_factory=lambda data: data["block_size"], ge=1)
    """Steps between rows of energies, stress and volume, a multiple of ``block_size``.
    Defaults to ``block_size``."""
    frames_every: int = Field(default_factory=lambda data: data["block_size"], ge=1)
    """Steps between frames (positions, momenta, cells), a multiple of ``block_size``.
    Defaults to ``block_size``."""
    seed: int | None
    """Random seed for reproducibility. None for time-based."""
    index_dtype: Literal["int32", "int64"] = "int64"
    """Integer dtype of all state indices. ``int32`` is faster on GPU."""
    float_dtype: Literal["float32", "float64"] = "float64"
    """Float dtype of the particle and system tables: positions, momenta, cells and
    their other floats. ``float32`` is faster on GPUs with slow double precision;
    give Lennard-Jones a ``float32`` ``kernel_dtype`` too."""

    @model_validator(mode="after")
    def _check_whole_blocks(self) -> MdRunConfig:
        """Reject a production length or output intervals that are not whole blocks."""
        steps = {
            "num_steps": self.num_steps,
            "thermo_every": self.thermo_every,
            "frames_every": self.frames_every,
        }
        for name, value in steps.items():
            if value % self.block_size:
                raise ValueError(
                    f"run.{name} ({value} steps) must be a multiple of run.block_size "
                    f"({self.block_size} steps): production and output run in whole "
                    "blocks."
                )
        return self


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
    rescale_to_mean_volume: bool = False
    """Scale each cell of a previous ``kups_md`` run given as input, and its
    positions, to the mean volume of that run's rows."""
    skin: float = 1.0
    """Verlet skin (Å) of the neighbor list. Pairs within cutoff + skin are listed
    once and listed again when atom motion or cell deformation could have brought a
    missing pair within the cutoff. The skin changes speed, not results: a larger
    skin lists less often but checks more pairs every step."""


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
    it only adds the system dimension to the cell. Momenta, gradients, energies
    and integrator parameters take the float dtype of the positions.

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

    dtype = p.positions.dtype
    if key is not None:
        momenta = _maxwell_boltzmann_momenta(key, p, config.temperature)
    else:
        momenta = jnp.zeros((n_particles, 3), dtype)

    md_particles = particles.set_data(
        MDParticles(
            positions=p.positions,
            masses=p.masses,
            atomic_numbers=p.atomic_numbers,
            charges=p.charges,
            labels=p.labels,
            system=p.system,
            position_gradients=jnp.zeros_like(p.positions),
            momenta=momenta,
        )
    )

    batched_cell = cell[None]
    systems = Table(
        (p.system.keys[0],),
        MDSystems(
            cell=batched_cell,
            integrator_params=_build_integrator_params(config, batched_cell),
            cell_gradients=tree_zeros_like(batched_cell),
            cell_momentum=jnp.zeros(batched_cell.vectors.shape, dtype),
            potential_energy=jnp.zeros(1, dtype),
        ),
    )

    return md_particles, systems


def md_state_from_ase(
    atoms: ase.Atoms | str | Path,
    config: MdParameters,
    *,
    key: Array | None = None,
    index_dtype: DTypeLike = int,
    float_dtype: DTypeLike = float,
) -> tuple[Table[ParticleId, MDParticles], Table[SystemId, MDSystems]]:
    """Build MD particles and system data from an ASE Atoms object or file.

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.
        config: MD configuration with temperature, timestep, and thermostat/barostat
            parameters.
        key: JAX PRNG key for Maxwell-Boltzmann momenta initialisation. If None,
            momenta are set to zero.
        index_dtype: Integer dtype of the particle indices.
        float_dtype: Float dtype of the particle and system tables.

    Returns:
        Tuple of (particles, systems) ready for use with MD integrators.

    Raises:
        ValueError: If ``config.rescale_to_mean_volume`` is set.
    """
    if config.rescale_to_mean_volume:
        raise ValueError(
            "md.rescale_to_mean_volume applies to previous kups_md runs, not to "
            f"structures such as {atoms}."
        )
    particles, cell, _ = particles_from_ase(
        atoms, index_dtype=index_dtype, float_dtype=float_dtype
    )
    return md_state_from_particles_and_cell(particles, cell, config, key=key)


def md_state_from_run(
    path: str | Path,
    config: MdParameters,
    *,
    key: Array | None = None,
    index_dtype: DTypeLike = int,
    float_dtype: DTypeLike = float,
) -> tuple[Table[ParticleId, MDParticles], Table[SystemId, MDSystems]]:
    """Build MD particles and systems from the final state of a ``kups_md`` run.

    Particles keep their order, positions and momenta, and systems their cells;
    integrator parameters come from ``config``. With
    ``config.rescale_to_mean_volume``, each cell and its positions are scaled to
    the mean volume of the run's rows, which keeps fractional coordinates, and cell
    momenta are reset.

    Args:
        path: Output file of a finished ``kups_md`` run.
        config: MD configuration of the new run.
        key: JAX PRNG key for new Maxwell-Boltzmann momenta. If None, the run's
            momenta are kept.
        index_dtype: Integer dtype of the particle and system indices.
        float_dtype: Float dtype of the particle and system tables.

    Returns:
        Tuple of (particles, systems) ready for use with MD integrators.

    Raises:
        ValueError: If the file holds no final state of a finished run, or no rows
            to average the volume over.
    """
    reader: HDF5StorageReader[MDLoggedData]
    with HDF5StorageReader(path) as reader:
        # killed runs have no actual_steps, and runs without steps no final state
        done = reader.file.attrs.get("actual_steps", 0)
        if done < 1 or done != reader.file.attrs.get("total_steps"):
            raise ValueError(f"{path} holds no final state of a finished kups_md run.")
        final = reader.focus_group(lambda s: s.final)[...]
        volumes = reader.focus_group(lambda s: s.step).read(select=lambda d: d.volume)
    # in the dtypes of this run, which may differ from those of the earlier one
    particles, systems = astype_floating(
        cast_indices(
            jax.tree.map(jnp.asarray, (final.atoms, final.systems)), index_dtype
        ),
        float_dtype,
    )
    if config.rescale_to_mean_volume:
        if len(volumes) == 0:
            raise ValueError(f"{path} has no rows to average the volume over.")
        cell = systems.data.cell
        mean = jnp.mean(volumes, axis=0, dtype=jnp.float64)
        scale = ((mean / cell.volume) ** (1 / 3)).astype(cell.volume.dtype)
        particle_scale = systems.set_data(scale)[particles.data.system]
        particles = bind(particles, lambda t: t.data.positions).apply(
            lambda positions: positions * particle_scale[:, None]
        )
        systems = bind(systems, lambda t: t.data.cell).set(cell * scale)
        systems = bind(systems, lambda t: t.data.cell_momentum).apply(jnp.zeros_like)
    if key is not None:
        momenta = _maxwell_boltzmann_momenta(key, particles.data, config.temperature)
        particles = bind(particles, lambda t: t.data.momenta).set(momenta)
    params = _build_integrator_params(config, systems.data.cell)
    return particles, bind(systems, lambda t: t.data.integrator_params).set(params)


def _maxwell_boltzmann_momenta(
    key: Array, particles: Particles, temperature: float
) -> Array:
    """Momenta drawn from the Maxwell-Boltzmann distribution at ``temperature``,
    p_i ~ N(0, sqrt(m_i k T)), without center-of-mass motion, in the dtype of the
    positions. They are drawn in the default float dtype, so that float32 states
    start from the same momenta."""
    std = jnp.sqrt(particles.masses * temperature * BOLTZMANN_CONSTANT)
    momenta = jax.random.normal(key, (len(particles.masses), 3)) * std[:, None]
    momenta = remove_center_of_mass_momentum(
        momenta, particles.masses, particles.system
    )
    return momenta.astype(particles.positions.dtype)


def _gao_barostat_mass(
    cell_vectors: Array, compressibility: Array, tau_P: Array
) -> Array:
    r"""Gao Eq. (14)/(15) fictitious cell mass tensor.

    .. math::

        M_{\alpha\beta} = \frac{3\,\det(h_0)}{\kappa\,(h_0)_{\alpha\alpha}^2}
                          \left(\frac{\tau_{\alpha\beta}}{2\pi}\right)^2

    For simplicity all $\tau_{\alpha\beta}$ default to a single barostat
    time-scale ``tau_P``. Gao writes the formula for the paper's upper-triangular
    cell matrix ``h``. kUPS stores ``V = h.T`` and ``p_h = p_h_paper.T``, so the
    paper row index becomes the lower-triangular column index here.
    """
    V0 = jnp.abs(jnp.linalg.det(cell_vectors))  # (n_sys,)
    h_diag = jnp.diagonal(cell_vectors, axis1=-2, axis2=-1)  # (n_sys, 3)
    # Per-paper-row scale. In kUPS' transposed lower-triangular convention this
    # is a per-column scale for active cell-momentum DOFs.
    per_column = (
        3.0
        * V0[..., None]
        * (tau_P[..., None] / (2.0 * jnp.pi)) ** 2
        / (compressibility[..., None] * h_diag**2)
    )  # (n_sys, 3)
    M = jnp.broadcast_to(
        per_column[..., None, :], (*per_column.shape, 3)
    )  # (n_sys, 3, 3)
    return jnp.tril(M)


def _build_integrator_params(
    config: MdParameters, cell: Cell[AnyPeriodicity]
) -> IntegratorParams:
    """Integrator parameters matching ``config.integrator``, one row per cell, in the
    float dtype of the cells."""

    def per_system(value: float) -> Array:
        return jnp.full(cell.volume.shape, value, cell.vectors.dtype)

    time_step = per_system(config.time_step * FEMTO_SECOND)
    temperature = per_system(config.temperature)
    match config.integrator:
        case "verlet":
            return VerletParams(time_step=time_step)
        case "baoab_langevin":
            return BAOABLangevinParams(
                time_step=time_step,
                temperature=temperature,
                friction_coefficient=per_system(
                    config.friction_coefficient / FEMTO_SECOND
                ),
            )
        case "csvr":
            return CSVRParams(
                time_step=time_step,
                temperature=temperature,
                thermostat_time_constant=per_system(
                    config.thermostat_time_constant * FEMTO_SECOND
                ),
            )
        case "csvr_npt":
            return CSVRNPTParams(
                time_step=time_step,
                temperature=temperature,
                thermostat_time_constant=per_system(
                    config.thermostat_time_constant * FEMTO_SECOND
                ),
                target_pressure=per_system(config.target_pressure * PASCAL),
                pressure_coupling_time=per_system(
                    config.pressure_coupling_time * FEMTO_SECOND
                ),
                compressibility=per_system(config.compressibility / PASCAL),
                minimum_scale_factor=per_system(config.minimum_scale_factor),
            )
        case "baoab_npt_langevin":
            tau_P = per_system(config.pressure_coupling_time * FEMTO_SECOND)
            kappa = per_system(config.compressibility / PASCAL)
            gamma = per_system(config.friction_coefficient / FEMTO_SECOND)
            barostat_mass = _gao_barostat_mass(cell.vectors, kappa, tau_P)
            barostat_friction = jnp.tril(
                jnp.broadcast_to(gamma[..., None, None], barostat_mass.shape)
            )
            return BAOABNPTLangevinParams(
                time_step=time_step,
                temperature=temperature,
                friction_coefficient=gamma,
                target_pressure=per_system(config.target_pressure * PASCAL),
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
    "md_state_from_run",
]
