# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and ASE initialisation for molecular dynamics simulations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import ase
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from pydantic import BaseModel, Field

from kups.application.mcmc.data import AdsorbateConfig, MotifParticles
from kups.application.utils.particles import (
    Particles,
    default_exclusion,
    particles_from_ase,
)
from kups.core.constants import BOLTZMANN_CONSTANT, FEMTO_SECOND, PASCAL
from kups.core.data import Index, Table
from kups.core.typing import (
    ExclusionId,
    GroupId,
    InclusionId,
    Label,
    MotifId,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass, field, key_chain, tree_zeros_like
from kups.core.utils.quaternion import Quaternion
from kups.core.utils.rigid_body import (
    initial_inertia_for_dynamics,
    inertia_tensor_diag,
    is_linear_motif,
    reconstruct_atom_positions,
)
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
        degrees_of_freedom: Effective DOF used by kinetic-energy thermostats
            (point particles: ``3N − 3``; rigid molecules:
            ``6 N_nonlinear + 5 N_linear − 3``), shape ``(n_systems,)``.
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
    degrees_of_freedom: Array

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


class _BaseMdParameters(BaseModel):
    """Fields common to every MD integrator."""

    temperature: float
    """Target temperature (K)."""
    time_step: float
    """Integration timestep (fs)."""
    initialize_momenta: bool = False
    """If True, initialise momenta from Maxwell-Boltzmann distribution."""


class VerletParameters(_BaseMdParameters):
    """NVE Verlet parameters."""

    integrator: Literal["verlet"] = "verlet"


class BaoabLangevinParameters(_BaseMdParameters):
    """NVT Langevin (BAOAB) parameters."""

    integrator: Literal["baoab_langevin"] = "baoab_langevin"
    friction_coefficient: float
    """Langevin friction coefficient (1/fs)."""


class CsvrParameters(_BaseMdParameters):
    """NVT CSVR (Bussi-Donadio-Parrinello) parameters."""

    integrator: Literal["csvr"] = "csvr"
    thermostat_time_constant: float
    """CSVR thermostat coupling time (fs)."""


class CsvrNptParameters(_BaseMdParameters):
    """NPT CSVR + stochastic cell rescaling (Bernetti-Bussi 2020) parameters."""

    integrator: Literal["csvr_npt"] = "csvr_npt"
    thermostat_time_constant: float
    """CSVR thermostat coupling time (fs)."""
    target_pressure: float
    """Target pressure for the barostat (Pa)."""
    pressure_coupling_time: float
    """Barostat coupling time (fs)."""
    compressibility: float
    """Isothermal compressibility (1/Pa)."""
    minimum_scale_factor: float
    """Minimum allowed box scaling factor per barostat step (dimensionless)."""


type MdParameters = Annotated[
    VerletParameters | BaoabLangevinParameters | CsvrParameters | CsvrNptParameters,
    Field(discriminator="integrator"),
]
"""Discriminated union of MD parameter records, keyed on ``integrator``.

Pydantic picks the variant matching the ``integrator`` field of the input.
Each variant carries only the fields its integrator actually reads."""


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
        _md_systems_from_params(config, unitcell, dof=max(3 * n_atoms - 3, 0)),
        label=SystemId,
    )

    return particles, systems


def _md_systems_from_params(
    config: _BaseMdParameters,
    unitcell: UnitCell,
    *,
    dof: int,
) -> MDSystems:
    """Build per-system MD parameters from a discriminated parameter variant.

    Fields that the chosen integrator does not read (e.g. ``friction_coefficient``
    on Verlet, NPT fields on CSVR) are populated with placeholder values that
    the integrator never touches.
    """
    return MDSystems(
        unitcell=unitcell,
        temperature=jnp.array([config.temperature]),
        time_step=jnp.array([config.time_step * FEMTO_SECOND]),
        friction_coefficient=jnp.array(
            [getattr(config, "friction_coefficient", 0.0) / FEMTO_SECOND]
        ),
        thermostat_time_constant=jnp.array(
            [getattr(config, "thermostat_time_constant", 1.0) * FEMTO_SECOND]
        ),
        target_pressure=jnp.array(
            [getattr(config, "target_pressure", 0.0) * PASCAL]
        ),
        pressure_coupling_time=jnp.array(
            [getattr(config, "pressure_coupling_time", 1.0) * FEMTO_SECOND]
        ),
        compressibility=jnp.array(
            [getattr(config, "compressibility", 0.0) / PASCAL]
        ),
        minimum_scale_factor=jnp.array(
            [getattr(config, "minimum_scale_factor", 1.0)]
        ),
        unitcell_gradients=tree_zeros_like(unitcell),
        potential_energy=jnp.array([0.0]),
        degrees_of_freedom=jnp.array([dof], dtype=jnp.int32),
    )


@dataclass
class MDRigidParticles(Particles):
    """Atom-level state for rigid-body MD.

    Mirrors :class:`kups.application.mcmc.data.MCMCParticles` but without the
    buffered-MCMC bookkeeping. Atoms in a rigid molecule do not carry their
    own momentum: per-group COM momentum and angular momentum live on the
    :class:`MDRigidGroup` table. Atom positions are derived each step by
    :class:`AtomReconstructionStep` from the COM and orientation.

    Attributes:
        group: Which rigid-body group each atom belongs to.
        motif: Which motif site each atom corresponds to (offsets into
            :class:`MotifParticles`).
        position_gradients: Energy gradient w.r.t. atom positions, shape
            ``(n_atoms, 3)``. Written by the potential evaluator and
            consumed by :class:`ForceAggregationStep`.
    """

    group: Index[GroupId]
    motif: Index[MotifParticleId]
    position_gradients: Array

    @property
    def forces(self) -> Array:
        """Per-atom forces, shape ``(n_atoms, 3)``."""
        return -self.position_gradients

    @property
    def exclusion(self) -> Index[ExclusionId]:
        """Exclusion index derived from the group index (intra-group pairs are excluded)."""
        return Index(
            tuple(ExclusionId(int(k)) for k in self.group.keys),
            self.group.indices,
            self.group.max_count,
        )

    @property
    def inclusion(self) -> Index[InclusionId]:
        """Inclusion index derived from the system index."""
        return Index(
            tuple(InclusionId(int(k)) for k in self.system.keys),
            self.system.indices,
            self.system.max_count,
        )


@dataclass
class MDRigidGroup:
    """Per-rigid-body state for rigid-body MD.

    Translational fields are named to satisfy ``HasPositions / HasMomenta /
    HasMasses / HasForces`` so that the existing :class:`MomentumStep`,
    :class:`PositionStep`, :class:`StochasticStep`, and
    :class:`StochasticCellRescalingStep` propagators can act on this table
    through a lens with zero new code.

    Attributes:
        system: Which simulation system each group belongs to.
        motif: Which motif species (rigid template) each group is.
        positions: Centre-of-mass position, shape ``(n_groups, 3)``.
        momenta: Centre-of-mass momentum (lab frame), shape ``(n_groups, 3)``.
        masses: Total group mass, shape ``(n_groups,)``.
        position_gradients: Energy gradient w.r.t. the COM, shape ``(n_groups, 3)``.
        quaternion: Orientation quaternion (body→lab), batched ``(n_groups,)``.
        angular_momentum: Lab-frame angular momentum, shape ``(n_groups, 3)``.
        inertia_diag: Body-frame principal moments, shape ``(n_groups, 3)``.
            Linear-motif symmetry axes carry ``inf`` to freeze the missing DOF.
        torque: Lab-frame torque, shape ``(n_groups, 3)``.
    """

    system: Index[SystemId]
    motif: Index[MotifId]
    positions: Array
    momenta: Array
    masses: Array
    position_gradients: Array
    quaternion: Quaternion
    angular_momentum: Array
    inertia_diag: Array
    torque: Array

    @property
    def forces(self) -> Array:
        """Net force on the COM (negative gradient), shape ``(n_groups, 3)``."""
        return -self.position_gradients

    @property
    def velocities(self) -> Array:
        """COM velocity, shape ``(n_groups, 3)``."""
        return self.momenta / self.masses[..., None]


class RigidVerletParameters(_BaseMdParameters):
    """Rigid-body NVE Verlet parameters."""

    integrator: Literal["rigid_verlet"] = "rigid_verlet"


class RigidBaoabLangevinParameters(_BaseMdParameters):
    """Rigid-body NVT Langevin (BAOAB) parameters."""

    integrator: Literal["rigid_baoab_langevin"] = "rigid_baoab_langevin"
    friction_coefficient: float
    """Langevin friction coefficient (1/fs)."""


class RigidCsvrParameters(_BaseMdParameters):
    """Rigid-body NVT CSVR parameters."""

    integrator: Literal["rigid_csvr"] = "rigid_csvr"
    thermostat_time_constant: float
    """CSVR thermostat coupling time (fs)."""


class RigidCsvrNptParameters(_BaseMdParameters):
    """Rigid-body NPT CSVR + stochastic cell rescaling parameters."""

    integrator: Literal["rigid_csvr_npt"] = "rigid_csvr_npt"
    thermostat_time_constant: float
    """CSVR thermostat coupling time (fs)."""
    target_pressure: float
    """Target pressure for the barostat (Pa)."""
    pressure_coupling_time: float
    """Barostat coupling time (fs)."""
    compressibility: float
    """Isothermal compressibility (1/Pa)."""
    minimum_scale_factor: float
    """Minimum allowed box scaling factor per barostat step (dimensionless)."""


type RigidMdParameters = Annotated[
    RigidVerletParameters
    | RigidBaoabLangevinParameters
    | RigidCsvrParameters
    | RigidCsvrNptParameters,
    Field(discriminator="integrator"),
]
"""Discriminated union of rigid-MD parameter records, keyed on ``integrator``."""


def _canonicalise_motif(
    adsorbate: AdsorbateConfig,
) -> tuple[Array, Array, Array, Array, bool]:
    """Re-centre a motif to its COM and rotate to its principal-axis frame.

    Returns:
        Tuple ``(canonical_positions, masses, charges, inertia_for_dynamics,
        is_linear)``. ``inertia_for_dynamics`` carries ``inf`` on the symmetry
        axis of linear motifs to freeze the missing rotational DOF.
    """
    positions = jnp.asarray(adsorbate.positions, dtype=float)
    masses = jnp.asarray(adsorbate.masses, dtype=float)
    charges = jnp.asarray(adsorbate.charges, dtype=float)
    total_mass = float(masses.sum())
    com = (masses[:, None] * positions).sum(axis=0) / total_mass
    centred = positions - com
    inertia_diag, eigvecs = inertia_tensor_diag(centred, masses)
    # Rotate so principal axes align with x, y, z.
    canonical = centred @ eigvecs
    linear = is_linear_motif(inertia_diag)
    inertia_for_dynamics = initial_inertia_for_dynamics(inertia_diag)
    return canonical, masses, charges, inertia_for_dynamics, linear


def _grid_positions(
    n_molecules: int, box_size: tuple[float, float, float]
) -> np.ndarray:
    """Lay out ``n_molecules`` COM positions on a regular grid inside a box.

    The grid side count is ``ceil(N^{1/3})``; only the first ``N`` positions
    are returned. Coordinates are centred on the box (``[-L/2, L/2]``).
    """
    n_per_side = int(math.ceil(n_molecules ** (1.0 / 3.0)))
    spacing = np.array(box_size) / n_per_side
    indices = np.indices((n_per_side, n_per_side, n_per_side)).reshape(3, -1).T
    coords = (indices + 0.5) * spacing - 0.5 * np.array(box_size)
    return coords[:n_molecules]


def build_rigid_state_from_grid(
    key: Array,
    adsorbates: tuple[AdsorbateConfig, ...],
    n_molecules: tuple[int, ...],
    box_size: tuple[float, float, float],
    config: RigidMdParameters,
) -> tuple[
    Table[ParticleId, MDRigidParticles],
    Table[GroupId, MDRigidGroup],
    Table[MotifParticleId, MotifParticles],
    Table[SystemId, MDSystems],
]:
    """Initialise a **single-system** rigid-body MD state on a regular grid.

    Each adsorbate species is canonicalised (COM at origin, principal axes
    aligned with x, y, z). Molecules are placed on a regular grid inside a
    box of size ``box_size`` with random orientations, and momenta are
    drawn from Maxwell-Boltzmann at ``config.temperature``. Per-system COM
    momentum is removed; angular momentum is *intentionally not* projected
    out (projecting out angular momentum is incorrect for a rigid PBC system).

    Multi-system batched setups are constructed by calling this helper once
    per system and stitching the results with :meth:`Table.union`, matching
    the pattern in :func:`kups.application.mcmc.data.mcmc_state_from_config`.

    Args:
        key: PRNG key for orientation and momentum sampling.
        adsorbates: Per-species configurations (TIP4P/2005 water etc.).
        n_molecules: Number of molecules per species (same length as ``adsorbates``).
        box_size: Cubic box edges (Å).
        config: Numerical / thermodynamic parameters.

    Returns:
        ``(particles, groups, motifs, systems)`` tables for one system.
    """
    chain = key_chain(key)
    assert len(adsorbates) == len(n_molecules)

    # 1) Canonicalise motif templates and assemble the motif table.
    motif_positions: list[Array] = []
    motif_masses: list[Array] = []
    motif_charges: list[Array] = []
    motif_atomic_numbers: list[Array] = []
    motif_labels: list[str] = []
    motif_species_idx: list[int] = []
    species_total_mass: list[float] = []
    species_inertia_diag: list[Array] = []
    species_is_linear: list[bool] = []

    for sp_idx, ads in enumerate(adsorbates):
        canonical, masses, charges, inertia_diag, linear = _canonicalise_motif(ads)
        motif_positions.append(canonical)
        motif_masses.append(masses)
        motif_charges.append(charges)
        motif_atomic_numbers.append(jnp.asarray(ads.atomic_numbers, dtype=int))
        motif_labels.extend(ads.symbols)
        motif_species_idx.extend([sp_idx] * len(ads.symbols))
        species_total_mass.append(float(masses.sum()))
        species_inertia_diag.append(inertia_diag)
        species_is_linear.append(linear)

    motif_positions_arr = jnp.concatenate(motif_positions, axis=0)
    motif_masses_arr = jnp.concatenate(motif_masses, axis=0)
    motif_charges_arr = jnp.concatenate(motif_charges, axis=0)
    motif_atomic_numbers_arr = jnp.concatenate(motif_atomic_numbers, axis=0)
    motif_species_idx_arr = jnp.asarray(motif_species_idx, dtype=int)
    n_species = len(adsorbates)

    motifs = Table.arange(
        MotifParticles(
            positions=motif_positions_arr,
            masses=motif_masses_arr,
            atomic_numbers=motif_atomic_numbers_arr,
            charges=motif_charges_arr,
            labels=Index.new(list(map(Label, motif_labels))),
            motif=Index.integer(
                motif_species_idx_arr, n=n_species, label=MotifId,
            ),
        ),
        label=MotifParticleId,
    )

    # 2) Build per-group state on a regular grid.
    total_groups = int(sum(n_molecules))
    grid = _grid_positions(total_groups, box_size)
    rng_key = next(chain)
    quaternions = Quaternion.random(rng_key, shape=(total_groups,))

    group_species: list[int] = []
    for sp_idx, n in enumerate(n_molecules):
        group_species.extend([sp_idx] * n)
    group_species_idx = jnp.asarray(group_species, dtype=int)

    com_positions = jnp.asarray(grid, dtype=float)
    total_masses = jnp.asarray(
        [species_total_mass[s] for s in group_species], dtype=float
    )
    inertia_diag = jnp.asarray(
        np.stack([np.asarray(species_inertia_diag[s]) for s in group_species]),
        dtype=float,
    )

    # 3) Sample momenta and angular momenta (Maxwell-Boltzmann).
    kT = config.temperature * BOLTZMANN_CONSTANT
    if config.initialize_momenta:
        sigma_p = jnp.sqrt(total_masses * kT)
        com_momenta = jax.random.normal(next(chain), (total_groups, 3)) * sigma_p[:, None]
        # Remove per-system COM drift (single system here).
        com_momenta = com_momenta - com_momenta.mean(axis=0, keepdims=True)

        # L_body_a ~ N(0, sqrt(I_a kT)); inf I → zero variance (frozen axis).
        inertia_for_sample = jnp.where(
            jnp.isfinite(inertia_diag), inertia_diag, 0.0
        )
        sigma_l = jnp.sqrt(inertia_for_sample * kT)
        l_body = jax.random.normal(next(chain), (total_groups, 3)) * sigma_l
        # Lab-frame: L_lab = q ⊗ L_body
        angular_momentum = l_body @ quaternions
    else:
        com_momenta = jnp.zeros((total_groups, 3))
        angular_momentum = jnp.zeros((total_groups, 3))

    # 4) Reconstruct atom positions from group state.
    atoms_per_species = [len(adsorbates[s].symbols) for s in range(n_species)]
    atoms_per_group_list: list[int] = []
    for sp_idx, n in enumerate(n_molecules):
        atoms_per_group_list.extend([atoms_per_species[sp_idx]] * n)
    n_atoms_total = sum(atoms_per_group_list)

    # Atom-level group / motif-particle indices.
    atom_group_idx = np.repeat(np.arange(total_groups), atoms_per_group_list)
    motif_offset_per_species = np.cumsum([0] + atoms_per_species)
    atom_motif_idx = np.concatenate(
        [
            motif_offset_per_species[s] + np.arange(atoms_per_species[s])
            for s in group_species
        ]
    )

    atom_positions = reconstruct_atom_positions(
        com_positions=com_positions,
        quaternion=quaternions,
        motif_positions=motif_positions_arr,
        group_idx=jnp.asarray(atom_group_idx),
        motif_idx=jnp.asarray(atom_motif_idx),
    )

    # Pull per-atom physical properties from the motif.
    atom_masses = motif_masses_arr[atom_motif_idx]
    atom_charges = motif_charges_arr[atom_motif_idx]
    atom_atomic_numbers = motif_atomic_numbers_arr[atom_motif_idx]
    atom_label_strings = [motif_labels[i] for i in atom_motif_idx.tolist()]

    particles = Table.arange(
        MDRigidParticles(
            positions=atom_positions,
            masses=atom_masses,
            atomic_numbers=atom_atomic_numbers,
            charges=atom_charges,
            labels=Index.new(list(map(Label, atom_label_strings))),
            system=Index.zeros(n_atoms_total, label=SystemId),
            group=Index.integer(
                jnp.asarray(atom_group_idx),
                n=total_groups,
                label=GroupId,
                max_count=max(atoms_per_group_list),
            ),
            motif=Index.integer(
                jnp.asarray(atom_motif_idx),
                n=int(motif_offset_per_species[-1]),
                label=MotifParticleId,
            ),
            position_gradients=jnp.zeros((n_atoms_total, 3)),
        ),
        label=ParticleId,
    )

    groups = Table.arange(
        MDRigidGroup(
            system=Index.zeros(total_groups, label=SystemId),
            motif=Index.integer(
                group_species_idx, n=n_species, label=MotifId,
            ),
            positions=com_positions,
            momenta=com_momenta,
            masses=total_masses,
            position_gradients=jnp.zeros((total_groups, 3)),
            quaternion=quaternions,
            angular_momentum=angular_momentum,
            inertia_diag=inertia_diag,
            torque=jnp.zeros((total_groups, 3)),
        ),
        label=GroupId,
    )

    # 5) Build the per-system table with the right DOF count.
    n_nonlinear = sum(n for n, sp in zip(n_molecules, species_is_linear) if not sp)
    n_linear = sum(n for n, sp in zip(n_molecules, species_is_linear) if sp)
    dof = max(6 * n_nonlinear + 5 * n_linear - 3, 0)

    box_matrix = jnp.diag(jnp.asarray(box_size, dtype=float))
    unitcell = TriclinicUnitCell.from_matrix(box_matrix[None])

    systems = Table.arange(
        _md_systems_from_params(config, unitcell, dof=dof),
        label=SystemId,
    )

    return particles, groups, motifs, systems


__all__ = [
    "MDParticles",
    "MDSystems",
    "MdRunConfig",
    "MdParameters",
    "md_state_from_ase",
    "MDRigidParticles",
    "MDRigidGroup",
    "RigidMdParameters",
    "build_rigid_state_from_grid",
]
