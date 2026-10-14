# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and ASE initialisation for structure relaxation."""

from __future__ import annotations

from pathlib import Path

import ase
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel

from kups.application.utils.particles import (
    Particles,
    default_exclusion,
    particles_from_ase,
)
from kups.core.cell import AnyPeriodicity, Cell, DeformedFrame, MatrixLogFrame
from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import bind
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.typing import ExclusionId, IsState, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, tree_zeros_like
from kups.potential.common.geometry import PositionsAndCell, PositionsAndCellIndex
from kups.relaxation.config import TransformationConfig


@dataclass
class RelaxParticles(Particles):
    """Particle data for structure relaxation.

    Extends ``Particles`` with energy gradients and derived properties
    (forces, inclusion/exclusion indices) needed by relaxation propagators.

    Attributes:
        position_gradients: Optimizer position-DOF gradient ``∂E/∂u_pos`` (the
            relaxation filter's output), shape ``(n_atoms, 3)``: ``∂E/∂q`` under
            ``cell_filter`` (reference-cartesian) or ``∂E/∂r`` under
            ``positions_only``. The force source and ASE-fmax convergence quantity.
    """

    position_gradients: Array
    exclusion: Index[ExclusionId] = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.exclusion is None:
            object.__setattr__(self, "exclusion", default_exclusion(len(self.charges)))

    @property
    def forces(self) -> Array:
        """Atomic forces, the negative position gradient."""
        return -self.position_gradients


@dataclass
class RelaxSystems:
    """System-level data for structure relaxation."""

    cell: Cell[AnyPeriodicity]
    """Cell geometry, one row per system."""
    cell_gradients: Cell[AnyPeriodicity]
    """Optimizer cell-DOF gradient ``∂E/∂u_cell`` (the relaxation filter's output),
    stored on :attr:`cell`'s frame (the lower-triangular log-deformation entries
    under ``cell_filter``). The ASE-fmax convergence quantity for the cell; the
    atoms-ride-the-cell coupling is already folded in by the filter pullback."""
    potential_energy: Array
    """Potential energy per system, shape ``(n_systems,)``."""


@dataclass
class RelaxState[OptState]:
    """Force-field-agnostic relaxation state.

    The potential is built with its parameters at construction time (via the
    adapters' ``parameters=``), so no force-field field lives on the state.
    """

    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: OptState
    step: Array


type HasRelaxData = IsState[RelaxParticles, RelaxSystems]


def relax_parameters(
    particles: Table[ParticleId, RelaxParticles],
    systems: Table[SystemId, RelaxSystems],
) -> PositionsAndCell:
    """Raw ``(positions, cell)``, shaped like the optimizer DOFs."""
    return PositionsAndCell(
        particles.map_data(lambda p: p.positions), systems.map_data(lambda s: s.cell)
    )


def relax_index_prefix(
    particles: Table[ParticleId, RelaxParticles],
    systems: Table[SystemId, RelaxSystems],
) -> PositionsAndCellIndex:
    """Index prefix mapping every DOF to its system (per-particle and per-cell)."""
    return PositionsAndCellIndex(particles.data.system, systems.index)


def relax_gradients(state: HasRelaxData) -> PositionsAndCell:
    """The cached DOF gradients ``∂E/∂u`` held in a relaxation state."""
    return PositionsAndCell(
        state.particles.map_data(lambda p: p.position_gradients),
        state.systems.map_data(lambda s: s.cell_gradients),
    )


class RelaxRunConfig(BaseModel):
    """Configuration for a relaxation run."""

    out_file: str | Path
    """Path to the HDF5 output file."""
    max_steps: int
    """Maximum number of optimisation steps."""
    seed: int | None
    """Random seed. None for time-based."""
    force_tolerance: float
    """Convergence threshold for max atomic force (eV/Å)."""
    optimizer: TransformationConfig
    """List of Optax transform specifications passed to `make_optimizer`."""
    optimize_cell: bool
    """Whether to also relax lattice vectors."""


def relax_systems(cell: Cell[AnyPeriodicity], n_atoms: Array) -> RelaxSystems:
    """System rows with zeroed caches and cells in the log-deformation frame.

    ``cell_factor = n_atoms`` (ASE's ``exp_cell_factor``) balances the extensive
    cell-virial gradient against the per-atom forces in the joint optimiser.
    Empty systems get factor 1, since a zero factor makes the cell NaN.

    Args:
        cell: Cells with a plain frame, one per system; an unbatched cell gains
            a leading axis.
        n_atoms: Atom count per system, shape ``(n_systems,)``.
    """
    if cell.vectors.ndim == 2:
        cell = cell[None]
    cell_factor = jnp.maximum(n_atoms, 1)
    cell = bind(cell, lambda x: x.frame).apply(
        lambda f: DeformedFrame.from_frame(
            f, cell_factor=cell_factor, deformation=MatrixLogFrame
        )
    )
    return RelaxSystems(
        cell=cell,
        cell_gradients=tree_zeros_like(cell),
        potential_energy=jnp.zeros(cell.vectors.shape[0], cell.vectors.dtype),
    )


def relax_particles(
    particles: Particles,
    *,
    position_gradients: Array | None = None,
    exclusion: Index[ExclusionId] | None = None,
) -> RelaxParticles:
    """Relaxation particle rows from plain particles, gradients zeroed by default."""
    return RelaxParticles(
        positions=particles.positions,
        masses=particles.masses,
        atomic_numbers=particles.atomic_numbers,
        charges=particles.charges,
        labels=particles.labels,
        system=particles.system,
        position_gradients=(
            jnp.zeros_like(particles.positions)
            if position_gradients is None
            else position_gradients
        ),
        exclusion=(
            default_exclusion(len(particles.positions))
            if exclusion is None
            else exclusion
        ),
    )


def relax_state_from_particles_and_cell(
    particles: Table[ParticleId, Particles], cell: Cell[AnyPeriodicity]
) -> tuple[Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]]:
    """Build relaxation particle and system tables from plain particles and a cell.

    Args:
        particles: Particles of one or more systems. Keys are preserved;
            out-of-bounds system references denote padding.
        cell: Their cells in system-key order, unbatched for a single system.

    Returns:
        Tuple of ``(particles, systems)`` ready for relaxation propagators.

    Raises:
        ValueError: If the system index does not have one entry per particle,
            or ``cell`` does not hold one cell per system.
    """
    system = particles.data.system
    if system.indices.shape != (len(particles),):
        raise ValueError("Expected one system reference per particle.")
    expected_shape = (system.num_labels, 3, 3)
    if cell.vectors.shape != expected_shape and not (
        system.num_labels == 1 and cell.vectors.shape == (3, 3)
    ):
        raise ValueError(
            f"Expected cell vectors with shape {expected_shape}, "
            f"got {cell.vectors.shape}."
        )
    n_atoms = system.counts.data.astype(particles.data.positions.dtype)
    return (
        particles.map_data(relax_particles),
        Table(system.keys, relax_systems(cell, n_atoms), _cls=SystemId),
    )


def relax_state_from_ase(
    atoms: ase.Atoms | str | Path,
) -> tuple[Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]]:
    """Build relaxation particle and system data from an ASE Atoms object or file.

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.

    Returns:
        Tuple of ``(particles, systems)`` ready for relaxation propagators.
    """
    p, cell, _ = particles_from_ase(atoms)
    return relax_state_from_particles_and_cell(p, cell)
