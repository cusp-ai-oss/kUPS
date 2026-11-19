# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and ASE initialisation for structure relaxation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import ase
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel

from kups.application.utils.data import Scalar
from kups.application.utils.particles import (
    Particles,
    default_exclusion,
    fixed_from_ase,
    particles_from_ase,
)
from kups.core import units
from kups.core.cell import (
    AnyPeriodicity,
    Cell,
    DeformedFrame,
    MatrixLogFrame,
    zero_gradient_cell,
)
from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.geometry import (
    PositionsAndCell,
    PositionsAndCellIndex,
    PositionsAndCellTree,
)
from kups.core.lens import bind
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.typing import ExclusionId, IsState, ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.utils.jax import dataclass, field
from kups.core.utils.ops import where_broadcast_last
from kups.relaxation.config import TransformationConfig


@dataclass
class RelaxParticles(Particles):
    """Particle data for structure relaxation.

    Extends ``Particles`` with energy gradients and derived properties
    (forces, inclusion/exclusion indices) needed by relaxation propagators.

    Attributes:
        position_gradients: Optimizer position-DOF gradient ``∂E/∂q`` (the
            ``FRECHET_FILTER`` output, reference-cartesian), shape
            ``(n_atoms, 3)``; equal to ``∂E/∂r`` while the cell is at its
            reference, so always for a fixed cell. The force source, kept
            unconstrained: fixed particles report their real force, which
            :func:`relax_gradients` zeroes for convergence.
        fixed: Whether the particle is held in place, shape ``(n_atoms,)``.
        exclusion: Exclusion group of each particle; defaults to one group per
            particle.
    """

    position_gradients: units.Force
    fixed: Array
    # pyrefly: ignore[bad-assignment]
    exclusion: Index[ExclusionId] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.exclusion is None:
            object.__setattr__(self, "exclusion", default_exclusion(len(self.charges)))

    @property
    def forces(self) -> units.Force:
        """Atomic forces, the negative position gradient."""
        return -self.position_gradients


@dataclass
class RelaxSystems:
    """System-level data for structure relaxation.

    Attributes:
        cell: Cell geometry, one row per system.
        cell_gradients: Total cell gradient, with atoms riding the cell at fixed
            fractional coordinates (the ``FRECHET_FILTER`` output), stored on
            :attr:`cell`'s frame (the matrix-log deformation). Kept
            unconstrained, so a fixed cell still has its stress, which
            :func:`relax_gradients` zeroes for convergence.
        potential_energy: Potential energy per system, shape ``(n_systems,)``.
        fixed_cell: Whether the cell is held fixed, shape ``(n_systems,)``.
    """

    cell: Cell[AnyPeriodicity]
    cell_gradients: Cell[AnyPeriodicity]
    potential_energy: units.Energy
    fixed_cell: Array


@dataclass
class RelaxState[OptState]:
    """Force-field-agnostic relaxation state.

    The potential is built with its parameters at construction time (via the
    adapters' ``parameters=``), so no force-field field lives on the state.

    Attributes:
        particles: Per-particle positions and position gradients.
        systems: Per-system cell, cell gradients, and energy.
        neighborlist_params: Capacity hints for the neighbor list.
        opt_state: Optimizer state.
        step: Optimizer step counter, shape ``(1,)``.
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
) -> PositionsAndCell[units.Length]:
    """Raw ``(positions, cell)``, shaped like the optimizer DOFs."""
    return PositionsAndCellTree(
        particles.map_data(lambda p: p.positions), systems.map_data(lambda s: s.cell)
    )


def relax_index_prefix(
    particles: Table[ParticleId, RelaxParticles],
    systems: Table[SystemId, RelaxSystems],
) -> PositionsAndCellIndex:
    """Index prefix mapping every DOF to its system (per-particle and per-cell)."""
    return PositionsAndCellIndex(particles.data.system, systems.index)


def constrain_gradients(
    state: HasRelaxData, gradients: PositionsAndCell[units.Force]
) -> PositionsAndCell[units.Force]:
    """``gradients`` with the rows of fixed particles and fixed cells zeroed.

    The one place constraints act: the optimizer, its line searches and
    convergence see these gradients, so a fixed particle or cell receives no
    update.

    Args:
        state: Relaxation state whose particles and systems say which rows are
            fixed.
        gradients: DOF gradients with one position row per particle and one
            cell row per system of ``state``.

    Returns:
        ``gradients`` with fixed rows set to zero.
    """
    particles, systems = state.particles.data, state.systems.data
    zero_cell = zero_gradient_cell(systems.cell)
    return PositionsAndCellTree(
        gradients.positions.map_data(
            lambda g: where_broadcast_last(particles.fixed, units.zeros_like(g), g)
        ),
        gradients.cell.map_data(
            lambda g: where_broadcast_last(systems.fixed_cell, zero_cell, g)
        ),
    )


def relax_gradients(state: HasRelaxData) -> PositionsAndCell[units.Force]:
    """The cached DOF gradients ``∂E/∂u``, constrained as the optimizer sees them."""
    cached = PositionsAndCellTree(
        state.particles.map_data(lambda p: p.position_gradients),
        state.systems.map_data(lambda s: s.cell_gradients),
    )
    return constrain_gradients(state, cached)


class RelaxRunConfig(BaseModel):
    """Configuration for a relaxation run."""

    out_file: str | Path
    """Path to the HDF5 output file."""
    max_steps: int
    """Maximum number of optimisation steps."""
    seed: int | None
    """Random seed. None for time-based."""
    force_tolerance: Annotated[units.Force, Scalar]
    """Convergence threshold for max atomic force."""
    optimizer: TransformationConfig
    """List of Optax transform specifications passed to `make_optimizer`.

    Their settings are bare numbers in the base units (Å, eV), e.g.
    ``max_step_size`` in Å, since the optimizer runs on magnitudes.
    """
    optimize_cell: bool
    """Whether to also relax lattice vectors."""


def relax_systems(
    cell: Cell[AnyPeriodicity], n_atoms: Array, *, fixed_cell: bool = True
) -> RelaxSystems:
    """System rows with zeroed caches and cells in the log-deformation frame.

    ``cell_factor = n_atoms × 1 Å`` is ASE's ``exp_cell_factor``: it balances the
    extensive cell virial against the per-atom forces in the joint optimiser,
    and the length makes the cell gradient a force, as ASE treats it. Empty
    systems get factor 1 Å, since a zero factor makes the cell NaN.

    Args:
        cell: Cells with a plain frame, one per system; an unbatched cell gains
            a leading axis.
        n_atoms: Atom count per system, shape ``(n_systems,)``.
        fixed_cell: Whether the cells are held fixed.
    """
    if cell.vectors.ndim == 2:
        cell = cell[None]
    cell_factor = jnp.maximum(n_atoms, 1) * u.angstrom
    cell = bind(cell, lambda x: x.frame).apply(
        lambda f: DeformedFrame.from_frame(
            f, cell_factor=cell_factor, deformation=MatrixLogFrame
        )
    )
    return RelaxSystems(
        cell=cell,
        cell_gradients=zero_gradient_cell(cell),
        potential_energy=units.zeros(
            cell.vectors.shape[0], u.eV, dtype=cell.vectors.dtype
        ),
        fixed_cell=jnp.full(cell.vectors.shape[0], fixed_cell),
    )


def relax_particles(
    particles: Particles,
    *,
    position_gradients: units.Force | None = None,
    fixed: Array | None = None,
    exclusion: Index[ExclusionId] | None = None,
) -> RelaxParticles:
    """Relaxation particle rows from plain particles.

    Gradients are zero and no particle is fixed unless given.
    """
    return RelaxParticles(
        positions=particles.positions,
        masses=particles.masses,
        atomic_numbers=particles.atomic_numbers,
        charges=particles.charges,
        labels=particles.labels,
        system=particles.system,
        position_gradients=(
            units.zeros(
                particles.positions.shape,
                u.eV / u.angstrom,
                dtype=particles.positions.dtype,
            )
            if position_gradients is None
            else position_gradients
        ),
        fixed=(
            jnp.zeros(particles.positions.shape[:-1], dtype=bool)
            if fixed is None
            else fixed
        ),
        exclusion=(
            default_exclusion(len(particles.positions))
            if exclusion is None
            else exclusion
        ),
    )


def relax_state_from_particles_and_cell(
    particles: Table[ParticleId, Particles],
    cell: Cell[AnyPeriodicity],
    *,
    fixed: Array | None = None,
    fixed_cell: bool = True,
) -> tuple[Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]]:
    """Build relaxation particle and system tables from plain particles and a cell.

    Args:
        particles: Particles of one or more systems. Keys are preserved;
            out-of-bounds system references denote padding.
        cell: Their cells in system-key order, unbatched for a single system.
        fixed: Whether each particle is held in place; none by default.
        fixed_cell: Whether the cells are held fixed.

    Returns:
        Tuple of ``(particles, systems)`` ready for relaxation propagators.

    Raises:
        ValueError: If the system index does not have one entry per particle,
            or ``cell`` does not hold one cell per system.
    """
    system = particles.data.system
    if system.shape != (len(particles),):
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
        particles.map_data(lambda p: relax_particles(p, fixed=fixed)),
        Table(
            system.keys,
            relax_systems(cell, n_atoms, fixed_cell=fixed_cell),
            _cls=SystemId,
        ),
    )


def relax_state_from_ase(
    atoms: ase.Atoms | str | Path, *, fixed_cell: bool = True
) -> tuple[Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]]:
    """Build relaxation particle and system data from an ASE Atoms object or file.

    Atoms held in place by ASE constraints are marked ``fixed``, see
    [fixed_from_ase][kups.application.utils.particles.fixed_from_ase].

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.
        fixed_cell: Whether the cell is held fixed.

    Returns:
        Tuple of ``(particles, systems)`` ready for relaxation propagators.

    Raises:
        ValueError: If ``atoms`` carries a constraint other than holding whole
            atoms in place.
    """
    p, cell, _ = particles_from_ase(atoms)
    return relax_state_from_particles_and_cell(
        p, cell, fixed=fixed_from_ase(atoms), fixed_cell=fixed_cell
    )
