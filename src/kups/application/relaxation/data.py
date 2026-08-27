# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and ASE initialisation for structure relaxation."""

from __future__ import annotations

from pathlib import Path

import ase
import jax.numpy as jnp
import optax
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
from kups.core.typing import ExclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, tree_zeros_like
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
    """Cell geometry, batched with shape (1,)."""
    cell_gradients: Cell[AnyPeriodicity]
    """Optimizer cell-DOF gradient ``∂E/∂u_cell`` (the relaxation filter's output),
    a matching-structure copy of :attr:`cell` (wrapper, periodicity, and frame
    leaves, including the ``MatrixLogFrame`` deformation). The ASE-fmax cell
    quantity; the atoms-ride-the-cell coupling is already folded in by the
    filter pullback."""
    potential_energy: Array
    """Potential energy per system, shape (1,)."""


@dataclass
class RelaxState:
    """Force-field-agnostic relaxation state.

    The potential is built with its parameters at construction time (via the
    adapters' ``parameters=``), so no force-field field lives on the state.
    """

    particles: Table[ParticleId, RelaxParticles]
    systems: Table[SystemId, RelaxSystems]
    neighborlist_params: UniversalNeighborlistParameters
    opt_state: optax.OptState
    step: Array


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


def relax_state_from_particles_and_cell(
    particles: Table[ParticleId, Particles],
    cell: Cell[AnyPeriodicity],
) -> tuple[Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]]:
    """Build one-system relaxation data from source-neutral particles and a cell.

    The supplied particles and cell must already describe the same system in the
    same kUPS coordinate frame. This builder does not transform geometry; it adds
    the system axis and an identity ``MatrixLogFrame`` deformation around the
    input frame, preserving the input's initial physical cell vectors.

    Args:
        particles: Non-empty particle table for exactly one complete system.
        cell: Unbatched cell with vectors of shape ``(3, 3)`` and an undeformed
            outer frame (not a :class:`DeformedFrame`).

    Returns:
        Relaxation particle and system tables preserving the input particle keys
        and referenced ``SystemId``.

    Raises:
        ValueError: If ``particles`` is empty, does not reference exactly one
            system with one reference per particle selecting that system, if
            ``cell.vectors`` is not shape ``(3, 3)``, or if ``cell.frame`` is a
            :class:`DeformedFrame`.
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
    if p.system.indices.shape != (n_particles,):
        raise ValueError(
            "particles must contain one system reference per particle; "
            f"got shape {p.system.indices.shape} for {n_particles} particles."
        )
    if bool(jnp.any(p.system.indices != 0)):
        raise ValueError(
            "particles system references must all select the sole SystemId."
        )
    if cell.vectors.shape != (3, 3):
        raise ValueError(
            f"cell.vectors must have shape (3, 3); got {cell.vectors.shape}."
        )
    if isinstance(cell.frame, DeformedFrame):
        raise ValueError(
            "cell.frame must be an undeformed frame, not a DeformedFrame; "
            "restart and rebasing semantics are out of scope."
        )

    relax_particles = particles.set_data(
        RelaxParticles(
            positions=p.positions,
            masses=p.masses,
            atomic_numbers=p.atomic_numbers,
            charges=p.charges,
            labels=p.labels,
            system=p.system,
            position_gradients=jnp.zeros_like(p.positions),
        ),
    )
    # cell_factor = per-system atom count (ASE's exp_cell_factor) balances the
    # extensive cell-virial gradient against the per-atom forces in the joint
    # optimiser. bincount over the system index gives one count per system.
    n_systems = p.system.num_labels
    cell_factor = jnp.bincount(p.system.indices, length=n_systems).astype(
        p.positions.dtype
    )
    deformed_cell = bind(cell[None], lambda x: x.frame).apply(
        lambda f: DeformedFrame.from_frame(
            f, cell_factor=cell_factor, deformation=MatrixLogFrame
        )
    )
    systems = Table(
        (p.system.keys[0],),
        RelaxSystems(
            cell=deformed_cell,
            cell_gradients=tree_zeros_like(deformed_cell),
            potential_energy=jnp.zeros(n_systems),
        ),
    )
    return relax_particles, systems


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
    particles, cell, _ = particles_from_ase(atoms)
    return relax_state_from_particles_and_cell(particles, cell)
