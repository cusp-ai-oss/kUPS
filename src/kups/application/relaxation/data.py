# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and ASE initialisation for structure relaxation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import ase
import optax
from jax import Array

from kups.application.utils.data import Scalar
from kups.application.utils.particles import (
    Particles,
    default_exclusion,
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
from kups.core.lens import bind
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.typing import ExclusionId, ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.units.pydantic import UnitsModel
from kups.core.utils.jax import dataclass, field
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

    position_gradients: units.Force
    exclusion: Index[ExclusionId] = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.exclusion is None:
            object.__setattr__(self, "exclusion", default_exclusion(len(self.charges)))

    @property
    def forces(self) -> units.Force:
        """Atomic forces, the negative position gradient."""
        return -self.position_gradients


@dataclass
class RelaxSystems:
    """System-level data for structure relaxation."""

    cell: Cell[AnyPeriodicity]
    """Cell geometry, batched with shape (1,)."""
    cell_gradients: Cell[AnyPeriodicity]
    """Optimizer cell-DOF gradient ``∂E/∂u_cell`` (the relaxation filter's output),
    stored on :attr:`cell`'s frame (the lower-triangular log-deformation entries
    under ``cell_filter``). The ASE-fmax convergence quantity for the cell; the
    atoms-ride-the-cell coupling is already folded in by the filter pullback."""
    potential_energy: units.Energy
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


class RelaxRunConfig(UnitsModel):
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
    """List of Optax transform specifications passed to `make_optimizer`."""
    optimize_cell: bool
    """Whether to also relax lattice vectors."""


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
    particles = p.set_data(
        RelaxParticles(
            positions=p.data.positions,
            masses=p.data.masses,
            atomic_numbers=p.data.atomic_numbers,
            charges=p.data.charges,
            labels=p.data.labels,
            system=p.data.system,
            position_gradients=units.zeros(
                p.data.positions.shape,
                u.eV / u.angstrom,
                dtype=p.data.positions.dtype,
            ),
        ),
    )
    # cell_factor = per-system atom count (ASE's exp_cell_factor) balances the
    # extensive cell-virial gradient against the per-atom forces in the joint
    # optimiser. bincount over the system index gives one count per system.
    n_systems = p.data.system.num_labels
    cell_factor = p.data.system.counts.data.astype(p.data.positions.dtype)
    cell = bind(cell[None], lambda x: x.frame).apply(
        lambda f: DeformedFrame.from_frame(
            f, cell_factor=cell_factor, deformation=MatrixLogFrame
        )
    )
    systems = Table.arange(
        RelaxSystems(
            cell=cell,
            cell_gradients=zero_gradient_cell(cell),
            potential_energy=units.zeros(n_systems, u.eV),
        ),
        label=SystemId,
    )
    return particles, systems
