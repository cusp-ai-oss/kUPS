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
from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.typing import ExclusionId, ParticleId, SystemId
from kups.core.unitcell import UnitCell
from kups.core.utils.jax import dataclass, field, tree_zeros_like
from kups.relaxation.optax.optimizer import TransformationConfig


@dataclass
class RelaxParticles(Particles):
    """Particle data for structure relaxation.

    Extends ``Particles`` with energy gradients and derived properties
    (forces, inclusion/exclusion indices) needed by relaxation propagators.

    Attributes:
        position_gradients: Energy gradient w.r.t. positions, shape ``(n_atoms, 3)``.
    """

    position_gradients: Array
    exclusion: Index[ExclusionId] = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self):
        if self.exclusion is None:
            object.__setattr__(self, "exclusion", default_exclusion(len(self.charges)))

    @property
    def forces(self) -> Array:
        """Atomic forces, the negative position gradient."""
        return -self.position_gradients


@dataclass
class RelaxSystems:
    """System-level data for structure relaxation."""

    unitcell: UnitCell
    """Unit cell geometry, batched with shape (1,)."""
    unitcell_gradients: UnitCell
    potential_energy: Array
    """Potential energy per system, shape (1,)."""

    @property
    def stress_tensor(self) -> Array:
        """Cauchy stress tensor, shape ``(..., 3, 3)``."""
        return (
            -self.unitcell_gradients.lattice_vectors
            / self.unitcell.volume[..., None, None]
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


class RelaxParameters(BaseModel):
    """Optimiser configuration for relaxation."""

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
    p, unitcell, _ = particles_from_ase(atoms)
    particles = p.set_data(
        RelaxParticles(
            positions=p.data.positions,
            masses=p.data.masses,
            atomic_numbers=p.data.atomic_numbers,
            charges=p.data.charges,
            labels=p.data.labels,
            system=p.data.system,
            position_gradients=jnp.zeros_like(p.data.positions),
        ),
    )
    unitcell = unitcell[None]
    systems = Table.arange(
        RelaxSystems(
            unitcell=unitcell,
            unitcell_gradients=tree_zeros_like(unitcell),
            potential_energy=jnp.array([0.0]),
        ),
        label=SystemId,
    )
    return particles, systems
