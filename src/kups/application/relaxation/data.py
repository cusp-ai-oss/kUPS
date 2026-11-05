# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures and ASE initialisation for structure relaxation."""

from __future__ import annotations

from pathlib import Path

import ase
import jax.numpy as jnp
import optax
from jax import Array
from jax.typing import ArrayLike
from pydantic import BaseModel

from kups.application.utils.particles import (
    Particles,
    _electronic_metadata,
    _prepare_ase,
    default_exclusion,
)
from kups.core.cell import AnyPeriodicity, Cell, DeformedFrame, MatrixLogFrame
from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import bind
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.typing import ExclusionId, ParticleId, SystemId
from kups.core.utils.jax import (
    dataclass,
    field,
    skip_post_init_if_disabled,
    tree_zeros_like,
)
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
    stored on :attr:`cell`'s frame (the matrix-log deformation under
    ``cell_filter``). The ASE-fmax convergence quantity for the cell; the
    atoms-ride-the-cell coupling is already folded in by the filter pullback."""
    potential_energy: Array
    """Potential energy per system, shape (1,)."""

    total_charge: Array = field(default=None, kw_only=True)  # type: ignore
    """Integer total charge per system, defaulting to zero."""
    spin_multiplicity: Array = field(default=None, kw_only=True)  # type: ignore
    """Integer spin multiplicity per system, defaulting to one."""

    @skip_post_init_if_disabled
    def __post_init__(self) -> None:
        if self.total_charge is None:
            object.__setattr__(
                self, "total_charge", jnp.zeros(self.cell.vectors.shape[:-2], dtype=int)
            )
        if self.spin_multiplicity is None:
            object.__setattr__(
                self,
                "spin_multiplicity",
                jnp.ones(self.cell.vectors.shape[:-2], dtype=int),
            )


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


def relax_state_from_ase(
    atoms: ase.Atoms | str | Path,
    *,
    total_charge: ArrayLike | None = None,
    spin_multiplicity: ArrayLike | None = None,
) -> tuple[Table[ParticleId, RelaxParticles], Table[SystemId, RelaxSystems]]:
    """Build relaxation particle and system data from an ASE Atoms object or file.

    Non-None metadata keywords override ``atoms.info["charge"]`` and
    ``atoms.info["spin"]`` independently. Missing info defaults to 0/1;
    present invalid values raise unless overridden. Partial charges are unchanged.

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.
        total_charge: Integer total charge, defaulting to zero.
        spin_multiplicity: Nonnegative integer multiplicity, defaulting to one.

    Returns:
        Tuple of ``(particles, systems)`` ready for relaxation propagators.

    Raises:
        TypeError: If selected electronic metadata are not integers.
        ValueError: If selected metadata have invalid shapes or values.
    """
    (p, cell, _), info = _prepare_ase(atoms)
    charge, spin = _electronic_metadata(total_charge, spin_multiplicity, info)
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
            cell_gradients=tree_zeros_like(cell),
            potential_energy=jnp.zeros(n_systems),
            total_charge=charge[None],
            spin_multiplicity=spin[None],
        ),
        label=SystemId,
    )
    return particles, systems
