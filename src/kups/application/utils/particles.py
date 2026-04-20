# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared particle data structures and ASE loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import ase
import ase.io
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.typing import ExclusionId, InclusionId, Label, ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell, to_lower_triangular
from kups.core.utils.jax import dataclass


@dataclass
class Particles:
    """Particle state shared across simulation types.

    Attributes:
        positions: Cartesian coordinates in the lower-triangular frame, shape (n_atoms, 3).
        masses: Atomic masses (amu), shape (n_atoms,).
        atomic_numbers: Atomic numbers, shape (n_atoms,).
        charges: Partial charges, shape (n_atoms,).
        labels: Per-atom string labels.
        system: Index mapping each particle to a system.
    """

    positions: Array
    masses: Array
    atomic_numbers: Array
    charges: Array
    labels: Index[Label]
    system: Index[SystemId]

    @property
    def inclusion(self) -> Index[InclusionId]:
        """System index re-labeled as InclusionId."""
        return Index(tuple(map(InclusionId, self.system.keys)), self.system.indices)


def default_exclusion(n: int) -> Index[ExclusionId]:
    """Build a default per-particle exclusion index (each atom excludes itself).

    Args:
        n: Number of particles.

    Returns:
        Index mapping each particle to a unique ExclusionId.
    """
    return Index.integer(jnp.arange(n), n=n, label=ExclusionId)


def particles_from_ase(
    atoms: ase.Atoms | str | Path,
) -> tuple[Table[ParticleId, Particles], UnitCell, Callable[[Array], Array]]:
    """Build particle data and unit cell from an ASE Atoms object or file path.

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.

    Returns:
        Tuple of (particles, unitcell, uc_transform) where uc_transform
        rotates Cartesian positions into the lower-triangular frame.
    """
    if isinstance(atoms, (str, Path)):
        atoms = next(ase.io.iread(atoms, index=-1, store_tags=True))
    L, uc_transform = to_lower_triangular(jnp.asarray(atoms.cell.array))
    unitcell = TriclinicUnitCell.from_matrix(L)
    # Rotate Cartesian positions into the lower-triangular frame.
    positions = uc_transform(jnp.asarray(atoms.positions))
    masses = jnp.asarray(atoms.get_masses())
    atomic_numbers = jnp.asarray(atoms.get_atomic_numbers())
    n_atoms = len(masses)
    charges = jnp.asarray(
        atoms.info.get(
            "_atom_type_partial_charge",
            atoms.info.get("_atom_site_charge", jnp.zeros((len(positions),))),
        )
    )
    labels = list(
        map(Label, atoms.info.get("_atom_site_label", atoms.get_chemical_symbols()))
    )
    particles = Table.arange(
        Particles(
            positions=positions,
            masses=masses,
            atomic_numbers=atomic_numbers,
            charges=charges,
            labels=Index.new(labels),
            system=Index.integer(jnp.zeros(n_atoms, dtype=int), label=SystemId),
        ),
        label=ParticleId,
    )
    return particles, unitcell, uc_transform
