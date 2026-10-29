# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared particle data structures and ASE loading utilities."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, Callable

import ase
import ase.io
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from kups.core import units
from kups.core.cell import AnyPeriodicity, Cell, TriclinicFrame, to_lower_triangular
from kups.core.data import Index, Table
from kups.core.typing import ExclusionId, InclusionId, Label, ParticleId, SystemId
from kups.core.units import E_A, E_K, E_KG, E_M, E_S, Measured
from kups.core.units import catalog as u
from kups.core.utils.jax import dataclass


@dataclass
class Particles:
    """Particle state shared across simulation types.

    Attributes:
        positions: Cartesian coordinates in the lower-triangular frame, shape (n_atoms, 3).
        masses: Atomic masses, shape (n_atoms,).
        atomic_numbers: Atomic numbers, shape (n_atoms,).
        charges: Partial charges, shape (n_atoms,).
        labels: Per-atom string labels.
        system: Index mapping each particle to a system.
    """

    positions: units.Length
    masses: units.Mass
    atomic_numbers: Array
    charges: units.Charge
    labels: Index[Label]
    system: Index[SystemId]

    @property
    def inclusion(self) -> Index[InclusionId]:
        """System index re-labeled as InclusionId."""
        return self.system.to_cls(InclusionId)


def default_exclusion(n: int) -> Index[ExclusionId]:
    """Build a default per-particle exclusion index (each atom excludes itself).

    Args:
        n: Number of particles.

    Returns:
        Index mapping each particle to a unique ExclusionId.
    """
    return Index.integer(jnp.arange(n), n=n, label=ExclusionId)


def particles_from_arrays(
    *,
    positions: units.Length,
    cell_vectors: units.Length,
    periodicity: (
        Sequence[bool | np.bool_ | Array] | np.ndarray[Any, np.dtype[np.bool_]] | Array
    ),
    masses: units.Mass,
    atomic_numbers: ArrayLike,
    labels: Sequence[str],
    charges: units.Charge | None = None,
) -> tuple[
    Table[ParticleId, Particles],
    Cell[AnyPeriodicity],
    Callable[[units.Length], units.Length],
]:
    """Build particle data and cell from source-neutral arrays.

    Numeric inputs follow the active JAX dtype configuration; exact dtype
    preservation is not guaranteed.

    Args:
        positions: Cartesian positions, shape ``(n, 3)``, expressed in the
            same Cartesian coordinate frame as ``cell_vectors``.
        cell_vectors: Three cell vectors stored row-wise, shape ``(3, 3)``.
            The caller must provide a valid cell or bounding frame.
        periodicity: Three boolean values selecting periodic axes.
        masses: Particle masses, shape ``(n,)``.
        atomic_numbers: Integer atomic numbers, shape ``(n,)``.
        labels: Non-string sequence of string labels, one per particle.
        charges: Partial charges, shape ``(n,)``. Omitted charges default to
            zeros.

    Returns:
        Tuple of ``(particles, cell, uc_transform)`` where ``uc_transform``
        rotates Cartesian coordinates into the lower-triangular cell frame.

    Raises:
        ValueError: If an input shape or particle/label count is invalid.
        TypeError: If a dimensioned input is plain magnitudes, or another input
            fails to convert or has an invalid type or dtype.
        units.DimensionError: If a quantity input has the wrong dimension.
    """
    _require_dimension("positions", positions, u.angstrom)
    _require_dimension("cell_vectors", cell_vectors, u.angstrom)
    _require_dimension("masses", masses, u.amu)
    atomic_numbers_array = _as_jax_array("atomic_numbers", atomic_numbers)

    _require_integer_dtype("atomic_numbers", atomic_numbers_array)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (n, 3); got {positions.shape}.")
    if cell_vectors.shape != (3, 3):
        raise ValueError(
            f"cell_vectors must have shape (3, 3); got {cell_vectors.shape}."
        )

    n_particles = positions.shape[0]
    if charges is None:
        charges = units.zeros(n_particles, u.e)
    _require_dimension("charges", charges, u.e)
    _require_particle_shape("masses", masses, n_particles)
    _require_particle_shape("atomic_numbers", atomic_numbers_array, n_particles)
    _require_particle_shape("charges", charges, n_particles)

    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a non-string sequence of strings.")
    try:
        label_values = list(labels)
    except TypeError as error:
        raise TypeError("labels must be a sequence of strings.") from error
    if len(label_values) != n_particles:
        raise ValueError(
            f"labels must contain {n_particles} values; got {len(label_values)}."
        )
    if not all(isinstance(label, str) for label in label_values):
        raise TypeError("labels must contain only strings.")

    periodicity_tuple = _normalize_periodicity(periodicity)

    return _build_particles_and_cell(
        positions=positions,
        cell_vectors=cell_vectors,
        periodicity=periodicity_tuple,
        masses=masses,
        atomic_numbers=atomic_numbers_array,
        charges=charges,
        labels=list(map(Label, label_values)),
    )


def particles_from_ase(
    atoms: ase.Atoms | str | Path,
) -> tuple[
    Table[ParticleId, Particles],
    Cell[AnyPeriodicity],
    Callable[[units.Length], units.Length],
]:
    """Build particle data and cell from an ASE Atoms object or file path.

    Results are cached when ``atoms`` is a file path.

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.

    Returns:
        Tuple of (particles, cell, uc_transform) where uc_transform
        rotates Cartesian positions into the lower-triangular frame.
    """
    if isinstance(atoms, (str, Path)):
        return _particles_from_path(atoms)
    return _particles_from_atoms(atoms)


@cache
def _particles_from_path(
    path: str | Path,
) -> tuple[
    Table[ParticleId, Particles],
    Cell[AnyPeriodicity],
    Callable[[units.Length], units.Length],
]:
    """Read an ASE-readable file and build cached particle data and cell."""
    return _particles_from_atoms(next(ase.io.iread(path, index=-1, store_tags=True)))


def _as_jax_array(name: str, value: ArrayLike) -> Array:
    """Convert one public numeric argument to a JAX array."""
    try:
        return jnp.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} could not be converted to a JAX array.") from error


def _require_dimension(
    name: str,
    value: units.Quantity[E_M, E_KG, E_S, E_A, E_K],
    unit: units.Unit[E_M, E_KG, E_S, E_A, E_K],
) -> None:
    """Require ``value`` to be a quantity of ``unit``'s dimension.

    Args:
        name: Argument name for error messages.
        value: The argument as passed.
        unit: A unit of the expected dimension.

    Raises:
        TypeError: If ``value`` is plain magnitudes.
        units.DimensionError: If ``value`` carries another dimension.
    """
    if not units.is_quantity(value):
        raise TypeError(
            f"{name} must be a quantity of {unit}'s dimension; multiply plain "
            "magnitudes by their unit."
        )
    if units.dim_of(value) != units.dim_of(unit):
        raise units.DimensionError(
            f"{name} must carry the dimension of {unit}; got {value.unit}."
        )


def _require_integer_dtype(name: str, value: Array) -> None:
    """Require an integer dtype."""
    if not jnp.issubdtype(value.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype; got {value.dtype}.")


def _require_particle_shape(name: str, value: Measured, n_particles: int) -> None:
    """Require a one-dimensional per-particle field."""
    if value.shape != (n_particles,):
        raise ValueError(f"{name} must have shape ({n_particles},); got {value.shape}.")


def _normalize_periodicity(
    periodicity: (
        Sequence[bool | np.bool_ | Array] | np.ndarray[Any, np.dtype[np.bool_]] | Array
    ),
) -> AnyPeriodicity:
    """Validate and normalize periodicity to an exact Python boolean tuple."""
    try:
        periodicity_array = jnp.asarray(periodicity)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("periodicity could not be converted to a JAX array.") from error
    if periodicity_array.shape != (3,):
        raise ValueError(
            f"periodicity must contain exactly three values; "
            f"got shape {periodicity_array.shape}."
        )
    if not jnp.issubdtype(periodicity_array.dtype, jnp.bool_):
        raise TypeError(
            f"periodicity must contain only boolean values; "
            f"got {periodicity_array.dtype}."
        )
    return (
        bool(periodicity_array[0]),
        bool(periodicity_array[1]),
        bool(periodicity_array[2]),
    )


def _build_particles_and_cell(
    *,
    positions: units.Length,
    cell_vectors: units.Length,
    periodicity: AnyPeriodicity,
    masses: units.Mass,
    atomic_numbers: Array,
    charges: units.Charge,
    labels: list[Label],
) -> tuple[
    Table[ParticleId, Particles],
    Cell[AnyPeriodicity],
    Callable[[units.Length], units.Length],
]:
    """Build particle data and cell from extracted source data."""
    L, uc_transform = to_lower_triangular(cell_vectors)
    cell = Cell.from_pbc(TriclinicFrame.from_matrix(L), periodicity)
    positions = uc_transform(positions)
    n_atoms = len(masses)
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
    return particles, cell, uc_transform


def _particles_from_atoms(
    atoms: ase.Atoms,
) -> tuple[
    Table[ParticleId, Particles],
    Cell[AnyPeriodicity],
    Callable[[units.Length], units.Length],
]:
    """Build particle data and cell from an ASE Atoms object."""
    cell_vectors = jnp.asarray(atoms.cell.array)
    pbc = (bool(atoms.pbc[0]), bool(atoms.pbc[1]), bool(atoms.pbc[2]))
    positions = jnp.asarray(atoms.positions)
    masses = jnp.asarray(atoms.get_masses())
    atomic_numbers = jnp.asarray(atoms.get_atomic_numbers())
    charges = jnp.asarray(
        atoms.info.get(
            "_atom_type_partial_charge",
            atoms.info.get("_atom_site_charge", jnp.zeros((len(positions),))),
        )
    )
    labels = list(
        map(Label, atoms.info.get("_atom_site_label", atoms.get_chemical_symbols()))
    )
    return _build_particles_and_cell(
        positions=positions * u.angstrom,
        cell_vectors=cell_vectors * u.angstrom,
        periodicity=pbc,
        masses=masses * u.amu,
        atomic_numbers=atomic_numbers,
        charges=charges * u.e,
        labels=labels,
    )
