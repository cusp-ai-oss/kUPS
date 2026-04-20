# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Data structures for application-layer simulation entities."""

from __future__ import annotations

from jax import Array

from kups.core.unitcell import UnitCell
from kups.core.utils.jax import dataclass


@dataclass
class Atoms:
    """Per-atom arrays shared by host and adsorbate representations.

    Attributes:
        positions: Cartesian coordinates, shape ``(n_atoms, 3)``.
        species: Integer species indices, shape ``(n_atoms,)``.
        charges: Partial charges, shape ``(n_atoms,)``.
        masses: Atomic masses (amu), shape ``(n_atoms,)``.
        atomic_numbers: Atomic numbers, shape ``(n_atoms,)``.
    """

    positions: Array
    species: Array
    charges: Array
    masses: Array
    atomic_numbers: Array


@dataclass
class Adsorbates:
    """Adsorbate thermodynamic parameters and atom data.

    Attributes:
        critical_temperatures: Critical temperatures per adsorbate species.
        critical_pressures: Critical pressures per adsorbate species.
        acentric_factors: Acentric factors per adsorbate species.
        atoms: Per-atom data for all adsorbate atoms.
    """

    critical_temperatures: Array
    critical_pressures: Array
    acentric_factors: Array
    atoms: Atoms


@dataclass
class Host:
    """Host framework (e.g. MOF or zeolite) with its unit cell.

    Attributes:
        atoms: Per-atom data for the host framework.
        unitcell: Periodic unit cell describing the simulation box.
    """

    atoms: Atoms
    unitcell: UnitCell


@dataclass
class Molecules:
    """Minimal molecule descriptor carrying species indices.

    Attributes:
        species: Integer species indices, shape ``(n_molecules,)``.
    """

    species: Array
