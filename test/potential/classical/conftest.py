# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for classical potential tests.

Positions and lattice vectors are real-space lengths; plain magnitudes handed to
these builders are read in Angstrom.
"""

import jax

from kups.core import units
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from test.units import as_length


@dataclass
class PointCloudParticles:
    """Simple particle data for testing."""

    positions: units.Length
    labels: Index[str]
    system: Index[SystemId]


@dataclass
class SystemData:
    """Simple system data for testing."""

    cell: Cell


def make_particles(
    positions: jax.Array | units.Length, species: list[str], system_ids: list[int]
) -> Table[ParticleId, PointCloudParticles]:
    labels = Index.new(species)
    system = Index.new(system_ids)
    return Table.arange(
        PointCloudParticles(as_length(positions), labels, system), label=ParticleId
    )


def make_systems(
    lattice_vectors: jax.Array | units.Length,
) -> Table[SystemId, SystemData]:
    cell = PeriodicCell(TriclinicFrame.from_matrix(as_length(lattice_vectors)))
    return Table.arange(SystemData(cell), label=SystemId)
