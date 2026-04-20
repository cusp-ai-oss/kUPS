# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for classical potential tests."""

import jax

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass


@dataclass
class PointCloudParticles:
    """Simple particle data for testing."""

    positions: jax.Array
    labels: Index[str]
    system: Index[SystemId]


@dataclass
class SystemData:
    """Simple system data for testing."""

    unitcell: UnitCell


def make_particles(
    positions: jax.Array, species: list[str], system_ids: list[int]
) -> Table[ParticleId, PointCloudParticles]:
    labels = Index.new(species)
    system = Index.new(system_ids)
    return Table.arange(
        PointCloudParticles(positions, labels, system), label=ParticleId
    )


def make_systems(lattice_vectors: jax.Array) -> Table[SystemId, SystemData]:
    unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
    return Table.arange(SystemData(unitcell), label=SystemId)
