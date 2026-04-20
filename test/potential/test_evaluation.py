# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Test evaluate_radius_graph_potential with UniversalNeighborlistParameters."""

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import ExclusionId, InclusionId, Label, ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    lennard_jones_energy,
)
from kups.potential.common.evaluation import evaluate_radius_graph_potential
from kups.potential.common.graph import PointCloud

from ..clear_cache import clear_cache  # noqa: F401


@dataclass
class _Particles:
    positions: jax.Array
    labels: Index[Label]
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class _Systems:
    unitcell: UnitCell


def test_evaluate_radius_graph_potential_lj():
    """Builds radius graph and evaluates LJ energy end-to-end."""
    positions = jnp.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [1.5, 1.5, 0.0]]
    )
    n = len(positions)

    particles = Table.arange(
        _Particles(
            positions=positions,
            labels=Index.new([Label("A"), Label("B"), Label("A"), Label("B")]),
            system=Index.new([SystemId(0)] * n),
            inclusion=Index.new([InclusionId(0)] * n),
            exclusion=Index.new([ExclusionId(i) for i in range(n)]),
        ),
        label=ParticleId,
    )
    systems = Table.arange(
        _Systems(unitcell=TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * 10.0)),
        label=SystemId,
    )
    lj_params = LennardJonesParameters(
        labels=(Label("A"), Label("B")),
        sigma=jnp.ones((2, 2)),
        epsilon=jnp.ones((2, 2)) * 0.5,
        cutoff=Table((SystemId(0),), jnp.array([5.0])),
    )

    result = evaluate_radius_graph_potential(
        PointCloud(particles, systems),
        lj_params,
        cutoffs=lj_params.cutoff,
        energy_fn=lennard_jones_energy,
    )

    energies = result.total_energies.data
    assert energies.shape == (1,)
    assert jnp.isfinite(energies).all()
    assert energies[0] != 0.0
