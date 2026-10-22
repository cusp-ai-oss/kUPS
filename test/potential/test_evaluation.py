# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-shot potential evaluation helpers."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import lens
from kups.core.typing import ExclusionId, InclusionId, Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.ewald import EwaldParameters
from kups.potential.classical.ewald.potential import TO_STANDARD_UNITS
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    lennard_jones_energy,
)
from kups.potential.common.evaluation import (
    evaluate_ewald_potential,
    evaluate_radius_graph_potential,
)
from kups.potential.common.graph import PointCloud


@dataclass
class _Particles:
    positions: jax.Array
    labels: Index[Label]
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class _Systems:
    cell: Cell


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
        _Systems(
            cell=PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0))
        ),
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


@dataclass
class _ChargedParticles(_Particles):
    charges: jax.Array


def _rock_salt(r0: float, systems: int) -> PointCloud:
    """Conventional rock-salt cells (8 ions) with nearest-neighbour distance r0."""
    corners = jnp.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float
    )
    charges = jnp.where(corners.sum(axis=1) % 2 == 0, 1.0, -1.0)
    n = 8 * systems
    particles = Table.arange(
        _ChargedParticles(
            positions=jnp.tile(corners * r0, (systems, 1)),
            labels=Index.new([Label("A")] * n),
            system=Index.new([SystemId(i // 8) for i in range(n)]),
            inclusion=Index.new([InclusionId(i // 8) for i in range(n)]),
            exclusion=Index.new([ExclusionId(i) for i in range(n)]),
            charges=jnp.tile(charges, systems),
        ),
        label=ParticleId,
    )
    cell = jnp.broadcast_to(jnp.eye(3) * 2 * r0, (systems, 3, 3))
    systems_table = Table.arange(
        _Systems(cell=PeriodicCell(TriclinicFrame.from_matrix(cell))), label=SystemId
    )
    return PointCloud(particles, systems_table)


@pytest.mark.parametrize("systems", [1, 2])
def test_evaluate_ewald_potential_rock_salt_madelung(systems):
    """The full Ewald sum reproduces the rock-salt Madelung energy."""
    r0 = 2.8
    cloud = _rock_salt(r0, systems)
    parameters = EwaldParameters.make(cloud.particles, cloud.systems, real_cutoff=5.0)
    result = evaluate_ewald_potential(
        cloud, parameters, gradient_lens=lens(lambda x: x.particles.data.positions)
    )
    expected = -4 * 1.747564594633 * TO_STANDARD_UNITS / r0
    npt.assert_allclose(result.total_energies.data, expected, rtol=1e-6)
    npt.assert_allclose(result.gradients, 0.0, atol=1e-6)
