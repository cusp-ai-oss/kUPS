# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-shot potential evaluation helpers."""

import jax.numpy as jnp
import pytest

from kups.core import units
from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import lens
from kups.core.typing import ExclusionId, InclusionId, Label, ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.utils.jax import dataclass
from kups.potential.classical.ewald import EwaldParameters
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    lennard_jones_energy,
)
from kups.potential.common.evaluation import (
    evaluate_ewald_potential,
    evaluate_radius_graph_potential,
)
from kups.potential.common.graph import PointCloud
from test.units import EV, A, as_length, assert_close


@dataclass
class _Particles:
    positions: units.Length
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
            positions=as_length(positions),
            labels=Index.new([Label("A"), Label("B"), Label("A"), Label("B")]),
            system=Index.new([SystemId(0)] * n),
            inclusion=Index.new([InclusionId(0)] * n),
            exclusion=Index.new([ExclusionId(i) for i in range(n)]),
        ),
        label=ParticleId,
    )
    systems = Table.arange(
        _Systems(
            cell=PeriodicCell(TriclinicFrame.from_matrix((jnp.eye(3)[None] * 10.0) * A))
        ),
        label=SystemId,
    )
    lj_params = LennardJonesParameters(
        labels=(Label("A"), Label("B")),
        sigma=jnp.ones((2, 2)) * A,
        epsilon=jnp.ones((2, 2)) * 0.5 * EV,
        cutoff=Table((SystemId(0),), jnp.array([5.0]) * A),
    )

    result = evaluate_radius_graph_potential(
        PointCloud(particles, systems),
        lj_params,
        cutoffs=lj_params.cutoff,
        energy_fn=lennard_jones_energy,
    )

    energies = result.total_energies.data
    assert energies.shape == (1,)
    magnitudes = energies.to(EV)
    assert jnp.isfinite(magnitudes).all()
    assert magnitudes[0] != 0.0


@dataclass
class _ChargedParticles(_Particles):
    charges: units.Charge


def _rock_salt(r0: float, systems: int) -> PointCloud:
    """Conventional rock-salt cells (8 ions) with nearest-neighbour distance r0 in Å."""
    corners = jnp.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float
    )
    charges = jnp.where(corners.sum(axis=1) % 2 == 0, 1.0, -1.0)
    n = 8 * systems
    particles = Table.arange(
        _ChargedParticles(
            positions=as_length(jnp.tile(corners * r0, (systems, 1))),
            labels=Index.new([Label("A")] * n),
            system=Index.new([SystemId(i // 8) for i in range(n)]),
            inclusion=Index.new([InclusionId(i // 8) for i in range(n)]),
            exclusion=Index.new([ExclusionId(i) for i in range(n)]),
            charges=jnp.tile(charges, systems) * u.e,
        ),
        label=ParticleId,
    )
    cell = jnp.broadcast_to(jnp.eye(3) * 2 * r0, (systems, 3, 3)) * A
    systems_table = Table.arange(
        _Systems(cell=PeriodicCell(TriclinicFrame.from_matrix(cell))), label=SystemId
    )
    return PointCloud(particles, systems_table)


@pytest.mark.parametrize("systems", [1, 2])
def test_evaluate_ewald_potential_rock_salt_madelung(systems):
    """The full Ewald sum reproduces the rock-salt Madelung energy."""
    r0 = 2.8
    cloud = _rock_salt(r0, systems)
    parameters = EwaldParameters.make(
        cloud.particles, cloud.systems, real_cutoff=5.0 * A
    )
    result = evaluate_ewald_potential(
        cloud, parameters, gradient_lens=lens(lambda x: x.particles.data.positions)
    )
    expected = -4 * 1.747564594633 * units.constants.k_e * u.e**2 / (r0 * A)
    assert_close(result.total_energies.data, expected, rtol=1e-6)
    assert_close(result.gradients, jnp.zeros((8 * systems, 3)), atol=1e-6)
