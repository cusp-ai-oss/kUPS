# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Source-neutral relaxation builders preserve table identity and geometry."""

from typing import assert_type

import ase
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.relaxation.data import (
    relax_index_prefix,
    relax_state_from_particles,
)
from kups.application.relaxation.simulation import IndexPrefix
from kups.application.utils.particles import particles_from_ase
from kups.core.cell import DeformedFrame
from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.typing import ParticleId, SystemId
from kups.potential.common.geometry import PositionsAndCellIndex


@pytest.mark.parametrize("batched", [False, True])
def test_builder_preserves_keys_geometry_and_system_counts(batched: bool) -> None:
    atoms = ase.Atoms("Ar2", positions=[[0, 0, 0], [1, 2, 3]], cell=[4, 5, 6], pbc=True)
    particles, cell, _ = particles_from_ase(atoms)
    keys = (SystemId(12), SystemId(42)) if batched else (SystemId(42),)
    references = jnp.array([0, 1] if batched else [0, 0])
    particles = Table(
        (ParticleId(7), ParticleId(19)),
        bind(particles.data).focus(lambda p: p.system).set(Index(keys, references)),
    )
    if batched:
        cell = jax.tree.map(lambda x: jnp.repeat(x[None], 2, axis=0), cell)
    relaxed, systems = relax_state_from_particles(particles, cell)
    index_prefix: IndexPrefix = relax_index_prefix
    prefix = index_prefix(particles=relaxed, systems=systems)
    assert_type(prefix, PositionsAndCellIndex)
    npt.assert_array_equal(prefix.positions.indices, relaxed.data.system.indices)
    npt.assert_array_equal(prefix.cell.indices, systems.index.indices)
    assert relaxed.keys == particles.keys
    assert systems.keys == keys
    npt.assert_array_equal(
        systems[relaxed.data.system].cell.vectors,
        cell.vectors if batched else jnp.repeat(cell.vectors[None], 2, axis=0),
    )
    npt.assert_array_equal(relaxed.data.positions, particles.data.positions)
    npt.assert_array_equal(relaxed.data.position_gradients, 0)
    npt.assert_array_equal(relaxed.data.system.counts.data, [1, 1] if batched else [2])
    frame = systems.data.cell.frame
    assert isinstance(frame, DeformedFrame)
    npt.assert_array_equal(
        frame.vectors, cell.vectors if batched else cell.vectors[None]
    )


def test_builder_rejects_mismatched_cell_count() -> None:
    atoms = ase.Atoms("Ar", cell=[4, 5, 6], pbc=True)
    particles, cell, _ = particles_from_ase(atoms)
    with pytest.raises(ValueError, match="cell vectors"):
        relax_state_from_particles(
            particles, jax.tree.map(lambda x: jnp.repeat(x[None], 2, axis=0), cell)
        )
