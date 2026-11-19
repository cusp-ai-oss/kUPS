# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Source-neutral relaxation builders preserve table identity and geometry."""

from typing import assert_type

import ase
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.application.relaxation.data import (
    relax_index_prefix,
    relax_state_from_particles_and_cell,
)
from kups.application.relaxation.simulation import IndexPrefixFn
from kups.application.utils.particles import particles_from_ase
from kups.core.cell import DeformedFrame, MatrixLogFrame
from kups.core.data import Index, Table
from kups.core.geometry import PositionsAndCellIndex
from kups.core.lens import bind
from kups.core.typing import ParticleId, SystemId
from kups.core.units import catalog as u
from kups.core.units import testing


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

        def stack_twice(x: Array) -> Array:
            return jnp.repeat(x[None], 2, axis=0)

        cell = jax.tree.map(stack_twice, cell)
    relaxed, systems = relax_state_from_particles_and_cell(particles, cell)
    index_prefix: IndexPrefixFn = relax_index_prefix
    prefix = index_prefix(particles=relaxed, systems=systems)
    assert_type(prefix, PositionsAndCellIndex)
    npt.assert_array_equal(prefix.positions.value, relaxed.data.system.value)
    npt.assert_array_equal(prefix.cell.value, systems.index.value)
    assert relaxed.keys == particles.keys
    assert systems.keys == keys
    vectors = cell.vectors.to(u.angstrom)
    npt.assert_array_equal(
        systems[relaxed.data.system].cell.vectors.to(u.angstrom),
        vectors if batched else jnp.repeat(vectors[None], 2, axis=0),
    )
    npt.assert_array_equal(
        relaxed.data.positions.to(u.angstrom), particles.data.positions.to(u.angstrom)
    )
    npt.assert_array_equal(relaxed.data.position_gradients.to(u.eV / u.angstrom), 0)
    npt.assert_array_equal(relaxed.data.system.counts.data, [1, 1] if batched else [2])
    frame = systems.data.cell.frame
    assert isinstance(frame, DeformedFrame)
    assert isinstance(frame.deformation, MatrixLogFrame)
    # ASE's exp_cell_factor: one atom count per system, times 1 Å.
    testing.assert_allclose(
        frame.deformation.cell_factor[:, 0, 0],
        jnp.array([1.0, 1.0] if batched else [2.0]),
        u.angstrom,
    )
    npt.assert_array_equal(
        frame.vectors.to(u.angstrom), vectors if batched else vectors[None]
    )


def test_builder_rejects_mismatched_cell_count() -> None:
    atoms = ase.Atoms("Ar", cell=[4, 5, 6], pbc=True)
    particles, cell, _ = particles_from_ase(atoms)

    def stack_twice(x: Array) -> Array:
        return jnp.repeat(x[None], 2, axis=0)

    with pytest.raises(ValueError, match="cell vectors"):
        relax_state_from_particles_and_cell(particles, jax.tree.map(stack_twice, cell))
