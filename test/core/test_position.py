# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for position utility functions."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import GroupId, ParticleId
from kups.core.unitcell import TriclinicUnitCell
from kups.core.utils.jax import dataclass
from kups.core.utils.position import (
    center_of_mass,
    to_absolute_positions,
    to_relative_positions,
)


@dataclass
class ParticlesWithMass:
    positions: jax.Array
    weights: jax.Array
    group: Index[GroupId]


def _make_group_index(ids: list[int], num_groups: int) -> Index[GroupId]:
    """Create an Index[GroupId] from integer group assignments."""
    labels = tuple(GroupId(i) for i in range(num_groups))
    return Index(labels, jnp.array(ids))


def _make_particles(
    positions: jax.Array, weights: jax.Array, ids: list[int], num_groups: int
) -> Table[ParticleId, ParticlesWithMass]:
    """Create Table particles from positions, weights, and group assignments."""
    group = _make_group_index(ids, num_groups)
    data = ParticlesWithMass(positions=positions, weights=weights, group=group)
    return Table.arange(data, label=ParticleId)


@pytest.fixture(scope="module")
def unitcells_2sys():
    """Two cubic unit cells: 4x4x4 and 6x6x6."""
    return TriclinicUnitCell.from_matrix(
        jnp.array(
            [
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
            ]
        ),
    )


@pytest.fixture(scope="module")
def unitcells_1sys():
    """Single 4x4x4 cubic unit cell."""
    return TriclinicUnitCell.from_matrix(
        jnp.array([[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]]),
    )


@pytest.fixture(scope="module")
def small_unitcells():
    """Single 2x2x2 cubic unit cell."""
    return TriclinicUnitCell.from_matrix(
        jnp.array([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]]),
    )


class TestCenterOfMass:
    """Test the center_of_mass function."""

    def test_center_of_mass_variants(self, unitcells_2sys, small_unitcells):
        """Test CoM with equal weights, different weights, wrapping, and single particle."""
        # Equal weights
        positions = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        particles = _make_particles(positions, jnp.ones(4), [0, 0, 1, 1], num_groups=2)
        com = center_of_mass(particles, unitcells_2sys[:2])
        expected = jnp.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]])
        npt.assert_allclose(com, expected, atol=1e-10)

        # Different weights
        positions2 = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]]
        )
        weights2 = jnp.array([3.0, 1.0, 1.0, 3.0])
        particles2 = _make_particles(positions2, weights2, [0, 0, 1, 1], num_groups=2)
        com2 = center_of_mass(particles2, unitcells_2sys[:2])
        expected2 = jnp.array([[0.25, 0.0, 0.0], [2.75, 2.0, 0.0]])
        npt.assert_allclose(com2, expected2, atol=1e-10)

        # Wrapping
        positions3 = jnp.array([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]])
        particles3 = _make_particles(positions3, jnp.ones(2), [0, 0], num_groups=1)
        com3 = center_of_mass(particles3, small_unitcells)
        npt.assert_allclose(com3, jnp.array([[0.0, 0.0, 0.0]]), atol=1e-10)

        # Single particle per group
        positions4 = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        particles4 = _make_particles(positions4, jnp.ones(2), [0, 1], num_groups=2)
        com4 = center_of_mass(particles4, unitcells_2sys[:2])
        expected_wrapped = unitcells_2sys[:2].wrap(positions4)
        npt.assert_allclose(com4, expected_wrapped, atol=1e-10)

    def test_center_of_mass_assertion_error(self, unitcells_2sys):
        """Test that assertion error is raised when unit cells don't match groups."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(positions, jnp.ones(2), [0, 0], num_groups=1)

        com = center_of_mass(particles, unitcells_2sys[:1])
        assert com.shape == (1, 3)

        with pytest.raises(
            AssertionError, match="Unit cells must match the number of groups"
        ):
            center_of_mass(particles, unitcells_2sys[:2])


class TestToRelative:
    """Test the to_relative_positions function."""

    def test_relative_positions(self, unitcells_1sys, small_unitcells):
        """Test basic relative positions, with provided CoM, with mass, and wrapping."""
        # Basic
        positions = jnp.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
        particles = _make_particles(positions, jnp.ones(2), [0, 0], num_groups=1)
        com = center_of_mass(particles, unitcells_1sys)
        rel_pos = to_relative_positions(particles, unitcells_1sys)
        expected = unitcells_1sys[0].wrap(positions - com[0])
        npt.assert_allclose(rel_pos, expected, atol=1e-10)

        # With provided CoM
        positions2 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        particles2 = _make_particles(positions2, jnp.ones(2), [0, 0], num_groups=1)
        provided_com = jnp.array([[1.5, 0.0, 0.0]])
        rel_pos2 = to_relative_positions(particles2, unitcells_1sys, provided_com)
        expected2 = jnp.array([[-1.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        npt.assert_allclose(rel_pos2, expected2, atol=1e-10)

        # With mass
        positions3 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        weights3 = jnp.array([1.0, 2.0])
        particles3 = _make_particles(positions3, weights3, [0, 0], num_groups=1)
        com3 = center_of_mass(particles3, unitcells_1sys)
        rel_pos3 = to_relative_positions(particles3, unitcells_1sys)
        expected3 = unitcells_1sys[0].wrap(positions3 - com3[0])
        npt.assert_allclose(rel_pos3, expected3, atol=1e-10)

        # Wrapping
        positions4 = jnp.array([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]])
        particles4 = _make_particles(positions4, jnp.ones(2), [0, 0], num_groups=1)
        rel_pos4 = to_relative_positions(particles4, small_unitcells)
        expected4 = jnp.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        npt.assert_allclose(rel_pos4, expected4, atol=1e-10)


class TestToAbsolute:
    """Test the to_absolute_positions function."""

    def test_absolute_positions(self, unitcells_1sys, small_unitcells):
        """Test basic absolute positions, with mass, and wrapping."""
        # Basic
        rel_positions = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(rel_positions, jnp.ones(2), [0, 0], num_groups=1)
        center_of_masses = jnp.array([[2.0, 1.0, 1.0]])
        abs_pos = to_absolute_positions(particles, unitcells_1sys, center_of_masses)
        expected = unitcells_1sys[0].wrap(rel_positions + center_of_masses[0])
        npt.assert_allclose(abs_pos, expected, atol=1e-10)

        # With mass
        rel_positions2 = jnp.array([[-0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        weights2 = jnp.array([2.0, 1.0])
        particles2 = _make_particles(rel_positions2, weights2, [0, 0], num_groups=1)
        center_of_masses2 = jnp.array([[1.0, 2.0, 3.0]])
        abs_pos2 = to_absolute_positions(particles2, unitcells_1sys, center_of_masses2)
        expected2 = unitcells_1sys[0].wrap(rel_positions2 + center_of_masses2[0])
        npt.assert_allclose(abs_pos2, expected2, atol=1e-10)

        # Wrapping
        rel_positions3 = jnp.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        particles3 = _make_particles(rel_positions3, jnp.ones(2), [0, 0], num_groups=1)
        center_of_masses3 = jnp.array([[1.8, 0.0, 0.0]])
        abs_pos3 = to_absolute_positions(particles3, small_unitcells, center_of_masses3)
        assert jnp.all(jnp.isfinite(abs_pos3))
        assert jnp.all(jnp.abs(abs_pos3) <= 1.0)


class TestRoundTripConsistency:
    """Test round-trip consistency between relative and absolute positions."""

    def test_round_trip(self, unitcells_2sys, unitcells_1sys):
        """Test to_relative -> to_absolute gives back original positions."""
        # Multi-group
        original_positions = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]
        )
        group_ids = [0, 0, 1, 1]
        particles = _make_particles(
            original_positions, jnp.ones(4), group_ids, num_groups=2
        )
        com = center_of_mass(particles, unitcells_2sys)
        rel_pos = to_relative_positions(particles, unitcells_2sys, com)
        rel_particles = _make_particles(rel_pos, jnp.ones(4), group_ids, num_groups=2)
        abs_pos = to_absolute_positions(rel_particles, unitcells_2sys, com)
        npt.assert_allclose(abs_pos, original_positions, atol=1e-6)

        # With mass
        original_positions2 = jnp.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]
        )
        weights = jnp.array([1.0, 2.0, 0.5])
        group_ids2 = [0, 0, 0]
        particles2 = _make_particles(
            original_positions2, weights, group_ids2, num_groups=1
        )
        com2 = center_of_mass(particles2, unitcells_1sys)
        rel_pos2 = to_relative_positions(particles2, unitcells_1sys, com2)
        rel_particles2 = _make_particles(rel_pos2, weights, group_ids2, num_groups=1)
        abs_pos2 = to_absolute_positions(rel_particles2, unitcells_1sys, com2)
        npt.assert_allclose(abs_pos2, original_positions2, atol=1e-10)


class TestJAXCompatibility:
    """Test JAX compatibility of the math functions."""

    def test_jax_transforms(self, small_unitcells):
        """Test JIT, gradient, and vmap for center_of_mass."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(positions, jnp.ones(2), [0, 0], num_groups=1)

        # JIT
        com_normal = center_of_mass(particles, small_unitcells)
        com_jit = jax.jit(center_of_mass)(particles, small_unitcells)
        npt.assert_allclose(com_normal, com_jit, atol=1e-10)

        # Gradient
        group = _make_group_index([0, 0], num_groups=1)

        def com_func(pos):
            data = ParticlesWithMass(
                positions=pos, weights=jnp.ones(pos.shape[0]), group=group
            )
            p = Table.arange(data, label=ParticleId)
            return jnp.sum(center_of_mass(p, small_unitcells))

        grad_positions = jnp.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
        grad = jax.grad(com_func)(grad_positions)
        assert jnp.all(jnp.isfinite(grad))
        expected_grad = jnp.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        npt.assert_allclose(grad, expected_grad, atol=1e-6)

        # Vmap
        batch_positions = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]],
                [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        )

        def process_batch(batch_pos):
            data = ParticlesWithMass(
                positions=batch_pos,
                weights=jnp.ones(batch_pos.shape[0]),
                group=group,
            )
            p = Table.arange(data, label=ParticleId)
            return center_of_mass(p, small_unitcells)

        batch_coms = jax.vmap(process_batch)(batch_positions)
        assert batch_coms.shape == (3, 1, 3)
        expected = jnp.array([process_batch(batch_positions[i]) for i in range(3)])
        npt.assert_allclose(batch_coms, expected, atol=1e-10)
