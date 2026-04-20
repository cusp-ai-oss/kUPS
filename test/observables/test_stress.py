# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for stress computation via different methods."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.data import Index, Table
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass
from kups.observables.stress import (
    stress_via_lattice_vector_gradients,
    stress_via_virial_theorem,
)


@dataclass
class _VirialParticles:
    positions: Array
    position_gradients: Array
    system: Index[SystemId]


@dataclass
class _VirialSystems:
    unitcell: UnitCell
    unitcell_gradients: UnitCell


def _make_systems(
    lattice_vectors: Array, lattice_grad: Array | None = None
) -> Table[SystemId, _VirialSystems]:
    """Helper: single-system Table with unit cell and gradients."""
    if lattice_grad is None:
        lattice_grad = jnp.zeros_like(lattice_vectors)
    uc = TriclinicUnitCell.from_matrix(lattice_vectors)
    uc_grad = TriclinicUnitCell.from_matrix(lattice_grad)
    n = lattice_vectors.shape[0]
    keys = tuple(SystemId(i) for i in range(n))
    return Table(keys, _VirialSystems(unitcell=uc, unitcell_gradients=uc_grad))


def _make_particles(
    positions: Array,
    position_gradients: Array | None = None,
    system_ids: Array | None = None,
    n_systems: int = 1,
) -> Table[ParticleId, _VirialParticles]:
    """Helper: particle Table with positions, gradients, and system index."""
    n = positions.shape[0]
    if position_gradients is None:
        position_gradients = positions  # default: use positions as gradients
    if system_ids is None:
        system_ids = jnp.zeros(n, dtype=int)
    sys_keys = tuple(SystemId(i) for i in range(n_systems))
    p_keys = tuple(ParticleId(i) for i in range(n))
    return Table(
        p_keys,
        _VirialParticles(
            positions=positions,
            position_gradients=position_gradients,
            system=Index(sys_keys, system_ids),
        ),
    )


class TestStressViaLatticeVectorGradients:
    """Test stress from energy gradients w.r.t. lattice vectors."""

    @pytest.mark.parametrize(
        "virial,box_size,expected_fn",
        [
            pytest.param(
                jnp.zeros((1, 3, 3)),
                1.0,
                lambda v, b: jnp.zeros((3, 3)),
                id="zero_gradient",
            ),
            pytest.param(
                jnp.eye(3)[None] * 2.0,
                3.0,
                lambda v, b: -v[0] / b**3,
                id="known_gradient",
            ),
            pytest.param(
                jnp.array([[[1.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]]),
                2.0,
                lambda v, b: -v[0] / b**3,
                id="asymmetric_virial",
            ),
        ],
    )
    def test_stress_from_virial(self, virial, box_size, expected_fn):
        """Stress = -virial / V for various virial tensors."""
        systems = _make_systems(jnp.eye(3)[None] * box_size, virial)
        result = stress_via_lattice_vector_gradients(systems)
        expected = expected_fn(virial, box_size)
        npt.assert_allclose(result.data[0], expected, rtol=1e-6, atol=1e-10)


class TestStressViaVirialTheorem:
    """Test stress via virial theorem: sigma = -1/V * (sum(grad_i ⊗ r_i) + h^T dU/dh)."""

    def test_known_virial_stress(self):
        """Two particles with known positions: verify stress = -sum(r_i ⊗ r_i)/V."""
        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        lv = jnp.eye(3)[None] * 2.0
        particles = _make_particles(positions)  # gradients = positions
        systems = _make_systems(lv)
        result = jax.jit(stress_via_virial_theorem)(particles, systems)

        volume = 8.0
        expected = (
            -(
                jnp.outer(positions[0], positions[0])
                + jnp.outer(positions[1], positions[1])
            )
            / volume
        )
        assert result.data.shape == (1, 3, 3)
        npt.assert_allclose(result.data[0], expected, rtol=1e-6)

    def test_single_particle_at_origin(self):
        """Single particle at origin gives zero stress."""
        particles = _make_particles(jnp.zeros((1, 3)))
        systems = _make_systems(jnp.eye(3)[None])
        result = jax.jit(stress_via_virial_theorem)(particles, systems)
        npt.assert_allclose(result.data, 0.0, atol=1e-10)

    def test_two_particles_numerical(self):
        """Two particles at known positions: verify each stress component."""
        positions = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        particles = _make_particles(positions)
        systems = _make_systems(jnp.eye(3)[None])
        result = jax.jit(stress_via_virial_theorem)(particles, systems)

        expected = -(
            jnp.outer(positions[0], positions[0])
            + jnp.outer(positions[1], positions[1])
        )
        assert result.data.shape == (1, 3, 3)
        npt.assert_allclose(result.data[0], expected, rtol=1e-6)

    def test_with_lattice_gradient_contribution(self):
        """Non-zero lattice gradient adds to stress via h^T @ dU/dh."""
        lv = jnp.eye(3)[None] * 2.0
        lattice_grad = jnp.eye(3)[None] * 3.0
        particles = _make_particles(
            jnp.zeros((1, 3)), position_gradients=jnp.zeros((1, 3))
        )
        systems = _make_systems(lv, lattice_grad)
        result = jax.jit(stress_via_virial_theorem)(particles, systems)

        # h^T @ dU/dh = (2I)^T @ (3I) = 6I, stress = -6I / V = -6I / 8
        expected = -jnp.eye(3) * 6.0 / 8.0
        npt.assert_allclose(result.data[0], expected, rtol=1e-6)


class TestStressMethodComparison:
    """Cross-validate that both stress methods produce consistent results."""

    def test_symmetric_positions_give_symmetric_stress(self):
        """Symmetric positions should produce symmetric stress tensors."""
        positions = jnp.array([[0.25, 0.25, 0.25], [-0.25, -0.25, -0.25]])
        lv = jnp.eye(3)[None] * 2.0

        # Lattice vector method: σ = -virial / V where virial = sum(r) ⊗ sum(r)
        sum_pos = jnp.sum(positions, axis=0)  # = [0, 0, 0]
        virial = jnp.outer(sum_pos, sum_pos)[None]
        systems_lv = _make_systems(lv, virial)
        result_lv = stress_via_lattice_vector_gradients(systems_lv)

        # Virial theorem method
        particles = _make_particles(positions)
        systems_vt = _make_systems(lv)
        result_vt = jax.jit(stress_via_virial_theorem)(particles, systems_vt)

        npt.assert_allclose(result_lv.data, 0.0, atol=1e-10)
        npt.assert_allclose(result_vt.data[0], result_vt.data[0].T, atol=1e-10)
