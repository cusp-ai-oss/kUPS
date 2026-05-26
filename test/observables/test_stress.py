# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for stress computation via the virial theorem."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax import Array

from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.observables.stress import stress_via_virial_theorem


@dataclass
class _VirialParticles:
    positions: Array
    position_gradients: Array
    system: Index[SystemId]


@dataclass
class _VirialSystems:
    cell: Cell
    cell_gradients: Cell


def _make_systems(
    lattice_vectors: Array, lattice_grad: Array | None = None
) -> Table[SystemId, _VirialSystems]:
    """Helper: Table with cell and cell-gradient for each row of
    ``lattice_vectors`` (one system per leading-axis entry)."""
    if lattice_grad is None:
        lattice_grad = jnp.zeros_like(lattice_vectors)
    cell = PeriodicCell(TriclinicFrame.from_matrix(lattice_vectors))
    cell_grad = PeriodicCell(TriclinicFrame.from_matrix(lattice_grad))
    n = lattice_vectors.shape[0]
    keys = tuple(SystemId(i) for i in range(n))
    return Table(keys, _VirialSystems(cell=cell, cell_gradients=cell_grad))


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


class TestStressViaVirialTheorem:
    """σ = -1/V sym[Σ r ⊗ ∂U/∂r + h^T·∂U/∂h]."""

    def test_known_virial_stress(self):
        """Two particles with `position_gradients == positions` (symmetric outer
        product); zero cell gradient: σ = -Σ p_i⊗p_i / V."""
        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        lv = jnp.eye(3)[None] * 2.0
        particles = _make_particles(positions)
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

    def test_with_diagonal_lattice_gradient(self):
        """Diagonal h and diagonal ∂U/∂h: σ = -h^T·∂U/∂h / V."""
        lv = jnp.eye(3)[None] * 2.0
        lattice_grad = jnp.eye(3)[None] * 3.0
        particles = _make_particles(
            jnp.zeros((1, 3)), position_gradients=jnp.zeros((1, 3))
        )
        systems = _make_systems(lv, lattice_grad)
        result = jax.jit(stress_via_virial_theorem)(particles, systems)

        # h^T @ ∂U/∂h = (2I) @ (3I) = 6I, σ = -6I / 8
        expected = -jnp.eye(3) * 6.0 / 8.0
        npt.assert_allclose(result.data[0], expected, rtol=1e-6)

    def test_periodic_pair_virial_combines_position_and_cell_before_symmetry(self):
        """A pair crossing PBC reconstructs the full minimum-image virial.

        The wrapped coordinate difference is ``r_j - r_i = [-8, 2, 1]`` and the
        periodic image shift is ``[1, 0, 0]``, giving minimum-image displacement
        ``d = [2, 2, 1]``. For a central pair with ``∂U/∂d = d``, the stress is
        ``-d⊗d/V``. This only comes out correctly if the lower-triangular
        position virial and cell-gradient virial are added before symmetrizing.
        """
        lv = (jnp.eye(3) * 10.0)[None]
        positions = jnp.array([[9.0, 1.0, 1.0], [1.0, 3.0, 2.0]])
        d = jnp.array([2.0, 2.0, 1.0])
        position_gradients = jnp.array([-d, d])
        # Full d = r_j - r_i + s @ h with s = [1, 0, 0]. Only lower-triangular
        # cell-gradient DOFs are stored.
        lattice_grad = jnp.tril(
            jnp.array([[[2.0, 2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        )

        particles = _make_particles(positions, position_gradients=position_gradients)
        systems = _make_systems(lv, lattice_grad)
        result = jax.jit(stress_via_virial_theorem)(particles, systems)

        expected = -jnp.outer(d, d) / 1000.0
        npt.assert_allclose(result.data[0], expected, atol=1e-12)

    def test_triclinic_cell_recovers_symmetric_cell_virial(self):
        """Non-diagonal lower-triangular ``h``: stress code recovers
        ``S = h^T·∂U/∂h`` from the stored lower triangle of ``∂U/∂h``.

        Build a symmetric ``S``, solve ``g = (h^T)^{-1}·S`` for the full
        ``∂U/∂h``, project to the lower triangle (what ``TriclinicFrame``
        stores), and verify ``stress_via_virial_theorem`` returns
        ``-S / V`` with no particle contribution.
        """
        rng = np.random.default_rng(0)
        h_np = np.array([[2.0, 0.0, 0.0], [0.7, 1.9, 0.0], [0.3, -0.5, 2.4]])
        S_np = rng.standard_normal((3, 3))
        S_np = S_np + S_np.T  # symmetric
        g_full = np.linalg.solve(h_np.T, S_np)
        g_tril = np.tril(g_full)  # what TriclinicFrame.from_matrix keeps

        lv = jnp.asarray(h_np)[None]
        lattice_grad = jnp.asarray(g_tril)[None]
        particles = _make_particles(
            jnp.zeros((1, 3)), position_gradients=jnp.zeros((1, 3))
        )
        systems = _make_systems(lv, lattice_grad)
        result = jax.jit(stress_via_virial_theorem)(particles, systems)

        volume = float(np.abs(np.linalg.det(h_np)))
        expected = -S_np / volume
        npt.assert_allclose(result.data[0], expected, atol=1e-10)

    def test_stress_is_symmetric(self):
        """Output is always symmetric, even with asymmetric position outer
        products and a triclinic lattice gradient."""
        rng = np.random.default_rng(1)
        positions = jnp.asarray(rng.standard_normal((4, 3)))
        position_gradients = jnp.asarray(rng.standard_normal((4, 3)))
        h_np = np.array([[2.0, 0.0, 0.0], [0.7, 1.9, 0.0], [0.3, -0.5, 2.4]])
        g_tril = np.tril(rng.standard_normal((3, 3)))
        lv = jnp.asarray(h_np)[None]
        lattice_grad = jnp.asarray(g_tril)[None]
        particles = _make_particles(positions, position_gradients=position_gradients)
        systems = _make_systems(lv, lattice_grad)
        result = jax.jit(stress_via_virial_theorem)(particles, systems)
        npt.assert_allclose(result.data[0], result.data[0].T, atol=1e-10)
