# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for stress computation via the virial theorem."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax import Array

from kups.core.cell import (
    AnyPeriodicity,
    Cell,
    PeriodicCell,
    TriclinicFrame,
    VacuumCell,
)
from kups.core.data import Index, Table
from kups.core.typing import GroupId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.observables.stress import (
    molecular_stress_via_virial_theorem,
    stress_via_virial_theorem,
    total_lattice_gradient,
)


@dataclass
class _VirialParticles:
    positions: Array
    position_gradients: Array
    system: Index[SystemId]


@dataclass
class _VirialSystems:
    cell: Cell[AnyPeriodicity]
    cell_gradients: Cell[AnyPeriodicity]


def _make_systems(
    lattice_vectors: Array,
    lattice_grad: Array | None = None,
    pbc: tuple[bool, bool, bool] = (True, True, True),
) -> Table[SystemId, _VirialSystems]:
    """Helper: Table with cell and cell-gradient for each row of
    ``lattice_vectors`` (one system per leading-axis entry)."""
    if lattice_grad is None:
        lattice_grad = jnp.zeros_like(lattice_vectors)
    cell = Cell.from_pbc(TriclinicFrame.from_matrix(lattice_vectors), pbc)
    cell_grad = Cell.from_pbc(TriclinicFrame.from_matrix(lattice_grad), pbc)
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


@dataclass
class _MolecularVirialParticles:
    positions: Array
    position_gradients: Array
    masses: Array
    group: Index[GroupId]
    system: Index[SystemId]


@dataclass
class _MolecularGroups:
    system: Index[SystemId]


def _make_molecular_particles(
    positions: Array,
    position_gradients: Array,
    group_ids: Array,
    n_groups: int,
    system_ids: Array | None = None,
    n_systems: int = 1,
    masses: Array | None = None,
) -> Table[ParticleId, _MolecularVirialParticles]:
    """Helper: particle Table carrying a group index as well as a system index.

    ``masses`` default to equal weights, which makes the centre of mass
    coincide with the geometric centroid.
    """
    n = positions.shape[0]
    if system_ids is None:
        system_ids = jnp.zeros(n, dtype=int)
    if masses is None:
        masses = jnp.ones(n)
    return Table(
        tuple(ParticleId(i) for i in range(n)),
        _MolecularVirialParticles(
            positions=positions,
            position_gradients=position_gradients,
            masses=masses,
            group=Index(tuple(GroupId(i) for i in range(n_groups)), group_ids),
            system=Index(tuple(SystemId(i) for i in range(n_systems)), system_ids),
        ),
    )


def _make_groups(
    group_system_ids: Array, n_systems: int = 1
) -> Table[GroupId, _MolecularGroups]:
    """Helper: group Table assigning each group to a system."""
    n = group_system_ids.shape[0]
    return Table(
        tuple(GroupId(i) for i in range(n)),
        _MolecularGroups(
            system=Index(tuple(SystemId(i) for i in range(n_systems)), group_system_ids)
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

    def test_vacuum_cell_has_zero_stress(self):
        """A fully non-periodic (vacuum) cell has no lattice-strain DOF, so every
        stress component is zeroed regardless of the particle/cell gradients."""
        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        lv = jnp.eye(3)[None] * 2.0
        lattice_grad = jnp.eye(3)[None] * 3.0
        particles = _make_particles(positions)
        systems = _make_systems(lv, lattice_grad, pbc=(False, False, False))
        result = jax.jit(stress_via_virial_theorem)(particles, systems)
        npt.assert_allclose(result.data, 0.0, atol=1e-12)

    def test_slab_zeroes_non_periodic_row_and_column(self):
        """A slab periodic in x,y keeps only the in-plane 2×2 block: the z row and
        column are zeroed, and the in-plane block matches the fully periodic
        result (same frame, hence same volume and virial)."""
        positions = jnp.array([[1.0, 2.0, 3.0], [0.5, -1.0, 0.7]])
        lv = jnp.eye(3)[None] * 2.0
        lattice_grad = jnp.tril(
            jnp.array([[[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.3, -0.4, 1.5]]])
        )
        particles = _make_particles(positions)
        slab = jax.jit(stress_via_virial_theorem)(
            particles, _make_systems(lv, lattice_grad, pbc=(True, True, False))
        ).data[0]
        full = jax.jit(stress_via_virial_theorem)(
            particles, _make_systems(lv, lattice_grad)
        ).data[0]
        npt.assert_allclose(slab[2, :], 0.0, atol=1e-12)
        npt.assert_allclose(slab[:, 2], 0.0, atol=1e-12)
        npt.assert_allclose(slab[:2, :2], full[:2, :2], atol=1e-12)


_H_TRICLINIC = jnp.array([[[2.0, 0.0, 0.0], [0.7, 1.9, 0.0], [0.3, -0.5, 2.4]]])


def _periodic(frame: TriclinicFrame) -> Table[SystemId, PeriodicCell]:
    return Table((SystemId(0),), PeriodicCell(frame))


class TestTotalLatticeGradient:
    """total = ∂E/∂h|_r + h⁻ᵀ·Σ rᵢ⊗∂E/∂rᵢ, dropping non-periodic lattice vectors."""

    def test_reduces_to_partial_at_zero_force(self):
        """With ∂E/∂rᵢ = 0 the coupling term vanishes: total == partial."""
        cell = _periodic(TriclinicFrame.from_matrix(_H_TRICLINIC))
        partial = _periodic(TriclinicFrame.from_matrix(jnp.tril(jnp.ones((1, 3, 3)))))
        positions = jnp.array([[0.5, 1.0, -0.3], [1.2, 0.1, 0.8]])
        system = Index((SystemId(0),), jnp.zeros(2, dtype=int))
        total = total_lattice_gradient(
            positions, jnp.zeros((2, 3)), cell, partial, system
        )
        npt.assert_allclose(
            total.data.frame.vectors, partial.data.frame.vectors, atol=1e-12
        )

    def test_matches_autodiff_through_fixed_fractional(self):
        """Validate against ``jax.grad`` of ``E(s@h, h)`` with fractional ``s`` held
        fixed -- the definition of the total lattice gradient."""
        rng = np.random.default_rng(3)
        frame = TriclinicFrame.from_matrix(_H_TRICLINIC)
        positions = jnp.asarray(rng.standard_normal((5, 3)))
        system = Index((SystemId(0),), jnp.zeros(5, dtype=int))
        c = 0.37

        def energy(fr: TriclinicFrame, r: Array) -> Array:
            return 0.5 * (r**2).sum() + c * (fr.vectors**2).sum()

        partial = jax.grad(lambda fr: energy(fr, positions))(frame)
        g = jax.grad(lambda r: energy(frame, r))(positions)
        total = total_lattice_gradient(
            positions, g, _periodic(frame), _periodic(partial), system
        )

        s = frame.to_fractional(positions)  # fixed fractional coordinates
        total_ref = jax.grad(lambda fr: energy(fr, fr.to_real(s)))(frame)
        npt.assert_allclose(total.data.frame.vectors, total_ref.vectors, atol=1e-10)

    def test_non_periodic_axes_drop_the_coupling(self):
        """A non-periodic basis vector carries no atoms: an open cell drops the
        coupling entirely (total == partial), a periodic one does not."""
        rng = np.random.default_rng(4)
        frame = TriclinicFrame.from_matrix(_H_TRICLINIC)
        partial_frame = TriclinicFrame.from_matrix(
            jnp.tril(jnp.asarray(rng.standard_normal((1, 3, 3))))
        )
        positions = jnp.asarray(rng.standard_normal((4, 3)))
        g = jnp.asarray(rng.standard_normal((4, 3)))
        system = Index((SystemId(0),), jnp.zeros(4, dtype=int))

        vacuum = total_lattice_gradient(
            positions,
            g,
            Table((SystemId(0),), VacuumCell(frame)),
            Table((SystemId(0),), VacuumCell(partial_frame)),
            system,
        )
        npt.assert_allclose(
            vacuum.data.frame.vectors, partial_frame.vectors, atol=1e-12
        )

        periodic = total_lattice_gradient(
            positions, g, _periodic(frame), _periodic(partial_frame), system
        )
        assert not jnp.allclose(periodic.data.frame.vectors, partial_frame.vectors)


class TestMolecularStressViaVirialTheorem:
    """σ referenced to group centres of mass rather than atom positions.

    One test per property that delegating the reference point to
    ``center_of_mass`` could have broken: equivalence with the atomic formula,
    minimum-image handling, and mass weighting itself. Tests that pass no
    explicit masses use equal weights, where the centre of mass coincides with
    the geometric centroid.
    """

    def test_singleton_groups_reduce_to_atomic_virial(self):
        """One particle per group makes the centroid the particle itself."""
        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        lv = jnp.eye(3)[None] * 20.0
        systems = _make_systems(lv)
        atomic = stress_via_virial_theorem(_make_particles(positions), systems)
        molecular = molecular_stress_via_virial_theorem(
            _make_molecular_particles(
                positions, positions, jnp.arange(2, dtype=int), n_groups=2
            ),
            _make_groups(jnp.zeros(2, dtype=int)),
            systems,
        )
        npt.assert_allclose(molecular.data, atomic.data, atol=1e-12)

    def test_centroid_uses_the_image_nearest_each_particle(self):
        """A group straddling the boundary must not be centred in mid-box.

        Particles at ``x = 9`` and ``x = 2`` in a 10 A box form one molecule
        across the boundary; its centroid is ``x = 0.5 (mod L)``, not the naive
        in-box mean ``5.5``. Each particle then pairs with the centroid image
        closest to itself -- ``x = 10.5`` for the particle at ``9`` -- so the
        virial stays continuous as a molecule drifts across the boundary.
        """
        L = 10.0
        positions = jnp.array([[9.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        gradients = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = molecular_stress_via_virial_theorem(
            _make_molecular_particles(
                positions, gradients, jnp.zeros(2, dtype=int), n_groups=1
            ),
            _make_groups(jnp.zeros(1, dtype=int)),
            _make_systems(jnp.eye(3)[None] * L),
        )
        expected = jnp.zeros((3, 3)).at[0, 0].set(-10.5 / L**3)
        npt.assert_allclose(result.data[0], expected, atol=1e-12)
        # The naive in-box centroid (5.5) would give a different answer.
        assert not jnp.allclose(result.data[0, 0, 0], -5.5 / L**3)

    def test_reference_point_is_mass_weighted(self):
        """A heteronuclear pair is referenced to its COM, not its centroid.

        Two particles at ``x = 0`` and ``x = 6`` with masses ``1`` and ``3``
        have COM ``x = 4.5`` but centroid ``x = 3``. Only the light particle
        carries a force, so the position virial reads off the reference point
        directly and the two conventions are cleanly separated.
        """
        L = 40.0  # large enough that no minimum-image wrapping intervenes
        positions = jnp.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        gradients = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = molecular_stress_via_virial_theorem(
            _make_molecular_particles(
                positions,
                gradients,
                jnp.zeros(2, dtype=int),
                n_groups=1,
                masses=jnp.array([1.0, 3.0]),
            ),
            _make_groups(jnp.zeros(1, dtype=int)),
            _make_systems(jnp.eye(3)[None] * L),
        )
        com_x, centroid_x = 4.5, 3.0
        npt.assert_allclose(result.data[0, 0, 0], -com_x / L**3, atol=1e-12)
        assert not jnp.allclose(result.data[0, 0, 0], -centroid_x / L**3)
