# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for blocking spheres potential."""

import jax
import jax.numpy as jnp

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist import Edges
from kups.core.typing import ExclusionId, InclusionId, MotifId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.blocking import (
    BlockingSpheresParameters,
    BlockingSpheresPotentialInput,
    blocking_spheres_energy,
)


@dataclass
class ParticleData:
    """Particle data with positions, system, and motif index."""

    positions: jax.Array
    system: Index[SystemId]
    motif: Index[MotifId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


def _make_particles(
    positions: jax.Array,
    system_ids: list[int],
    motif_ids: list[int],
) -> Table[ParticleId, ParticleData]:
    system = Index.new([SystemId(i) for i in system_ids])
    motif = Index.new([MotifId(i) for i in motif_ids])
    inclusion = Index(
        tuple(InclusionId(lab) for lab in system.keys),
        system.indices,
    )
    exclusion = Index.arange(len(system_ids), label=ExclusionId)
    return Table.arange(
        ParticleData(positions, system, motif, inclusion, exclusion),
        label=ParticleId,
    )


def create_test_edges(
    particles: Table[ParticleId, ParticleData], indices: list[list[int]]
) -> Edges:
    """Helper function to create Edges with proper shifts for testing."""
    if not indices:
        return Edges(
            indices=Index(particles.keys, jnp.zeros((0, 2), dtype=int)),
            shifts=jnp.zeros((0, 1, 3)),
        )
    indices_array = jnp.array(indices)
    n_edges = indices_array.shape[0]
    shifts = jnp.zeros((n_edges, 1, 3))
    return Edges(indices=Index(particles.keys, indices_array), shifts=shifts)


_jit_blocking_spheres_energy = jax.jit(blocking_spheres_energy)


class TestBlockingSpheresEnergy:
    """Test blocking_spheres_energy function."""

    @classmethod
    def setup_class(cls):
        """Set up test data for energy calculations."""
        cls.radii = jnp.array([1.0, 2.0])
        cls.sphere_positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        cls.parameters = BlockingSpheresParameters(
            radii=cls.radii,
            positions=cls.sphere_positions,
            system=Index.new([SystemId(i) for i in [0, 0]]),
            motif=Index.new([MotifId(i) for i in [0, 0]]),
        )
        cls.particle_positions = jnp.array(
            [
                [0.5, 0.0, 0.0],  # Inside first sphere
                [3.0, 0.0, 0.0],  # Between spheres
                [4.0, 0.0, 0.0],  # Inside second sphere
                [10.0, 0.0, 0.0],  # Outside both spheres
            ]
        )
        cls.particles = _make_particles(
            cls.particle_positions, [0, 0, 0, 0], [0, 0, 0, 0]
        )

    def test_energy_scenarios(self):
        """Merged: inside + outside + boundary + multiple + no_edges + with_unit_cell."""
        energy_fn = _jit_blocking_spheres_energy

        # Inside sphere -> infinite energy
        edges_in = create_test_edges(self.particles, [[0, 0]])
        inp_in = BlockingSpheresPotentialInput(
            parameters=self.parameters,
            particles=self.particles,
            unitcell=None,
            edges=edges_in,
        )
        result_in = energy_fn(inp_in)
        assert jnp.isinf(result_in.data.data[0])

        # Outside sphere -> zero energy
        edges_out = create_test_edges(self.particles, [[3, 0]])
        inp_out = BlockingSpheresPotentialInput(
            parameters=self.parameters,
            particles=self.particles,
            unitcell=None,
            edges=edges_out,
        )
        result_out = energy_fn(inp_out)
        assert result_out.data.data[0] == 0.0

        # On boundary -> finite energy
        boundary_particles = _make_particles(jnp.array([[1.0, 0.0, 0.0]]), [0], [0])
        edges_bnd = create_test_edges(boundary_particles, [[0, 0]])
        inp_bnd = BlockingSpheresPotentialInput(
            parameters=self.parameters,
            particles=boundary_particles,
            unitcell=None,
            edges=edges_bnd,
        )
        result_bnd = energy_fn(inp_bnd)
        assert jnp.isfinite(result_bnd.data.data[0])

        # Multiple particles + spheres -> inf if any overlap
        edges_multi = create_test_edges(
            self.particles,
            [[0, 0], [1, 0], [2, 1], [3, 1]],
        )
        inp_multi = BlockingSpheresPotentialInput(
            parameters=self.parameters,
            particles=self.particles,
            unitcell=None,
            edges=edges_multi,
        )
        result_multi = energy_fn(inp_multi)
        assert jnp.isinf(result_multi.data.data[0])

        # No edges -> zero energy
        edges_none = create_test_edges(self.particles, [])
        inp_none = BlockingSpheresPotentialInput(
            parameters=self.parameters,
            particles=self.particles,
            unitcell=None,
            edges=edges_none,
        )
        result_none = energy_fn(inp_none)
        assert result_none.data.data[0] == 0.0

        # With unit cell (still inside) -> infinite
        uc_particles = _make_particles(jnp.array([[0.5, 0.0, 0.0]]), [0], [0])
        edges_uc = create_test_edges(uc_particles, [[0, 0]])
        inp_uc = BlockingSpheresPotentialInput(
            parameters=self.parameters,
            particles=uc_particles,
            unitcell=None,
            edges=edges_uc,
        )
        result_uc = energy_fn(inp_uc)
        assert jnp.isinf(result_uc.data.data[0])

    def test_edge_cases(self):
        """Merged: zero_radius + negative_radius + very_large_distances."""
        # Different shapes from class fixture, so use raw function
        energy_fn = blocking_spheres_energy

        # Zero radius sphere
        params_zero = BlockingSpheresParameters(
            radii=jnp.array([0.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        particles_z = _make_particles(jnp.array([[0.0, 0.0, 0.0]]), [0], [0])
        edges_z = create_test_edges(particles_z, [[0, 0]])
        inp_z = BlockingSpheresPotentialInput(
            parameters=params_zero,
            particles=particles_z,
            unitcell=None,
            edges=edges_z,
        )
        result_z = energy_fn(inp_z)
        assert jnp.isfinite(result_z.data.data[0])

        # Negative radius sphere -> never blocks
        params_neg = BlockingSpheresParameters(
            radii=jnp.array([-1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        particles_n = _make_particles(jnp.array([[0.0, 0.0, 0.0]]), [0], [0])
        edges_n = create_test_edges(particles_n, [[0, 0]])
        inp_n = BlockingSpheresPotentialInput(
            parameters=params_neg,
            particles=particles_n,
            unitcell=None,
            edges=edges_n,
        )
        result_n = energy_fn(inp_n)
        assert jnp.isfinite(result_n.data.data[0])
        assert result_n.data.data[0] == 0.0

        # Very large distances -> zero energy
        params_far = BlockingSpheresParameters(
            radii=jnp.array([1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        particles_f = _make_particles(jnp.array([[1e6, 0.0, 0.0]]), [0], [0])
        edges_f = create_test_edges(particles_f, [[0, 0]])
        inp_f = BlockingSpheresPotentialInput(
            parameters=params_far,
            particles=particles_f,
            unitcell=None,
            edges=edges_f,
        )
        result_f = energy_fn(inp_f)
        assert jnp.isfinite(result_f.data.data[0])
        assert result_f.data.data[0] == 0.0

    def test_jit_gradient(self):
        """Test that gradients can be computed through JIT compiled function."""

        def energy_wrapper(positions):
            parameters = BlockingSpheresParameters(
                radii=jnp.array([1.0]),
                positions=jnp.array([[0.0, 0.0, 0.0]]),
                system=Index.new([SystemId(0)]),
                motif=Index.new([MotifId(0)]),
            )
            particles = _make_particles(positions, [0], [0])
            edges = create_test_edges(particles, [[0, 0]])
            inp = BlockingSpheresPotentialInput(
                parameters=parameters,
                particles=particles,
                unitcell=None,
                edges=edges,
            )
            return blocking_spheres_energy(inp).data.data[0]

        positions = jnp.array([[2.0, 0.0, 0.0]])
        grad_fn = jax.jit(jax.grad(energy_wrapper))
        gradient = grad_fn(positions)
        assert jnp.all(jnp.isfinite(gradient))

    def test_multiple_batches(self):
        """Test energy calculation with multiple batches."""
        parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0, 1.5]),
            positions=jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            system=Index.new([SystemId(i) for i in [0, 1]]),
            motif=Index.new([MotifId(i) for i in [0, 0]]),
        )
        particles = _make_particles(
            jnp.array([[0.5, 0.0, 0.0], [6.0, 0.0, 0.0]]),
            [0, 1],
            [0, 0],
        )
        edges = create_test_edges(particles, [[0, 0], [1, 1]])
        inp = BlockingSpheresPotentialInput(
            parameters=parameters,
            particles=particles,
            unitcell=None,
            edges=edges,
        )
        result = blocking_spheres_energy(inp)
        assert jnp.isinf(result.data.data[0])
        assert jnp.isfinite(result.data.data[1])
