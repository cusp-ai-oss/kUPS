# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for blocking spheres potential."""

import jax
import jax.numpy as jnp

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist import Edges
from kups.core.typing import GroupId, MotifId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.blocking import (
    BlockingSpheresParameters,
    BlockingSpheresPotentialInput,
    blocking_spheres_energy,
)


@dataclass
class ParticleData:
    """Particle data with positions, system, and group index."""

    positions: jax.Array
    system: Index[SystemId]
    group: Index[GroupId]


@dataclass
class GroupData:
    """Group data carrying motif assignment."""

    motif: Index[MotifId]


def _make_particles(
    positions: jax.Array,
    system_ids: list[int],
    group_ids: list[int],
) -> Table[ParticleId, ParticleData]:
    system = Index.new([SystemId(i) for i in system_ids])
    group = Index.new([GroupId(i) for i in group_ids])
    return Table.arange(ParticleData(positions, system, group), label=ParticleId)


def _make_groups(motif_ids: list[int]) -> Table[GroupId, GroupData]:
    motif = Index.new([MotifId(i) for i in motif_ids])
    return Table.arange(GroupData(motif), label=GroupId)


def _make_cell(n_systems: int, box_size: float = 1000.0):
    """Large cubic per-system cells, so wrap is effectively identity."""
    cells = PeriodicCell(
        TriclinicFrame.from_matrix(
            jnp.broadcast_to(jnp.eye(3) * box_size, (n_systems, 3, 3))
        )
    )
    return Table(tuple(SystemId(i) for i in range(n_systems)), cells)


def create_test_edges(
    particles: Table[ParticleId, ParticleData], indices: list[list[int]]
) -> Edges:
    """Build Edges with zero shifts from a list of [particle_idx, sphere_idx] pairs."""
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
        """Set up test data: two spheres in system 0, motif 0."""
        cls.parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0, 2.0]),
            positions=jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0), SystemId(0)]),
            motif=Index.new([MotifId(0), MotifId(0)]),
        )
        cls.particle_positions = jnp.array(
            [
                [0.5, 0.0, 0.0],  # Inside first sphere
                [3.0, 0.0, 0.0],  # Between spheres
                [4.0, 0.0, 0.0],  # Inside second sphere
                [10.0, 0.0, 0.0],  # Outside both spheres
            ]
        )
        # 4 particles, each in its own group; all groups have motif 0.
        cls.particles = _make_particles(
            cls.particle_positions, [0, 0, 0, 0], [0, 1, 2, 3]
        )
        cls.groups = _make_groups([0, 0, 0, 0])
        cls.cell = _make_cell(1)

    def _make_input(self, particles, edges, groups=None, cell=None):
        return BlockingSpheresPotentialInput(
            parameters=self.parameters,
            particles=particles,
            groups=groups if groups is not None else self.groups,
            cell=cell if cell is not None else self.cell,
            edges=edges,
        )

    def test_energy_scenarios(self):
        """Merged: inside + outside + boundary + multiple + no_edges."""
        energy_fn = _jit_blocking_spheres_energy

        # Inside sphere -> infinite energy
        result_in = energy_fn(
            self._make_input(
                self.particles, create_test_edges(self.particles, [[0, 0]])
            )
        )
        assert jnp.isinf(result_in.data.data[0])

        # Outside sphere -> zero energy
        result_out = energy_fn(
            self._make_input(
                self.particles, create_test_edges(self.particles, [[3, 0]])
            )
        )
        assert result_out.data.data[0] == 0.0

        # On boundary -> finite energy (dist == radius is not strictly less)
        boundary_particles = _make_particles(jnp.array([[1.0, 0.0, 0.0]]), [0], [0])
        result_bnd = energy_fn(
            self._make_input(
                boundary_particles,
                create_test_edges(boundary_particles, [[0, 0]]),
                groups=_make_groups([0]),
            )
        )
        assert jnp.isfinite(result_bnd.data.data[0])

        # Multiple particles + spheres -> inf if any overlap
        result_multi = energy_fn(
            self._make_input(
                self.particles,
                create_test_edges(self.particles, [[0, 0], [1, 0], [2, 1], [3, 1]]),
            )
        )
        assert jnp.isinf(result_multi.data.data[0])

        # No edges -> zero energy
        result_none = energy_fn(
            self._make_input(self.particles, create_test_edges(self.particles, []))
        )
        assert result_none.data.data[0] == 0.0

    def test_move_into_sphere_then_out(self):
        """Position-driven energy: a particle moving into a sphere yields inf,
        moving out yields 0. This is what the MCMC layer relies on to accept
        or reject proposed moves: each call evaluates the post-move energy.
        """
        parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        groups = _make_groups([0])
        cell = _make_cell(1)

        def energy_at(pos):
            particles = _make_particles(jnp.array([pos]), [0], [0])
            return blocking_spheres_energy(
                BlockingSpheresPotentialInput(
                    parameters=parameters,
                    particles=particles,
                    groups=groups,
                    cell=cell,
                    edges=create_test_edges(particles, [[0, 0]]),
                )
            ).data.data[0]

        # Move into the sphere -> inf (move would be rejected by MCMC).
        assert jnp.isinf(energy_at([0.3, 0.0, 0.0]))
        # Move back out -> 0 (move would be accepted).
        assert energy_at([2.5, 0.0, 0.0]) == 0.0
        # Move into a different point inside the sphere -> inf.
        assert jnp.isinf(energy_at([0.0, 0.5, 0.0]))

    def test_pbc_wrapping_blocks_across_boundary(self):
        """A particle on the far side of the box and a sphere at the origin
        are within blocking range when periodic boundary conditions wrap them
        together.
        """
        L = 10.0
        parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        # Particle at L - 0.4: real-space distance ~9.6, but PBC distance ~0.4 < 1.0.
        particles = _make_particles(jnp.array([[L - 0.4, 0.0, 0.0]]), [0], [0])
        cell = Table(
            (SystemId(0),),
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * L)),
        )
        result = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=parameters,
                particles=particles,
                groups=_make_groups([0]),
                cell=cell,
                edges=create_test_edges(particles, [[0, 0]]),
            )
        )
        assert jnp.isinf(result.data.data[0])

    def test_motif_filtering(self):
        """A sphere only blocks particles whose group's motif matches the sphere's motif."""
        # Two spheres, one per motif; both centered at origin so geometry alone wouldn't discriminate.
        parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0, 1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0), SystemId(0)]),
            motif=Index.new([MotifId(0), MotifId(1)]),
        )
        particles = _make_particles(jnp.array([[0.5, 0.0, 0.0]]), [0], [0])
        # The particle's group has motif 1 -> sphere 0 (motif 0) does NOT block,
        # sphere 1 (motif 1) DOES block.
        groups = _make_groups([1])
        cell = _make_cell(1)

        # Match: motif 1 sphere blocks
        result_match = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=parameters,
                particles=particles,
                groups=groups,
                cell=cell,
                edges=create_test_edges(particles, [[0, 1]]),
            )
        )
        assert jnp.isinf(result_match.data.data[0])

        # Mismatch: motif 0 sphere does not block
        result_miss = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=parameters,
                particles=particles,
                groups=groups,
                cell=cell,
                edges=create_test_edges(particles, [[0, 0]]),
            )
        )
        assert result_miss.data.data[0] == 0.0

    def test_motif_outside_sphere_keyspace_does_not_block(self):
        """A particle whose group's motif is absent from the sphere motif keyspace
        is never blocked, regardless of position. Without this, a host config
        whose adsorbates include species without spheres would crash the energy
        function.
        """
        # Sphere keyspace covers only motif 0.
        parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        # Particle is geometrically inside, but its group has motif 1
        # (not present in any sphere) -> must not block.
        particles = _make_particles(jnp.array([[0.5, 0.0, 0.0]]), [0], [0])
        groups = _make_groups([1])
        result = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=parameters,
                particles=particles,
                groups=groups,
                cell=_make_cell(1),
                edges=create_test_edges(particles, [[0, 0]]),
            )
        )
        assert result.data.data[0] == 0.0

    def test_oob_group_sentinel_does_not_block(self):
        """Buffer-slot semantics: a particle whose group is the OOB sentinel
        (e.g. an empty MCMC buffer slot) is not blocked, even when its position
        falls inside a sphere matching another particle's motif.
        """
        parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        # Two groups so the OOB sentinel is a meaningful out-of-vocab index;
        # both groups carry motif 0 so a clamping bug would silently block.
        groups = Table.arange(
            GroupData(Index.new([MotifId(0), MotifId(0)])),
            label=GroupId,
        )
        # Particle 0's group is masked out -> OOB sentinel; particle 1 is real.
        group = Index.new([GroupId(0), GroupId(1)]).apply_mask(jnp.array([False, True]))
        positions = jnp.array([[0.5, 0.0, 0.0], [10.0, 0.0, 0.0]])
        particles = Table.arange(
            ParticleData(positions, Index.new([SystemId(0), SystemId(0)]), group),
            label=ParticleId,
        )
        # Edge connects the OOB-group particle to the sphere it sits inside.
        result = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=parameters,
                particles=particles,
                groups=groups,
                cell=_make_cell(1),
                edges=create_test_edges(particles, [[0, 0]]),
            )
        )
        assert result.data.data[0] == 0.0

    def test_edge_cases(self):
        """Merged: zero_radius + negative_radius + very_large_distances."""
        # Zero radius sphere -> dist (=0) is not strictly less than 0 -> finite
        params_zero = BlockingSpheresParameters(
            radii=jnp.array([0.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        particles_z = _make_particles(jnp.array([[0.0, 0.0, 0.0]]), [0], [0])
        groups_z = _make_groups([0])
        result_z = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=params_zero,
                particles=particles_z,
                groups=groups_z,
                cell=_make_cell(1),
                edges=create_test_edges(particles_z, [[0, 0]]),
            )
        )
        assert jnp.isfinite(result_z.data.data[0])

        # Negative radius sphere -> never blocks
        params_neg = BlockingSpheresParameters(
            radii=jnp.array([-1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        result_n = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=params_neg,
                particles=particles_z,
                groups=groups_z,
                cell=_make_cell(1),
                edges=create_test_edges(particles_z, [[0, 0]]),
            )
        )
        assert result_n.data.data[0] == 0.0

        # Very large distance -> zero energy
        params_far = BlockingSpheresParameters(
            radii=jnp.array([1.0]),
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0)]),
            motif=Index.new([MotifId(0)]),
        )
        particles_f = _make_particles(jnp.array([[1e6, 0.0, 0.0]]), [0], [0])
        result_f = blocking_spheres_energy(
            BlockingSpheresPotentialInput(
                parameters=params_far,
                particles=particles_f,
                groups=_make_groups([0]),
                cell=_make_cell(1, box_size=1e9),
                edges=create_test_edges(particles_f, [[0, 0]]),
            )
        )
        assert result_f.data.data[0] == 0.0

    def test_jit_gradient(self):
        """Gradients can flow through the JIT-compiled energy."""

        def energy_wrapper(positions):
            parameters = BlockingSpheresParameters(
                radii=jnp.array([1.0]),
                positions=jnp.array([[0.0, 0.0, 0.0]]),
                system=Index.new([SystemId(0)]),
                motif=Index.new([MotifId(0)]),
            )
            particles = _make_particles(positions, [0], [0])
            inp = BlockingSpheresPotentialInput(
                parameters=parameters,
                particles=particles,
                groups=_make_groups([0]),
                cell=_make_cell(1),
                edges=create_test_edges(particles, [[0, 0]]),
            )
            return blocking_spheres_energy(inp).data.data[0]

        gradient = jax.jit(jax.grad(energy_wrapper))(jnp.array([[2.0, 0.0, 0.0]]))
        assert jnp.all(jnp.isfinite(gradient))

    def test_multiple_batches(self):
        """Energy is summed per-system across two systems."""
        parameters = BlockingSpheresParameters(
            radii=jnp.array([1.0, 1.5]),
            positions=jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            system=Index.new([SystemId(0), SystemId(1)]),
            motif=Index.new([MotifId(0), MotifId(0)]),
        )
        particles = _make_particles(
            jnp.array([[0.5, 0.0, 0.0], [6.0, 0.0, 0.0]]),
            [0, 1],
            [0, 1],
        )
        inp = BlockingSpheresPotentialInput(
            parameters=parameters,
            particles=particles,
            groups=_make_groups([0, 0]),
            cell=_make_cell(2),
            edges=create_test_edges(particles, [[0, 0], [1, 1]]),
        )
        result = blocking_spheres_energy(inp)
        assert jnp.isinf(result.data.data[0])
        assert jnp.isfinite(result.data.data[1])


class TestBlockingSpheresParametersFromData:
    """Test BlockingSpheresParameters.from_data."""

    def test_flattens_nested_sequence(self):
        """Nested [system][motif][sphere] sequence flattens with correct assignments."""

        @dataclass
        class _Sphere:
            center: tuple[float, float, float]
            radius: float

        data = [
            [
                [_Sphere((0.0, 0.0, 0.0), 1.0), _Sphere((1.0, 0.0, 0.0), 0.5)],
                [_Sphere((2.0, 0.0, 0.0), 2.0)],
            ],
            [[_Sphere((3.0, 0.0, 0.0), 1.5)]],
        ]
        params = BlockingSpheresParameters.from_data(data)
        assert params.radii.tolist() == [1.0, 0.5, 2.0, 1.5]
        assert params.system.indices.tolist() == [0, 0, 0, 1]
        assert params.motif.indices.tolist() == [0, 0, 1, 0]
        assert params.positions.shape == (4, 3)
