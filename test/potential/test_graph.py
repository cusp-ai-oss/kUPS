# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.data import WithIndices
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import view
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    Edges,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.unitcell import TriclinicUnitCell, UnitCell
from kups.core.utils.jax import dataclass, key_chain
from kups.potential.common.graph import (
    EdgeSetGraphConstructor,
    FullGraphSumComposer,
    GraphPotentialInput,
    HyperGraph,
    LocalGraphSumComposer,
    PointCloud,
    PointCloudConstructor,
    RadiusGraphConstructor,
    UpdatedEdges,
)

from ..clear_cache import clear_cache  # noqa: F401


@dataclass
class _PointData:
    positions: jax.Array
    species: jax.Array
    system: Index
    inclusion: Index
    exclusion: Index


@dataclass
class _SystemData:
    unitcell: UnitCell
    cutoff: jax.Array


def _make_particles(
    positions: jax.Array,
    species: jax.Array,
    system_ids: jax.Array,
    *,
    exclusion_ids: jax.Array | None = None,
) -> Table[ParticleId, _PointData]:
    n_sys = max(1, int(system_ids.max()) + 1) if system_ids.size > 0 else 0
    system = Index.integer(system_ids, n=n_sys, label=SystemId)
    inclusion = Index(system.keys, system.indices, _cls=SystemId)
    if exclusion_ids is not None:
        exclusion = Index.integer(exclusion_ids, label=ParticleId)
    elif len(positions) > 0:
        exclusion = Index.integer(jnp.arange(len(positions)), label=ParticleId)
    else:
        exclusion = Index((), jnp.zeros((0,), dtype=int), _cls=ParticleId)
    data = _PointData(positions, species, system, inclusion, exclusion)
    return Table.arange(data, label=ParticleId)


def _make_systems(
    lattice_vectors: jax.Array, cutoff: jax.Array
) -> Table[SystemId, _SystemData]:
    unitcell = TriclinicUnitCell.from_matrix(lattice_vectors)
    data = _SystemData(unitcell, cutoff)
    return Table.arange(data, label=SystemId)


def _make_edges(
    particles: Table[ParticleId, _PointData],
    indices: jax.Array,
    shifts: jax.Array,
) -> Edges[int]:
    return Edges(
        indices=Index(particles.keys, indices),
        shifts=shifts,
    )


class TestPointCloud:
    def test_creation(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        species = jnp.array([1, 2])
        particles = _make_particles(positions, species, jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))

        pc = PointCloud(particles, systems)
        assert pc.batch_size == 1
        npt.assert_array_equal(pc.particles.data.positions, positions)

    def test_multiple_batches(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        species = jnp.array([1, 2, 1])
        particles = _make_particles(positions, species, jnp.array([0, 0, 1]))
        systems = _make_systems(jnp.eye(3)[None].repeat(2, axis=0) * 5.0, jnp.ones(2))

        pc = PointCloud(particles, systems)
        assert pc.batch_size == 2

    def test_empty(self):
        positions = jnp.empty((0, 3))
        species = jnp.empty((0,), dtype=int)
        particles = _make_particles(positions, species, jnp.empty((0,), dtype=int))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))

        pc = PointCloud(particles, systems)
        assert pc.batch_size == 0  # no particles → no system labels
        assert pc.particles.data.positions.shape == (0, 3)


class TestHyperGraph:
    def test_creation(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        species = jnp.array([1, 2, 1])
        particles = _make_particles(positions, species, jnp.array([0, 0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(
            particles,
            jnp.array([[0, 1], [1, 2]]),
            jnp.array([[[0, 0, 0]], [[1, 0, 0]]]),
        )

        hg = HyperGraph(particles, systems, edges)
        assert hg.batch_size == 1
        assert hg.edges.degree == 2
        npt.assert_array_equal(hg.edge_offsets, edges.shifts)

    def test_edge_shifts(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        particles = _make_particles(
            positions, jnp.array([1, 2, 1]), jnp.array([0, 0, 0])
        )
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(
            particles,
            jnp.array([[0, 1], [1, 2]]),
            jnp.array([[[0, 0, 0]], [[1, 0, 0]]]),
        )

        hg = HyperGraph(particles, systems, edges)
        edge_shifts = hg.edge_shifts
        assert edge_shifts.shape == (2, 1, 3)

        # Edge 0→1: no shift
        npt.assert_allclose(edge_shifts[0, 0], positions[1] - positions[0])
        # Edge 1→2: with unit cell shift [5, 0, 0]
        npt.assert_allclose(
            edge_shifts[1, 0], positions[2] - positions[1] + jnp.array([5.0, 0.0, 0.0])
        )

    def test_edge_batch_mask(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        particles = _make_particles(
            positions, jnp.array([1, 2, 1]), jnp.array([0, 0, 1])
        )
        systems = _make_systems(jnp.eye(3)[None].repeat(2, axis=0) * 5.0, jnp.ones(2))
        edges = _make_edges(
            particles,
            jnp.array([[0, 1], [2, 0]]),
            jnp.array([[[0, 0, 0]], [[0, 0, 0]]]),
        )

        hg = HyperGraph(particles, systems, edges)
        batch_mask = hg.edge_batch_mask
        assert isinstance(batch_mask, Index)
        npt.assert_array_equal(batch_mask.value, [0, 1])


class TestHyperGraphSorted:
    def _make_graph(self, positions, species, system_ids, edge_indices, edge_shifts):
        particles = _make_particles(positions, species, system_ids)
        n_sys = int(system_ids.max()) + 1
        systems = _make_systems(
            jnp.eye(3)[None].repeat(n_sys, axis=0) * 10.0, jnp.ones(n_sys)
        )
        edges = _make_edges(particles, edge_indices, edge_shifts)
        return HyperGraph(particles, systems, edges)

    def test_sorted_already_sorted(self):
        hg = self._make_graph(
            positions=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            species=jnp.array([1, 2, 3]),
            system_ids=jnp.array([0, 0, 1]),
            edge_indices=jnp.array([[0, 1], [1, 2]]),
            edge_shifts=jnp.array([[[0, 0, 0]], [[0, 0, 0]]]),
        )
        s = hg.sorted_by_system()
        npt.assert_array_equal(s.particles.data.positions, hg.particles.data.positions)
        npt.assert_array_equal(s.particles.data.species, hg.particles.data.species)
        npt.assert_array_equal(
            s.particles.data.system.indices, hg.particles.data.system.indices
        )
        npt.assert_array_equal(s.edges.indices.indices, hg.edges.indices.indices)
        npt.assert_array_equal(s.edges.shifts, hg.edges.shifts)

    def test_sorted_basic(self):
        # Particles in unsorted system order: [1, 0, 1, 0]
        hg = self._make_graph(
            positions=jnp.array(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            ),
            species=jnp.array([10, 20, 30, 40]),
            system_ids=jnp.array([1, 0, 1, 0]),
            edge_indices=jnp.array([[0, 1], [2, 3]]),
            edge_shifts=jnp.array([[[0, 0, 0]], [[1, 0, 0]]]),
        )
        s = hg.sorted_by_system()
        # argsort([1,0,1,0], stable) = [1,3,0,2]; inverse = [2,0,3,1]
        npt.assert_array_equal(s.particles.data.system.indices, [0, 0, 1, 1])
        npt.assert_array_equal(
            s.particles.data.positions,
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        )
        npt.assert_array_equal(s.particles.data.species, [20, 40, 10, 30])
        # Edge [0,1] → [2,0], edge [2,3] → [3,1]
        npt.assert_array_equal(s.edges.indices.indices, [[2, 0], [3, 1]])

    def test_sorted_preserves_edge_shifts(self):
        shifts = jnp.array([[[1, 0, 0]], [[0, 1, 0]]])
        hg = self._make_graph(
            positions=jnp.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
            species=jnp.array([1, 2, 3, 4]),
            system_ids=jnp.array([1, 0, 1, 0]),
            edge_indices=jnp.array([[0, 1], [2, 3]]),
            edge_shifts=shifts,
        )
        npt.assert_array_equal(hg.sorted_by_system().edges.shifts, shifts)

    def test_sorted_edge_vectors_preserved(self):
        hg = self._make_graph(
            positions=jnp.array(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            ),
            species=jnp.array([10, 20, 30, 40]),
            system_ids=jnp.array([1, 0, 1, 0]),
            edge_indices=jnp.array([[0, 1], [2, 3]]),
            edge_shifts=jnp.array([[[1, 0, 0]], [[0, 1, 0]]]),
        )
        original_vecs = hg.edge_shifts
        sorted_vecs = hg.sorted_by_system().edge_shifts
        npt.assert_allclose(sorted_vecs, original_vecs)

    def test_sort_edges(self):
        # Edges: first edge connects particles in system 1, second in system 0
        # With sort_edges=True, edges should be reordered by system
        hg = self._make_graph(
            positions=jnp.array(
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
            ),
            species=jnp.array([10, 20, 30, 40]),
            system_ids=jnp.array([1, 0, 1, 0]),
            edge_indices=jnp.array([[0, 2], [1, 3]]),
            edge_shifts=jnp.array([[[1, 0, 0]], [[0, 1, 0]]]),
        )
        s = hg.sorted_by_system(sort_edges=True)
        # After particle sort: system indices = [0, 0, 1, 1]
        # Particle sort_order = [1, 3, 0, 2], inverse = [2, 0, 3, 1]
        # Edge [0,2] → [2,3] (system 1), edge [1,3] → [0,1] (system 0)
        # Edge systems: [1, 0] → after edge sort: [0, 1]
        # So edge [0,1] (sys 0) comes first, then [2,3] (sys 1)
        npt.assert_array_equal(s.edges.indices.indices, [[0, 1], [2, 3]])
        npt.assert_array_equal(s.edges.shifts, [[[0, 1, 0]], [[1, 0, 0]]])


class TestEdgeSetGraphConstructor:
    def test_basic(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        species = jnp.array([1, 2])
        particles = _make_particles(positions, species, jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(particles, jnp.array([[0, 1]]), jnp.array([[[0, 0, 0]]]))

        state = {"particles": particles, "systems": systems, "edges": edges}
        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=None,
        )

        graph = constructor(state, None)
        assert isinstance(graph, HyperGraph)
        assert graph.edges.degree == 2
        npt.assert_array_equal(graph.particles.data.positions, positions)

    @pytest.mark.parametrize(
        "n_particles,n_graphs,n_edges,degree",
        [
            (10, 1, 5, 2),
            (10, 1, 5, 3),
            (10, 2, 5, 2),
        ],
    )
    def test_parametrized(self, n_particles, n_graphs, n_edges, degree):
        @jax.vmap
        def get_edges(key):
            return jax.random.choice(key, n_particles, shape=(degree,), replace=False)

        chain = key_chain(jax.random.key(0))
        positions = jax.random.normal(next(chain), shape=(n_particles, 3))
        species = jax.random.randint(next(chain), (n_particles,), 0, 10)
        system_ids = jax.random.randint(next(chain), (n_particles,), 0, n_graphs)

        particles = _make_particles(positions, species, system_ids)
        systems = _make_systems(
            jax.random.normal(next(chain), shape=(n_graphs, 3, 3)),
            jnp.ones(n_graphs),
        )
        edge_indices = get_edges(jax.random.split(next(chain), n_edges))
        edges = _make_edges(
            particles,
            edge_indices,
            jax.random.randint(next(chain), (n_edges, degree - 1, 3), 0, 3),
        )

        state = {"particles": particles, "systems": systems, "edges": edges}
        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=None,
        )

        result = constructor(state, None)
        npt.assert_allclose(result.particles.data.positions, positions)
        npt.assert_allclose(result.edges.indices.indices, edges.indices.indices)
        npt.assert_allclose(result.edges.shifts, edges.shifts)
        assert result.edges.degree == degree


class TestRadiusGraphConstructor:
    def _make_state(self, positions, system_ids, cutoff_val, nnlist):
        species = jnp.zeros(len(positions), dtype=int)
        particles = _make_particles(positions, species, system_ids)
        n_sys = int(system_ids.max()) + 1
        systems = _make_systems(
            jnp.eye(3)[None].repeat(n_sys, axis=0) * 10.0,
            jnp.full(n_sys, cutoff_val),
        )
        return {
            "particles": particles,
            "systems": systems,
            "neighborlist": nnlist,
        }

    def _make_constructor(self):
        return RadiusGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            cutoffs=view(lambda x: x["systems"].map_data(lambda d: d.cutoff)),
            neighborlist=view(lambda x: x["neighborlist"]),
            probe=None,
        )

    def _make_nnlist(self):
        return DenseNearestNeighborList(
            avg_candidates=FixedCapacity(50),
            avg_edges=FixedCapacity(50),
            avg_image_candidates=FixedCapacity(50),
        )

    def test_basic(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        nnlist = self._make_nnlist()
        state = self._make_state(positions, jnp.array([0, 0]), 1.5, nnlist)
        constructor = self._make_constructor()

        graph = constructor(state, None)
        assert isinstance(graph, HyperGraph)
        assert graph.edges.degree == 2
        # Should find bidirectional edges between the two close particles
        valid = graph.edges.indices.indices < 2
        valid_edges = graph.edges.indices.indices[valid.all(-1)]
        edge_set = {tuple(e.tolist()) for e in valid_edges}
        assert (0, 1) in edge_set and (1, 0) in edge_set

    def test_cutoff_filters(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        nnlist = self._make_nnlist()
        # Cutoff too small — no edges
        state = self._make_state(positions, jnp.array([0, 0]), 0.1, nnlist)
        graph = self._make_constructor()(state, None)
        valid = graph.edges.indices.indices < 2
        assert not valid.all(-1).any()

    def test_multiple_systems(self):
        # 2 systems, 3 particles each, only within-system edges
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        )
        system_ids = jnp.array([0, 0, 0, 1, 1, 1])
        nnlist = self._make_nnlist()
        state = self._make_state(positions, system_ids, 1.5, nnlist)
        graph = self._make_constructor()(state, None)
        assert graph.batch_size == 2


class TestPointCloudConstructor:
    def test_basic(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        species = jnp.array([1, 2])
        particles = _make_particles(positions, species, jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))

        state = {"particles": particles, "systems": systems}
        constructor = PointCloudConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            probe_particles=None,
        )

        graph = constructor(state, None)
        assert isinstance(graph, HyperGraph)
        assert graph.edges.degree == 0
        assert len(graph.edges) == 0
        npt.assert_array_equal(graph.particles.data.positions, positions)
        assert graph.batch_size == 1

    def test_multiple_systems(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        species = jnp.array([1, 2, 1])
        particles = _make_particles(positions, species, jnp.array([0, 0, 1]))
        systems = _make_systems(jnp.eye(3)[None].repeat(2, axis=0) * 5.0, jnp.ones(2))

        state = {"particles": particles, "systems": systems}
        constructor = PointCloudConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            probe_particles=None,
        )

        graph = constructor(state, None)
        assert graph.batch_size == 2
        assert graph.edges.degree == 0


class TestGraphPotentialInput:
    def test_named_tuple(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(positions, jnp.array([1, 2]), jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(particles, jnp.array([[0, 1]]), jnp.array([[[0, 0, 0]]]))

        graph = HyperGraph(particles, systems, edges)
        params = {"sigma": 1.0, "epsilon": 0.5}
        inp = GraphPotentialInput(params, graph)

        assert inp.parameters is params
        assert inp.graph is graph
        assert isinstance(inp, tuple)


class TestLocalGraphSumComposer:
    def _make_composer(self):
        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=None,
        )
        return LocalGraphSumComposer(
            graph_constructor=constructor,
            parameter_view=view(lambda x: x["params"]),
        )

    def _make_state(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(positions, jnp.array([1, 2]), jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(particles, jnp.array([[0, 1]]), jnp.array([[[0, 0, 0]]]))
        return {
            "particles": particles,
            "systems": systems,
            "edges": edges,
            "params": {"sigma": 1.0},
        }

    def test_no_patch(self):
        composer = self._make_composer()
        state = self._make_state()
        result = composer(state, None)
        assert len(result) == 1
        assert result[0].weight == 1
        assert isinstance(result[0].inp, GraphPotentialInput)
        assert result[0].inp.parameters == {"sigma": 1.0}


class TestFullGraphSumComposer:
    def _make_composer(self):
        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=None,
        )
        return FullGraphSumComposer(
            graph_constructor=constructor,
            parameter_view=view(lambda x: x["params"]),
        )

    def _make_state(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(positions, jnp.array([1, 2]), jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(particles, jnp.array([[0, 1]]), jnp.array([[[0, 0, 0]]]))
        return {
            "particles": particles,
            "systems": systems,
            "edges": edges,
            "params": {"epsilon": 0.5},
        }

    def test_no_patch(self):
        composer = self._make_composer()
        state = self._make_state()
        result = composer(state, None)
        assert len(result) == 1
        assert result[0].weight == 1
        assert result[0].inp.parameters == {"epsilon": 0.5}


# --- Patch / incremental update tests ---


@dataclass
class _SimplePointData:
    """Minimal point data for patch tests (positions + system)."""

    positions: jax.Array
    system: Index


def _make_simple_particles(
    positions: jax.Array, system_ids: jax.Array
) -> Table[int, _SimplePointData]:
    system = Index.new(system_ids)
    return Table.arange(_SimplePointData(positions, system))


@dataclass
class _SimplePatch:
    """A patch that just returns stored state regardless of accept mask."""

    new_state: dict

    def __call__(self, state, accept):
        return self.new_state


def _make_probe_update(
    particles: Table[ParticleId, _PointData],
    particle_idx: int,
    new_positions: jax.Array,
) -> WithIndices[ParticleId, _PointData]:
    """Build a probe particle update with labels matching the original particles."""
    d = particles.data
    idx_arr = jnp.array([particle_idx])
    updated_data = _PointData(
        new_positions,
        d.species[idx_arr],
        Index(d.system.keys, d.system.indices[idx_arr], _cls=d.system.cls),
        Index(d.inclusion.keys, d.inclusion.indices[idx_arr], _cls=d.inclusion.cls),
        Index(d.exclusion.keys, d.exclusion.indices[idx_arr], _cls=d.exclusion.cls),
    )
    return WithIndices(Index(particles.keys, idx_arr), updated_data)


class TestPointCloudConstructorWithPatch:
    def test_fallback_no_probe(self):
        """patch + no probe → applies patch and recurses with patch=None."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        new_positions = jnp.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])
        particles = _make_particles(positions, jnp.array([1, 2]), jnp.array([0, 0]))
        new_particles = _make_particles(
            new_positions, jnp.array([1, 2]), jnp.array([0, 0])
        )
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))

        state = {"particles": particles, "systems": systems}
        new_state = {"particles": new_particles, "systems": systems}

        constructor = PointCloudConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            probe_particles=None,
        )

        graph = constructor(state, _SimplePatch(new_state))
        # Should use the new_state's particles
        npt.assert_allclose(graph.particles.data.positions, new_positions)

    def test_with_probe(self):
        """patch + probe → applies incremental particle update."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        particles = _make_particles(
            positions, jnp.array([1, 2, 3]), jnp.array([0, 0, 0])
        )
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        probe_result = _make_probe_update(particles, 1, jnp.array([[5.0, 5.0, 5.0]]))

        state = {"particles": particles, "systems": systems}
        constructor = PointCloudConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            probe_particles=lambda s, p: probe_result,
        )

        graph = constructor(state, _SimplePatch(state), old_graph=False)
        npt.assert_allclose(graph.particles.data.positions[1], [5.0, 5.0, 5.0])
        npt.assert_allclose(graph.particles.data.positions[0], positions[0])
        npt.assert_allclose(graph.particles.data.positions[2], positions[2])

    def test_with_probe_old_graph(self):
        """old_graph=True → returns original particles (no update applied)."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(positions, jnp.array([1, 2]), jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        probe_result = _make_probe_update(particles, 0, jnp.array([[9.0, 9.0, 9.0]]))

        state = {"particles": particles, "systems": systems}
        constructor = PointCloudConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            probe_particles=lambda s, p: probe_result,
        )

        graph = constructor(state, _SimplePatch(state), old_graph=True)
        npt.assert_allclose(graph.particles.data.positions, positions)


class TestEdgeSetGraphConstructorWithPatch:
    def _make_scene(self):
        """3 particles, 2 edges: (0,1) and (1,2)."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        species = jnp.array([1, 2, 3])
        particles = _make_particles(positions, species, jnp.array([0, 0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(
            particles,
            jnp.array([[0, 1], [1, 2]]),
            jnp.array([[[0, 0, 0]], [[0, 0, 0]]]),
        )
        return particles, systems, edges

    def test_fallback_no_probe(self):
        """patch + no probe → applies patch and recurses."""
        particles, systems, edges = self._make_scene()
        new_positions = jnp.array([[9.0, 0.0, 0.0], [8.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        new_particles = _make_particles(
            new_positions, jnp.array([1, 2, 3]), jnp.array([0, 0, 0])
        )

        state = {"particles": particles, "systems": systems, "edges": edges}
        new_state = {"particles": new_particles, "systems": systems, "edges": edges}

        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=None,
        )

        graph = constructor(state, _SimplePatch(new_state))
        npt.assert_allclose(graph.particles.data.positions, new_positions)

    def _make_edge_probe(self, particles, particle_idx, new_pos):
        update = _make_probe_update(particles, particle_idx, new_pos)
        empty_edges = UpdatedEdges(
            indices=jnp.array([], dtype=int),
            edge_data=Edges(
                indices=Index(particles.keys, jnp.zeros((0, 2), dtype=int)),
                shifts=jnp.zeros((0, 1, 3), dtype=int),
            ),
        )

        @dataclass
        class _Probe:
            _p: WithIndices
            _e: UpdatedEdges
            _c: FixedCapacity

            @property
            def particles(self):
                return self._p

            @property
            def edges(self):
                return self._e

            @property
            def capacity(self):
                return self._c

        return _Probe(_p=update, _e=empty_edges, _c=FixedCapacity(10))

    def test_with_probe(self):
        """patch + probe → updates particles and collects affected edges."""
        particles, systems, edges = self._make_scene()
        probe_result = self._make_edge_probe(particles, 1, jnp.array([[5.0, 5.0, 5.0]]))

        state = {"particles": particles, "systems": systems, "edges": edges}
        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=lambda s, p: probe_result,
        )

        graph = constructor(state, _SimplePatch(state), old_graph=False)
        npt.assert_allclose(graph.particles.data.positions[1], [5.0, 5.0, 5.0])
        # Both edges involve particle 1, so both should be in the result
        assert len(graph.edges) > 0

    def test_with_probe_old_graph(self):
        """old_graph=True → particles not updated but affected edges still collected."""
        particles, systems, edges = self._make_scene()
        probe_result = self._make_edge_probe(particles, 1, jnp.array([[5.0, 5.0, 5.0]]))

        state = {"particles": particles, "systems": systems, "edges": edges}
        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=lambda s, p: probe_result,
        )

        graph = constructor(state, _SimplePatch(state), old_graph=True)
        npt.assert_allclose(graph.particles.data.positions, particles.data.positions)


class TestRadiusGraphConstructorWithPatch:
    def _make_nnlist(self):
        return DenseNearestNeighborList(
            avg_candidates=FixedCapacity(50),
            avg_edges=FixedCapacity(50),
            avg_image_candidates=FixedCapacity(50),
        )

    def test_fallback_no_probe(self):
        """patch + no probe → applies patch and recurses."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        new_positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        species = jnp.zeros(2, dtype=int)

        particles = _make_particles(positions, species, jnp.array([0, 0]))
        new_particles = _make_particles(new_positions, species, jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 10.0, jnp.array([1.5]))
        nnlist = self._make_nnlist()

        state = {"particles": particles, "systems": systems, "neighborlist": nnlist}
        new_state = {
            "particles": new_particles,
            "systems": systems,
            "neighborlist": nnlist,
        }

        constructor = RadiusGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            cutoffs=view(lambda x: x["systems"].map_data(lambda d: d.cutoff)),
            neighborlist=view(lambda x: x["neighborlist"]),
            probe=None,
        )

        graph = constructor(state, _SimplePatch(new_state))
        npt.assert_allclose(graph.particles.data.positions, new_positions)

    def _make_radius_probe(self, particles, nnlist, particle_idx, new_pos):
        update = _make_probe_update(particles, particle_idx, new_pos)

        @dataclass
        class _Probe:
            _p: WithIndices
            _nn: DenseNearestNeighborList

            @property
            def particles(self):
                return self._p

            @property
            def neighborlist_after(self):
                return self._nn

            @property
            def neighborlist_before(self):
                return self._nn

        return _Probe(_p=update, _nn=nnlist)

    def test_with_probe(self):
        """patch + probe → updates particle, rebuilds edges from probe's nnlist."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        particles = _make_particles(
            positions, jnp.zeros(3, dtype=int), jnp.array([0, 0, 0])
        )
        systems = _make_systems(jnp.eye(3)[None] * 10.0, jnp.array([1.5]))
        nnlist = self._make_nnlist()
        probe_result = self._make_radius_probe(
            particles, nnlist, 1, jnp.array([[0.5, 0.0, 0.0]])
        )

        state = {"particles": particles, "systems": systems, "neighborlist": nnlist}
        constructor = RadiusGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            cutoffs=view(lambda x: x["systems"].map_data(lambda d: d.cutoff)),
            neighborlist=view(lambda x: x["neighborlist"]),
            probe=lambda s, p: probe_result,
        )

        graph = constructor(state, _SimplePatch(state), old_graph=False)
        npt.assert_allclose(graph.particles.data.positions[1], [0.5, 0.0, 0.0])
        assert graph.edges.degree == 2

    def test_with_probe_old_graph(self):
        """old_graph=True → returns original particle positions."""
        positions = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        particles = _make_particles(
            positions, jnp.zeros(2, dtype=int), jnp.array([0, 0])
        )
        systems = _make_systems(jnp.eye(3)[None] * 10.0, jnp.array([1.5]))
        nnlist = self._make_nnlist()
        probe_result = self._make_radius_probe(
            particles, nnlist, 0, jnp.array([[9.0, 9.0, 9.0]])
        )

        state = {"particles": particles, "systems": systems, "neighborlist": nnlist}
        constructor = RadiusGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            cutoffs=view(lambda x: x["systems"].map_data(lambda d: d.cutoff)),
            neighborlist=view(lambda x: x["neighborlist"]),
            probe=lambda s, p: probe_result,
        )

        graph = constructor(state, _SimplePatch(state), old_graph=True)
        npt.assert_allclose(graph.particles.data.positions, positions)


class TestLocalGraphSumComposerWithPatch:
    def test_with_patch(self):
        """patch + probe → old_graph(-1) + new_graph(+1) with add_previous_total."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        particles = _make_particles(positions, jnp.array([1, 2]), jnp.array([0, 0]))
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        probe_result = _make_probe_update(particles, 1, jnp.array([[5.0, 0.0, 0.0]]))

        state = {"particles": particles, "systems": systems, "params": {"sigma": 1.0}}

        constructor = PointCloudConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            probe_particles=lambda s, p: probe_result,
        )
        composer = LocalGraphSumComposer(
            graph_constructor=constructor,
            parameter_view=view(lambda x: x["params"]),
        )

        result = composer(state, _SimplePatch(state))
        assert len(result) == 2
        assert result.add_previous_total is True
        assert result[0].weight == -1
        assert result[1].weight == 1
        # old graph has original positions
        npt.assert_allclose(result[0].inp.graph.particles.data.positions, positions)
        # new graph has updated particle 1
        npt.assert_allclose(
            result[1].inp.graph.particles.data.positions[1], [5.0, 0.0, 0.0]
        )


class TestFullGraphSumComposerWithPatch:
    def test_with_patch(self):
        """patch → applies patch to state, then builds single full graph."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        new_positions = jnp.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])
        particles = _make_particles(positions, jnp.array([1, 2]), jnp.array([0, 0]))
        new_particles = _make_particles(
            new_positions, jnp.array([1, 2]), jnp.array([0, 0])
        )
        systems = _make_systems(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        edges = _make_edges(particles, jnp.array([[0, 1]]), jnp.array([[[0, 0, 0]]]))

        state = {
            "particles": particles,
            "systems": systems,
            "edges": edges,
            "params": {"epsilon": 0.5},
        }
        new_state = {
            "particles": new_particles,
            "systems": systems,
            "edges": edges,
            "params": {"epsilon": 0.5},
        }

        constructor = EdgeSetGraphConstructor(
            particles=view(lambda x: x["particles"]),
            systems=view(lambda x: x["systems"]),
            edges=view(lambda x: x["edges"]),
            probe=None,
        )
        composer = FullGraphSumComposer(
            graph_constructor=constructor,
            parameter_view=view(lambda x: x["params"]),
        )

        result = composer(state, _SimplePatch(new_state))
        assert len(result) == 1
        assert result.add_previous_total is False
        npt.assert_allclose(result[0].inp.graph.particles.data.positions, new_positions)
