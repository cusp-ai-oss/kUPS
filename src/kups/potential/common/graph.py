# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Graph construction from atomic coordinates for potential evaluation.

This module builds molecular graphs (point clouds, hypergraphs) from
``Indexed`` particle data for potential energy evaluation. Graphs support
periodic boundary conditions, multiple independent systems, and efficient
incremental construction for Monte Carlo via probes.

Key components:

- **[PointCloud][kups.potential.common.graph.PointCloud]**: Indexed particles and systems
- **[HyperGraph][kups.potential.common.graph.HyperGraph]**: Point cloud with typed edges
- **[RadiusGraphConstructor][kups.potential.common.graph.RadiusGraphConstructor]**: Builds pairwise graphs from neighbor lists
- **[EdgeSetGraphConstructor][kups.potential.common.graph.EdgeSetGraphConstructor]**: Builds graphs from explicit edge lists (bonds, angles)
- **[PointCloudConstructor][kups.potential.common.graph.PointCloudConstructor]**: Builds zero-order graphs (no edges)
- **[LocalGraphSumComposer][kups.potential.common.graph.LocalGraphSumComposer]**: Incremental energy update plans
- **[FullGraphSumComposer][kups.potential.common.graph.FullGraphSumComposer]**: Full recomputation plans
"""

from __future__ import annotations

from typing import (
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity
from kups.core.data import Index, Table, WithIndices
from kups.core.lens import View, bind
from kups.core.neighborlist import Edges, NearestNeighborList
from kups.core.patch import Patch, Probe
from kups.core.typing import (
    HasExclusionIndex,
    HasInclusionIndex,
    HasPositions,
    HasPositionsAndSystemIndex,
    HasSystemIndex,
    HasUnitCell,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, isin, jit
from kups.potential.common.energy import Sum, SumComposer, Summand

Params = TypeVar("Params", covariant=True)
Part = TypeVar("Part", covariant=True, bound=HasPositionsAndSystemIndex)
Sys = TypeVar("Sys", covariant=True, bound=HasUnitCell)
Degree = TypeVar("Degree", bound=int)


@dataclass
class PointCloud(Generic[Part, Sys]):
    """Indexed particles and systems, the base for all graph representations.

    Generic in ``Part`` (particle data with positions and system assignment)
    and ``Sys`` (system data with unit cell).

    Attributes:
        particles: Indexed particle data with positions and system assignment.
        systems: Indexed system data with unit cell information.
    """

    particles: Table[ParticleId, Part]
    systems: Table[SystemId, Sys]

    @property
    def batch_size(self) -> int:
        return self.particles.data.system.num_labels


@dataclass
class HyperGraph(PointCloud[Part, Sys], Generic[Part, Sys, Degree]):
    """Point cloud with edges representing particle interactions.

    Generic in ``Part`` (particle data), ``Sys`` (system data), and
    ``Degree`` (number of particles per edge: 2=pairs, 3=triplets).

    Attributes:
        particles: Inherited -- indexed particle data.
        systems: Inherited -- indexed system data.
        edges: Edge connectivity with ``Index[ParticleId]`` indices and periodic shifts.
    """

    edges: Edges[Degree]

    @property
    def edge_offsets(self) -> Array:
        return self.edges.shifts

    @property
    def edge_shifts(self) -> Array:
        return self.edges.difference_vectors(self.particles, self.systems)

    @property
    def edge_batch_mask(self) -> Index[SystemId]:
        return self.particles[self.edges.indices[:, 0]].system

    @overload
    def sorted_by_system(
        self,
        sort_edges: bool = ...,
        *,
        return_sort_order: Literal[False] = ...,
    ) -> HyperGraph[Part, Sys, Degree]: ...
    @overload
    def sorted_by_system(
        self,
        sort_edges: bool = ...,
        *,
        return_sort_order: Literal[True],
    ) -> tuple[HyperGraph[Part, Sys, Degree], Array]: ...
    def sorted_by_system(
        self,
        sort_edges: bool = False,
        *,
        return_sort_order: bool = False,
    ) -> HyperGraph[Part, Sys, Degree] | tuple[HyperGraph[Part, Sys, Degree], Array]:
        """Sort particles by system index and remap edges accordingly.

        Args:
            sort_edges: If True, also sort edges by the system of their first
                particle.
            return_sort_order: If True, also return the sort permutation.
        """
        n = len(self.particles)
        sort_order = jnp.argsort(self.particles.data.system.indices, stable=True)
        sorted_data = bind(self.particles.data).at((sort_order,)).get()
        sorted_particles = Table(
            self.particles.keys, sorted_data, _cls=self.particles.cls
        )
        # Size n+1: padded edge indices (>= n) map to n, the sentinel slot.
        inverse_order = (
            jnp.full(n + 1, fill_value=n, dtype=sort_order.dtype)
            .at[sort_order]
            .set(jnp.arange(n, dtype=sort_order.dtype))
        )
        remapped_indices = Index(
            self.edges.indices.keys,
            inverse_order[self.edges.indices.indices],
            max_count=self.edges.indices.max_count,
            _cls=self.edges.indices.cls,
        )
        sorted_edges = Edges(indices=remapped_indices, shifts=self.edges.shifts)
        if sort_edges:
            edge_systems = sorted_particles[remapped_indices[:, 0]].system.indices
            edge_order = jnp.argsort(edge_systems, stable=True)
            sorted_edges = Edges(
                indices=remapped_indices[edge_order],
                shifts=sorted_edges.shifts[edge_order],
            )
        result = HyperGraph(sorted_particles, self.systems, sorted_edges)
        if return_sort_order:
            return result, sort_order
        return result


class GraphConstructor[
    State,
    Ptch: Patch,
    P: HasPositionsAndSystemIndex,
    S: HasUnitCell,
    Degree: int,
](Protocol):
    """Protocol for constructing molecular graphs from simulation state."""

    def __call__(
        self, state: State, patch: Ptch | None, old_graph: bool = False
    ) -> HyperGraph[P, S, Degree]:
        """Construct a hypergraph from state.

        Args:
            state: Current simulation state.
            patch: Optional patch for incremental construction.
            old_graph: If True, return graph for the pre-update configuration.

        Returns:
            HyperGraph with particles, systems, and edges.
        """
        ...


@runtime_checkable
class IsRadiusGraphPoints(
    HasPositions,
    HasSystemIndex,
    HasInclusionIndex,
    HasExclusionIndex,
    Protocol,
): ...


@runtime_checkable
class IsRadiusGraphProbe[P: IsRadiusGraphPoints](Protocol):
    """Probe result for radius graph incremental updates."""

    @property
    def particles(self) -> WithIndices[ParticleId, P]: ...
    @property
    def neighborlist_after(self) -> NearestNeighborList: ...
    @property
    def neighborlist_before(self) -> NearestNeighborList: ...


@dataclass
class RadiusGraphConstructor[
    State,
    Ptch: Patch,
    P: IsRadiusGraphPoints,
    S: HasUnitCell,
](GraphConstructor[State, Ptch, P, S, Literal[2]]):
    """Constructs pairwise graphs from neighbor lists (Degree=2).

    Attributes:
        particles: View extracting ``Indexed[ParticleId, P]`` from state.
        systems: View extracting ``Indexed[SystemId, S]`` (unit cell).
        cutoffs: View extracting ``Indexed[SystemId, Array]`` from state.
        neighborlist: View extracting the ``NearestNeighborList`` from state.
        probe: Optional probe for incremental particle + neighbor list changes.
    """

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    cutoffs: View[State, Table[SystemId, Array]] = field(static=True)
    neighborlist: View[State, NearestNeighborList] = field(static=True)
    probe: Probe[State, Ptch, IsRadiusGraphProbe[P]] | None = field(static=True)

    @jit(static_argnames=("old_graph",))
    def __call__(
        self, state: State, patch: Ptch | None, old_graph: bool = False
    ) -> HyperGraph[P, S, Literal[2]]:
        lh = self.particles(state)
        systems = self.systems(state)
        cutoffs = self.cutoffs(state)

        if patch is not None and self.probe is None and not old_graph:
            new_state = patch(
                state, systems.set_data(jnp.ones(len(systems), dtype=jnp.bool_))
            )
            return self(new_state, None, old_graph=True)

        if patch is None:
            nnlist = self.neighborlist(state)
            edges = nnlist(lh, None, systems, cutoffs)
        else:
            assert self.probe is not None, "Expected probe to be set."
            probe = self.probe(state, patch)
            update = probe.particles
            indices = update.indices
            if not old_graph:
                rh = update.data
                lh = lh.update(indices, rh)
                nnlist = probe.neighborlist_after
            else:
                rh = lh[indices]
                nnlist = probe.neighborlist_before
            rh_indexed = Table.arange(rh, label=lh.cls)
            edges = nnlist(lh, rh_indexed, systems, cutoffs, rh_index_remap=indices)
        return HyperGraph(lh, systems, edges)


class UpdatedEdges[Degree: int](NamedTuple):
    """Updated edge information for incremental graph construction."""

    indices: Array
    edge_data: Edges[Degree]


class IsEdgeSetGraphProbe[P: HasPositionsAndSystemIndex, Degree: int](Protocol):
    """Probe result for edge-set graph incremental updates."""

    @property
    def particles(self) -> WithIndices[ParticleId, P]: ...
    @property
    def edges(self) -> UpdatedEdges[Degree]: ...
    @property
    def capacity(self) -> Capacity[int]: ...


@dataclass
class EdgeSetGraphConstructor[
    State,
    Ptch: Patch,
    P: HasPositionsAndSystemIndex,
    S: HasUnitCell,
    Degree: int,
](GraphConstructor[State, Ptch, P, S, Degree]):
    """Constructs graphs from predefined edge lists (bonds, angles, dihedrals).

    Attributes:
        particles: View extracting ``Indexed[ParticleId, P]``.
        systems: View extracting ``Indexed[SystemId, S]``.
        edges: View extracting ``Edges[Degree]`` from state.
        probe: Optional probe for incremental particle + edge changes.
    """

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    edges: View[State, Edges[Degree]] = field(static=True)
    probe: Probe[State, Ptch, IsEdgeSetGraphProbe[P, Degree]] | None = field(
        static=True
    )

    @jit(static_argnames=("old_graph",))
    def __call__(
        self, state: State, patch: Ptch | None, old_graph: bool = False
    ) -> HyperGraph[P, S, Degree]:
        particles = self.particles(state)
        edges = self.edges(state)
        systems = self.systems(state)

        if patch is not None and self.probe is None and not old_graph:
            new_state: State = patch(
                state, systems.set_data(jnp.ones(len(systems), dtype=jnp.bool_))
            )
            return self(new_state, None, old_graph=True)

        if patch is None:
            return HyperGraph(particles, systems, edges)

        assert self.probe is not None, "Expected probe to be set."

        probe = self.probe(state, patch)
        update = probe.particles
        indices = update.indices
        update_edges = probe.edges
        edge_idx, update_edge_data = update_edges

        if not old_graph:
            edges = bind(edges).at(edge_idx).set(update_edge_data)
            particles = particles.update(indices, update.data)

        capacity = probe.capacity
        affected_edges = isin(
            edges.indices.indices, indices.indices, edges.indices.num_labels
        ).any(-1)
        required_edges = jnp.sum(affected_edges)
        capacity = capacity.generate_assertion(required_edges)
        oob = len(particles)
        affected_edge_idx = jnp.where(
            affected_edges, size=capacity.size, fill_value=oob
        )[0]
        edge_idx = jnp.concatenate([affected_edge_idx, edge_idx])
        edge_idx = jnp.unique(edge_idx, size=edge_idx.size, fill_value=oob)
        edges = Edges(
            indices=Index(
                edges.indices.keys,
                edges.indices.indices.at[edge_idx].get(**edges.indices.scatter_args),
            ),
            shifts=edges.shifts.at[edge_idx].get(mode="fill", fill_value=0),
        )
        return HyperGraph(particles, systems, edges)


@dataclass
class PointCloudConstructor[
    State,
    Ptch: Patch,
    P: HasPositionsAndSystemIndex,
    S: HasUnitCell,
](GraphConstructor[State, Ptch, P, S, Literal[0]]):
    """Constructs zero-order graphs (Degree=0, no edges).

    Attributes:
        particles: View extracting ``Indexed[ParticleId, P]``.
        systems: View extracting ``Indexed[SystemId, S]``.
        probe_particles: Optional probe returning ``WithIndices[ParticleId, P]``.
    """

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    probe_particles: Probe[State, Ptch, WithIndices[ParticleId, P]] | None = field(
        static=True
    )

    @jit(static_argnames=("old_graph",))
    def __call__(
        self, state: State, patch: Ptch | None, old_graph: bool = False
    ) -> HyperGraph[P, S, Literal[0]]:
        particles = self.particles(state)
        systems = self.systems(state)
        edges = Edges(
            indices=Index(particles.keys, jnp.zeros((0, 0), dtype=int)),
            shifts=jnp.zeros((0, 0, 3), dtype=int),
        )
        if patch is not None and self.probe_particles is None and not old_graph:
            new_state: State = patch(
                state, systems.set_data(jnp.ones(len(systems), dtype=jnp.bool_))
            )
            return self(new_state, None, old_graph=True)

        if patch is None:
            return HyperGraph(particles, systems, edges)

        assert self.probe_particles is not None, "Expected probe_particles to be set."
        update = self.probe_particles(state, patch)
        indices = update.indices
        if not old_graph:
            particles = particles.update(indices, update.data)
        return HyperGraph(particles, systems, edges)


class GraphPotentialInput(NamedTuple, Generic[Params, Part, Sys, Degree]):
    """Input bundle for graph-based potential energy functions."""

    parameters: Params
    graph: HyperGraph[Part, Sys, Degree]


@dataclass
class LocalGraphSumComposer[
    State,
    Ptch: Patch,
    P: HasPositionsAndSystemIndex,
    S: HasUnitCell,
    Degree: int,
    Params,
](SumComposer[State, GraphPotentialInput[Params, P, S, Degree], Ptch]):
    """Composer for local potentials with incremental updates.

    Without a patch, returns a single full-graph summand. With a patch,
    returns ``old_graph`` (weight −1) + ``new_graph`` (weight +1) with
    ``add_previous_total=True``, enabling O(k) energy updates.
    """

    graph_constructor: GraphConstructor[State, Ptch, P, S, Degree] = field(static=True)
    parameter_view: View[State, Params] = field(static=True)

    def __call__(
        self, state: State, patch: Ptch | None
    ) -> Sum[GraphPotentialInput[Params, P, S, Degree]]:
        params = self.parameter_view(state)

        if patch is None:
            graph = self.graph_constructor(state, None)
            return Sum(Summand(GraphPotentialInput(params, graph)))

        old_graph = self.graph_constructor(state, patch, old_graph=True)
        new_graph = self.graph_constructor(state, patch, old_graph=False)
        return Sum(
            Summand(GraphPotentialInput(params, old_graph), -1),
            Summand(GraphPotentialInput(params, new_graph), 1),
            add_previous_total=True,
        )


@dataclass
class FullGraphSumComposer[
    State,
    Ptch: Patch,
    P: HasPositionsAndSystemIndex,
    S: HasUnitCell,
    Degree: int,
    Params,
](SumComposer[State, GraphPotentialInput[Params, P, S, Degree], Ptch]):
    """Composer for global potentials requiring full recomputation.

    Always applies the patch (if any) to the state and then builds a single
    full graph.
    """

    graph_constructor: GraphConstructor[State, Ptch, P, S, Degree] = field(static=True)
    parameter_view: View[State, Params] = field(static=True)

    def __call__(
        self, state: State, patch: Ptch | None
    ) -> Sum[GraphPotentialInput[Params, P, S, Degree]]:
        if patch is not None:
            state = patch(state, Table((SystemId(0),), jnp.ones((1,), dtype=jnp.bool_)))
        params = self.parameter_view(state)
        graph = self.graph_constructor(state, None)
        return Sum(Summand(GraphPotentialInput(params, graph)))
