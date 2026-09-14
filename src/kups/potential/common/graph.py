# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Graph construction from atomic coordinates for potential evaluation.

This module builds molecular graphs (point clouds, hypergraphs) from
``Indexed`` particle data for potential energy evaluation. Graphs honor the
cell's per-axis ``periodic`` mask via the neighbor list, support multiple
independent systems, and allow efficient incremental construction for Monte
Carlo via probes.

Key components:

- **[PointCloud][kups.potential.common.graph.PointCloud]**: Indexed particles and systems
- **[HyperGraph][kups.potential.common.graph.HyperGraph]**: Point cloud with typed edges
- **[GraphConstructor][kups.potential.common.graph.GraphConstructor]**: Builds graphs from a degree-parametrized neighbor list
- **[GraphInputConstructor][kups.potential.common.graph.GraphInputConstructor]**: Full or incremental potential inputs
"""

from __future__ import annotations

from typing import (
    Any,
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

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table, WithIndices
from kups.core.lens import Lens, View, bind, lens
from kups.core.neighborlist import Edges, NeighborList
from kups.core.patch import Patch, Probe
from kups.core.typing import (
    HasCell,
    HasExclusionIndex,
    HasInclusionIndex,
    HasPositions,
    HasPositionsAndSystemIndex,
    HasSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, jit
from kups.potential.common.energy import InputConstructor
from kups.potential.common.geometry import Geometry, PositionsAndSystemIndex

Params = TypeVar("Params", covariant=True)
Part = TypeVar("Part", covariant=True, bound=HasPositionsAndSystemIndex)
Sys = TypeVar("Sys", covariant=True, bound=HasCell[AnyPeriodicity])
Degree = TypeVar("Degree", bound=int)


@dataclass
class PointCloud(Generic[Part, Sys]):
    """Indexed particles and systems, the base for all graph representations.

    Generic in ``Part`` (particle data with positions and system assignment)
    and ``Sys`` (system data with cell).

    Attributes:
        particles: Indexed particle data with positions and system assignment.
        systems: Indexed system data with cell information.
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


@runtime_checkable
class IsRadiusGraphPoints(
    HasPositions,
    HasSystemIndex,
    HasInclusionIndex,
    HasExclusionIndex,
    Protocol,
): ...


@runtime_checkable
class IsGraphProbe[P: IsRadiusGraphPoints, Degree: int](Protocol):
    """Probe result for incremental graph construction."""

    @property
    def particles(self) -> WithIndices[ParticleId, P]: ...
    @property
    def neighborlist_after(self) -> NeighborList[Degree]: ...
    @property
    def neighborlist_before(self) -> NeighborList[Degree]: ...


@dataclass
class GraphConstructor[
    State,
    Ptch: Patch[Any],
    P: IsRadiusGraphPoints,
    S: HasCell[AnyPeriodicity],
    Degree: int,
]:
    """Constructs graphs from a degree-parametrized neighbor list.

    Attributes:
        particles: View extracting ``Indexed[ParticleId, P]``.
        systems: View extracting ``Indexed[SystemId, S]`` (cell).
        neighborlist: View extracting the neighbor list implementation for the state.
        probe: Optional probe for incremental particle + neighbor list changes.
    """

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    neighborlist: View[State, NeighborList[Degree]] = field(static=True)
    probe: Probe[State, Ptch, IsGraphProbe[P, Degree]] | None = field(static=True)

    @jit(static_argnames=("old_graph",))
    def __call__(
        self, state: State, patch: Ptch | None, old_graph: bool = False
    ) -> HyperGraph[P, S, Degree]:
        keys = self.particles(state)
        systems = self.systems(state)

        if patch is not None and self.probe is None:
            if old_graph:
                return self(state, None, old_graph=True)
            new_state: State = patch(
                state, systems.set_data(jnp.ones(len(systems), dtype=jnp.bool_))
            )
            return self(new_state, None, old_graph=True)

        if patch is None:
            nnlist = self.neighborlist(state)
            edges = nnlist(keys, systems)
        else:
            assert self.probe is not None, "Expected probe to be set."
            probe = self.probe(state, patch)
            update = probe.particles
            indices = update.indices
            if not old_graph:
                keys = keys.update(indices, update.data)
                nnlist = probe.neighborlist_after
            else:
                nnlist = probe.neighborlist_before
            edges = nnlist(keys, systems, queried_keys=indices)
        return HyperGraph(keys, systems, edges)


class GraphPotentialInput(NamedTuple, Generic[Params, Part, Sys, Degree]):
    """Input bundle for graph-based potential energy functions."""

    parameters: Params
    graph: HyperGraph[Part, Sys, Degree]


class IsGraphInput(Protocol):
    """A potential input exposing a ``graph`` of particles and systems."""

    @property
    def graph(
        self,
    ) -> PointCloud[HasPositionsAndSystemIndex, HasCell[AnyPeriodicity]]: ...


def _pointcloud_geometry(
    cloud: PointCloud[HasPositionsAndSystemIndex, HasCell[AnyPeriodicity]],
) -> Geometry:
    return Geometry(
        cloud.particles.map_data(
            lambda p: PositionsAndSystemIndex(p.positions, p.system)
        ),
        cloud.systems.map_data(lambda s: s.cell),
    )


POINTCLOUD_GEOMETRY: Lens[Any, Geometry] = lens(_pointcloud_geometry)
"""Geometry adapter shared by graph, fused, and reciprocal-space inputs."""


GRAPH_GEOMETRY: Lens[IsGraphInput, Geometry] = lens(lambda inp: inp.graph).nest(
    POINTCLOUD_GEOMETRY
)
"""Adapter from a graph-bearing potential input to ``Geometry``."""


@dataclass
class GraphInputConstructor[
    State,
    Ptch: Patch[Any],
    P: IsRadiusGraphPoints,
    S: HasCell[AnyPeriodicity],
    Degree: int,
    Params,
](InputConstructor[State, GraphPotentialInput[Params, P, S, Degree], Ptch]):
    """Construct full or affected graph inputs for the shared sum composers.

    A graph probe restricts proposals to affected interactions, keeping
    parameters from the current state. Without a probe, apply the patch to
    the full state before reading parameters and constructing the graph.
    ``old_input`` selects the current state in either case.
    """

    graph_constructor: GraphConstructor[State, Ptch, P, S, Degree] = field(static=True)
    parameter_view: View[State, Params] = field(static=True)

    def __call__(
        self,
        state: State,
        patch: Ptch | None,
        old_input: bool = False,
    ) -> GraphPotentialInput[Params, P, S, Degree]:
        if patch is not None and self.graph_constructor.probe is None:
            if not old_input:
                systems = self.graph_constructor.systems(state)
                state = patch(
                    state, systems.set_data(jnp.ones(len(systems), dtype=jnp.bool_))
                )
            patch = None
        return GraphPotentialInput(
            self.parameter_view(state),
            self.graph_constructor(state, patch, old_graph=old_input),
        )
