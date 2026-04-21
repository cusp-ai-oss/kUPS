# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Local machine-learned interatomic potential (MLIAP) with message passing.

This module provides infrastructure for local MLIAPs that use a single round of
message passing. The key feature is efficient incremental energy updates during
Monte Carlo simulations by caching node embeddings and aggregated messages.

Key components:

- **[LocalMLIAPData][kups.potential.mliap.local.LocalMLIAPData]**: Configuration
  holding model functions and cache
- **[LocalMLIAPComposer][kups.potential.mliap.local.LocalMLIAPComposer]**: Composes
  state into inputs for energy evaluation
- **[local_mliap_energy][kups.potential.mliap.local.local_mliap_energy]**: Computes
  energy with automatic full/incremental dispatch

The architecture follows a message-passing neural network pattern:
1. Node initialization: atomic_numbers → node embeddings
2. Edge function: (node_i, node_j, r_ij) → messages
3. Message aggregation: sum messages per node
4. Readout: (node_emb, msg_sum) → per-atom energies

Example:
    ```python
    def init_fn(atomic_numbers): return embedding_table[atomic_numbers]
    def edge_fn(n1, n2, r): return mlp(r) * n2
    def readout_fn(emb, msg): return linear(emb + msg)

    config = LocalMLIAPData(
        cutoff=jnp.array([6.0]),
        init_function=init_fn,
        edge_function=edge_fn,
        readout_function=readout_fn,
        cache=LocalMLIAPCache(...),
    )
    ```
"""

import json
import zipfile
from pathlib import Path
from typing import Any, Literal, Protocol, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table, WithIndices
from kups.core.lens import Lens, View, bind
from kups.core.neighborlist import Edges, NearestNeighborList
from kups.core.patch import Accept, Patch, Probe, WithPatch
from kups.core.potential import EMPTY_LENS, Energy, Potential, PotentialOut
from kups.core.typing import (
    HasCache,
    HasPositionsAndAtomicNumbers,
    HasSystemIndex,
    HasUnitCell,
    MaybeCached,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, sequential_vmap_with_vjp, tree_map
from kups.core.utils.ops import where_broadcast_last
from kups.potential.common.energy import (
    PotentialFromEnergy,
    Sum,
    Summand,
)
from kups.potential.common.graph import (
    IsRadiusGraphPoints,
    IsRadiusGraphProbe,
    PointCloud,
    RadiusGraphConstructor,
)


def _make_dtype_converter(precision: Literal["32bit", "64bit"]):
    """Create a dtype conversion function based on precision."""

    def convert_dtype(x: jnp.ndarray) -> jnp.ndarray:
        is_int = jnp.issubdtype(x.dtype, jnp.integer)
        match precision:
            case "32bit":
                dtype = jnp.int32 if is_int else jnp.float32
            case "64bit":
                dtype = jnp.int64 if is_int else jnp.float64
        return x.astype(dtype)

    return convert_dtype


class NodeInitFunction(Protocol):
    """Protocol for node initialization function.

    Maps atomic numbers `(n_atoms,)` to node embeddings `(n_atoms, embed_dim)`.
    """

    def __call__(self, atomic_numbers: Array) -> Array: ...


class ReadoutFunction(Protocol):
    """Protocol for readout function.

    Computes per-atom energies `(n_atoms,)` from node embeddings and
    aggregated messages, both of shape `(n_atoms, embed_dim)`.
    """

    def __call__(self, node_emb: Array, msg_sum: Array) -> Array: ...


class EdgeFunction(Protocol):
    """Protocol for edge/message function.

    Computes edge messages `(n_edges, msg_dim)` from source/target node
    embeddings `(n_edges, embed_dim)` and displacement vectors `(n_edges, 3)`.
    """

    def __call__(
        self, node1: Array, node2: Array, difference_vectors: Array
    ) -> Array: ...


class IsLocalMLIAPParticleData(HasPositionsAndAtomicNumbers, HasSystemIndex, Protocol):
    """Protocol for particle data required by LocalMLIAP.

    Must provide positions, atomic numbers, and system index.
    """

    ...


class IsLocalMLIAPGraphParticles(
    IsLocalMLIAPParticleData, IsRadiusGraphPoints, Protocol
):
    """Combined protocol for local MLIAP particles in radius graph context."""

    ...


@dataclass
class LocalMLIAPCache:
    """Cache for incremental energy updates.

    Stores intermediate values from the last full computation to enable
    efficient incremental updates when only a subset of atoms change.

    Attributes:
        node_init: Cached node embeddings from init_function, shape `(n_atoms, embed_dim)`
        msg_sum: Cached aggregated messages per node, shape `(n_atoms, msg_dim)`
    """

    node_init: Array
    msg_sum: Array


@dataclass
class LocalMLIAPData:
    """Configuration for a local MLIAP model.

    Bundles the model functions, cutoff, and cache together with a lens
    for updating the cache in the state.

    Attributes:
        cutoff: Interaction cutoff radius [Å], shape `(n_systems,)`
        init_function: Maps atomic numbers to node embeddings
        edge_function: Computes messages from node pairs and displacements
        readout_function: Computes per-atom energies from embeddings and messages
        cache: Cached values for incremental updates
    """

    cutoff: Table[SystemId, Array]
    init_function: NodeInitFunction = field(static=True)
    edge_function: EdgeFunction = field(static=True)
    readout_function: ReadoutFunction = field(static=True)
    cache: LocalMLIAPCache

    @staticmethod
    def from_zip_file(zip_file: str | Path, n_atoms: int) -> "LocalMLIAPData":
        """Load a local MLIAP model from a zip archive.

        Expects ``node_init.jax``, ``edge.jax``, ``readout.jax``,
        and ``metadata.json`` (with ``cutoff`` and ``precision`` keys).

        Args:
            zip_file: Path to the ``.zip`` archive.
            n_atoms: Number of atoms to allocate cache for.

        Returns:
            Loaded local MLIAP model with initialized cache.
        """
        with zipfile.ZipFile(zip_file, "r") as zf:
            with zf.open("metadata.json") as f:
                metadata = json.loads(f.read().decode())
            with zf.open("node_init.jax") as f:
                exported_init = jax.export.deserialize(f.read())  # type: ignore
            with zf.open("edge.jax") as f:
                exported_edge = jax.export.deserialize(f.read())  # type: ignore
            with zf.open("readout.jax") as f:
                exported_readout = jax.export.deserialize(f.read())  # type: ignore

        convert = _make_dtype_converter(metadata["precision"])

        def init_fn(atomic_numbers):
            return sequential_vmap_with_vjp(exported_init.call)(convert(atomic_numbers))

        def edge_fn(node1, node2, difference_vectors):
            fn = sequential_vmap_with_vjp(exported_edge.call)
            return fn(convert(node1), convert(node2), convert(difference_vectors))

        def readout_fn(node_emb, msg_sum):
            return sequential_vmap_with_vjp(exported_readout.call)(
                convert(node_emb), convert(msg_sum)
            )

        # Infer dimensions
        sample_atomic_numbers = jnp.zeros((1,), dtype=int)
        node_emb_dim = jax.eval_shape(init_fn, sample_atomic_numbers).shape[-1]
        sample_node = jnp.zeros((1, node_emb_dim))
        msg_dim = jax.eval_shape(
            edge_fn, sample_node, sample_node, jnp.zeros((1, 3))
        ).shape[-1]

        return LocalMLIAPData(
            cutoff=Table((SystemId(0),), jnp.array([metadata["cutoff"]], float)),
            init_function=init_fn,
            edge_function=edge_fn,
            readout_function=readout_fn,
            cache=LocalMLIAPCache(
                node_init=jnp.zeros((n_atoms, node_emb_dim)),
                msg_sum=jnp.zeros((n_atoms, msg_dim)),
            ),
        )


@dataclass
class LocalMLIAPInput[
    State,
    P: IsLocalMLIAPGraphParticles,
    S: HasUnitCell,
]:
    """Input bundle for local MLIAP energy computation.

    Contains all data needed to compute energies, supporting both full
    computation and incremental updates.

    Type Parameters:
        State: Simulation state type
        P: Particle data type (positions, atomic numbers, system, inclusion/exclusion)
        S: System data type (must have unit cell)

    Attributes:
        point_cloud: Current particle positions and systems
        point_cloud_changes: Changed particles for incremental update (None for full)
        edges: Current edges within cutoff
        edges_deleted: Old edges to remove for incremental update (None for full)
        config: Model configuration with functions and cache
        cache_lens: Lens to access/update cache in state
    """

    point_cloud: PointCloud[P, S]
    point_cloud_changes: WithIndices[ParticleId, P] | None
    edges: Edges[Literal[2]]
    edges_deleted: Edges[Literal[2]] | None
    config: LocalMLIAPData
    cache_lens: Lens[State, LocalMLIAPCache] = field(static=True)


@dataclass
class LocalMLIAPPatch[State](Patch[State]):
    """Patch to update the MLIAP cache in state.

    Applied after energy computation to update cached node embeddings
    and aggregated messages for systems where moves were accepted.

    Type Parameters:
        State: Simulation state type

    Attributes:
        cache: New cache values to apply
        system_idx: System index for masking
        lens: Lens to access cache in state
    """

    cache: LocalMLIAPCache
    system_idx: Index[SystemId]
    lens: Lens[State, LocalMLIAPCache] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        """Apply cache update to state.

        Args:
            state: Current simulation state
            accept: Boolean mask per system indicating accepted moves

        Returns:
            Updated state with new cache values where mask is True
        """
        new_cache = self.cache
        mask = accept[self.system_idx]
        return self.lens.apply(
            state,
            lambda cache: tree_map(
                lambda a, b: where_broadcast_last(mask, a, b), new_cache, cache
            ),
        )


def local_mliap_energy_full[
    State,
    P: IsLocalMLIAPGraphParticles,
    S: HasUnitCell,
](
    inp: LocalMLIAPInput[State, P, S],
) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
    """Compute full MLIAP energy from scratch.

    Performs complete message passing: initializes node embeddings,
    computes all edge messages, aggregates, and applies readout.
    Updates the cache with new values.

    Args:
        inp: Input containing point cloud, edges, and model config

    Returns:
        Total energy per system and patch to update cache
    """
    point_cloud = inp.point_cloud
    n = point_cloud.particles.size
    edges = inp.edges
    # Initial node embeddings
    node_emb = inp.config.init_function(point_cloud.particles.data.atomic_numbers)
    # Edge embeddings / messages
    edge_emb = inp.config.edge_function(
        node_emb[edges.indices.indices[:, 0]],
        node_emb[edges.indices.indices[:, 1]],
        edges.difference_vectors(point_cloud.particles, point_cloud.systems)[:, 0],
    )
    # Aggregate messages
    msg_sum = jax.ops.segment_sum(edge_emb, edges.indices.indices[:, 0], n)
    # Node-wise energies
    energies = inp.config.readout_function(node_emb, msg_sum)
    # Total energies per system
    energy = point_cloud.particles.data.system.sum_over(energies)
    new_cache = LocalMLIAPCache(node_emb, msg_sum)
    return WithPatch(
        energy,
        LocalMLIAPPatch(
            new_cache,
            point_cloud.particles.data.system,
            inp.cache_lens,
        ),
    )


def local_mliap_energy_update[
    State,
    P: IsLocalMLIAPGraphParticles,
    S: HasUnitCell,
](
    inp: LocalMLIAPInput[State, P, S],
) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
    """Compute MLIAP energy incrementally using cached values.

    Only recomputes embeddings and messages for changed atoms, subtracting
    old contributions and adding new ones. Much faster than full computation
    when only a small subset of atoms change.

    Args:
        inp: Input with point_cloud_changes and edges_deleted set

    Returns:
        Total energy per system and patch to update cache

    Raises:
        AssertionError: If point_cloud_changes or edges_deleted is None
    """
    assert inp.point_cloud_changes is not None and inp.edges_deleted is not None, (
        "If point cloud changes are provided, edges_deleted must also be provided."
    )
    old_point_cloud = inp.point_cloud
    changes = inp.point_cloud_changes
    change_indices = changes.indices
    change_raw = change_indices.indices_in(old_point_cloud.particles.keys)
    new_point_cloud = bind(old_point_cloud, lambda x: x.particles).apply(
        lambda p: p.update(change_indices, changes.data)
    )
    n = old_point_cloud.particles.size

    # Compute changed initial embeddings
    old_node_emb = inp.config.cache.node_init
    node_emb = old_node_emb.at[change_raw].set(
        inp.config.init_function(changes.data.atomic_numbers)
    )
    # Compute updated and deleted messages
    old_edge_emb = inp.config.edge_function(
        old_node_emb[inp.edges_deleted.indices.indices[:, 0]],
        old_node_emb[inp.edges_deleted.indices.indices[:, 1]],
        inp.edges_deleted.difference_vectors(
            old_point_cloud.particles, old_point_cloud.systems
        )[:, 0],
    )
    new_edge_emb = inp.config.edge_function(
        node_emb[inp.edges.indices.indices[:, 0]],
        node_emb[inp.edges.indices.indices[:, 1]],
        inp.edges.difference_vectors(
            new_point_cloud.particles, new_point_cloud.systems
        )[:, 0],
    )
    # Update message sums
    msg_sum = (
        inp.config.cache.msg_sum
        - jax.ops.segment_sum(old_edge_emb, inp.edges_deleted.indices.indices[:, 0], n)
        + jax.ops.segment_sum(new_edge_emb, inp.edges.indices.indices[:, 0], n)
    )
    # Node-wise readout
    energies = inp.config.readout_function(node_emb, msg_sum)
    # Total energies per system
    energy = new_point_cloud.particles.data.system.sum_over(energies)
    new_cache = LocalMLIAPCache(node_emb, msg_sum)
    return WithPatch(
        energy,
        LocalMLIAPPatch(
            new_cache,
            new_point_cloud.particles.data.system,
            inp.cache_lens,
        ),
    )


def local_mliap_energy[
    State,
    P: IsLocalMLIAPGraphParticles,
    S: HasUnitCell,
](
    inp: LocalMLIAPInput[State, P, S],
) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
    """Compute MLIAP energy with automatic full/incremental dispatch.

    Automatically chooses between full computation and incremental update
    based on whether point_cloud_changes is provided.

    Args:
        inp: Input bundle with point cloud, edges, and config

    Returns:
        Total energy per system and patch to update cache
    """
    if inp.point_cloud_changes is None:
        return local_mliap_energy_full(inp)
    else:
        return local_mliap_energy_update(inp)


@dataclass
class LocalMLIAPComposer[
    State,
    P: IsLocalMLIAPGraphParticles,
    S: HasUnitCell,
    Ptch: Patch,
]:
    """Composes simulation state into LocalMLIAP input.

    Extracts particles, edges, and model config from state, handling both
    full computation (patch=None) and incremental updates (patch provided).

    Type Parameters:
        State: Simulation state type
        P: Particle data type (positions + system + inclusion/exclusion + atomic numbers)
        S: System data type (unit cell + cutoff)
        Ptch: Patch type for incremental updates

    Attributes:
        particles: View to extract indexed particle data from state
        systems: View to extract indexed system data from state
        neighborlist: View to extract full neighbor list from state
        model: Lens to access model config in state
        probe: Probe to detect particle changes from patch
    """

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    cutoffs: View[State, Table[SystemId, Array]] = field(static=True)
    neighborlist: View[State, NearestNeighborList] = field(static=True)
    model: Lens[State, LocalMLIAPData] = field(static=True)
    probe: Probe[State, Ptch, IsRadiusGraphProbe[P]] | None = field(static=True)

    def __call__(
        self, state: State, patch: Ptch | None
    ) -> Sum[LocalMLIAPInput[State, P, S]]:
        """Compose state and patch into LocalMLIAP input.

        Args:
            state: Current simulation state
            patch: Proposed changes (None for full computation)

        Returns:
            Sum containing single LocalMLIAPInput summand
        """

        radius_graph_constr = RadiusGraphConstructor(
            self.particles,
            self.systems,
            self.cutoffs,
            self.neighborlist,
            self.probe,
        )
        pc = PointCloud(self.particles(state), self.systems(state))
        conf = self.model.get(state)
        cache_lens = self.model.focus(lambda x: x.cache)
        if patch is None:
            graph = radius_graph_constr(state, None)
            result = LocalMLIAPInput(pc, None, graph.edges, None, conf, cache_lens)
        else:
            assert self.probe is not None
            new_graph = radius_graph_constr(state, patch)
            old_graph = radius_graph_constr(state, patch, old_graph=True)
            result = LocalMLIAPInput(
                pc,
                self.probe(state, patch).particles,
                new_graph.edges,
                old_graph.edges,
                conf,
                cache_lens,
            )
        return Sum(Summand(result))


def make_local_mliap_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
    P: IsLocalMLIAPGraphParticles,
    S: HasUnitCell,
](
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, S]],
    cutoffs_view: View[State, Table[SystemId, Array]],
    neighborlist_view: View[State, NearestNeighborList],
    model_lens: Lens[State, LocalMLIAPData],
    probe: Probe[State, Ptch, IsRadiusGraphProbe[P]] | None,
    gradient_lens: Lens[LocalMLIAPInput[State, P, S], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create a local MLIAP potential with single message passing.

    Constructs a potential from model functions (init, edge, readout) with
    support for efficient incremental updates via caching.

    Args:
        particles_view: Extracts indexed particle data (positions, atomic numbers)
        systems_view: Extracts indexed system data (unit cell)
        cutoffs_view: Extracts cutoffs as ``Indexed[SystemId, Array]``
        neighborlist_view: Extracts full neighbor list
        model_lens: Lens to access [LocalMLIAPData][kups.potential.mliap.local.LocalMLIAPData]
            containing model functions and cache
        probe: Detects particle changes and provides updated/old neighbor lists
            for incremental updates
        gradient_lens: Specifies which gradients to compute
        hessian_lens: Specifies which Hessians to compute
        hessian_idx_view: Index structure for Hessian computation
        patch_idx_view: Index structure for cached output updates (optional)
        out_cache_lens: Lens to cache location for incremental updates (optional)

    Returns:
        Configured local MLIAP [Potential][kups.core.potential.Potential]
    """
    composer = LocalMLIAPComposer[State, P, S, Ptch](
        particles=particles_view,
        systems=systems_view,
        cutoffs=cutoffs_view,
        neighborlist=neighborlist_view,
        model=model_lens,
        probe=probe,
    )
    potential = PotentialFromEnergy(
        composer=composer,
        energy_fn=local_mliap_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential


class IsLocalMLIAPState[Model](Protocol):
    """Protocol for states providing all inputs for the local MLIAP potential."""

    @property
    def particles(self) -> Table[ParticleId, IsLocalMLIAPGraphParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def neighborlist(self) -> NearestNeighborList: ...
    @property
    def local_mliap_model(self) -> Model: ...


@overload
def make_local_mliap_from_state[State, Gradient, Hessian](
    state: Lens[State, IsLocalMLIAPState[MaybeCached[LocalMLIAPData, Any]]],
    probe: None = None,
    gradient_lens: Lens[LocalMLIAPInput, Gradient] = EMPTY_LENS,
    hessian_lens: Lens[Gradient, Hessian] = EMPTY_LENS,
    hessian_idx_view: Lens[State, Hessian] = EMPTY_LENS,
    out_idx_view: None = None,
) -> Potential[State, Gradient, Hessian, Patch]: ...


@overload
def make_local_mliap_from_state[State, Ptch: Patch, Gradient, Hessian](
    state: Lens[State, IsLocalMLIAPState[HasCache[LocalMLIAPData, PotentialOut]]],
    probe: Probe[State, Ptch, IsRadiusGraphProbe[IsLocalMLIAPGraphParticles]],
    gradient_lens: Lens[LocalMLIAPInput, Gradient] = EMPTY_LENS,
    hessian_lens: Lens[Gradient, Hessian] = EMPTY_LENS,
    hessian_idx_view: Lens[State, Hessian] = EMPTY_LENS,
    out_idx_view: Lens[State, PotentialOut[Gradient, Hessian]] | None = None,
) -> Potential[State, Gradient, Hessian, Ptch]: ...


def make_local_mliap_from_state(
    state: Any,
    probe: Any = None,
    gradient_lens: Any = EMPTY_LENS,
    hessian_lens: Any = EMPTY_LENS,
    hessian_idx_view: Any = EMPTY_LENS,
    out_idx_view: Any = None,
) -> Any:
    """Create a local MLIAP potential from a typed state, optionally with incremental updates.

    Convenience wrapper around
    [make_local_mliap_potential][kups.potential.mliap.local.make_local_mliap_potential].
    When ``probe`` is ``None``, extracts views from a state satisfying
    [IsLocalMLIAPState][kups.potential.mliap.local.IsLocalMLIAPState].
    When ``probe`` is provided, additionally wires the ``PotentialOut`` cache for
    efficient incremental caching across Monte Carlo steps.

    Args:
        state: Lens into the sub-state providing particles, unit cell, neighbor list,
            and local MLIAP model.
        probe: Detects which particles and neighbor-list edges changed since the last
            step. ``None`` for full-only computation.
        gradient_lens: Specifies which gradients to compute (e.g., forces).
        hessian_lens: Specifies which Hessians to compute.
        hessian_idx_view: Index structure for Hessian updates.
        out_idx_view: Index into the cached output for partial updates. Only used when
            ``probe`` is not ``None``. Defaults to full re-indexing of
            ``local_mliap_out_cache.total_energies``.

    Returns:
        Configured local MLIAP [Potential][kups.core.potential.Potential].
    """

    def _model(x: Any) -> LocalMLIAPData:
        m = x.local_mliap_model
        return m.data if isinstance(m, HasCache) else m

    if probe is None:
        return make_local_mliap_potential(
            state.focus(lambda x: x.particles),
            state.focus(lambda x: x.systems),
            state.focus(lambda x: _model(x).cutoff),
            state.focus(lambda x: x.neighborlist),
            state.focus(_model),
            None,
            gradient_lens,
            hessian_lens,
            hessian_idx_view,
            None,
            None,
        )

    if out_idx_view is None:
        out_idx_view = state.focus(
            lambda x: bind(
                x.local_mliap_model.cache, lambda x: x.total_energies.data
            ).apply(lambda x: jnp.arange(x.size, dtype=int))
        )

    return make_local_mliap_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        state.focus(lambda x: x.local_mliap_model.data.cutoff),
        state.focus(lambda x: x.neighborlist),
        state.focus(lambda x: x.local_mliap_model.data),
        probe,
        gradient_lens,
        hessian_lens,
        hessian_idx_view,
        out_idx_view,
        state.focus(lambda x: x.local_mliap_model.cache),
    )
