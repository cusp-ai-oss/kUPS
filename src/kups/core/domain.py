# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain decomposition: particle ownership and the per-device interaction graph.

Replicate-all model: every device stores all N particles (cheap, no comms) and
``origin: Index[OriginDeviceId]`` marks the owner. Each device builds only the
edges incident on the atoms it owns by passing them as the neighbor list's
``queried_keys`` — only owned atoms drive the cell-list stencil, so the costly
candidate/distance work is O(N/D) while the key side is just cheap-hashed.
There is no ghost table: the owned-incident edges (global ids + image shifts)
are themselves the device's graph. How the stock energy functions reduce owned-only
and combine across the mesh is the ``Decomposition`` seam — see
``kups.potential.common.graph.Decomposition``; ``Sharded`` here is its
domain-decomposed implementation.

Layered like the rest of kUPS: pure algorithms (generic over protocols,
``Capacity``-guarded) wrapped by state-coupled components holding
``View``/``Lens`` fields with ``make_*_from_state`` factories.
"""

from typing import Any, Literal, Protocol, override, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.capacity import Capacity
from kups.core.cell import AnyPeriodicity
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import Edges, NeighborList, NeighborListPoints
from kups.core.patch import Patch
from kups.core.sharding import shard_axis
from kups.core.typing import (
    HasCell,
    HasOrigin,
    IsState,
    OriginDeviceId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.core.utils.ops import where_broadcast_last
from kups.potential.common.graph import Decomposition, HyperGraph


@runtime_checkable
class IsDecomposedParticle(NeighborListPoints, HasOrigin, Protocol):
    """A neighbor-list point that also knows which device owns it (``origin``)."""


@dataclass
class Sharded[P: IsDecomposedParticle](Decomposition[P]):
    """Domain-decomposed placement: mask non-owned rows, ``psum`` across the mesh.

    Only ever constructed inside a ``shard_map`` over the ``OriginDeviceId``
    mesh axis, so the type itself is the gate — no runtime sharding detection.
    """

    @override
    def owned_only(self, particles: Table[ParticleId, P], x: Array) -> Array:
        owned = particles.data.origin.indices == jax.lax.axis_index(
            shard_axis(OriginDeviceId)
        )
        return where_broadcast_last(owned, x, 0)

    @override
    def combine_across_shards(self, x: Array) -> Array:
        return jax.lax.psum(x, shard_axis(OriginDeviceId))


def partition_equal_counts(n_particles: int, n_devices: int) -> Index[OriginDeviceId]:
    """Assign owners as equal-count contiguous chunks in table order.

    Ownership never affects correctness, only the locality of the per-device
    cell-list queries, and table order is typically already coherent; a
    space-filling-curve partitioner is the upgrade path if that ever shows
    up in profiles.
    """
    return Index.integer(
        np.arange(n_particles) * n_devices // n_particles,
        n=n_devices,
        label=OriginDeviceId,
    )


def owned_subset[P: IsDecomposedParticle](
    particles: Table[ParticleId, P], device_id: Array | int, cap_owned: Capacity[int]
) -> Index[ParticleId]:
    """Global ids of the atoms ``device_id`` owns, packed into a ``cap_owned`` buffer.

    Padding slots hold the OOB sentinel ``len(particles)`` so downstream
    gathers/queries drop them. Overflowing ``cap_owned`` would silently lose
    owned atoms, so the required size is recorded as a runtime assertion (it
    raises under the assertion interpreter; see ``kups.core.result``). The
    recorded requirement is the max owned count over every device — the shared
    capacity must cover them all, and computing it from the replicated
    ``origin`` labels keeps the assertion device-invariant, so it can leave a
    ``shard_map`` through a replicated assertion context.
    """
    n = len(particles)
    origin = particles.data.origin
    cap_owned = cap_owned.generate_assertion(origin.counts.data.max())
    owned_idx = jnp.where(
        origin.indices == device_id, size=cap_owned.size, fill_value=n
    )[0]
    return Index(particles.keys, owned_idx)


def sharded_local_edges[P: IsDecomposedParticle, S: HasCell[AnyPeriodicity]](
    particles: Table[ParticleId, P],
    systems: Table[SystemId, S],
    nl: NeighborList[Literal[2]],
    device_id: Array | int,
    cap_owned: Capacity[int],
) -> Edges[Literal[2]]:
    """One device's owned-incident shard of the radius graph: edges touching its owned atoms.

    Passes the owned ids as ``queried_keys``, so the neighbor list returns
    exactly the self-graph edges incident on owned atoms, both orientations of
    every owned-incident pair. Correctness comes from the reduction, not the
    edge set: ``reduce_edges_to_systems`` counts only energy attributed to
    *owned* first nodes, so across the mesh each pair is counted once —
    owned-owned pairs contribute both orientations on one device, owned-unowned
    pairs one orientation on each of the two owners. The symmetric ``/2`` stays
    an explicit factor in the energy function.

    ``nl`` carries its own cutoffs; the O(N/D) cost is the caller's choice of a
    ``CellListNeighborList`` — a dense list is correct but O(N²) and not
    shard_map-safe. Padding rows need no inclusion mask: ``owned_subset`` pads
    with the OOB sentinel id ``len(particles)``, the pipeline lifts
    ``queried_keys`` back to key space via a fill-mode ``subset`` gather, and
    the padded rows' candidates are dropped before they become edges.
    """
    owned = owned_subset(particles, device_id, cap_owned)
    return nl(particles, systems, queried_keys=owned)


@dataclass
class ShardedRadiusGraphConstructor[
    State,
    P: IsDecomposedParticle,
    S: HasCell[AnyPeriodicity],
]:
    """Build the calling device's owned-incident shard of the radius graph.

    A ``GraphConstructor``-shaped callable for full rebuilds: returns a
    ``HyperGraph`` over the global particle table whose edges are this device's
    owned-incident shard (``sharded_local_edges``), tagged ``Sharded`` so the
    stock energy functions reduce owned-only with no DD code of their own. The
    device id is read from ``jax.lax.axis_index`` — valid only inside
    ``shard_map`` over the ``OriginDeviceId`` axis. There is no incremental
    path: a non-``None`` patch raises rather than silently recomputing.
    """

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    neighborlist: View[State, NeighborList[Literal[2]]] = field(static=True)
    cap_owned: Capacity[int] = field(static=True)

    def __call__(
        self, state: State, patch: Patch[Any] | None = None, old_graph: bool = False
    ) -> HyperGraph[P, S, Literal[2]]:
        if patch is not None:
            raise NotImplementedError(
                "ShardedRadiusGraphConstructor has no incremental path; "
                "compose it without a probe so patches stay None."
            )
        particles = self.particles(state)
        systems = self.systems(state)
        axis = shard_axis(OriginDeviceId)
        # An origin label outside the mesh would be owned by no device and its
        # atoms silently dropped from every reduction; psum(1) is the static
        # mesh axis size.
        n_devices = jax.lax.psum(1, axis)
        assert particles.data.origin.num_labels == n_devices, (
            f"origin was partitioned for {particles.data.origin.num_labels} devices "
            f"but the {axis} mesh axis has {n_devices}"
        )
        edges = sharded_local_edges(
            particles,
            systems,
            self.neighborlist(state),
            jax.lax.axis_index(axis),
            self.cap_owned,
        )
        return HyperGraph(particles, systems, edges, decomposition=Sharded())


def make_sharded_radius_graph_from_state[
    State,
    P: IsDecomposedParticle,
    S: HasCell[AnyPeriodicity],
](
    state_lens: Lens[State, IsState[P, S]],
    neighborlist: View[State, NeighborList[Literal[2]]],
    cap_owned: Capacity[int],
) -> ShardedRadiusGraphConstructor[State, P, S]:
    """Wire a ``ShardedRadiusGraphConstructor`` from a state lens and a neighbor-list view.

    ``neighborlist`` is a view so callers compose it the same way the stock
    potential factories do (a ``NeighborListFactory`` applied to the state and
    the potential's cutoffs).
    """
    return ShardedRadiusGraphConstructor(
        particles=state_lens.focus(lambda s: s.particles),
        systems=state_lens.focus(lambda s: s.systems),
        neighborlist=neighborlist,
        cap_owned=cap_owned,
    )
