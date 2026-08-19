# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Domain decomposition: particle ownership and the per-device interaction graph.

Replicate-all model: every device stores all N particles (cheap, no comms) and
``origin: Index[OriginDeviceId]`` marks the owner. Each device builds only the
edges incident on its OWNED atoms by passing them as the neighbor list's
``queried_keys`` — only owned atoms drive the cell-list stencil, so the costly
candidate/distance work is O(N/D) while the key side is just cheap-hashed.
There is no ghost table: the owned-incident edges (global ids + image shifts)
ARE the device's graph. How the stock energy functions then reduce owned-only
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
    HasPositionsAndSystemIndex,
    IsState,
    OriginDeviceId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.core.utils.segment import segment_sum
from kups.core.utils.math import triangular_3x3_matmul
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
        return jnp.where(owned.reshape(owned.shape + (1,) * (x.ndim - 1)), x, 0.0)

    @override
    def combine_across_shards(self, x: Array) -> Array:
        return jax.lax.psum(x, shard_axis(OriginDeviceId))


class Partitioner[P: HasPositionsAndSystemIndex, S: HasCell[AnyPeriodicity]](Protocol):
    """Assigns each particle an owner device (any strategy)."""

    def __call__(
        self,
        particles: Table[ParticleId, P],
        systems: Table[SystemId, S],
        n_devices: int,
    ) -> Index[OriginDeviceId]: ...


def _morton_key(quantised: Array, bits: int) -> Array:
    """Interleave bits of integer coords ``(N, 3)`` into one Z-order key ``(N,)``."""
    key = jnp.zeros(quantised.shape[0], dtype=jnp.int32)
    for b in range(bits):
        for axis in range(3):
            key |= ((quantised[:, axis] >> b) & 1) << (3 * b + axis)
    return key


@dataclass
class MortonPartitioner:
    """Assign owners along a Z-order (Morton) space-filling curve, cut into equal chunks.

    Steps: positions -> fractional coords -> quantise each axis to ``bits``
    bins -> interleave the per-axis bits into one Morton key (so spatially-near
    particles get adjacent keys) -> sort -> cut the sorted order into
    ``n_devices`` equal-count contiguous chunks, one per device. Coherent
    (compact domains), balanced (counts differ by <=1), deterministic and
    shape-static. Re-applying as positions drift IS migration. (Distinct from
    the cell-list ``_cell_hash``, a row-major bin for neighbor search rather
    than a load-balancing curve.)
    """

    bits: int = field(static=True, default=10)

    def __call__(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, HasCell[AnyPeriodicity]],
        n_devices: int,
    ) -> Index[OriginDeviceId]:
        # 3 * bits interleaved bits must fit the int32 Morton key.
        assert 3 * self.bits <= 31, f"bits={self.bits} overflows the int32 Morton key"
        # Per-particle cell (gather each particle's own system cell, as the
        # neighbor-list pipeline does) so multi-system states fold correctly;
        # for one system this is N identical copies.
        cell = systems[particles.data.system].cell
        # Real -> fractional in [0, 1), then quantise each axis to `bits` bins.
        frac, _ = cell.fold(
            triangular_3x3_matmul(cell.inverse_vectors, particles.data.positions)
        )
        q = jnp.clip(
            (frac * (1 << self.bits)).astype(jnp.int32), 0, (1 << self.bits) - 1
        )
        # Position along the Z-order curve, cut into equal-count chunks.
        rank = jnp.argsort(jnp.argsort(_morton_key(q, self.bits)))
        return Index.integer(
            (rank * n_devices) // len(particles), n=n_devices, label=OriginDeviceId
        )


def owned_subset[P: IsDecomposedParticle](
    particles: Table[ParticleId, P], device_id: Array | int, cap_owned: Capacity[int]
) -> Index[ParticleId]:
    """Global ids of the atoms ``device_id`` owns, packed into a ``cap_owned`` buffer.

    Padding slots hold the OOB sentinel ``len(particles)`` so downstream
    gathers/queries drop them. Overflowing ``cap_owned`` would silently lose
    owned atoms, so the required size is recorded as a runtime assertion (it
    raises under the assertion interpreter; see ``kups.core.result``). The
    recorded requirement is the max owned count over ALL devices — the shared
    capacity must cover every device, and computing it from the replicated
    ``origin`` labels keeps the assertion device-invariant, so it can leave a
    ``shard_map`` through a replicated assertion context.
    """
    n = len(particles)
    origin = particles.data.origin
    counts = segment_sum(
        jnp.ones(n, dtype=jnp.int32), origin.indices, origin.num_labels, mode="drop"
    )
    cap_owned = cap_owned.generate_assertion(counts.max())
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
class Repartitioner[State, P: HasPositionsAndSystemIndex, S: HasCell[AnyPeriodicity]]:
    """Migration: run a partitioner and write ``origin`` back into the state."""

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    origin: Lens[State, Index[OriginDeviceId]] = field(static=True)
    partitioner: Partitioner[P, S] = field(static=True)
    n_devices: int = field(static=True)

    def __call__(self, state: State) -> State:
        origin = self.partitioner(
            self.particles(state), self.systems(state), self.n_devices
        )
        return self.origin.set(state, origin)


def make_repartitioner_from_state[
    State,
    P: IsDecomposedParticle,
    S: HasCell[AnyPeriodicity],
](
    state_lens: Lens[State, IsState[P, S]],
    partitioner: Partitioner[P, S],
    n_devices: int,
) -> Repartitioner[State, P, S]:
    """Wire a ``Repartitioner`` from a single state lens (the kUPS ``*_from_state`` shorthand)."""
    return Repartitioner(
        state_lens.focus(lambda s: s.particles),
        state_lens.focus(lambda s: s.systems),
        state_lens.focus(lambda s: s.particles).focus(lambda p: p.data.origin),
        partitioner,
        n_devices,
    )


@dataclass
class ShardedRadiusGraphConstructor[
    State,
    P: IsDecomposedParticle,
    S: HasCell[AnyPeriodicity],
]:
    """Build the calling device's owned-incident shard of the radius graph.

    A ``GraphConstructor``-shaped callable for full rebuilds: returns a
    ``HyperGraph`` over the GLOBAL particle table whose edges are this device's
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
        # An origin label outside the mesh would be owned by NO device and its
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
