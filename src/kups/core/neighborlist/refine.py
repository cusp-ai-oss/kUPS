# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Refinement neighbor lists that post-process precomputed edges.

These let one expensive base neighbor list be shared across multiple potentials
with different masking rules
([`RefineMaskNeighborList`][kups.core.neighborlist.refine.RefineMaskNeighborList])
or tighter cutoffs
([`RefineCutoffNeighborList`][kups.core.neighborlist.refine.RefineCutoffNeighborList]).
"""

from __future__ import annotations

from typing import Literal, overload

import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity
from kups.core.data import Index, Table
from kups.core.neighborlist.common import (
    Candidates,
    make_batch_with_mic,
)
from kups.core.neighborlist.compact import MaskOnlyCompactor, ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    TouchesQueriedKeysMask,
)
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.types import (
    CandidateBatch,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, jit


@dataclass
class PrecomputedEdgesSelector:
    """Selector that wraps precomputed ``Edges`` for both refine variants.

    Precomputed self-graph edges are already in ``keys`` space. A disjoint
    bipartite ``queries`` call may still use query positions for the second
    column, matching the edge convention of the original candidate set.

    Attributes:
        candidates: Precomputed edges (indices in keys-space).
        recompute_mic_shifts: When ``True``, drop the precomputed shifts and
            recompute minimum-image shifts on the current positions
            (``RefineCutoffNeighborList`` — the precomputed shifts may be
            stale relative to the current cell). When ``False``, reuse
            ``candidates.shifts`` directly (``RefineMaskNeighborList``).
            ``is_minimum_image`` is always all-True (no image replication).
    """

    candidates: Edges[Literal[2]]
    recompute_mic_shifts: bool = field(static=True, default=False)

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        if self.recompute_mic_shifts:
            indices = self.candidates.indices.indices
            query = ctx.edge_query_table
            raw_candidates = Candidates(
                key_idx=Index(ctx.keys.keys, indices[:, 0]),
                query_idx=Index(query.keys, indices[:, 1]),
            )
            return make_batch_with_mic(raw_candidates, ctx.keys, query, ctx.systems)
        indices = self.candidates.indices.indices
        edges = Edges(Index(ctx.keys.keys, indices), self.candidates.shifts)
        return CandidateBatch(
            edges=edges,
            is_minimum_image=jnp.ones((len(self.candidates),), dtype=bool),
            query_keys=ctx.edge_query_table.keys,
        )


def _resolve_precomputed_inputs(
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints] | None,
    queried_keys: Index[ParticleId] | None,
) -> tuple[
    Table[ParticleId, NeighborListPoints],
    Table[ParticleId, NeighborListPoints] | None,
]:
    """Return the tables that precomputed refine candidates address."""
    assert queries is None or queried_keys is None, (
        "Refine neighbor lists cannot combine queries with queried_keys."
    )
    return keys, queries


@dataclass
class RefineMaskNeighborList:
    """Refine a precomputed neighbor list by applying inclusion/exclusion masks.

    This neighbor list takes an existing set of candidate edges and filters them
    based on segmentation masks, without recomputing distances. Enables sharing
    a single base neighbor list across multiple potentials with different
    interaction rules.

    **Key benefit**: Compute expensive neighbor list once, apply different masks
    for different potentials (e.g., Lennard-Jones excludes 1-4 interactions,
    Coulomb has different exclusions).

    Attributes:
        candidates: Precomputed edges to refine

    Use cases:
        - Multiple potentials sharing one neighbor list with different exclusions
        - Excluding bonded pairs (1-2, 1-3, 1-4) from non-bonded interactions
        - Applying group-specific interaction rules
        - Multi-scale simulations with different interaction levels

    Example:
        ```python
        # Compute base neighbor list once
        base_edges = base_nl(particles, cells)

        # Share across potentials with different masks
        lj_nl = RefineMaskNeighborList(candidates=base_edges)
        lj_edges = lj_nl(lj_particles, cells)  # 1-4 exclusions

        coulomb_nl = RefineMaskNeighborList(candidates=base_edges)
        coulomb_edges = coulomb_nl(coulomb_particles, cells)  # 1-2 exclusions only
        ```
    """

    candidates: Edges[Literal[2]]

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> Edges[Literal[2]]: ...

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]: ...

    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        resolved_keys, resolved_queries = _resolve_precomputed_inputs(
            keys, queries, queried_keys
        )
        pipeline = Pipeline[Literal[2]](
            selector=PrecomputedEdgesSelector(self.candidates),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                TouchesQueriedKeysMask(),
                ExclusionMask(),
            ),
            compactor=MaskOnlyCompactor(),
        )
        if resolved_queries is not None:
            return pipeline(resolved_keys, systems, queries=resolved_queries)
        return pipeline(resolved_keys, systems, queried_keys=queried_keys)


@dataclass
class RefineCutoffNeighborList:
    """Refine precomputed edges by re-checking distances with new cutoffs.

    This neighbor list takes an existing set of candidate edges and filters them
    by computing actual distances and comparing to cutoffs. Enables sharing a
    single conservative neighbor list across multiple potentials with different
    cutoff distances.

    **Key benefit**: Compute expensive neighbor list once with maximum cutoff,
    then refine for each potential with its specific cutoff (e.g., Lennard-Jones
    at 10 Å, Coulomb at 15 Å).

    Attributes:
        candidates: Precomputed edges to refine (should be conservative/over-inclusive).
        avg_edges: Capacity for output edge array.
        cutoffs: Per-system cutoff distances used by this refinement.

    Use cases:
        - Multiple potentials sharing one neighbor list with different cutoffs
        - Multi-stage neighbor list construction (coarse then fine)
        - Adaptive cutoffs that change during simulation
        - Using a static "super" neighbor list with varying actual cutoffs

    Example:
        ```python
        # Compute base neighbor list once with maximum cutoff
        max_cutoff = 15.0  # Maximum of all potential cutoffs
        base_edges = base_nl(particles, cells)

        # Share across potentials with different cutoffs
        lj_nl = RefineCutoffNeighborList(
            candidates=base_edges, avg_edges=cap1, cutoffs=lj_cutoffs
        )
        lj_edges = lj_nl(particles, cells)  # LJ cutoff

        coulomb_nl = RefineCutoffNeighborList(
            candidates=base_edges, avg_edges=cap2, cutoffs=coulomb_cutoffs
        )
        coulomb_edges = coulomb_nl(particles, cells)  # Coulomb cutoff
        ```
    """

    candidates: Edges[Literal[2]]
    avg_edges: Capacity[int]
    cutoffs: Table[SystemId, Array]

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> Edges[Literal[2]]: ...

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]: ...

    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        resolved_keys, resolved_queries = _resolve_precomputed_inputs(
            keys, queries, queried_keys
        )
        query_size = (
            queried_keys.size
            if queried_keys is not None
            else (queries.size if queries is not None else keys.size)
        )
        cutoffs = Table.broadcast_to(self.cutoffs, systems)
        pipeline = Pipeline[Literal[2]](
            selector=PrecomputedEdgesSelector(
                self.candidates, recompute_mic_shifts=True
            ),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                TouchesQueriedKeysMask(),
                ExclusionMask(),
            ),
            compactor=ReduceCompactor(avg_edges=self.avg_edges.multiply(query_size)),
        )
        if resolved_queries is not None:
            return pipeline(resolved_keys, systems, queries=resolved_queries)
        return pipeline(resolved_keys, systems, queried_keys=queried_keys)
