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

from typing import Literal

import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity
from kups.core.data import Index, Table
from kups.core.neighborlist.common import Candidates, make_batch_with_mic
from kups.core.neighborlist.compact import MaskOnlyCompactor, ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
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

    Precomputed edges use the same index convention as the call that produced
    them: public lh-space edges when an ``rh`` remap is supplied, and raw
    rh-space indices for a disjoint ``rh`` without a remap. Remapped ``rh``
    rows are overlaid onto ``lh`` before this selector runs.

    Attributes:
        candidates: Precomputed edges (indices in lh-space).
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
            raw_candidates = Candidates(
                lhs=Index(ctx.lh.keys, indices[:, 0]),
                rhs=Index(ctx.rh.keys, indices[:, 1]),
            )
            return make_batch_with_mic(raw_candidates, ctx.lh, ctx.rh, ctx.systems)
        indices = self.candidates.indices.indices
        edges = Edges(Index(ctx.lh.keys, indices), self.candidates.shifts)
        return CandidateBatch(
            edges=edges,
            is_minimum_image=jnp.ones((len(self.candidates),), dtype=bool),
        )


def _resolve_precomputed_inputs(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints] | None,
    rh_index_remap: Index[ParticleId] | None,
) -> tuple[
    Table[ParticleId, NeighborListPoints],
    Table[ParticleId, NeighborListPoints] | None,
]:
    """Return the tables that precomputed refine candidates address.

    With a remap, refine candidates are public ``Edges`` outputs, so both
    columns are in lh-space. Overlay rh rows onto the corresponding lh slots
    and run the pipeline as a self-refinement. Without a remap, rh is disjoint
    and the candidate right column is already in rh-space.
    """
    if rh is None:
        return lh, None
    if rh_index_remap is None:
        return lh, rh
    return lh.update(rh_index_remap, rh.data), None


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
        base_edges = base_nl(particles, None, cells, cutoffs, None)

        # Share across potentials with different masks
        lj_nl = RefineMaskNeighborList(candidates=base_edges)
        lj_edges = lj_nl(lj_particles, None, cells, cutoffs, None)  # 1-4 exclusions

        coulomb_nl = RefineMaskNeighborList(candidates=base_edges)
        coulomb_edges = coulomb_nl(coulomb_particles, None, cells, cutoffs, None)  # 1-2 exclusions only
        ```
    """

    candidates: Edges[Literal[2]]

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        resolved_lh, resolved_rh = _resolve_precomputed_inputs(lh, rh, rh_index_remap)
        pipeline = Pipeline[Literal[2]](
            selector=PrecomputedEdgesSelector(self.candidates),
            masks=(InBoundsMask(), InclusionMatchMask(), ExclusionMask()),
            compactor=MaskOnlyCompactor(),
        )
        return pipeline(resolved_lh, resolved_rh, systems, None)


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

    Use cases:
        - Multiple potentials sharing one neighbor list with different cutoffs
        - Multi-stage neighbor list construction (coarse then fine)
        - Adaptive cutoffs that change during simulation
        - Using a static "super" neighbor list with varying actual cutoffs

    Example:
        ```python
        # Compute base neighbor list once with maximum cutoff
        max_cutoff = 15.0  # Maximum of all potential cutoffs
        base_edges = base_nl(particles, None, cells, max_cutoff, None)

        # Share across potentials with different cutoffs
        lj_nl = RefineCutoffNeighborList(candidates=base_edges, avg_edges=cap1)
        lj_edges = lj_nl(particles, None, cells, cutoff=10.0, None)  # LJ cutoff

        coulomb_nl = RefineCutoffNeighborList(candidates=base_edges, avg_edges=cap2)
        coulomb_edges = coulomb_nl(particles, None, cells, cutoff=15.0, None)  # Coulomb cutoff
        ```
    """

    candidates: Edges[Literal[2]]
    avg_edges: Capacity[int]

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        resolved_lh, resolved_rh = _resolve_precomputed_inputs(lh, rh, rh_index_remap)
        rh_size = rh.size if rh is not None else lh.size
        cutoffs = Table.broadcast_to(cutoffs, systems)
        pipeline = Pipeline[Literal[2]](
            selector=PrecomputedEdgesSelector(
                self.candidates, recompute_mic_shifts=True
            ),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                ExclusionMask(),
            ),
            compactor=ReduceCompactor(avg_edges=self.avg_edges.multiply(rh_size)),
        )
        return pipeline(resolved_lh, resolved_rh, systems, None)
