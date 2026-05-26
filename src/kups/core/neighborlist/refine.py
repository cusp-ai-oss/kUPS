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
from kups.core.neighborlist.common import _Candidates, basic_neighborlist
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import NeighborListPoints, NeighborListSystems
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit
from kups.core.utils.ops import where_broadcast_last


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
        lh_c = self.candidates.indices[:, 0]
        rh_c = self.candidates.indices[:, 1]
        lh_d, rh_d = lh[lh_c], lh[rh_c]
        lh_incl, rh_incl = Index.match(lh_d.inclusion, rh_d.inclusion)
        lh_excl, rh_excl = Index.match(lh_d.exclusion, rh_d.exclusion)
        mask = lh_incl == rh_incl
        mask &= lh_excl != rh_excl
        indices = where_broadcast_last(mask, self.candidates.indices.indices, lh.size)
        shifts = where_broadcast_last(mask, self.candidates.shifts, 0)
        return Edges(Index(self.candidates.indices.keys, indices), shifts)


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
        rh_remap_raw = (
            rh_index_remap.indices_in(lh.keys) if rh_index_remap is not None else None
        )

        if rh_remap_raw is not None:
            assert rh is not None
            inv_rh_index_remap = jnp.full(lh.size, rh.size, dtype=int)
            inv_rh_index_remap = inv_rh_index_remap.at[rh_remap_raw].set(
                jnp.arange(rh.size, dtype=int)
            )
        else:
            inv_rh_index_remap = None

        def _cand_selector(
            lh: Table[ParticleId, NeighborListPoints],
            rh: Table[ParticleId, NeighborListPoints],
            systems: Table[SystemId, NeighborListSystems],
        ) -> _Candidates:
            rh_c = self.candidates.indices[:, 1].indices
            if inv_rh_index_remap is not None:
                rh_c = inv_rh_index_remap.at[rh_c].get(mode="fill", fill_value=len(lh))
            return _Candidates(self.candidates.indices[:, 0], Index(rh.keys, rh_c))

        rh_size = rh.size if rh is not None else lh.size
        return basic_neighborlist(
            lh,
            rh,
            systems,
            cutoffs,
            rh_index_remap,
            candidate_selector=_cand_selector,
            max_num_edges=self.avg_edges.multiply(rh_size),
            consider_images=False,
        )
