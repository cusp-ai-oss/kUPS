# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Distance-agnostic pair list connecting all particles in an inclusion segment.

[`all_connected_neighborlist`][kups.core.neighborlist.all_connected.all_connected_neighborlist]
ignores the cutoff and emits every pair sharing the same inclusion segment that
has differing exclusion segment IDs. Used by Ewald summation to enumerate the
real-space exclusion list.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.data import Index, Table, subselect
from kups.core.neighborlist.common import _Candidates, _compact_edges
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import NeighborListPoints, NeighborListSystems
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.math import triangular_3x3_matmul


def all_connected_neighborlist(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints] | None,
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    rh_index_remap: Index[ParticleId] | None = None,
) -> Edges[Literal[2]]:
    """Neighbor list connecting all pairs sharing the same inclusion segment, ignoring distance.

    Connects every particle pair that belongs to the same inclusion segment and has
    differing exclusion segment IDs. The cutoff is ignored for neighbor selection;
    the cell is used only to compute minimum-image shifts.

    Requires ``max_count`` to be set on the inclusion ``Index``.
    """
    if rh is None:
        rh = lh
        rh_index_remap = Index.arange(len(lh), label=ParticleId)

    ngraphs = lh.data.inclusion.num_labels
    max_count = lh.data.inclusion.max_count
    assert max_count is not None, "inclusion.max_count must be set"
    capacity = FixedCapacity(max_count).multiply(min(lh.size, rh.size))
    out_of_bounds = max(lh.size, rh.size)

    lh_sys = systems[lh.data.system]
    rh_sys = systems[rh.data.system]

    selection_result = subselect(
        lh.data.inclusion.indices,
        rh.data.inclusion.indices,
        output_buffer_size=capacity,
        num_segments=ngraphs,
    )
    candidates = _Candidates(
        lhs=Index(lh.keys, selection_result.scatter_idxs),
        rhs=Index(rh.keys, selection_result.gather_idxs),
    )
    lh_idx, rh_idx = candidates.lhs, candidates.rhs
    lh_data, rh_data = lh[lh_idx], rh[rh_idx]
    lh_excl, rh_excl = Index.match(lh_data.exclusion, rh_data.exclusion)
    mask = lh_excl != rh_excl
    if rh_index_remap is not None:
        lh_i, rh_i = Index.match(lh_idx, rh.set_data(rh_index_remap)[rh_idx])
        mask &= ~lh_idx.isin(rh_index_remap) | (lh_i >= rh_i)

    lh_frac = triangular_3x3_matmul(lh_sys.cell.inverse_vectors, lh.data.positions)
    rh_frac = triangular_3x3_matmul(rh_sys.cell.inverse_vectors, rh.data.positions)
    shifts = jnp.round(lh_frac[lh_idx.indices] - rh_frac[rh_idx.indices]).astype(int)
    return _compact_edges(
        candidates,
        mask,
        shifts,
        rh_index_remap.indices_in(lh.keys) if rh_index_remap is not None else None,
        capacity,
        out_of_bounds,
    )
