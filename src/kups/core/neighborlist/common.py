# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Algorithmic core of the neighbor list module.

Provides the [`basic_neighborlist`][kups.core.neighborlist.common.basic_neighborlist]
engine and the shared helpers (candidate selection, distance/PBC filtering,
periodic-image expansion, edge compaction) that every concrete neighbor list
implementation composes.

The public-facing implementations (``CellListNeighborList``,
``DenseNearestNeighborList``, ``AllDenseNearestNeighborList``,
``RefineCutoffNeighborList``) all delegate to ``basic_neighborlist`` with
different
[`CandidateSelector`][kups.core.neighborlist.common.CandidateSelector] strategies.
"""

from __future__ import annotations

from typing import Literal, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import NeighborListPoints, NeighborListSystems
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, isin
from kups.core.utils.math import triangular_3x3_matmul


def _num_cells(
    systems: NeighborListSystems,
    cutoff: Array,
    *,
    eps: float = 1e-6,
) -> Array:
    inv_norms: jax.Array = jnp.linalg.norm(systems.cell.inverse_vectors, axis=-1)
    face_lengths = 1.0 / jnp.where(inv_norms < eps, jnp.ones_like(inv_norms), inv_norms)
    num_bins = jnp.maximum((face_lengths / cutoff[..., None]).astype(int), 1)
    return num_bins


@dataclass
class _Candidates:
    lhs: Index[ParticleId]
    rhs: Index[ParticleId]


class CandidateSelector(Protocol):
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
    ) -> _Candidates: ...


def _generate_image_offsets(images: jax.Array, out_size: Capacity[int]) -> jax.Array:
    """Generate centered coordinate grids from odd dimension specifications.

    Args:
        images: Array of shape (n, 3) containing odd numbers.
        out_size: Total number of output rows (sum of products of each row in images).

    Returns:
        Array of shape (m, 3) with centered coordinates.

    Example:
        ```python
        images = jnp.array([[3, 3, 1], [1, 1, 1]])
        out_size = FixedCapacity(10)  # 3*3*1 + 1*1*1 = 10
        coords = _generate_image_offsets(images, out_size)
        # First 9 rows (3x3x1 grid centered at origin, starting with [0,0,0]):
        # [[ 0,  0, 0],  # center first
        #  [ 1,  0, 0],
        #  [-1,  1, 0],
        #  [ 0,  1, 0],
        #  [ 1,  1, 0],
        #  [-1, -1, 0],
        #  [ 0, -1, 0],
        #  [ 1, -1, 0],
        #  [-1,  0, 0]]
        # Last 1 row (1x1x1 grid):
        # [[0, 0, 0]]
        ```
    """
    # Calculate total elements per row and cumulative sums for indexing
    counts = jnp.prod(images, axis=1)
    cumsum = jnp.cumsum(counts)
    out_size = out_size.generate_assertion(cumsum[-1])

    # Map each output index to its corresponding row in images
    indices = jnp.arange(out_size.size)
    row_indices = jnp.searchsorted(cumsum, indices, side="right")
    prev_cumsum = jnp.concatenate([jnp.zeros(1, dtype=counts.dtype), cumsum[:-1]])
    local_indices = indices - prev_cumsum[row_indices]
    dims = images[row_indices]

    # Convert flat local indices to 3D grid coordinates (i, j, k)
    ab = dims[:, 0] * dims[:, 1]
    a = dims[:, 0]
    half = (dims - 1) // 2

    # Shift indices so that [0,0,0] (the center) comes first
    center_flat = half[:, 0] + half[:, 1] * a + half[:, 2] * ab
    shifted = (local_indices + center_flat) % counts[row_indices]

    i = shifted % a
    j = (shifted // a) % dims[:, 1]
    k = shifted // ab

    # Center coordinates around origin by subtracting half the grid dimensions
    coords = jnp.stack([i, j, k], axis=1)
    return coords - half


def _candidate_image_counts(cells, cutoffs: Array) -> Array:
    """Return per-system, per-axis image counts for candidate replication.

    Periodic axes covered by the minimum-image convention use one image. Wider
    periodic cutoffs use a symmetric integer-shift stencil; open axes and
    non-finite cutoff/height ratios use one image.
    """
    ratio = cutoffs[..., None] / cells.perpendicular_lengths
    images = jnp.where(
        ratio <= 0.5,
        1,
        2 * jnp.ceil(ratio).astype(int) + 1,
    )
    images = jnp.where(jnp.isfinite(ratio), images, 1)
    return jnp.where(jnp.array(cells.periodic), images, 1)


def _get_candidate_images(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Array,
    out_size: Capacity[int],
) -> tuple[Array, Array, Array]:
    cells = systems.data.cell
    images = _candidate_image_counts(cells, cutoffs)
    images_per_sys = jnp.prod(images, axis=-1).astype(int)

    cand_sys_ids = lh.data.system.indices[candidates.lhs.indices]
    cand_per_sys = jnp.bincount(cand_sys_ids, length=systems.size)
    total_cand = jnp.vdot(cand_per_sys, images_per_sys)
    out_size = out_size.generate_assertion(total_cand)
    num_cands = candidates.lhs.size
    if out_size.size <= num_cands:
        offset = jnp.zeros((num_cands, 3), dtype=lh.data.positions.dtype)
        idx = jnp.arange(num_cands)
        return idx, offset, jnp.zeros((num_cands,), dtype=bool)

    offsets = _generate_image_offsets(images[cand_sys_ids], out_size)
    images_per_particle = images_per_sys[cand_sys_ids]
    idx = jnp.arange(num_cands + 1).repeat(
        jnp.pad(images_per_particle, (0, 1)),
        total_repeat_length=out_size.size,
    )
    has_been_replicated = (
        (images_per_particle > 1).at[idx].get(mode="fill", fill_value=False)
    )
    return idx, offsets, has_been_replicated


def _build_segmentation_mask(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    rh_index_remap: Array | None,
    out_of_bounds: int,
) -> Array:
    """Build a mask based on segmentation constraints."""
    ngraphs = lh.data.inclusion.num_labels
    lh_idx = candidates.lhs.indices
    rh_idx = candidates.rhs.indices
    rh_idx_out = (
        rh_index_remap.at[rh_idx].get(mode="fill", fill_value=out_of_bounds)
        if rh_index_remap is not None
        else rh_idx
    )
    lh_incl = lh.data.inclusion.indices[lh_idx]
    rh_incl = rh.data.inclusion.indices[rh_idx]

    mask = lh_incl == rh_incl
    mask &= (
        (lh.data.inclusion.indices < ngraphs)
        .at[lh_idx]
        .get(mode="fill", fill_value=False)
    )
    mask &= (
        (rh.data.inclusion.indices < ngraphs)
        .at[rh_idx]
        .get(mode="fill", fill_value=False)
    )

    if rh_index_remap is not None:
        mask &= ~isin(lh_idx, rh_index_remap, lh.size) | (lh_idx >= rh_idx_out)

    return mask


def _compute_distances_pbc_sq(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    shifts: Array | None = None,
) -> tuple[Array, Array]:
    """Compute squared distances honoring the cell's per-axis periodicity.

    If ``shifts`` is None, minimum-image shifts are computed via rounding on
    periodic axes; non-periodic axes get a zero shift (no MIC fold).
    """
    lattice_vecs = systems.map_data(lambda s: s.cell.vectors)
    vecs = lattice_vecs[lh.data.system[candidates.lhs.indices]]
    deltas = (
        lh.data.positions[candidates.lhs.indices]
        - rh.data.positions[candidates.rhs.indices]
    )
    if shifts is None:
        shifts = systems.data.cell.minimum_image_shifts(deltas)
    deltas -= shifts
    real_deltas = triangular_3x3_matmul(vecs, deltas)
    dist_sq = jnp.einsum("...d,...d->...", real_deltas, real_deltas)
    return dist_sq, shifts


@dataclass
class _DistanceCutoffResult:
    candidates: _Candidates
    original_candidate_idx: Array
    mask: Array
    shifts: Array
    is_minimum_interaction: Array


def _apply_distance_cutoff_wo_images(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
) -> _DistanceCutoffResult:
    """Apply distance cutoff with periodic boundaries, without image generation."""
    dist_sq, shifts = _compute_distances_pbc_sq(candidates, lh, rh, systems)
    cand_sys = lh.data.system[candidates.lhs.indices]
    return _DistanceCutoffResult(
        candidates,
        jnp.arange(candidates.lhs.size),
        dist_sq < cutoffs[cand_sys] ** 2,
        shifts,
        jnp.ones((candidates.lhs.size,), dtype=bool),
    )


def _apply_distance_cutoff_w_images(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    max_image_candidates: Capacity[int],
) -> _DistanceCutoffResult:
    """Apply distance cutoff with periodic boundaries and image generation."""
    idx, shifts, has_been_replicated = _get_candidate_images(
        candidates, lh, systems, cutoffs.data, max_image_candidates
    )
    if idx.size == candidates.lhs.size:
        return _apply_distance_cutoff_wo_images(candidates, lh, rh, systems, cutoffs)

    min_dist_sq, min_shifts = _compute_distances_pbc_sq(candidates, lh, rh, systems)
    candidates_w_images = bind(candidates).at(idx).get()
    dist_sq, shifts = _compute_distances_pbc_sq(
        candidates_w_images, lh, rh, systems, shifts
    )
    shifts = jnp.where(has_been_replicated[:, None], shifts, min_shifts[idx])
    dist_sq = jnp.where(has_been_replicated, dist_sq, min_dist_sq[idx])
    cand_sys = lh.data.system[candidates_w_images.lhs.indices]
    return _DistanceCutoffResult(
        candidates_w_images,
        idx,
        dist_sq < cutoffs[cand_sys] ** 2,
        shifts,
        (min_shifts[idx] == shifts).all(axis=-1),
    )


def _compute_distances_and_apply_cutoff(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    consider_images: bool,
    max_image_candidates: Capacity[int] | None,
) -> _DistanceCutoffResult:
    """Compute distances and apply cutoff filter."""
    if not consider_images:
        return _apply_distance_cutoff_wo_images(candidates, lh, rh, systems, cutoffs)
    if max_image_candidates is None:
        max_image_candidates = FixedCapacity(
            candidates.lhs.size,
            "Cutoff is larger than half the cell length, "
            "we need to generate additional images. "
            "Please provide a editable max_candidates.",
        )
    return _apply_distance_cutoff_w_images(
        candidates, lh, rh, systems, cutoffs, max_image_candidates
    )


def _compact_edges(
    candidates: _Candidates,
    mask: Array,
    shifts: Array,
    rh_index_remap: Array | None,
    max_num_edges: Capacity[int],
    out_of_bounds: int,
) -> Edges[Literal[2]]:
    """Compact valid edges and format output."""
    num_edges = mask.sum()
    max_num_edges = max_num_edges.generate_assertion(num_edges)
    sort_idxs = jnp.where(mask, size=max_num_edges.size, fill_value=mask.size)[0]
    shifts = shifts.at[sort_idxs].get(
        mode="fill", fill_value=0, indices_are_sorted=True
    )
    rh_idx_out = (
        rh_index_remap.at[candidates.rhs.indices].get(
            mode="fill", fill_value=out_of_bounds
        )
        if rh_index_remap is not None
        else candidates.rhs.indices
    )
    lh_edge, rh_edge = (
        c.at[sort_idxs].get(
            mode="fill", fill_value=out_of_bounds, indices_are_sorted=True
        )
        for c in (candidates.lhs.indices, rh_idx_out)
    )

    if rh_index_remap is not None:
        shifts = jnp.concatenate([shifts, -shifts], axis=0)
        lh_edge, rh_edge = (
            jnp.concatenate([lh_edge, rh_edge], axis=0),
            jnp.concatenate([rh_edge, lh_edge], axis=0),
        )

    shifts = jnp.expand_dims(shifts, axis=-2)
    edge_indices = Index(candidates.lhs.keys, jnp.stack([lh_edge, rh_edge], axis=-1))
    return Edges(edge_indices, shifts)


def _filter_candidates(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    rh_index_remap: Index[ParticleId] | None,
    *,
    max_num_edges: Capacity[int],
    max_image_candidates: Capacity[int] | None,
    consider_images: bool,
) -> Edges[Literal[2]]:
    out_of_bounds = max(rh.data.positions.shape[0], lh.data.positions.shape[0])

    # Convert Index[ParticleId] to raw array for leaf functions
    rh_remap_raw: Array | None = (
        rh_index_remap.indices_in(lh.keys) if rh_index_remap is not None else None
    )
    if rh_remap_raw is not None and rh_remap_raw.size == 0:
        rh_remap_raw = jnp.full((1,), out_of_bounds, dtype=int)

    mask = _build_segmentation_mask(candidates, lh, rh, rh_remap_raw, out_of_bounds)

    distance_result = _compute_distances_and_apply_cutoff(
        candidates,
        lh,
        rh,
        systems,
        cutoffs,
        consider_images,
        max_image_candidates,
    )
    mask = mask.at[distance_result.original_candidate_idx].get(
        mode="fill", fill_value=False
    )
    mask &= distance_result.mask
    shifts = distance_result.shifts
    candidates = distance_result.candidates

    # Exclusion mask: drop edges where exclusion segments match on minimum image
    lh_excl, rh_excl = Index.match(
        lh[candidates.lhs].exclusion, rh[candidates.rhs].exclusion
    )
    mask &= (lh_excl != rh_excl) | ~distance_result.is_minimum_interaction

    result = _compact_edges(
        candidates, mask, shifts, rh_remap_raw, max_num_edges, out_of_bounds
    )
    return result


def basic_neighborlist(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints] | None,
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    rh_index_remap: Index[ParticleId] | None,
    *,
    candidate_selector: CandidateSelector,
    max_num_edges: Capacity[int],
    max_image_candidates: Capacity[int] | None = None,
    consider_images: bool = True,
) -> Edges[Literal[2]]:
    """Core neighbor list construction algorithm with pluggable candidate selection."""
    cutoffs = Table.broadcast_to(cutoffs, systems)
    if rh is None:
        rh = lh

    # Transform coordinates to fractional using per-particle system data
    lh_inv = systems[lh.data.system].cell.inverse_vectors
    lh = (
        bind(lh)
        .focus(lambda x: x.data.positions)
        .apply(lambda r: triangular_3x3_matmul(lh_inv, r))
    )
    rh_inv = systems[rh.data.system].cell.inverse_vectors
    rh = (
        bind(rh)
        .focus(lambda x: x.data.positions)
        .apply(lambda r: triangular_3x3_matmul(rh_inv, r))
    )

    candidates = candidate_selector(lh, rh, systems)

    return _filter_candidates(
        candidates,
        lh,
        rh,
        systems,
        cutoffs,
        rh_index_remap,
        max_num_edges=max_num_edges,
        max_image_candidates=max_image_candidates,
        consider_images=consider_images,
    )
