# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared algorithmic helpers for neighbor list selectors and masks.

Contains:

- ``num_cells`` — per-axis spatial bin counts (used by the cell-list
  selector and by ``parameters.estimate``).
- ``Candidates`` — private intermediate struct used inside individual
  selector algorithms while raw ``(key_idx, query_idx)`` index arrays are being
  built. Not the pipeline carrier (see
  [`CandidateBatch`][kups.core.neighborlist.types.CandidateBatch]).
- ``candidate_image_counts`` — per-axis periodic-image window width a cutoff
  reaches; used by ``_get_candidate_images`` and by ``parameters.estimate``.
- ``_generate_image_offsets``, ``_get_candidate_images`` — image-expansion
  primitives (per-pair anchored windows).
- ``replicate_for_images`` — adapts raw ``Candidates`` into a
  ``CandidateBatch`` with shifts and ``is_minimum_image`` set, replicating
  each pair across its anchored image window when ``cutoff > perp/2``.
- ``make_batch_with_mic`` — pack raw candidates with minimum-image shifts
  and ``is_minimum_image=all-True`` (used by selectors that don't replicate).
- ``real_distance_sq`` — squared real-space distance between candidate
  pairs given fractional shifts; used by ``DistanceCutoffMask``.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.cell import AnyPeriodicity, Cell, MaterializedFrame
from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import (
    CandidateBatch,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


def num_cells(
    systems: NeighborListSystems,
    cutoff: Array,
    *,
    eps: float = 1e-6,
    dtype: DTypeLike = int,
) -> Array:
    # Reciprocal vectors are the columns of inverse_vectors (row-vector lattices).
    inv_norms: jax.Array = jnp.linalg.norm(systems.cell.inverse_vectors, axis=-2)
    face_lengths = 1.0 / jnp.where(inv_norms < eps, jnp.ones_like(inv_norms), inv_norms)
    cutoff = jnp.where(cutoff > 0, cutoff, jnp.inf)  # a zero cutoff gets one bin
    num_bins = jnp.maximum((face_lengths / cutoff[..., None]).astype(dtype), 1)
    return num_bins


@dataclass
class Candidates:
    """Private intermediate produced inside selector algorithms.

    Not the pipeline carrier — selectors convert ``Candidates`` into a
    ``CandidateBatch`` (via ``replicate_for_images`` or
    ``make_batch_with_mic``) before returning.
    """

    key_idx: Index[ParticleId]
    query_idx: Index[ParticleId]


def lift_query_candidates(candidates: Candidates, ctx: PipelineContext) -> Candidates:
    """Convert query-local self-update candidates to ``ctx.keys`` positions."""
    if ctx.queried_keys is None:
        return candidates
    queried = Table.arange(ctx.queried_keys, label=ParticleId)
    return Candidates(candidates.key_idx, queried[candidates.query_idx])


def _generate_image_offsets(images: jax.Array, out_size: Capacity[int]) -> jax.Array:
    """Generate 0-based coordinate ramps from per-row window widths.

    Each row of ``images`` gives a per-axis width; the output enumerates that
    axis-aligned window as coordinates in ``{0 .. width - 1}``, the center
    element ``(width - 1) // 2`` emitted first per block (so the row closest to
    the minimum image leads, giving a stable tie-break downstream).

    Args:
        images: Array of shape (n, 3) of per-axis window widths (>= 1).
        out_size: Total number of output rows (sum of products of each row in images).

    Returns:
        Array of shape (m, 3) with 0-based window coordinates.

    Example:
        ```python
        images = jnp.array([[3, 3, 1], [1, 1, 1]])
        out_size = FixedCapacity(10)  # 3*3*1 + 1*1*1 = 10
        coords = _generate_image_offsets(images, out_size)
        # First 9 rows (3x3x1 window, center (1,1,0) first):
        # [[1, 1, 0],  # center first
        #  [2, 1, 0],
        #  [0, 2, 0],
        #  [1, 2, 0],
        #  [2, 2, 0],
        #  [0, 0, 0],
        #  [1, 0, 0],
        #  [2, 0, 0],
        #  [0, 1, 0]]
        # Last 1 row (1x1x1 window):
        # [[0, 0, 0]]
        ```
    """
    # Calculate total elements per row and cumulative sums for indexing
    counts = jnp.prod(images, axis=1, dtype=images.dtype)
    cumsum = jnp.cumsum(counts, dtype=images.dtype)
    out_size = out_size.generate_assertion(cumsum[-1])

    # Map each output index to its corresponding row in images
    indices = jnp.arange(out_size.size, dtype=images.dtype)
    row_indices = jnp.searchsorted(cumsum, indices, side="right")
    prev_cumsum = jnp.concatenate([jnp.zeros(1, dtype=counts.dtype), cumsum[:-1]])
    local_indices = indices - prev_cumsum[row_indices]
    dims = images[row_indices]

    # Convert flat local indices to 3D grid coordinates (i, j, k)
    ab = dims[:, 0] * dims[:, 1]
    a = dims[:, 0]
    half = (dims - 1) // 2

    # Shift indices so the window center comes first
    center_flat = half[:, 0] + half[:, 1] * a + half[:, 2] * ab
    shifted = (local_indices + center_flat) % counts[row_indices]

    i = shifted % a
    j = (shifted // a) % dims[:, 1]
    k = shifted // ab

    return jnp.stack([i, j, k], axis=1)


def candidate_image_counts(
    cells: Cell[AnyPeriodicity], cutoffs: Array, dtype: DTypeLike = int
) -> Array:
    """Return per-system, per-axis periodic-image window widths.

    Each periodic axis needs ``ceil(2 * cutoff / perpendicular_length)``
    consecutive integer shifts -- the tight count of lattice planes a sphere of
    radius ``cutoff`` can reach along that axis under the strict ``< cutoff``
    distance mask (the max number of integers in an open interval of that
    width). At ``ratio <= 0.5`` this collapses to one image, recovering the
    minimum-image convention. Open axes and non-finite ratios use one image.
    ``perpendicular_lengths`` is the correct per-axis measure for arbitrary
    skew (it equals ``1 / |column of inverse_vectors|``).
    """
    ratio = cutoffs[..., None] / cells.perpendicular_lengths
    images = jnp.maximum(jnp.ceil(2 * ratio), 1).astype(dtype)
    images = jnp.where(jnp.isfinite(ratio), images, 1)
    return jnp.where(jnp.array(cells.periodic), images, 1)


def _get_candidate_images(
    candidates: Candidates,
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Array,
    out_size: Capacity[int],
) -> tuple[Array, Array]:
    """Replicate each candidate across its periodic-image window.

    Returns ``(idx, offsets)``: ``idx`` maps each replicated row back to its
    original candidate; ``offsets`` are integer fractional shifts. Each
    candidate's window is a per-axis run of ``candidate_image_counts``
    consecutive shifts anchored at ``ceil(separation - cutoff/perp)``, which
    brackets every image of the pair within ``cutoff`` (the reciprocal-projection
    bound ``|separation_i - n_i| <= cutoff / perpendicular_length_i``).

    If the image buffer is no larger than the candidate buffer, the result is one
    zero-shift copy per candidate, which is exact only if no candidate has a
    second image. The capacity check therefore asks for the total number of
    images, and for at least one row more than the candidates if any candidate
    has a second image. The capacity fix then grows the buffer past the candidate
    buffer, and the candidates are copied once per image.
    """
    cells = systems.data.cell
    dtype = candidates.key_idx.raw_indices.dtype
    key_points = keys[candidates.key_idx]
    # Out-of-bounds candidates take no images.
    images, ratio = (
        systems.set_data(
            (
                candidate_image_counts(cells, cutoffs, dtype),
                cutoffs[:, None] / cells.perpendicular_lengths,
            )
        )
        .at(key_points.system, args={"mode": "fill", "fill_value": 0})
        .get()
    )
    per_candidate = jnp.prod(images, axis=-1, dtype=dtype)
    num_cands = candidates.key_idx.size
    total = per_candidate.sum(dtype=dtype)
    required = jnp.where(
        jnp.any(per_candidate > 1), jnp.maximum(total, num_cands + 1), total
    )
    out_size = out_size.generate_assertion(required)
    if num_cands == 0 or out_size.size <= num_cands:
        offset = jnp.zeros((num_cands, 3), dtype=keys.data.positions.dtype)
        return jnp.arange(num_cands, dtype=dtype), offset

    window = _generate_image_offsets(images, out_size)
    idx = jnp.arange(num_cands + 1, dtype=dtype).repeat(
        jnp.pad(per_candidate, (0, 1)), total_repeat_length=out_size.size
    )
    separation = key_points.positions - queries[candidates.query_idx].positions
    # Anchor each window at ceil(separation - ratio) so it brackets every in-range
    # image of the pair. Where the ratio is non-finite (infinite cutoff or
    # degenerate axis) the window has width 1 and sits on the minimum image;
    # non-periodic axes use no shift.
    mic = cells.minimum_image_shifts(separation)
    anchor = jnp.where(jnp.isfinite(ratio), jnp.ceil(separation - ratio), mic)
    anchor = jnp.where(jnp.array(cells.periodic), anchor, 0).astype(dtype)
    offsets = (window + anchor.at[idx].get(mode="fill", fill_value=0)).astype(
        keys.data.positions.dtype
    )
    return idx, offsets


def _minimum_image_shifts(
    candidates: Candidates,
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> Array:
    """Compute minimum-image fractional shifts for each candidate pair."""
    deltas = (
        keys[candidates.key_idx].positions - queries[candidates.query_idx].positions
    )
    return systems.data.cell.minimum_image_shifts(deltas)


def _minimum_image_mask(
    replicated: Candidates,
    offsets: Array,
    idx: Array,
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    num_candidates: int,
) -> Array:
    """Flag the closest periodic image of each candidate (one ``True`` per pair).

    Marks, per original candidate, the replicated copy with the smallest real
    distance via a segment argmin. This is the exact minimum image even on skewed
    cells, where round-based shifts can pick the wrong copy. Distance ties resolve
    to the first emitted (window center) copy.
    """
    frames = systems.map_data(lambda s: s.cell.frame.materialize())
    key_points = keys[replicated.key_idx]
    dist_sq = real_distance_sq(
        key_points.positions,
        queries[replicated.query_idx].positions,
        frames[key_points.system],
        offsets,
    )
    num_segments = num_candidates + 1
    seg_min = jax.ops.segment_min(dist_sq, idx, num_segments=num_segments)
    rows = jnp.arange(dist_sq.shape[0], dtype=idx.dtype)
    first = jax.ops.segment_min(
        jnp.where(dist_sq == seg_min[idx], rows, dist_sq.shape[0]),
        idx,
        num_segments=num_segments,
    )
    # idx == num_candidates marks sentinel padding rows; never flag those.
    return (rows == first[idx]) & (idx < num_candidates)


def make_batch_with_mic(
    candidates: Candidates,
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> CandidateBatch[Literal[2]]:
    """Pack raw candidates with minimum-image shifts; ``is_minimum_image=all-True``."""
    min_shifts = _minimum_image_shifts(candidates, keys, queries, systems)
    return candidates_to_batch(
        candidates,
        min_shifts,
        jnp.ones((candidates.key_idx.size,), dtype=bool),
    )


def candidates_to_batch(
    candidates: Candidates,
    shifts: Array,
    is_minimum_image: Array,
) -> CandidateBatch[Literal[2]]:
    """Pack ``(candidates, flat shifts, is_min)`` into a ``CandidateBatch[2]``."""
    # Bipartite candidates index the union of the disjoint key and query ids.
    indices = Index.stack(candidates.key_idx, candidates.query_idx, axis=-1)
    edges: Edges[Literal[2]] = Edges(indices, jnp.expand_dims(shifts, axis=-2))
    return CandidateBatch(edges=edges, is_minimum_image=is_minimum_image)


def replicate_for_images(
    candidates: Candidates,
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    max_image_candidates: Capacity[int] | None,
) -> CandidateBatch[Literal[2]]:
    """Replicate candidates across their periodic-image windows.

    For each candidate pair:
    - If ``cutoff[sys] / perp_axes <= 0.5`` on every axis: emit 1 copy with MIC
      shifts (the minimum image is the only image in range).
    - Otherwise: emit the per-axis window of integer shifts anchored at the pair's
      separation (see ``_get_candidate_images``); ``is_minimum_image`` flags the
      closest copy per pair (via real-distance argmin) so ``ExclusionMask`` keeps
      non-minimum image periodic copies of excluded pairs. Over-emitted copies are
      pruned downstream by ``DistanceCutoffMask``.

    Args:
        candidates: Raw candidate pair indices.
        keys, queries, systems: Pipeline tables (fractional coords).
        cutoffs: Per-system cutoff.
        max_image_candidates: Capacity for replicated-candidates buffer.
            When ``None``, falls back to ``FixedCapacity(candidates.key_idx.size)``
            with an error message — pass an editable capacity if image
            replication is expected.

    Returns:
        ``CandidateBatch`` with shifts populated and ``is_minimum_image`` set.
    """
    cutoffs_t = Table.broadcast_to(cutoffs, systems)
    if max_image_candidates is None:
        max_image_candidates = FixedCapacity(
            candidates.key_idx.size,
            "Cutoff is larger than half the cell length, "
            "we need to generate additional images. "
            "Please provide a editable max_candidates.",
        )

    idx, offsets = _get_candidate_images(
        candidates, keys, queries, systems, cutoffs_t.data, max_image_candidates
    )

    if idx.size == candidates.key_idx.size:
        # No replication needed — MIC shifts cover everything.
        min_shifts = _minimum_image_shifts(candidates, keys, queries, systems)
        return candidates_to_batch(
            candidates, min_shifts, jnp.ones((candidates.key_idx.size,), dtype=bool)
        )

    replicated = bind(candidates).at(idx).get()
    is_min = _minimum_image_mask(
        replicated, offsets, idx, keys, queries, systems, candidates.key_idx.size
    )
    return candidates_to_batch(replicated, offsets, is_min)


def real_distance_sq(
    key_positions: Array,
    query_positions: Array,
    frames: MaterializedFrame,
    shifts: Array,
) -> Array:
    """Squared real-space distance between already-broadcast candidate pairs.

    Args:
        key_positions: Fractional left endpoint positions, shape ``(n, 3)``.
        query_positions: Fractional right endpoint positions, shape ``(n, 3)``.
        frames: Materialized cell frames broadcast to the candidate key system.
        shifts: ``(n, 3)`` fractional shifts.

    Returns:
        ``(n,)`` array of squared distances in real coordinates.
    """
    deltas = key_positions - query_positions - shifts
    real_deltas = frames.to_real(deltas)
    return jnp.einsum("...d,...d->...", real_deltas, real_deltas)
