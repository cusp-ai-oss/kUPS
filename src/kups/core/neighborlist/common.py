# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared algorithmic helpers for neighbor list selectors and masks.

Contains:

- ``num_cells`` — per-axis spatial bin counts (used by the cell-list
  selector and by ``parameters.estimate``).
- ``Candidates`` — private intermediate struct used inside individual
  selector algorithms while raw ``(lhs, rhs)`` index arrays are being built.
  Not the pipeline carrier (see
  [`CandidateBatch`][kups.core.neighborlist.types.CandidateBatch]).
- ``_generate_image_offsets``, ``_get_candidate_images`` — image-expansion
  primitives.
- ``replicate_for_images`` — adapts raw ``Candidates`` into a
  ``CandidateBatch`` with shifts and ``is_minimum_image`` set, replicating
  per image multiplicity when ``cutoff > perp/2``.
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

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.cell import MaterializedFrame
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
) -> Array:
    inv_norms: jax.Array = jnp.linalg.norm(systems.cell.inverse_vectors, axis=-1)
    face_lengths = 1.0 / jnp.where(inv_norms < eps, jnp.ones_like(inv_norms), inv_norms)
    num_bins = jnp.maximum((face_lengths / cutoff[..., None]).astype(int), 1)
    return num_bins


@dataclass
class Candidates:
    """Private intermediate produced inside selector algorithms.

    Not the pipeline carrier — selectors convert ``Candidates`` into a
    ``CandidateBatch`` (via ``replicate_for_images`` or
    ``make_batch_with_mic``) before returning.
    """

    lhs: Index[ParticleId]
    rhs: Index[ParticleId]


def edge_rhs_table(ctx: PipelineContext) -> Table[ParticleId, NeighborListPoints]:
    """Return the table addressed by the second edge column."""
    return ctx.rh if ctx.rh is not None else ctx.lh


def query_table(ctx: PipelineContext) -> Table[ParticleId, NeighborListPoints]:
    """Return the table used to enumerate rhs/query candidates.

    ``rh`` is reserved for true bipartite calls. ``for_indices`` selects a
    self-graph update subset from ``lh`` and is lifted back to ``lh`` index
    space before masks and compaction see the batch.
    """
    if ctx.rh is not None:
        return ctx.rh
    if ctx.for_indices is None:
        return ctx.lh
    return ctx.lh.subset(Index(ctx.lh.keys, ctx.for_indices))


def lift_query_candidates(candidates: Candidates, ctx: PipelineContext) -> Candidates:
    """Convert query-local self-update candidates to ``ctx.lh`` positions."""
    if ctx.for_indices is None:
        return candidates
    oob = ctx.lh.size
    rhs = ctx.for_indices.at[candidates.rhs.indices].get(mode="fill", fill_value=oob)
    return Candidates(lhs=candidates.lhs, rhs=Index(ctx.lh.keys, rhs))


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
    candidates: Candidates,
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


def _minimum_image_shifts(
    candidates: Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> Array:
    """Compute minimum-image fractional shifts for each candidate pair."""
    deltas = (
        lh.data.positions[candidates.lhs.indices]
        - rh.data.positions[candidates.rhs.indices]
    )
    return systems.data.cell.minimum_image_shifts(deltas)


def make_batch_with_mic(
    candidates: Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> CandidateBatch[Literal[2]]:
    """Pack raw candidates with minimum-image shifts; ``is_minimum_image=all-True``."""
    min_shifts = _minimum_image_shifts(candidates, lh, rh, systems)
    return candidates_to_batch(
        candidates,
        min_shifts,
        jnp.ones((candidates.lhs.size,), dtype=bool),
    )


def candidates_to_batch(
    candidates: Candidates,
    shifts: Array,
    is_minimum_image: Array,
) -> CandidateBatch[Literal[2]]:
    """Pack ``(candidates, flat shifts, is_min)`` into a ``CandidateBatch[2]``."""
    indices_2d = jnp.stack([candidates.lhs.indices, candidates.rhs.indices], axis=-1)
    edges: Edges[Literal[2]] = Edges(
        Index(candidates.lhs.keys, indices_2d),
        jnp.expand_dims(shifts, axis=-2),
    )
    return CandidateBatch(
        edges=edges,
        is_minimum_image=is_minimum_image,
        rhs_keys=candidates.rhs.keys,
    )


def replicate_for_images(
    candidates: Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    max_image_candidates: Capacity[int] | None,
) -> CandidateBatch[Literal[2]]:
    """Replicate candidates by image multiplicity, attaching shifts and is-min flag.

    For each candidate pair:
    - If ``max(cutoff[sys] / perp_axes) <= 0.5``: emit 1 copy with MIC shifts.
    - Otherwise: emit per-image copies with replicated shifts; the
      ``is_minimum_image`` flag is set per copy so ``ExclusionMask`` can keep
      non-minimum image periodic copies of excluded pairs.

    Args:
        candidates: Raw candidate pair indices.
        lh, rh, systems: Pipeline tables (fractional coords).
        cutoffs: Per-system cutoff.
        max_image_candidates: Capacity for replicated-candidates buffer.
            When ``None``, falls back to ``FixedCapacity(candidates.lhs.size)``
            with an error message — pass an editable capacity if image
            replication is expected.

    Returns:
        ``CandidateBatch`` with shifts populated and ``is_minimum_image`` set.
    """
    cutoffs_t = Table.broadcast_to(cutoffs, systems)
    if max_image_candidates is None:
        max_image_candidates = FixedCapacity(
            candidates.lhs.size,
            "Cutoff is larger than half the cell length, "
            "we need to generate additional images. "
            "Please provide a editable max_candidates.",
        )

    idx, image_shifts, has_been_replicated = _get_candidate_images(
        candidates, lh, systems, cutoffs_t.data, max_image_candidates
    )
    min_shifts = _minimum_image_shifts(candidates, lh, rh, systems)

    if idx.size == candidates.lhs.size:
        # No replication needed — MIC shifts cover everything.
        return candidates_to_batch(
            candidates, min_shifts, jnp.ones((candidates.lhs.size,), dtype=bool)
        )

    replicated = bind(candidates).at(idx).get()
    final_shifts = jnp.where(
        has_been_replicated[:, None], image_shifts, min_shifts[idx]
    )
    is_min = (min_shifts[idx] == final_shifts).all(axis=-1)
    return candidates_to_batch(replicated, final_shifts, is_min)


def real_distance_sq(
    lhs_positions: Array,
    rhs_positions: Array,
    frames: MaterializedFrame,
    shifts: Array,
) -> Array:
    """Squared real-space distance between already-broadcast candidate pairs.

    Args:
        lhs_positions: Fractional left endpoint positions, shape ``(n, 3)``.
        rhs_positions: Fractional right endpoint positions, shape ``(n, 3)``.
        frames: Materialized cell frames broadcast to the candidate lhs system.
        shifts: ``(n, 3)`` fractional shifts.

    Returns:
        ``(n,)`` array of squared distances in real coordinates.
    """
    deltas = lhs_positions - rhs_positions - shifts
    real_deltas = frames.to_real(deltas)
    return jnp.einsum("...d,...d->...", real_deltas, real_deltas)
