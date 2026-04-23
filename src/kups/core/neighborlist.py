# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor list construction and edge representations for molecular systems.

This module provides multiple neighbor list algorithms for finding interacting
pairs of particles within cutoff distances, with different performance and
accuracy trade-offs.

## Core Components

- **[Edges][kups.core.neighborlist.Edges]**: Represents connections between particles with periodic shifts
- **[NearestNeighborList][kups.core.neighborlist.NearestNeighborList]**: Protocol for neighbor search implementations
- **[RefineMaskNeighborList][kups.core.neighborlist.RefineMaskNeighborList]**: Applies inclusion/exclusion masks for selective interactions

## Neighbor List Implementations

### Primary Implementations

1. **[CellListNeighborList][kups.core.neighborlist.CellListNeighborList]** (Recommended when cutoff << box size)
    - O(N) complexity using spatial hashing
    - Best when cutoff / box_size < 0.3 (cutoff much smaller than box)
    - Requires periodic boundary conditions
    - Efficiency improves as cutoff/box ratio decreases

2. **[DenseNearestNeighborList][kups.core.neighborlist.DenseNearestNeighborList]**
    - O(N²/K) complexity (K = number of systems)
    - Best when cutoff / box_size ~ 1 (cutoff comparable to box)
    - Works with or without periodic boundaries
    - More efficient when few cells would fit in box

3. **[AllDenseNearestNeighborList][kups.core.neighborlist.AllDenseNearestNeighborList]**
    - O(N²) complexity across all systems
    - Only for single-system simulations or testing
    - Crosses system boundaries (use with caution!)

### Refinement Implementations

These allow sharing a single base neighbor list across multiple potentials
with different cutoffs or interaction rules (e.g., Lennard-Jones and Coulomb).

4. **[RefineMaskNeighborList][kups.core.neighborlist.RefineMaskNeighborList]**
    - Applies inclusion/exclusion masks to precomputed edges
    - Use for bonded exclusions or group-specific interactions
    - No distance recalculation
    - Share one neighbor list, apply different masks per potential

5. **[RefineCutoffNeighborList][kups.core.neighborlist.RefineCutoffNeighborList]**
    - Refines precomputed edges with new cutoff distances
    - Use for multi-stage construction or adaptive cutoffs
    - Recalculates distances
    - Share one conservative neighbor list, apply different cutoffs per potential

## Features

All neighbor lists handle:
- Periodic boundary conditions via shift vectors
- Multiple systems in parallel with segmentation
- Automatic capacity management for variable neighbor counts
- Integration with JAX transformations (JIT, vmap, etc.)

## Choosing an Implementation

```python
# When cutoff << box size (cutoff/box < 0.3)
# Example: 10 Å cutoff, 50 Å box → use CellList
nl = CellListNeighborList.new(state, lens=lens(lambda s: s.nl_params))

# When cutoff ~ box size (cutoff/box ~ 1)
# Example: 15 Å cutoff, 20 Å box → use Dense
nl = DenseNearestNeighborList.new(state, lens=lens(lambda s: s.nl_params))

# Share one neighbor list across multiple potentials with different masks
base_edges = base_nl(particles, None, cells, cutoffs, None)
lj_nl = RefineMaskNeighborList(candidates=base_edges)  # Exclude 1-4 interactions
coulomb_nl = RefineMaskNeighborList(candidates=base_edges)  # Different exclusions

# Share one neighbor list across potentials with different cutoffs
base_edges = base_nl(particles, None, cells, max_cutoff, None)
lj_nl = RefineCutoffNeighborList(candidates=base_edges, avg_edges=cap1)  # r_cut = 10 Å
coulomb_nl = RefineCutoffNeighborList(candidates=base_edges, avg_edges=cap2)  # r_cut = 15 Å
```
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.capacity import Capacity, FixedCapacity, LensCapacity
from kups.core.data import Index, Sliceable, Table, subselect
from kups.core.data.wrappers import WithIndices
from kups.core.lens import Lens, bind, lens
from kups.core.typing import (
    HasExclusionIndex,
    HasInclusionIndex,
    HasPositions,
    HasPositionsAndSystemIndex,
    HasSystemIndex,
    HasUnitCell,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, isin, jit, no_jax_tracing
from kups.core.utils.math import next_higher_power, triangular_3x3_matmul
from kups.core.utils.ops import where_broadcast_last


@dataclass
class Edges[Degree: int](Sliceable):
    """Represents edges (connections) between particles in a molecular system.

    An edge connects `Degree` particles, where degree=2 represents pairwise
    interactions (bonds), degree=3 represents three-body interactions (angles), etc.

    For periodic systems, edges include shift vectors that indicate how many
    unit cells to traverse when computing distances between connected particles.

    Type Parameters:
        Degree: Number of particles connected by each edge (static type check)

    Attributes:
        indices: Particle indices for each edge, shape `(n_edges, Degree)`
        shifts: Periodic shift vectors, shape `(n_edges, Degree-1, 3)`.
            Shift vectors for the 2nd through Degree-th particle relative to the first.

    Example:
        ```python
        # Pairwise edges (bonds) between particles
        edges = Edges(
            indices=jnp.array([[0, 1], [1, 2], [0, 2]]),  # 3 edges
            shifts=jnp.array([[[0, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]])  # 3rd edge crosses boundary
        )
        ```
    """

    # The degree is purely for type checking and does not affect runtime behavior
    indices: Index[ParticleId]  # (n_edges, Degree)
    shifts: Array  # (n_edges, Degree - 1, 3)

    def __post_init__(self):
        # Resolve the underlying array for validation
        raw = self.indices.indices if isinstance(self.indices, Index) else self.indices
        if not isinstance(raw, Array):
            return
        assert jnp.issubdtype(raw.dtype, jnp.integer), (
            f"Indices must be of integer type, got {raw.dtype}"
        )
        target_shape = (
            *self.indices.shape[:-1],
            self.indices.shape[-1] - 1 if self.indices.shape[-1] > 1 else 0,
            3,
        )
        assert self.shifts.shape == target_shape, (
            f"Shifts must have shape {target_shape}, got {self.shifts.shape}"
        )

    def difference_vectors(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, HasUnitCell],
    ) -> Array:
        """Compute difference vectors between connected particles.

        For each edge, computes the vector from the first particle to each
        subsequent particle, accounting for periodic boundary conditions.

        Args:
            particles: Particle positions with system index information.
            systems: System data with unit cell for periodic boundary conditions.

        Returns:
            Array of shape `(n_edges, Degree-1, 3)` containing difference vectors.
        """

        shifts = self.absolute_shifts(particles, systems)
        pos = particles[self.indices].positions
        return pos[:, 1:] - pos[:, :1] + shifts

    def absolute_shifts(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, HasUnitCell],
    ) -> Array:
        """Compute absolute shift vectors for all particles in each edge.

        Converts relative shifts to absolute Cartesian shift vectors.

        Args:
            particles: Particle data with system index information.
            systems: System data with unit cell for periodic boundary conditions.

        Returns:
            Array of shape `(n_edges, Degree-1, 3)` containing absolute shift vectors.
        """
        lattice = systems.map_data(lambda x: x.unitcell.lattice_vectors)
        vecs = lattice[particles[self.indices[:, 0]].system]
        return triangular_3x3_matmul(vecs[:, None], self.shifts)

    @property
    def degree(self) -> int:
        return self.indices.shape[-1]

    def __len__(self) -> int:
        return self.indices.shape[0]


class NeighborListPoints(
    HasPositions,
    HasSystemIndex,
    HasInclusionIndex,
    HasExclusionIndex,
    Protocol,
): ...


class NeighborListSystems(HasUnitCell, Protocol): ...


class NearestNeighborList(Protocol):
    """Protocol for neighbor list construction algorithms.

    Implementations find pairs of particles within a cutoff distance, handling
    periodic boundary conditions and inclusion/exclusion masks.
    """

    def __call__[P: NeighborListPoints](
        self,
        lh: Table[ParticleId, P],
        rh: Table[ParticleId, P] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        """Find all particle pairs within the cutoff distance.

        Args:
            lh: Left-hand particles to find neighbors for
            rh: Right-hand particles to search within (or None for self-neighbors)
            systems: Indexed system data with unit cell information
            cutoffs: Indexed cutoff data per system
            rh_index_remap: Optional index mapping rh particles back to lh
                particle IDs for self-interaction exclusion. When ``None``,
                rh is treated as disjoint from lh.

        Returns:
            Edges connecting particle pairs within cutoff
        """
        ...


def _cell_hash(coordinate: Array, num_cells: Array):
    """Calculate the cell hash for a given coordinate and number of cells."""
    factor = jnp.cumprod(num_cells, axis=-1) / num_cells
    return (jnp.floor(coordinate * num_cells) * factor).astype(int).sum(axis=-1)


def _cell_stencil(dim: int):
    with jax.ensure_compile_time_eval():
        return jnp.stack(
            jnp.meshgrid(*[jnp.arange(-1, 2) for _ in range(dim)], indexing="ij"),
            axis=-1,
        ).reshape(-1, dim)


def _wrap_coordinates(coordinates: Array):
    """Wrap coordinates to be within the bounds of the grid."""
    return jnp.where(
        coordinates < 0,
        1 + coordinates,
        jnp.where(coordinates >= 1, coordinates - 1, coordinates),
    )


def _num_cells(
    systems: NeighborListSystems,
    cutoff: Array,
    *,
    eps: float = 1e-6,
) -> Array:
    inv_norms: jax.Array = jnp.linalg.norm(
        systems.unitcell.inverse_lattice_vectors, axis=-1
    )
    face_lengths = 1.0 / jnp.where(inv_norms < eps, jnp.ones_like(inv_norms), inv_norms)
    num_bins = jnp.maximum((face_lengths / cutoff[..., None]).astype(int), 1)
    return num_bins


def _cell_list_subselect(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Array,
    max_num_cells: Capacity[int],
    max_num_candidates: Capacity[int],
) -> _Candidates:
    key_positions = _wrap_coordinates(lh.data.positions)
    query_positions = _wrap_coordinates(rh.data.positions)

    num_cells = systems.map_data(partial(_num_cells, cutoff=cutoffs))
    max_num_cells = max_num_cells.generate_assertion(
        jnp.max(jnp.prod(num_cells.data, axis=-1))
    )
    num_systems = systems.size
    cell_oob = max_num_cells.size * num_systems

    dim = key_positions.shape[-1]
    assert query_positions.shape[-1] == dim, (
        f"Queries must have the same dimensionality as keys, "
        f"got {query_positions.shape[-1]} != {dim}"
    )

    # Raw system IDs for hash offset computation
    lh_system_ids = lh.data.system.indices
    rh_system_ids = rh.data.system.indices

    key_hashes = (
        _cell_hash(key_positions, num_cells[lh.data.system])
        + lh_system_ids * max_num_cells.size
    )

    # Expand neighborhood around query points: for each query, tile across stencil
    stencil = _cell_stencil(dim)
    query_neighborhood_positions = _wrap_coordinates(
        jax.vmap(lambda s: query_positions + s[None] / num_cells[rh.data.system])(
            stencil
        ).reshape(-1, dim)
    )
    query_original = Index(rh.keys, jnp.tile(jnp.arange(len(rh)), len(stencil)))
    query_system = rh.data.system[query_original.indices]
    query_neighborhood_hashes = (
        _cell_hash(query_neighborhood_positions, num_cells[query_system])
        + rh_system_ids[query_original.indices] * max_num_cells.size
    )
    unique_queries = jnp.unique(
        jnp.stack([query_neighborhood_hashes, query_original.indices], axis=-1),
        axis=0,
        size=len(query_original),
        fill_value=jnp.array([cell_oob, len(rh)]),
    )
    query_neighborhood_hashes = unique_queries[:, 0]
    query_original = Index(rh.keys, unique_queries[:, 1])

    selection_result = subselect(
        key_hashes,
        query_neighborhood_hashes,
        output_buffer_size=max_num_candidates,
        num_segments=cell_oob,
        is_sorted=True,  # unique sorts the neighborhood hashes
    )
    lhs = Index(lh.keys, selection_result.scatter_idxs)
    rhs = Index(
        query_original.keys,
        query_original.indices.at[selection_result.gather_idxs].get(
            **query_original.scatter_args
        ),
    )
    return _Candidates(lhs=lhs, rhs=rhs)


def _all_subselect(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> _Candidates:
    lh_indices, rh_indices = jnp.indices((len(lh), len(rh))).reshape(2, -1)
    return _Candidates(lhs=Index(lh.keys, lh_indices), rhs=Index(rh.keys, rh_indices))


def _dense_subselect(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    max_num_candidates: Capacity[int],
) -> _Candidates:
    selection_result = subselect(
        lh.data.system.indices,
        rh.data.system.indices,
        output_buffer_size=max_num_candidates,
        num_segments=systems.size,
    )
    return _Candidates(
        lhs=Index(lh.keys, selection_result.scatter_idxs),
        rhs=Index(rh.keys, selection_result.gather_idxs),
    )


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


def _get_candidate_images(
    candidates: _Candidates,
    lh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Array,
    out_size: Capacity[int],
) -> tuple[Array, Array, Array]:
    unitcells = systems.data.unitcell
    images = jnp.ceil(2 * cutoffs[..., None] / unitcells.perpendicular_lengths).astype(
        int
    )
    images = jnp.where(jnp.isfinite(cutoffs[..., None]), images, 1)
    images += images % 2 == 0
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
    """Compute squared distances with periodic boundary conditions.

    If ``shifts`` is None, minimum-image shifts are computed via rounding.
    """
    lattice_vecs = systems.map_data(lambda s: s.unitcell.lattice_vectors)
    vecs = lattice_vecs[lh.data.system[candidates.lhs.indices]]
    deltas = (
        lh.data.positions[candidates.lhs.indices]
        - rh.data.positions[candidates.rhs.indices]
    )
    if shifts is None:
        shifts = jnp.round(deltas)
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
            "Cutoff is larger than half the unit cell length, "
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
    lh_inv = systems[lh.data.system].unitcell.inverse_lattice_vectors
    lh = (
        bind(lh)
        .focus(lambda x: x.data.positions)
        .apply(lambda r: triangular_3x3_matmul(lh_inv, r))
    )
    rh_inv = systems[rh.data.system].unitcell.inverse_lattice_vectors
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


@no_jax_tracing
def _estimate_avg_num_edges(
    num_particles: int | Array,
    volume: float | Array,
    cutoff: float | Array,
    base: float = 2.0,
    multiplier: float = 1.0,
) -> int:
    """Estimate average number of neighbors per particle for neighbor list allocation.

    Calculates expected neighbors within cutoff radius based on particle density,
    with tolerance factor for small systems. Result is rounded up to next power of base.

    Args:
        num_particles: Total number of particles in the system.
        volume: Total volume of the simulation box.
        cutoff: Cutoff radius for neighbor interactions.
        base: Base for power rounding (default 2.0).
        multiplier: Multiplied with the estimate to create a buffer (default 1.0).

    Returns:
        Conservative estimate rounded to next power of base for array allocation.
    """
    # avg_edges ≈ (N/V) * (4π/3 * r³), i.e. uniform-density sphere of radius cutoff
    avg_particle_density = num_particles / volume
    cutoff_volume = 4 / 3 * jnp.pi * cutoff**3
    avg_particles_in_cutoff = cutoff_volume * avg_particle_density
    estimate = multiplier * avg_particles_in_cutoff
    return int(next_higher_power(jnp.array(estimate), base=base))


@dataclass
class AllDenseNearestNeighborList:
    """Dense O(N²) neighbor list considering all pairs across all systems.

    This implementation generates all possible particle pairs without spatial
    optimization. It is only suitable for very small systems or testing.

    **Warning**: This crosses system boundaries! Only use for single-system
    simulations. For multiple systems, use
    [DenseNearestNeighborList][kups.core.neighborlist.DenseNearestNeighborList]
    instead.

    Complexity: O(N²) where N is the total number of particles across all systems.

    Attributes:
        avg_edges: Capacity manager for edge array.
        avg_image_candidates: Capacity manager for image candidate pairs.

    Example:
        ```python
        # Construct from state and a lens to the neighbor list parameters:
        nl = AllDenseNearestNeighborList.new(state, lens(lambda s: s.nl_params))

        # Or, if the state implements IsNeighborListState:
        nl = AllDenseNearestNeighborList.from_state(state)

        edges = nl(particles, None, unit_cells, cutoffs, None)
        ```
    """

    avg_edges: Capacity[int]
    avg_image_candidates: Capacity[int]

    @classmethod
    def new[S](
        cls, state: S, lens: Lens[S, IsAllDenseNeighborListParams]
    ) -> AllDenseNearestNeighborList:
        params = lens.get(state)
        return AllDenseNearestNeighborList(
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            avg_image_candidates=LensCapacity(
                params.avg_image_candidates,
                lens.focus(lambda x: x.avg_image_candidates),
            ),
        )

    @classmethod
    def from_state(
        cls, state: IsNeighborListState[IsAllDenseNeighborListParams]
    ) -> AllDenseNearestNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params))

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        if lh.data.inclusion.num_labels >= 2:
            logging.warning(
                "AllDenseNearestNeighborList is intended for single-system simulations. "
                "Performance may be degraded when using multiple systems. "
                "Consider using DenseNearestNeighborList or CellListNeighborList instead."
            )
        rh_size = rh.size if rh is not None else lh.size
        return basic_neighborlist(
            lh,
            rh,
            systems,
            cutoffs,
            rh_index_remap,
            candidate_selector=_all_subselect,
            max_num_edges=self.avg_edges.multiply(rh_size),
            max_image_candidates=self.avg_image_candidates.multiply(rh_size)
            if self.avg_image_candidates
            else None,
        )


@dataclass
class DenseNearestNeighborList:
    """Dense O(N²) neighbor list respecting system boundaries.

    This implementation generates all particle pairs within each system
    separately, avoiding cross-system interactions. Efficient when the cutoff
    is comparable to the box size (cutoff/box ~ 1).

    Complexity: O(N² / K²) where N is total particles and K is number of systems.

    Attributes:
        avg_candidates: Capacity for candidate pair storage.
        avg_edges: Capacity for final edge array.
        avg_image_candidates: Capacity for image candidate pairs.

    When to use:
        - When cutoff/box_size ~ 1 (cutoff comparable to box dimensions)
        - Small box relative to cutoff (few cells would fit)
        - Non-periodic systems

    Example:
        ```python
        # Example: 15 Å cutoff in 20 Å box → cutoff/box = 0.75
        nl = DenseNearestNeighborList.new(state, lens(lambda s: s.nl_params))

        # Or, if the state implements IsNeighborListState:
        nl = DenseNearestNeighborList.from_state(state)

        edges = nl(particles, None, systems, cutoffs, None)
        ```
    """

    avg_candidates: Capacity[int]
    avg_edges: Capacity[int]
    avg_image_candidates: Capacity[int]

    @classmethod
    def new[S](
        cls, state: S, lens: Lens[S, IsDenseNeighborlistParams]
    ) -> DenseNearestNeighborList:
        params = lens.get(state)
        return DenseNearestNeighborList(
            avg_candidates=LensCapacity(
                params.avg_candidates, lens.focus(lambda x: x.avg_candidates)
            ),
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            avg_image_candidates=LensCapacity(
                params.avg_image_candidates,
                lens.focus(lambda x: x.avg_image_candidates),
            ),
        )

    @classmethod
    def from_state(
        cls, state: IsNeighborListState[IsDenseNeighborlistParams]
    ) -> DenseNearestNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params))

    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        rh_size = rh.size if rh is not None else lh.size
        selector = partial(
            _dense_subselect,
            max_num_candidates=self.avg_candidates.multiply(rh_size),
        )
        return basic_neighborlist(
            lh,
            rh,
            systems,
            cutoffs,
            rh_index_remap,
            candidate_selector=selector,
            max_num_edges=self.avg_edges.multiply(rh_size),
            max_image_candidates=self.avg_image_candidates.multiply(rh_size),
        )


@dataclass
class CellListNeighborList:
    """Efficient O(N) neighbor list using spatial hashing with cell lists.

    This is the recommended implementation when the cutoff is much smaller than
    the box size. It divides space into a grid of cells and only checks pairs in
    neighboring cells, achieving linear scaling with system size.

    **Requires periodic boundary conditions** (UnitCell).

    Complexity: O(N) for well-distributed particles where cutoff << box size.
    Efficiency improves as cutoff/box ratio decreases.

    Attributes:
        avg_candidates: Capacity for candidate pair storage (from cell list).
        avg_edges: Capacity for final edge array.
        cells: Capacity for cell hash table (grows with box_size³/cutoff³).
        avg_image_candidates: Capacity for image candidate pairs.

    Algorithm:
        1. Partition space into grid cells of size ~cutoff
        2. Hash each particle to its cell
        3. For each particle, check only neighboring 27 cells (3D)
        4. Filter candidates by actual distance

    When to use:
        - When cutoff/box_size << 1 (cutoff much smaller than box)
        - Typically cutoff/box < 0.3 for good efficiency
        - Periodic boundary conditions required

    Example:
        ```python
        # Example: 10 Å cutoff in 50 Å box → cutoff/box = 0.2 -- Good for CellList
        nl = CellListNeighborList.new(state, lens(lambda s: s.nl_params))

        # Or, if the state implements IsNeighborListState:
        nl = CellListNeighborList.from_state(state)

        edges = nl(particles, None, unit_cells, cutoffs, None)
        ```
    """

    avg_candidates: Capacity[int]
    avg_edges: Capacity[int]
    cells: Capacity[int]
    avg_image_candidates: Capacity[int]

    @classmethod
    def new[S](cls, state: S, lens: Lens[S, IsCellListParams]) -> CellListNeighborList:
        params = lens.get(state)
        return CellListNeighborList(
            avg_candidates=LensCapacity(
                params.avg_candidates, lens.focus(lambda x: x.avg_candidates)
            ),
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            avg_image_candidates=LensCapacity(
                params.avg_image_candidates,
                lens.focus(lambda x: x.avg_image_candidates),
            ),
            cells=LensCapacity(params.cells, lens.focus(lambda x: x.cells), base=1),
        )

    @classmethod
    def from_state(
        cls, state: IsNeighborListState[IsCellListParams]
    ) -> CellListNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params))

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        rh_size = rh.size if rh is not None else lh.size
        return basic_neighborlist(
            lh,
            rh,
            systems,
            cutoffs,
            rh_index_remap,
            candidate_selector=partial(
                _cell_list_subselect,
                cutoffs=cutoffs.data,
                max_num_cells=self.cells,
                max_num_candidates=self.avg_candidates.multiply(rh_size),
            ),
            max_num_edges=self.avg_edges.multiply(rh_size),
            max_image_candidates=self.avg_image_candidates.multiply(rh_size),
        )


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
    the unit cell is used only to compute minimum-image shifts.

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

    lh_frac = triangular_3x3_matmul(
        lh_sys.unitcell.inverse_lattice_vectors, lh.data.positions
    )
    rh_frac = triangular_3x3_matmul(
        rh_sys.unitcell.inverse_lattice_vectors, rh.data.positions
    )
    shifts = jnp.round(lh_frac[lh_idx.indices] - rh_frac[rh_idx.indices]).astype(int)
    return _compact_edges(
        candidates,
        mask,
        shifts,
        rh_index_remap.indices_in(lh.keys) if rh_index_remap is not None else None,
        capacity,
        out_of_bounds,
    )


class IsAllDenseNeighborListParams(Protocol):
    """Protocol for parameters required by ``AllDenseNearestNeighborList``."""

    @property
    def avg_edges(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...


class IsCellListParams(Protocol):
    """Protocol for parameters required by ``CellListNeighborList``."""

    @property
    def avg_candidates(self) -> int: ...
    @property
    def avg_edges(self) -> int: ...
    @property
    def cells(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...


class IsDenseNeighborlistParams(Protocol):
    """Protocol for parameters required by ``DenseNearestNeighborList``."""

    @property
    def avg_candidates(self) -> int: ...
    @property
    def avg_edges(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...


class IsUniversalNeighborlistParams(Protocol):
    """Protocol for parameters required by any neighbor list implementation.

    A superset of ``IsAllDenseNeighborListParams``, ``IsDenseNeighborlistParams``,
    and ``IsCellListParams``. Satisfying this protocol allows constructing any
    of the three neighbor list types.
    """

    @property
    def avg_edges(self) -> int: ...
    @property
    def avg_candidates(self) -> int: ...
    @property
    def avg_image_candidates(self) -> int: ...
    @property
    def cells(self) -> int: ...


class IsNeighborListState[P](Protocol):
    """Protocol for states that expose neighbor list parameters.

    A state satisfying this protocol can be passed to ``from_state()`` on any
    neighbor list class. The type parameter ``P`` determines which neighbor
    list types the state can construct (e.g., ``IsAllDenseNeighborListParams``,
    ``IsDenseNeighborlistParams``, ``IsCellListParams``, or
    ``IsUniversalNeighborlistParams``).
    """

    @property
    def neighborlist_params(self) -> P: ...


@dataclass
class UniversalNeighborlistParameters:
    """Concrete parameter dataclass satisfying ``IsUniversalNeighborlistParams``.

    Holds the capacity hints needed by every neighbor list implementation.
    Use the ``estimate()`` classmethod to compute reasonable initial values
    from system geometry rather than guessing manually.

    Attributes:
        avg_edges: Average number of edges per particle (for edge capacity).
        avg_candidates: Average number of candidate pairs per particle.
        avg_image_candidates: Average number of image candidate pairs per particle.
        cells: Maximum number of spatial hash cells across all systems.
    """

    avg_edges: int = field(static=True)
    avg_candidates: int = field(static=True)
    avg_image_candidates: int = field(static=True)
    cells: int = field(static=True)

    @classmethod
    @no_jax_tracing
    def estimate(
        cls,
        particles_per_system: Table[SystemId, Array],
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        *,
        base: float = 2,
        multiplier: float = 1.0,
    ) -> UniversalNeighborlistParameters:
        """Estimate parameters for all neighbor list types from system geometry.

        Computes conservative initial capacities based on particle density
        and cutoff radii. The estimates are rounded up to the next power of
        ``base`` to amortize future resizing.

        Args:
            particles_per_system: Number of particles per system.
            systems: System data with unit cell information.
            cutoffs: Cutoff distance per system.
            base: Base for power-of rounding (default 2).
            multiplier: Safety factor applied to the estimate (default 1.0).

        Returns:
            A ``UniversalNeighborlistParameters`` instance with estimated values.
        """
        sys = Table.join(systems, particles_per_system, cutoffs)
        total_candidates = total_edges = max_cells = 0
        for _, (s, n_p, c) in sys:
            num_cells = _num_cells(s, c).prod()
            total_candidates += min(n_p / num_cells * (3**3), n_p)
            total_edges += _estimate_avg_num_edges(
                n_p, s.unitcell.volume, c, base, multiplier
            )
            max_cells = max(num_cells, max_cells)
        total_candidates = next_higher_power(
            jnp.array(total_candidates * multiplier / sys.size), base=base
        )
        return UniversalNeighborlistParameters(
            avg_edges=int(total_edges // sys.size),
            avg_candidates=int(total_candidates),
            avg_image_candidates=int(total_candidates),  # Image candidates ~ candidates
            cells=int(max_cells),
        )


class NeighborListChangesResult(NamedTuple):
    added: Edges[Literal[2]]
    removed: Edges[Literal[2]]


@partial(jit, static_argnames=("compaction",))
def neighborlist_changes(
    neighborlist: NearestNeighborList,
    lh: Table[ParticleId, NeighborListPoints],
    rh: WithIndices[ParticleId, Table[ParticleId, NeighborListPoints]],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    compaction: float = 0.5,
) -> NeighborListChangesResult:
    """Compute added/removed edges from a particle change in a single call.

    Appends proposed positions to the particle array and queries both old
    and new interactions at once, then splits the result by filtering
    edge indices into ``removed`` (before) and ``added`` (after) sets.

    Args:
        neighborlist: Neighbor list implementation.
        lh: Full original particle table.
        rh: Proposed changes — ``rh.indices`` maps entries to particle IDs
            in ``lh``, ``rh.data`` holds the new particle data.
        systems: Per-system data (unit cells, etc.).
        cutoffs: Per-system cutoff distances.
        compaction: Fraction of total edges allocated per output (0–1).
            0.5 means each of added/removed gets half the buffer.
            1.0 means no compaction — full buffer with masking only.

    Returns:
        ``NeighborListChangesResult(added, removed)``.
    """
    N, k = lh.size, rh.data.size
    p_idx = rh.indices.indices_in(lh.keys)

    # Build a single query with new particles on the left-hand side
    # (original particles + new particles) and both old and new particles
    # on the right-hand side (old positions at changed indices + new positions).
    lh_combined = Table.union((lh, rh.data))
    rh_combined = Table.union((Table.arange(lh[rh.indices], label=ParticleId), rh.data))
    combined_remap = Index(
        lh_combined.keys, jnp.concatenate([p_idx, jnp.arange(k) + N])
    )

    # single neighborlist call
    all_edges = neighborlist(lh_combined, rh_combined, systems, cutoffs, combined_remap)

    # split into removed / added
    raw = all_edges.indices.indices  # (n_edges, 2)
    c0, c1 = raw[:, 0], raw[:, 1]
    # Removed mask checks for edges that exist in the original set (both indices < N).
    removed_mask = (c0 < N) & (c1 < N)

    # is_stale mask checks that both edges need to be in the original set
    # or one needs to be in the original set and the other needs to be in the new set.
    is_stale = isin(c0, p_idx, N + k) & (c0 < N) | isin(c1, p_idx, N + k) & (c1 < N)
    # Added mask checks for edges that involve at least one new particle.
    added_mask = (c0 < N + k) & (c1 < N + k) & ((c0 >= N) | (c1 >= N)) & ~is_stale

    # remap appended indices N+m -> p_idx[m]
    remapped = jnp.where(raw >= N, p_idx[raw - N], raw)

    # compact each output
    n_total = raw.shape[0]
    shifts = all_edges.shifts

    def _mask_only(mask: Array, indices: Array, shifts: Array) -> Edges[Literal[2]]:
        idx = where_broadcast_last(mask, indices, N)
        sh = where_broadcast_last(mask, shifts, 0)
        return Edges(Index(lh.keys, idx), sh)

    def _compact(mask: Array, indices: Array, label: str) -> Edges[Literal[2]]:
        count = mask.sum()
        runtime_assert(
            count <= capacity,
            f"neighborlist_changes: {label} edges ({{count}}) exceed "
            f"capacity ({{capacity}})",
            fmt_args={"count": count, "capacity": jnp.array(capacity)},
        )
        sel: Array = jnp.where(mask, size=capacity, fill_value=n_total - 1)[0]
        valid = mask.at[sel].get(mode="fill", fill_value=False)
        return _mask_only(valid, indices[sel], shifts[sel])

    if compaction >= 1.0:
        return NeighborListChangesResult(
            _mask_only(added_mask, remapped, shifts),
            _mask_only(removed_mask, remapped, shifts),
        )

    capacity = int(n_total * compaction)
    return NeighborListChangesResult(
        _compact(added_mask, remapped, "added"),
        _compact(removed_mask, remapped, "removed"),
    )
