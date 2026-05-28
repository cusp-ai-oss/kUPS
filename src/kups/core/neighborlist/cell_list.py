# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Efficient O(N) neighbor list using spatial hashing with cell lists."""

from __future__ import annotations

from functools import partial
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity, LensCapacity
from kups.core.data import Index, Table, subselect
from kups.core.lens import Lens, lens
from kups.core.neighborlist.common import (
    Candidates,
    num_cells,
    replicate_for_images,
)
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    ExclusionMask,
    InBoundsMask,
    InclusionMatchMask,
    RemapDedupMask,
)
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit


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


def _cell_hash(coordinate: Array, num_cells: Array):
    """Hash folded fractional coordinates into row-major cell bins.

    Boundary values are clamped into the valid bin range for each axis.
    """
    factor = jnp.cumprod(num_cells, axis=-1) // num_cells
    bin_idx = jnp.clip(jnp.floor(coordinate * num_cells).astype(int), 0, num_cells - 1)
    return (bin_idx * factor).sum(axis=-1)


def _cell_stencil(dim: int):
    with jax.ensure_compile_time_eval():
        return jnp.stack(
            jnp.meshgrid(*[jnp.arange(-1, 2) for _ in range(dim)], indexing="ij"),
            axis=-1,
        ).reshape(-1, dim)


def _cell_list_subselect(
    lh: Table[ParticleId, NeighborListPoints],
    rh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Array,
    max_num_cells: Capacity[int],
    max_num_candidates: Capacity[int],
) -> Candidates:
    cell = systems.data.cell
    key_positions, _ = cell.fold(lh.data.positions)
    query_positions, _ = cell.fold(rh.data.positions)

    bins = systems.map_data(partial(num_cells, cutoff=cutoffs))
    max_num_cells = max_num_cells.generate_assertion(
        jnp.max(jnp.prod(bins.data, axis=-1))
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
        _cell_hash(key_positions, bins[lh.data.system])
        + lh_system_ids * max_num_cells.size
    )

    # Expand neighborhood around query points: for each query, tile across stencil
    stencil = _cell_stencil(dim)
    raw_shifted = jax.vmap(lambda s: query_positions + s[None] / bins[rh.data.system])(
        stencil
    ).reshape(-1, dim)
    query_original = Index(rh.keys, jnp.tile(jnp.arange(len(rh)), len(stencil)))
    query_system = rh.data.system[query_original.indices]

    shifted, in_cell = cell.fold(raw_shifted)
    hashes = (
        _cell_hash(shifted, bins[query_system])
        + rh_system_ids[query_original.indices] * max_num_cells.size
    )
    # Stencil offsets that left the box on a non-periodic axis route to
    # cell_oob so they produce no key matches (cross-boundary candidates
    # are excluded). For fully-periodic cells fold guarantees in_cell is
    # all-True, making the where a no-op.
    query_neighborhood_hashes = jnp.where(in_cell, hashes, cell_oob)

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
    return Candidates(lhs=lhs, rhs=rhs)


@dataclass
class CellListSelector:
    """Selector for the cell-list algorithm.

    Calls the raw spatial-hash candidate emission, then replicates per image
    multiplicity when ``max(cutoff/perp) > 0.5``.
    """

    cutoffs: Table[SystemId, Array]
    max_cells: Capacity[int]
    max_candidates: Capacity[int]
    max_image_candidates: Capacity[int]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        candidates = _cell_list_subselect(
            ctx.lh,
            ctx.rh,
            ctx.systems,
            cutoffs=self.cutoffs.data,
            max_num_cells=self.max_cells,
            max_num_candidates=self.max_candidates,
        )
        return replicate_for_images(
            candidates,
            ctx.lh,
            ctx.rh,
            ctx.systems,
            self.cutoffs,
            self.max_image_candidates,
        )


@dataclass
class CellListNeighborList:
    """Efficient O(N) neighbor list using spatial hashing with cell lists.

    This is the recommended implementation when the cutoff is much smaller than
    the box size. It divides space into a grid of cells and only checks pairs in
    neighboring cells, achieving linear scaling with system size.

    Honors the cell's per-axis ``periodic`` mask: stencil offsets that cross a
    non-periodic face are routed to an out-of-bounds bin (no key matches), and
    minimum-image shifts are zero on non-periodic axes. The fully-periodic path
    is byte-identical to the original (gated at trace time on ``all(periodic)``)
    so PBC kernels see no overhead.

    Complexity: O(N) for well-distributed particles where cutoff << box size.
    Efficiency improves as cutoff/box ratio decreases.

    Attributes:
        avg_candidates: Capacity for candidate pair storage (from cell list).
        avg_edges: Capacity for final edge array.
        cells: Capacity for cell hash table (grows with box_size³/cutoff³).
        avg_image_candidates: Capacity for image candidate pairs.
        cutoffs: Per-system cutoff distances used by this neighbor list.

    Algorithm:
        1. Partition space into grid cells of size ~cutoff
        2. Hash each particle to its cell
        3. For each particle, check only neighboring 27 cells (3D)
        4. Filter candidates by actual distance

    When to use:
        - When cutoff/box_size << 1 (cutoff much smaller than box)
        - Typically cutoff/box < 0.3 for good efficiency
        - On non-periodic axes positions must lie inside ``[0, L)`` in real
          coordinates (the caller's invariant; out-of-range positions are
          silently routed to the OOB bin)

    Example:
        ```python
        # Example: 10 Å cutoff in 50 Å box → cutoff/box = 0.2 -- Good for CellList
        nl = CellListNeighborList.new(state, lens(lambda s: s.nl_params), cutoffs)

        # Or, if the state implements IsNeighborListState:
        nl = CellListNeighborList.from_state(state, cutoffs)

        edges = nl(particles, None, systems)
        ```
    """

    avg_candidates: Capacity[int]
    avg_edges: Capacity[int]
    cells: Capacity[int]
    avg_image_candidates: Capacity[int]
    cutoffs: Table[SystemId, Array]

    @classmethod
    def new[S](
        cls,
        state: S,
        lens: Lens[S, IsCellListParams],
        cutoffs: Table[SystemId, Array],
    ) -> CellListNeighborList:
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
            cutoffs=cutoffs,
        )

    @classmethod
    def from_state(
        cls,
        state: IsNeighborListState[IsCellListParams],
        cutoffs: Table[SystemId, Array],
    ) -> CellListNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params), cutoffs)

    @jit
    def __call__(
        self,
        lh: Table[ParticleId, NeighborListPoints],
        rh: Table[ParticleId, NeighborListPoints] | None,
        systems: Table[SystemId, NeighborListSystems],
        rh_index_remap: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        rh_size = rh.size if rh is not None else lh.size
        cutoffs = Table.broadcast_to(self.cutoffs, systems)
        pipeline = Pipeline[Literal[2]](
            selector=CellListSelector(
                cutoffs=cutoffs,
                max_cells=self.cells,
                max_candidates=self.avg_candidates.multiply(rh_size),
                max_image_candidates=self.avg_image_candidates.multiply(rh_size),
            ),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                RemapDedupMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                ExclusionMask(),
            ),
            compactor=ReduceCompactor(avg_edges=self.avg_edges.multiply(rh_size)),
        )
        return pipeline(lh, rh, systems, rh_index_remap)
