# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Efficient O(N) neighbor list using spatial hashing with cell lists."""

from __future__ import annotations

from typing import Literal, Protocol, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.capacity import Capacity, LensCapacity
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Index, Table, subselect
from kups.core.lens import Lens, lens
from kups.core.neighborlist.common import (
    Candidates,
    lift_query_candidates,
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
    QueriedKeysDedupMask,
)
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.postprocess import MirrorPairEdges
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, jit


def cell_hash(coordinate: Array, num_cells: Array) -> Array:
    """Hash folded fractional coordinates into row-major cell bins.

    Coordinates outside nonperiodic faces are assigned to the boundary bins.

    Args:
        coordinate: Fractional coordinates, folded on periodic axes, ``(..., dim)``.
        num_cells: Per-axis bin counts broadcastable to ``coordinate``.

    Returns:
        Row-major bin ids of shape ``(...,)``.
    """
    factor = jnp.cumprod(num_cells, axis=-1) // num_cells
    bin_idx = jnp.clip(jnp.floor(coordinate * num_cells).astype(int), 0, num_cells - 1)
    return (bin_idx * factor).sum(axis=-1)


def cell_stencil(dim: int) -> Array:
    """All ``3**dim`` neighbor-cell offsets in ``{-1, 0, 1}**dim``, ``(3**dim, dim)``."""
    with jax.ensure_compile_time_eval():
        return jnp.stack(
            jnp.meshgrid(*[jnp.arange(-1, 2) for _ in range(dim)], indexing="ij"),
            axis=-1,
        ).reshape(-1, dim)


def assign_cells(
    positions: Array,
    system: Array,
    active: Array,
    bins: Array,
    max_cells: int,
    cell: Cell[AnyPeriodicity],
) -> tuple[Array, Array]:
    """Fold fractional positions and assign active rows to global cell ids.

    ``max_cells`` is the allocated cell capacity per system, which may exceed
    its bin count. Inactive rows use the sentinel ``len(bins) * max_cells``.
    Open-boundary outliers share the edge bins; their unfolded coordinates
    remain available for exact distance filtering.
    """
    active = active & (system >= 0) & (system < bins.shape[0])
    system = jnp.where(active, system, 0)
    frac, _ = cell.fold(jnp.where(active[:, None], positions, 0.0))
    ids = cell_hash(frac, bins[system]) + system * max_cells
    return frac, jnp.where(active, ids, bins.shape[0] * max_cells)


def neighbor_cells(
    cells: Array,
    bins: Array,
    max_cells: int,
    cell: Cell[AnyPeriodicity],
    *,
    width: int = 27,
    promise_unique: bool = False,
) -> Array:
    """Distinct neighboring cells using the unit cell's boundary rules.

    ``max_cells`` is the allocated capacity per system. Invalid or duplicate
    entries use the sentinel ``len(bins) * max_cells``.
    """
    sentinel_cell = bins.shape[0] * max_cells
    if max_cells == 1:
        return jnp.pad(
            cells[:, None],
            ((0, 0), (0, width - 1)),
            constant_values=sentinel_cell,
        )
    system, local = cells // max_cells, cells % max_cells
    cell_bins = bins[jnp.minimum(system, bins.shape[0] - 1)]
    factor = jnp.cumprod(cell_bins, axis=-1) // cell_bins
    coords = (local[:, None] // factor) % cell_bins
    neighbor = coords[:, None, :] + cell_stencil(bins.shape[-1])[None]
    wrapped = jnp.where(
        jnp.asarray(cell.periodic), neighbor % cell_bins[:, None], neighbor
    )
    valid = ((wrapped >= 0) & (wrapped < cell_bins[:, None])).all(axis=-1)
    valid &= ((cells < sentinel_cell) & (local < cell_bins.prod(axis=-1)))[:, None]
    neighbor = (wrapped * factor[:, None]).sum(-1) + (system * max_cells)[:, None]
    neighbor = jnp.where(valid, neighbor, sentinel_cell)
    if promise_unique:
        runtime_assert(
            ((bins >= 3) | ~jnp.asarray(cell.periodic)).all(),
            "promise_unique_cells requires at least three cells along every periodic axis",
        )
    else:
        neighbor = jnp.sort(neighbor, axis=-1)
        duplicate = jnp.pad(neighbor[:, 1:] == neighbor[:, :-1], ((0, 0), (1, 0)))
        neighbor = jnp.sort(jnp.where(duplicate, sentinel_cell, neighbor), axis=-1)
    runtime_assert(
        (neighbor[:, width:] == sentinel_cell).all(),
        "stencil_width too small for the spatial bin counts",
    )
    return neighbor[:, :width]


class CellListLookup(Protocol):
    """Cached occupant lookup, independent of any per-particle payload type."""

    def select(self, ctx: PipelineContext) -> Candidates: ...


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


def _cell_list_subselect(
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Array,
    max_num_cells: Capacity[int],
    max_num_candidates: Capacity[int],
    promise_unique_cells: bool = False,
) -> Candidates:
    if keys.size == 0 or queries.size == 0:
        return Candidates(
            Index(keys.keys, jnp.zeros(0, int), _cls=keys.cls),
            Index(queries.keys, jnp.zeros(0, int), _cls=queries.cls),
        )
    bins = num_cells(systems.data, cutoffs)
    max_num_cells = max_num_cells.generate_assertion(bins.prod(axis=-1).max())

    def assign(points: NeighborListPoints) -> Array:
        return assign_cells(
            points.positions,
            points.system.indices_in(systems.keys),
            points.inclusion.valid_mask,
            bins,
            max_num_cells.size,
            systems.data.cell,
        )[1]

    key_hashes = assign(keys.data)
    query_cells = neighbor_cells(
        assign(queries.data),
        bins,
        max_num_cells.size,
        systems.data.cell,
        width=1 if max_num_cells.size == 1 else 3 ** bins.shape[-1],
        promise_unique=promise_unique_cells,
    )

    selection_result = subselect(
        key_hashes,
        query_cells.ravel(),
        output_buffer_size=max_num_candidates,
        num_segments=bins.shape[0] * max_num_cells.size,
    )
    key_idx = Index(keys.keys, selection_result.scatter_idxs, _cls=keys.cls)
    query_idx = Index(
        queries.keys,
        selection_result.gather_idxs // query_cells.shape[1],
        _cls=queries.cls,
    )
    return Candidates(key_idx=key_idx, query_idx=query_idx)


@dataclass
class CellListSelector:
    """Selector for the cell-list algorithm.

    Joins cell hashes, or reads occupants from an optional initialized cache,
    then replicates per image multiplicity when ``max(cutoff/perp) > 0.5``.
    Set ``promise_unique_cells`` when the caller guarantees at least three cells
    along every periodic axis,
    so each query's stencil cells are already distinct; this skips the
    per-query cell deduplication in the uncached lookup.
    """

    cutoffs: Table[SystemId, Array]
    max_cells: Capacity[int]
    max_candidates: Capacity[int]
    max_image_candidates: Capacity[int]
    promise_unique_cells: bool = field(default=False, static=True)
    cache: CellListLookup | None = field(default=None, kw_only=True)

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        query = ctx.query_table
        candidates = (
            self.cache.select(ctx)
            if self.cache is not None
            else _cell_list_subselect(
                ctx.keys,
                query,
                ctx.systems,
                cutoffs=self.cutoffs.data,
                max_num_cells=self.max_cells,
                max_num_candidates=self.max_candidates,
                promise_unique_cells=self.promise_unique_cells,
            )
        )
        candidates = lift_query_candidates(candidates, ctx)
        return replicate_for_images(
            candidates,
            ctx.keys,
            ctx.edge_query_table,
            ctx.systems,
            self.cutoffs,
            self.max_image_candidates,
        )


@dataclass
class CellListNeighborList:
    """Neighbor list using spatial hashing with cell lists.

    This is the recommended implementation when the cutoff is much smaller than
    the box size. It divides space into a grid of cells and only checks pairs in
    neighboring cells.

    Honors the cell's per-axis ``periodic`` mask: stencil offsets that cross a
    non-periodic face are routed to an out-of-bounds bin (no key matches), and
    minimum-image shifts are zero on non-periodic axes.

    ``cache`` optionally supplies an initialized ``CellListCache`` for the
    current particles and cutoff. It retains occupant slots and neighboring
    cells between calls; accepted moves must update it along with the particles.
    Rebuild it when the cell, cutoff, or particle row layout changes.
    Without a cache, the same grid algorithm joins particle and query cell hashes.

    Candidate storage is O(N) for well-distributed particles at fixed density
    and cutoff. Uncached lookup sorts cell hashes; cached lookup gathers slots.
    Efficiency improves as cutoff/box ratio decreases.

    Attributes:
        avg_candidates: Capacity for candidate pair storage (from cell list).
        avg_edges: Capacity for final edge array.
        cells: Capacity for cell hash table (grows with box_size³/cutoff³).
        avg_image_candidates: Capacity for image candidate pairs.
        cutoffs: Per-system cutoff distances used by this neighbor list.
        cache: Current cell occupants, or ``None`` to rebuild the hash lookup.

    Algorithm:
        1. Partition space into grid cells of size ~cutoff
        2. Hash each particle to its cell
        3. For each particle, check only neighboring 27 cells (3D)
        4. Filter candidates by actual distance

    When to use:
        - When cutoff/box_size << 1 (cutoff much smaller than box)
        - Typically cutoff/box < 0.3 for good efficiency
        - Nonperiodic outliers remain searchable in boundary bins; many outliers
          can increase the candidate count.

    Example:
        ```python
        # Example: 10 Å cutoff in 50 Å box → cutoff/box = 0.2 -- Good for CellList
        nl = CellListNeighborList.new(state, lens(lambda s: s.nl_params), cutoffs)

        # Or, if the state implements IsNeighborListState:
        nl = CellListNeighborList.from_state(state, cutoffs)

        edges = nl(particles, systems)
        ```
    """

    avg_candidates: Capacity[int]
    avg_edges: Capacity[int]
    cells: Capacity[int]
    avg_image_candidates: Capacity[int]
    cutoffs: Table[SystemId, Array]
    cache: CellListLookup | None = field(default=None, kw_only=True)

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

    def selector(
        self, query_size: int, systems: Table[SystemId, NeighborListSystems]
    ) -> CellListSelector:
        """Build the candidate selector shared by graph and pair evaluation."""
        return CellListSelector(
            cutoffs=Table.broadcast_to(self.cutoffs, systems),
            max_cells=self.cells,
            max_candidates=self.avg_candidates.multiply(query_size),
            max_image_candidates=self.avg_image_candidates.multiply(query_size),
            cache=self.cache,
        )

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
        assert queries is None or queried_keys is None, (
            "Neighbor-list calls cannot combine queries with queried_keys."
        )
        query_size = (
            queried_keys.size
            if queried_keys is not None
            else (queries.size if queries is not None else keys.size)
        )
        cutoffs = Table.broadcast_to(self.cutoffs, systems)
        pipeline = Pipeline[Literal[2]](
            selector=self.selector(query_size, systems),
            masks=(
                InBoundsMask(),
                InclusionMatchMask(),
                QueriedKeysDedupMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                ExclusionMask(),
            ),
            compactor=ReduceCompactor(avg_edges=self.avg_edges.multiply(query_size)),
            postprocessors=(MirrorPairEdges(),),
        )
        if queries is not None:
            return pipeline(keys, systems, queries=queries)
        return pipeline(keys, systems, queried_keys=queried_keys)
