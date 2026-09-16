# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Persistent occupancy storage for the cell-list algorithm.

A [`CellListCache`][kups.core.neighborlist.cell_list_cache.CellListCache] bins the active
particles of every system into spatial cells sized from the cutoff and
stores, per cell, the slots of its occupants together with a deduplicated
stencil of neighbor cells. The candidate neighbors of any point are the
occupants of its stencil cells, read as a fixed-width lane array
(``stencil_width * cell_capacity``). Fused potentials consume these chunks
directly. ``CellListNeighborList(cache=...)`` uses these occupants through its
normal selector, mask, compaction, and postprocessing pipeline. Both lookup paths
share the binning and neighboring-cell functions in ``cell_list``.

Slots are sorted by ``(system, cell)`` at build time so consecutive slots are
spatially local, and a
[`CellListCacheUpdatePatch`][kups.core.neighborlist.cell_list_cache.CellListCacheUpdatePatch]
moves the slots of accepted Monte Carlo proposals between touched cells, so a
table persists across a simulation instead of being rebuilt per step. Slot
storage is not re-sorted as particles move.

Candidate chunks include all periodic images within the cutoff, using the same
image windows and exclusion rules as the neighbor lists. Static capacities are
guarded by ``runtime_assert`` rather than the adaptive
[`Capacity`][kups.core.capacity.Capacity] machinery.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.capacity import FixedCapacity
from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table
from kups.core.data.index import SupportsSorting
from kups.core.lens import Lens
from kups.core.neighborlist.cell_list import assign_cells, neighbor_cells
from kups.core.neighborlist.common import (
    Candidates,
    candidate_image_counts,
    make_batch_with_mic,
    num_cells,
    replicate_for_images,
)
from kups.core.neighborlist.types import (
    CandidateBatch,
    NeighborListPoints,
    PipelineContext,
)
from kups.core.patch import Accept, Patch
from kups.core.typing import ExclusionId, HasCell, InclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, no_jax_tracing, tree_map


@dataclass
class CellListCacheParameters:
    """Static shapes of a cell table.

    Attributes:
        chunk_size: Query rows processed per local scan step; the
            padded slot count is a multiple of it.
        max_cells_per_system: Capacity for spatial bins per system
            (``prod(bins) <= max_cells_per_system`` is asserted).
        cell_capacity: Maximum active particles per bin (asserted).
        stencil_width: Distinct neighbor cells per cell. 27 is always valid;
            with fewer than three bins along an axis the stencil wraps onto
            duplicates and ``prod(min(bins, 3))`` suffices (asserted).
            Smaller widths shrink every candidate gather.
        key_chunk_size: Optional keys per block: columns of each neighboring cell
            for ``key_layout="cells"``, or particle slots for ``"slots"``.
            Empty blocks skip pair evaluation; ``None`` uses one block.
            This changes traversal, not the occupancy limit.
        key_layout: Read neighboring cells, or traverse all particle slots.
            Slot traversal still applies the exact distance and pair masks.
        max_images_per_pair: Capacity for each pair's periodic-image window.
            Estimated from the cutoff and cell geometry; one in the
            minimum-image regime.
    """

    chunk_size: int = field(static=True, default=32)
    max_cells_per_system: int = field(static=True, default=512)
    cell_capacity: int = field(static=True, default=8)
    stencil_width: int = field(static=True, default=27)
    key_chunk_size: int | None = field(static=True, default=None, kw_only=True)
    key_layout: Literal["cells", "slots"] = field(
        static=True, default="cells", kw_only=True
    )
    max_images_per_pair: int = field(static=True, default=1, kw_only=True)

    def __post_init__(self) -> None:
        if min(self.chunk_size, self.max_cells_per_system, self.cell_capacity) <= 0:
            raise ValueError("Cell table capacities and chunk_size must be positive")
        if self.key_chunk_size is not None and self.key_chunk_size < 1:
            raise ValueError("key_chunk_size must be positive")
        if self.key_layout not in ("cells", "slots"):
            raise ValueError("key_layout must be 'cells' or 'slots'")
        if not 1 <= self.stencil_width <= 27:
            raise ValueError("stencil_width must be between 1 and 27")
        if self.max_images_per_pair < 1:
            raise ValueError("max_images_per_pair must be positive")

    @classmethod
    @no_jax_tracing
    def estimate(
        cls,
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, HasCell[AnyPeriodicity]],
        cutoffs: Table[SystemId, Array],
        *,
        chunk_size: int = 64,
        occupancy_factor: float = 2.0,
        occupancy_headroom: int = 32,
        key_chunk_size: int | Literal["auto"] | None = None,
        key_layout: Literal["cells", "slots", "auto"] = "cells",
    ) -> Self:
        """Estimate parameters from a concrete state.

        Sizes ``cell_capacity`` from the exact per-cell occupancy of the
        active particles with multiplicative and additive headroom to absorb
        density growth (e.g. GCMC insertions); violations at runtime fail the
        assertions in [`build_cell_list_cache`][kups.core.neighborlist.cell_list_cache.build_cell_list_cache].

        Args:
            particles: Current (possibly buffered) particle table.
            systems: System table with cells.
            cutoffs: Per-system cutoff radii.
            chunk_size: Query rows per local scan step.
            occupancy_factor: Multiplier on the observed maximum occupancy.
            occupancy_headroom: Additive slack on top of the scaled occupancy.
            key_chunk_size: Keys per block, or ``"auto"`` to round cell occupancy
                up to a multiple of 32. For slot traversal, rounds total occupancy
                plus additive headroom up to a multiple of 128. Later blocks
                remain available as density grows; ``None`` uses one block.
            key_layout: ``"auto"`` selects slots when a single system's stencil
                covers all cells and the estimated slot block is smaller than
                the combined cell blocks. Otherwise retains cell traversal.

        Returns:
            Static parameters for the table.
        """
        cutoff = Table.broadcast_to(cutoffs, systems).data
        bins = num_cells(systems.data, cutoff)
        images = candidate_image_counts(systems.data.cell, cutoff)
        max_cells = int(jnp.prod(bins, axis=-1).max())
        n_cells = systems.size * max_cells
        cell = cell_rows(particles, systems, bins, max_cells, ()).cell
        max_occupancy = int(jnp.bincount(cell, length=n_cells + 1)[:n_cells].max())
        stencil_width = int(jnp.prod(jnp.minimum(bins, 3), axis=-1).max())
        cell_chunk_size = max(32, -(-max_occupancy // 32) * 32)
        occupancy = int((cell < n_cells).sum()) + occupancy_headroom
        slot_chunk_size = max(128, -(-occupancy // 128) * 128)
        if key_layout == "auto":
            limits = jnp.where(jnp.asarray(systems.data.cell.periodic), 3, 2)
            key_layout = (
                "slots"
                if systems.size == 1
                and bool((bins <= limits).all())
                and slot_chunk_size < stencil_width * cell_chunk_size
                else "cells"
            )
        if key_chunk_size == "auto":
            key_chunk_size = (
                slot_chunk_size if key_layout == "slots" else cell_chunk_size
            )
        return cls(
            chunk_size=chunk_size,
            key_chunk_size=key_chunk_size,
            key_layout=key_layout,
            max_cells_per_system=max_cells,
            cell_capacity=max(
                1, int(max_occupancy * occupancy_factor) + occupancy_headroom
            ),
            stencil_width=stencil_width,
            max_images_per_pair=int(images.prod(axis=-1).max()),
        )


class CellRows[Data](NamedTuple):
    """Binned particle rows: what a cell table stores per slot.

    Attributes:
        frac: Folded fractional positions, ``(..., 3)``.
        system: System index (0 where inactive).
        inclusion: Inclusion index; out of bounds for inactive rows.
        exclusion: Exclusion index.
        data: Caller payload (e.g. kernel features), leaves ``(..., ...)``.
        cell: Global cell id; the sentinel cell where inactive.
    """

    frac: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]
    data: Data
    cell: Array

    @property
    def positions(self) -> Array:
        """Fractional coordinates for the neighbor pipeline."""
        return self.frac

    @staticmethod
    def concatenate[D](*parts: CellRows[D]) -> CellRows[D]:
        """Join rows, aligning index vocabularies and preserving count bounds."""

        def concatenate[Key: SupportsSorting](
            *leaves: Array | Index[Key],
        ) -> Array | Index[Key]:
            if isinstance(leaves[0], Index):
                indices = [x for x in leaves if isinstance(x, Index)]
                if len(indices) != len(leaves):
                    raise TypeError("Expected matching Index leaves")
                return Index.concatenate(*indices)
            arrays = [x for x in leaves if isinstance(x, Array)]
            if len(arrays) != len(leaves):
                raise TypeError("Expected matching array leaves")
            return jnp.concatenate(arrays)

        return jax.tree.map(concatenate, *parts, is_leaf=lambda x: isinstance(x, Index))

    def pad(self, padding: int, *, sentinel_cell: int) -> Self:
        """Append inactive rows with finite geometry and zero-filled payloads."""
        fills = CellRows(
            jnp.array(0.0),
            tree_map(lambda _: jnp.array(0), self.system),
            tree_map(lambda _: jnp.array(self.inclusion.num_labels), self.inclusion),
            tree_map(lambda _: jnp.array(self.exclusion.num_labels), self.exclusion),
            tree_map(lambda _: jnp.array(0), self.data),
            jnp.array(sentinel_cell),
        )
        return jax.tree.map(
            lambda x, fill: jnp.concatenate(
                [x, jnp.full((padding, *x.shape[1:]), fill, dtype=x.dtype)]
            ),
            self,
            fills,
        )


def cell_rows[Data](
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, HasCell[AnyPeriodicity]],
    bins: Array,
    max_cells: int,
    data: Data,
) -> CellRows[Data]:
    """Fold and bin particles into global cells ``system * max_cells + local``.

    Particles with an out-of-bounds inclusion index (the buffered-row
    convention) are inactive and land in the sentinel cell ``n_systems *
    max_cells``.

    Args:
        particles: Particle table (positions, system, inclusion, exclusion).
        systems: System table with cells.
        bins: Per-system bin counts, ``(n_systems, 3)``.
        max_cells: Allocated cell capacity per system.
        data: Per-particle payload carried along.

    Returns:
        The binned rows.
    """
    points = particles.data
    active = points.inclusion.indices < points.inclusion.num_labels
    system = Index(
        systems.keys, jnp.where(active, points.system.indices_in(systems.keys), 0)
    )
    frames = systems.map_data(lambda s: s.cell.frame.materialize())
    frac = frames[system].to_fractional(
        jnp.where(active[:, None], points.positions, 0.0)
    )
    frac, cell = assign_cells(
        frac, system.indices, active, bins, max_cells, systems.data.cell
    )
    active = cell < bins.shape[0] * max_cells
    inclusion = Index(
        points.inclusion.keys,
        jnp.where(active, points.inclusion.indices, points.inclusion.num_labels),
    )
    return CellRows(frac, system, inclusion, points.exclusion, data, cell)


class CellListCache[Data](NamedTuple):
    """Slot-sorted particle rows with per-cell occupancy and stencil tables.

    Slots ``0 .. n-1`` hold the particles, slots up to ``n_pad`` are padding,
    and slot ``n_pad`` is the inactive sentinel filling empty table entries.
    Inactive rows occupy no cell entry and emit no candidates.

    Attributes:
        rows: Per-slot rows, leaves ``(n_pad + 1, ...)``.
        slot_of_row: Slot of each original particle row, ``(n,)``.
        cells: Slot ids per cell, ``(n_cells + 1, cell_capacity)``, sentinel
            filled; the last row is the always-empty sentinel cell.
        stencil: Neighbor cell ids per cell, ``(n_cells + 1, stencil_width)``,
            invalid entries routed to the sentinel cell.
        bins: Per-system bin counts, ``(n_systems, 3)``.
    """

    rows: CellRows[Data]
    slot_of_row: Array
    cells: Array
    stencil: Array
    bins: Array

    @property
    def sentinel_slot(self) -> int:
        return self.rows.frac.shape[0] - 1

    @property
    def sentinel_cell(self) -> int:
        return self.stencil.shape[0] - 1

    @property
    def max_cells_per_system(self) -> int:
        """Allocated capacity, derived from the stored array shapes."""
        return self.sentinel_cell // self.bins.shape[0]

    def candidates(self, cell: Array) -> Array:
        """Candidate slots of the given cells, ``(..., stencil_width * cell_capacity)``."""
        return self.cells[self.stencil[cell]].reshape(
            *cell.shape,
            self.stencil.shape[1] * self.cells.shape[1],
        )

    def select(self, ctx: PipelineContext) -> Candidates:
        """Look up occupants for ``CellListSelector`` using the stored grid.

        The cache must correspond to ``ctx.keys`` in original row order.
        Candidate ids are mapped back to those rows; the shared selector uses
        original coordinates when computing periodic shifts.
        """
        query = ctx.query_table
        _, cell = assign_cells(
            query.data.positions,
            query.data.system.indices_in(ctx.systems.keys),
            query.data.inclusion.valid_mask,
            self.bins,
            self.max_cells_per_system,
            ctx.systems.data.cell,
        )
        slots = (
            self.candidates(cell) if ctx.keys.size else jnp.zeros((query.size, 0), int)
        )
        row_of_slot = jnp.full(self.sentinel_slot + 1, ctx.keys.size, dtype=int)
        row_of_slot = row_of_slot.at[self.slot_of_row].set(jnp.arange(ctx.keys.size))
        return Candidates(
            Index(ctx.keys.keys, row_of_slot[slots].ravel(), _cls=ctx.keys.cls),
            Index(
                query.keys,
                jnp.repeat(jnp.arange(query.size), slots.shape[1]),
                _cls=query.cls,
            ),
        )

    def candidate_chunks(self, cell: Array, chunk_size: int) -> Array:
        """Group candidate columns into blocks, with a leading chunk axis.

        Columns from every neighboring cell share a block. Entirely unused
        blocks can be skipped without reducing cell occupancy capacity.
        The shape is ``(n_chunks, *cell.shape, stencil_width * chunk_size)``;
        the last block is sentinel-padded.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        width, capacity = self.stencil.shape[1], self.cells.shape[1]
        indices = self.candidates(cell).reshape(*cell.shape, width, capacity)
        padding = -capacity % chunk_size
        indices = jnp.pad(
            indices,
            ((0, 0),) * (indices.ndim - 1) + ((0, padding),),
            constant_values=self.sentinel_slot,
        )
        n_chunks = (capacity + padding) // chunk_size
        indices = indices.reshape(*cell.shape, width, n_chunks, chunk_size)
        return jnp.moveaxis(indices, -2, 0).reshape(
            n_chunks, *cell.shape, width * chunk_size
        )

    def bin_rows[D](
        self,
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, HasCell[AnyPeriodicity]],
        data: D,
    ) -> CellRows[D]:
        """Bin further particles (e.g. proposal queries) into this table's cells."""
        return cell_rows(particles, systems, self.bins, self.max_cells_per_system, data)


def cell_candidates[Data](
    keys: CellRows[Data],
    queries: CellRows[Data],
    index: Array,
    systems: Table[SystemId, HasCell[AnyPeriodicity]],
    cutoffs: Table[SystemId, Array],
    max_images_per_pair: int,
) -> tuple[CandidateBatch[Literal[2]], PipelineContext]:
    """Adapt a cell-table query to the standard neighbor pipeline.

    ``index`` contains key slots per query row. The same adapter handles
    within-system query pairs, so geometry and masks have a single implementation.
    Periodic images expand only this chunk, with capacity bounded per pair.
    """
    key_table = Table.arange(keys, label=ParticleId)
    query_table = Table.arange(queries, label=ParticleId)
    ctx = PipelineContext(key_table, query_table, systems, None)
    candidates = Candidates(
        Index(key_table.keys, index.ravel()),
        Index(query_table.keys, jnp.repeat(jnp.arange(index.shape[0]), index.shape[1])),
    )
    if max_images_per_pair == 1:
        return make_batch_with_mic(candidates, key_table, query_table, systems), ctx
    return replicate_for_images(
        candidates,
        key_table,
        query_table,
        systems,
        cutoffs,
        FixedCapacity(index.size * max_images_per_pair),
    ), ctx


def build_cell_list_cache[Data](
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, HasCell[AnyPeriodicity]],
    cutoffs: Table[SystemId, Array],
    data: Data,
    parameters: CellListCacheParameters,
) -> CellListCache[Data]:
    """Bin, sort, and tabulate the particles of a point cloud.

    Args:
        particles: Particle table (positions, system, inclusion, exclusion).
        systems: System table with cells.
        cutoffs: Per-system cutoff radii sizing the bins.
        data: Per-particle payload to carry per slot, leaves ``(n, ...)``.
        parameters: Static table shapes.

    Returns:
        The cell table.
    """
    n = particles.size
    n_pad = -(-n // parameters.chunk_size) * parameters.chunk_size
    max_cells = parameters.max_cells_per_system
    n_cells = systems.size * max_cells
    cutoff = Table.broadcast_to(cutoffs, systems).data
    runtime_assert(
        (
            candidate_image_counts(systems.data.cell, cutoff).prod(axis=-1)
            <= parameters.max_images_per_pair
        ).all(),
        "max_images_per_pair exceeded; re-estimate CellListCacheParameters for "
        "the current cutoffs and cells.",
    )
    bins = num_cells(systems.data, cutoff)
    runtime_assert(
        (jnp.prod(bins, axis=-1) <= max_cells).all(),
        "max_cells_per_system exceeded; increase CellListCacheParameters.max_cells_per_system.",
    )

    rows = cell_rows(particles, systems, bins, max_cells, data)
    order = jnp.argsort(rows.cell)
    sorted_cell = rows.cell[order]
    rank = jnp.arange(n) - jnp.searchsorted(sorted_cell, sorted_cell, side="left")
    runtime_assert(
        ((rank < parameters.cell_capacity) | (sorted_cell == n_cells)).all(),
        "cell_capacity exceeded; increase CellListCacheParameters.cell_capacity.",
    )
    cells = (
        jnp.full((n_cells + 1, parameters.cell_capacity), n_pad, dtype=int)
        .at[sorted_cell, rank]
        .set(jnp.arange(n), mode="drop")
        # Inactive particles were scattered into the sentinel-cell row; reset it.
        .at[n_cells]
        .set(n_pad)
    )

    sorted_rows: CellRows[Data] = jax.tree.map(lambda x: x[order], rows)
    padded_rows = sorted_rows.pad(n_pad + 1 - n, sentinel_cell=n_cells)
    return CellListCache(
        rows=padded_rows,
        slot_of_row=jnp.zeros(n, dtype=order.dtype).at[order].set(jnp.arange(n)),
        cells=cells,
        stencil=neighbor_cells(
            jnp.arange(n_cells + 1),
            bins,
            max_cells,
            systems.data.cell,
            width=parameters.stencil_width,
        ),
        bins=bins,
    )


@dataclass
class CellListCacheUpdatePatch[State, Data](Patch[State]):
    """Accept-conditional update of a persistent cell table.

    Accepted rows refresh their coordinates and payload. Only rows changing
    cells are removed and reinserted (overflow asserted); moves within a cell
    retain membership. An entirely rejected proposal skips both operations.
    Slots retain their original row identity, so ``slot_of_row`` stays valid.
    Spatial sort order is not maintained after moves. Changed rows are grouped
    by destination cell to assign insertion ranks with linear storage.

    Attributes:
        slots: Slots of the changed rows, ``(k,)``.
        system_idx: System of each changed row (for the accept gather).
        new: Rows after the update, leaves ``(k, ...)``.
        lens: Lens to the cell table in the state.
    """

    slots: Array
    system_idx: Index[SystemId]
    new: CellRows[Data]
    lens: Lens[State, CellListCache[Data]] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        table = self.lens.get(state)
        n_pad, n_cells = table.sentinel_slot, table.sentinel_cell
        old_cell = table.rows.cell[self.slots]
        accept = Table.broadcast_to(accept, Table(table.rows.system.keys, table.bins))
        old_accept = accept.at(
            table.rows.system[self.slots], args={"mode": "fill", "fill_value": False}
        ).get()
        new_accept = accept.at(
            self.system_idx, args={"mode": "fill", "fill_value": False}
        ).get()
        # Buffered deletions have no new system; insertions have no old cell.
        # Match Table.update_if's acceptance over the old/new system indices.
        ok = (self.slots < n_pad) & (((old_cell != n_cells) & old_accept) | new_accept)

        def apply(table: CellListCache[Data]) -> CellListCache[Data]:
            # Insertion/deletion also changes the cell via the inactive sentinel.
            relocate = ok & (old_cell != self.new.cell)

            def move_cells(cells: Array) -> Array:
                column = jnp.argmax(cells[old_cell] == self.slots[:, None], axis=-1)
                column = jnp.where(
                    relocate & (old_cell != n_cells), column, cells.shape[1]
                )
                cells = cells.at[old_cell, column].set(n_pad, mode="drop")
                # Rank incoming rows per cell to assign distinct free columns.
                insert = relocate & (self.new.cell != n_cells)
                target = jnp.where(insert, self.new.cell, n_cells)
                order = jnp.argsort(target, stable=True)
                sorted_target = target[order]
                sorted_rank = jnp.arange(target.size) - jnp.searchsorted(
                    sorted_target, sorted_target, side="left"
                )
                rank = jnp.zeros_like(sorted_rank).at[order].set(sorted_rank)
                free_count = jnp.cumsum(cells[target] == n_pad, axis=-1)
                position = jnp.argmax(free_count == (rank + 1)[:, None], axis=-1)
                runtime_assert(
                    ((free_count[:, -1] > rank) | ~insert).all(),
                    "cell_capacity exceeded during cell table update; increase CellListCacheParameters.cell_capacity.",
                )
                position = jnp.where(insert, position, cells.shape[1])
                return cells.at[target, position].set(self.slots, mode="drop")

            cells = jax.lax.cond(
                jnp.any(relocate), move_cells, lambda cells: cells, table.cells
            )

            def rewrite(current: Array, new: Array) -> Array:
                mask = ok.reshape(ok.shape + (1,) * (new.ndim - 1))
                return current.at[self.slots].set(
                    jnp.where(mask, new, current[self.slots])
                )

            rows: CellRows[Data] = jax.tree.map(rewrite, table.rows, self.new)
            return table._replace(rows=rows, cells=cells)

        table = jax.lax.cond(jnp.any(ok), apply, lambda table: table, table)
        return self.lens.set(state, table)
