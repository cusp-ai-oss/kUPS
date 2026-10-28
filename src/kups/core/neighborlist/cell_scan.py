# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Cell-list neighbor list whose candidate stage is a scan kernel.

Keys are sorted by cell. For each query, a Pallas kernel scans a window of cells
around the query's cell and emits only pairs within the cutoff, so the pipeline masks
see about 7x fewer candidates than in ``CellListNeighborList``. The window covers the
cutoff on each axis (one cell per side unless the cutoff exceeds the cell), wraps on
periodic axes and each pair carries its image shift. Infinite cutoffs need open axes.

Triton kernel on CUDA GPUs with compute capability 8.0+, Pallas interpreter
elsewhere. The kernel uses int32 and float32.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from kups.core.assertion import runtime_assert
from kups.core.capacity import Capacity, LensCapacity
from kups.core.data import Index, Table
from kups.core.lens import Lens
from kups.core.neighborlist.cell_list import IsCellListParams, assign_cells
from kups.core.neighborlist.common import (
    Candidates,
    candidates_to_batch,
    lift_query_candidates,
    num_cells,
)
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.pipeline import build_radius_graph
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.patch import IdPatch, WithPatch
from kups.core.typing import DTypeLike, ParticleId, SystemId
from kups.core.utils.jax import dataclass, jit
from kups.core.utils.subselect import argsort_indices

BLOCK, WARPS = 16, 1  # flat on an L4 for BLOCK 8-64, WARPS 1-4
# Å added to the kernel's float32 cutoff test, so it keeps every pair the exact
# (float64) DistanceCutoffMask keeps
SLACK = 1e-3


def triton_available() -> bool:
    """Whether the scan kernel runs on Triton: a CUDA GPU of compute capability 8.0+."""
    device = jax.devices()[0]
    capability = float(getattr(device, "compute_capability", 0) or 0)
    return device.platform == "gpu" and capability >= 8.0


def _scan_kernel(
    kx_ref: jax.Ref, ky_ref: jax.Ref, kz_ref: jax.Ref, knx_ref: jax.Ref, kny_ref: jax.Ref, knz_ref: jax.Ref,
    korder_ref: jax.Ref, start_ref: jax.Ref,
    qx_ref: jax.Ref, qy_ref: jax.Ref, qz_ref: jax.Ref, qnx_ref: jax.Ref, qny_ref: jax.Ref, qnz_ref: jax.Ref,
    qsys_ref: jax.Ref, qcell_ref: jax.Ref, qoffset_ref: jax.Ref,
    frame_ref: jax.Ref, rc2_ref: jax.Ref, bins_ref: jax.Ref, radius_ref: jax.Ref,
    *out_refs: jax.Ref,
    periodic: tuple[bool, ...], max_cells: int, write: bool,
) -> None:  # fmt: skip
    """Count (``write=False``) or write the in-cutoff keys in query ``q``'s window of cells.

    Key and query positions are folded fractional coordinates; ``kn``/``qn`` are the integer
    offsets that unfold them. A pair's shift (``delta = key - query - shift`` on the unfolded
    coordinates) is ``kn - qn - w`` for a key visited in window image ``w``.
    """
    q = pl.program_id(0)
    s = qsys_ref[q]
    h = [frame_ref[s, k] for k in range(9)]
    rc2 = rc2_ref[s]
    bins = [bins_ref[s, a] for a in range(3)]
    qpos = (qx_ref[q], qy_ref[q], qz_ref[q])
    qn = (qnx_ref[q], qny_ref[q], qnz_ref[q])
    qcell = [qcell_ref[q, a] for a in range(3)]  # negative: inactive query
    radius = [radius_ref[s, a] for a in range(3)]
    base = qoffset_ref[q] if write else 0

    def scan_cell(offset: tuple[Array, ...], n_found: Array) -> Array:
        coord: list[Array] = []
        image: list[Array] = []
        inside = qcell[0] >= 0
        for a in range(3):
            c = qcell[a] + offset[a]
            w = jnp.floor_divide(c, bins[a]) if periodic[a] else jnp.zeros_like(c)
            coord.append(c - w * bins[a])
            image.append(w.astype(jnp.float32))
            if not periodic[a]:
                inside &= (c >= 0) & (c < bins[a])
        cell = (
            s * max_cells + coord[0] + coord[1] * bins[0] + coord[2] * bins[0] * bins[1]
        )
        # out-of-range ids (cell capacity too small: the capacity assertion fails and the
        # caller retries) read as empty instead of out of bounds
        inside &= (
            cell < start_ref.shape[0] - 2
        )  # last slot: sentinel cell of inactive keys
        cell = jnp.where(inside, cell, 0)
        start, end = start_ref[cell], start_ref[cell + 1]
        end = jnp.where(inside, end, start)

        def chunk_body(chunk: Array, n_found: Array) -> Array:
            p = start + chunk * BLOCK + jnp.arange(BLOCK, dtype=jnp.int32)
            valid = p < end
            kpos = [
                plgpu.load(r.at[p], mask=valid, other=0.0)
                for r in (kx_ref, ky_ref, kz_ref)
            ]
            delta = [kpos[a] + image[a] - qpos[a] for a in range(3)]
            real = [
                delta[0] * h[c] + delta[1] * h[3 + c] + delta[2] * h[6 + c]
                for c in range(3)
            ]
            hit = valid & (
                real[0] * real[0] + real[1] * real[1] + real[2] * real[2] <= rc2
            )
            if write:
                key_ref, sx_ref, sy_ref, sz_ref = out_refs
                # misses and capacity overflow go to the trailing scratch slot, so every
                # store is in bounds and unmasked (also under the interpreter)
                scratch = key_ref.shape[0] - 1
                slot = (
                    base
                    + n_found
                    + jnp.cumsum(hit.astype(jnp.int32), dtype=jnp.int32)
                    - 1
                )
                slot = jnp.where(hit & (slot < scratch), slot, scratch)
                kn = [
                    plgpu.load(r.at[p], mask=valid, other=0.0)
                    for r in (knx_ref, kny_ref, knz_ref)
                ]
                for a in range(3):
                    plgpu.store(
                        (sx_ref, sy_ref, sz_ref)[a].at[slot], kn[a] - qn[a] - image[a]
                    )
                plgpu.store(
                    key_ref.at[slot], plgpu.load(korder_ref.at[p], mask=valid, other=0)
                )
            return n_found + jnp.sum(hit.astype(jnp.int32), dtype=jnp.int32)

        n_chunks = (end - start + BLOCK - 1) // BLOCK
        return jax.lax.fori_loop(0, n_chunks, chunk_body, n_found)

    def scan_axis(axis: int, offset: tuple[Array, ...], n_found: Array) -> Array:
        """Visit window offsets ``-radius..radius`` on ``axis`` and the axes below it."""
        if axis < 0:
            return scan_cell(offset, n_found)
        return jax.lax.fori_loop(
            -radius[axis],
            radius[axis] + 1,
            lambda d, n: scan_axis(axis - 1, (d, *offset), n),
            n_found,
        )

    n_found = scan_axis(2, (), jnp.zeros((), jnp.int32))
    if not write:
        out_refs[0][q] = n_found


def _closest_copy(
    candidates: Candidates,
    shifts: Array,
    filled: Array,
    keys: Table[ParticleId, NeighborListPoints],
    queries: Table[ParticleId, NeighborListPoints],
    query_vectors: Array,
) -> Array:
    """Flag the closest copy of each (key, query) pair; ties go to the first emitted."""
    key, query = candidates.key_idx.raw_indices, candidates.query_idx.raw_indices
    delta = keys.data.positions[key] - queries.data.positions[query] - shifts
    dist_sq = jnp.sum(
        jnp.einsum("ni,nij->nj", delta, query_vectors[query]) ** 2, axis=-1
    )
    n = key.size
    if n == 0:
        return filled
    # group the copies of each pair; unfilled rows share the out-of-range pair and sort last
    rows = jnp.arange(n, dtype=key.dtype)
    sorted_query, sorted_key, order = jax.lax.sort((query, key, rows), num_keys=2)
    new_pair = (sorted_query[1:] != sorted_query[:-1]) | (
        sorted_key[1:] != sorted_key[:-1]
    )
    segment = jnp.cumsum(jnp.pad(new_pair, (1, 0)), dtype=key.dtype)
    best = jax.ops.segment_min(dist_sq[order], segment, num_segments=n)
    first = jax.ops.segment_min(
        jnp.where(dist_sq[order] == best[segment], order, n), segment, num_segments=n
    )
    return filled & jnp.zeros(n, bool).at[order].set(order == first[segment])


@dataclass
class CellScanSelector:
    """Candidate selector: in-cutoff pairs within each query's window of cells (scan kernel)."""

    cutoffs: Table[SystemId, Array]
    max_cells: Capacity[int]
    max_candidates: Capacity[int]

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        keys, queries, systems = ctx.keys, ctx.query_table, ctx.systems
        dtype = keys.data.system.raw_indices.dtype
        if keys.size == 0 or queries.size == 0:  # no kernel launch over an empty grid
            empty = Candidates(
                Index(keys.keys, jnp.zeros(0, dtype), _cls=keys.cls),
                Index(queries.keys, jnp.zeros(0, dtype), _cls=queries.cls),
            )
            shifts = jnp.zeros((0, 3), keys.data.positions.dtype)
            return candidates_to_batch(
                lift_query_candidates(empty, ctx), shifts, jnp.zeros(0, bool)
            )
        cell = systems.data.cell
        periodic = tuple(bool(p) for p in cell.periodic)
        cutoffs = self.cutoffs.data
        width = cell.perpendicular_lengths
        bins = num_cells(systems.data, cutoffs, dtype=dtype)
        max_cells = self.max_cells.generate_assertion(bins.prod(axis=-1).max()).size
        # cells scanned on each side: enough to cover the cutoff; on open axes at most the grid
        reach = jnp.ceil(cutoffs[:, None] * bins / width)
        is_periodic = jnp.asarray(periodic)
        runtime_assert(
            jnp.all(jnp.isfinite(reach) | ~is_periodic),
            "CellScanNeighborList needs finite cutoffs on periodic axes; use a dense neighbor list",
        )
        reach = jnp.where(is_periodic, reach, jnp.minimum(reach, bins - 1))
        radius = jnp.where(jnp.isfinite(reach), reach, 0).astype(jnp.int32)

        def assign(points: NeighborListPoints) -> tuple[Array, Array, Array]:
            system = points.system.indices_in(systems.keys)
            folded, ids = assign_cells(
                points.positions,
                system,
                points.inclusion.valid_mask,
                bins,
                max_cells,
                cell,
            )
            return folded, ids, jnp.clip(system, 0, systems.size - 1)

        # keys sorted by cell; start[c] .. start[c + 1] are the keys of cell c
        kfold, key_cells, _ = assign(keys.data)
        order = argsort_indices(key_cells)
        n_slots = bins.shape[0] * max_cells + 1  # + sentinel cell of inactive keys
        # cell offsets by counting: searchsorted would run a binary-search while loop, a
        # host round trip per iteration on GPU
        per_cell = jax.ops.segment_sum(
            jnp.ones_like(key_cells), key_cells, n_slots, indices_are_sorted=False
        )
        start = jnp.cumulative_sum(per_cell, include_initial=True, dtype=dtype)
        qfold, query_cells, qsys = assign(queries.data)
        qcell = jnp.floor(qfold * bins[qsys]).astype(dtype)
        qcell = jnp.clip(qcell, 0, bins[qsys] - 1)
        qcell = jnp.where((query_cells < n_slots - 1)[:, None], qcell, -1)

        def f32(x: Array) -> Array:
            return x.astype(jnp.float32)

        def i32(x: Array) -> Array:
            return x.astype(jnp.int32)

        kn = keys.data.positions[order] - kfold[order]  # integer unfolding offsets
        qn = queries.data.positions - qfold
        args = (
            *f32(kfold[order]).T, *f32(kn).T, i32(order), i32(start),
            *f32(qfold).T, *f32(qn).T, i32(qsys), i32(qcell),
        )  # fmt: skip
        system_args = (
            f32(cell.vectors.reshape(-1, 9)),
            f32((cutoffs + SLACK) ** 2),
            i32(bins),
            radius,
        )
        n_q = queries.size
        call = partial(
            pl.pallas_call,
            grid=(n_q,),
            compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1),
            interpret=not triton_available(),
        )
        kernel = partial(
            _scan_kernel,
            periodic=periodic,
            max_cells=max_cells,
        )
        # pass 1 counts hits per query; their prefix sum gives each query its
        # write offset and the total sizes the output; pass 2 writes
        counts = call(
            partial(kernel, write=False),
            out_shape=jax.ShapeDtypeStruct((n_q,), jnp.int32),
        )(*args, jnp.zeros(n_q, jnp.int32), *system_args)
        total = jnp.sum(counts, dtype=dtype)
        size = self.max_candidates.generate_assertion(total).size

        def out(dtype: DTypeLike) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct((size + 1,), dtype)

        key_idx, sx, sy, sz = (
            x[:size]
            for x in call(
                partial(kernel, write=True),
                out_shape=(out(jnp.int32), *(out(jnp.float32),) * 3),
            )(*args, (jnp.cumsum(counts) - counts).astype(jnp.int32), *system_args)
        )

        filled = jnp.arange(size, dtype=dtype) < total
        query_idx = jnp.repeat(
            jnp.arange(n_q, dtype=dtype), counts, total_repeat_length=size
        )
        candidates = Candidates(
            Index(
                keys.keys,
                jnp.where(filled, key_idx.astype(dtype), keys.size),
                _cls=keys.cls,
            ),
            Index(
                queries.keys,
                jnp.where(filled, query_idx, n_q),
                _cls=queries.cls,
            ),
        )
        shifts = jnp.where(filled[:, None], jnp.stack([sx, sy, sz], -1), 0.0)
        shifts = shifts.astype(keys.data.positions.dtype)
        # copies of a pair lie a lattice vector apart, so two fall within the cutoff
        # only where a periodic width is at most twice the cutoff; then the minimum
        # image is the closest copy (exact also on skewed cells), else every copy is
        revisits = jnp.any((width <= 2 * (cutoffs[:, None] + SLACK)) & is_periodic)
        is_min = jax.lax.cond(
            revisits,
            lambda: _closest_copy(
                candidates, shifts, filled, keys, queries, cell.vectors[qsys]
            ),
            lambda: filled,
        )
        return candidates_to_batch(
            lift_query_candidates(candidates, ctx), shifts, is_min
        )


@dataclass
class CellScanNeighborList:
    """``CellListNeighborList`` with a scan-kernel candidate stage (see module docstring).

    Attributes:
        avg_edges: Capacity per query for the final edges and the in-cutoff candidates.
        cells: Capacity for the cell hash table.
        cutoffs: Per-system cutoff distances.
    """

    avg_edges: Capacity[int]
    cells: Capacity[int]
    cutoffs: Table[SystemId, Array]

    @classmethod
    def new[S](
        cls,
        state: S,
        lens: Lens[S, IsCellListParams],
        cutoffs: Table[SystemId, Array],
    ) -> CellScanNeighborList:
        params = lens.get(state)
        return CellScanNeighborList(
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            cells=LensCapacity(params.cells, lens.focus(lambda x: x.cells)),
            cutoffs=cutoffs,
        )

    @classmethod
    def from_state[S](
        cls,
        state: S,
        lens: Lens[S, IsNeighborListState[IsCellListParams]],
        cutoffs: Table[SystemId, Array],
    ) -> CellScanNeighborList:
        """Seed from the ``neighborlist_params`` of the state at ``lens``."""
        return cls.new(state, lens.focus(lambda s: s.neighborlist_params), cutoffs)

    def selector(
        self, query_size: int, systems: Table[SystemId, NeighborListSystems]
    ) -> CellScanSelector:
        return CellScanSelector(
            cutoffs=Table.broadcast_to(self.cutoffs, systems),
            max_cells=self.cells,
            max_candidates=self.avg_edges.multiply(query_size),
        )

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]: ...
    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]: ...
    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]:
        edges = build_radius_graph(
            self, keys, systems, queries=queries, queried_keys=queried_keys
        )
        return WithPatch(edges, IdPatch[Any]())
