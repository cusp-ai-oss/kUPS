# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor lists backed by the NVIDIA ALCHEMI toolkit (``nvalchemiops``).

The toolkit's CUDA/Warp kernels (exposed through ``nvalchemiops.jax.neighbors``)
enumerate, for each central atom, a fixed-width row of neighbor atom ids plus
integer periodic shifts. Empty slots are padded with ``fill_value = total_atoms``
(an out-of-bounds sentinel). This is consumed here as a
[`CandidateSelector`][kups.core.neighborlist.types.CandidateSelector]: the full
padded matrix is emitted as candidates and the standard pipeline masks
(``InBoundsMask`` drops padding, ``DistanceCutoffMask`` applies the cutoff,
``ExclusionMask`` handles exclusions) filter at the end.

A single static Cartesian ``cutoff`` (a Python float) parameterises the list;
the kernel needs it as a host value, so it cannot be derived from a traced array.
The toolkit's grid and periodic-shift sizes are computed in-trace from the cell
geometry (matching the kernel's own formulae) and carried by resizable
``Capacity`` objects -- no host-side sizing call is needed to ``jax.jit`` the
list. ``cell_list`` additionally passes an explicit ``neighbor_search_radius`` so
the grid's MIN_CELLS promotion never silently drops neighbors.

``nvalchemiops`` is an optional, CUDA/Warp-backed dependency, imported lazily;
constructing a neighbor list without it raises a clear error. Partial lists
(kUPS ``queried_keys``) map onto the toolkit's compact ``target_indices`` rows.
Bipartite ``queries`` are not supported.
"""

from __future__ import annotations

from types import ModuleType
from typing import Literal, Protocol, overload

import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.capacity import Capacity, FixedCapacity, LensCapacity
from kups.core.data import Index, Table
from kups.core.lens import Lens, lens
from kups.core.neighborlist.common import Candidates, candidates_to_batch
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
    NeighborList,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, jit

NvalchemiMethod = Literal["naive", "cell_list"]

# The toolkit promotes the cell grid to at least this many cells per periodic
# (or already-subdivided) axis; mirrored in-trace so the search radius matches.
_MIN_CELLS_PER_DIMENSION = 4


def _import_nvalchemi() -> ModuleType:
    """Import ``nvalchemiops.jax.neighbors`` or raise an install hint."""
    try:
        from nvalchemiops.jax import neighbors  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "Nvalchemi neighbor lists require the optional 'nvalchemiops' package "
            "(CUDA/Warp backed). Install it into the environment to use "
            "NvalchemiCellListNeighborList / NvalchemiNaiveNeighborList."
        ) from exc
    return neighbors


def _toolkit_cell_grid(
    perp: Array, pbc: Array, cutoff: float
) -> tuple[Array, Array, Array]:
    """Per-system cell-list grid sizing matching the toolkit's build kernel.

    Args:
        perp: Per-axis perpendicular cell lengths, shape ``(num_systems, 3)``.
        pbc: Per-axis periodicity mask, shape ``(num_systems, 3)``.
        cutoff: Cartesian cutoff.

    Returns:
        ``(natural, promoted, radius)``: the natural grid ``max(floor(perp/cutoff), 1)``,
        the MIN_CELLS-promoted grid the toolkit actually builds, and the atom-centric
        search radius covering the cutoff on the promoted grid, all ``(num_systems, 3)``.
    """
    natural = jnp.maximum((perp / cutoff).astype(int), 1)
    k = jnp.ceil(jnp.log2(jnp.maximum(_MIN_CELLS_PER_DIMENSION / natural, 1.0))).astype(
        int
    )
    promote = (pbc | (natural > 1)) & (natural < _MIN_CELLS_PER_DIMENSION)
    promoted = jnp.where(promote, natural * (2**k), natural)
    radius = jnp.where(
        (promoted == 1) & ~pbc,
        0,
        jnp.maximum(jnp.ceil(promoted * cutoff / perp).astype(jnp.int32), 1),
    )
    return natural, promoted, radius


def _toolkit_shifts(perp: Array, pbc: Array, cutoff: float) -> tuple[Array, Array]:
    """Per-system periodic-shift sizing for the naive kernel.

    Args:
        perp: Per-axis perpendicular cell lengths, shape ``(num_systems, 3)``.
        pbc: Per-axis periodicity mask, shape ``(num_systems, 3)``.
        cutoff: Cartesian cutoff.

    Returns:
        ``(shift_range, num_shifts)``: per-axis maximum shift index ``(num_systems, 3)``
        and per-system shift count ``prod(2 * shift_range + 1)`` ``(num_systems,)``. The
        count is an upper bound (full grid); out-of-range image threads are dropped by
        the kernel's distance test.
    """
    shift_range = jnp.ceil(cutoff / perp).astype(jnp.int32) * pbc.astype(jnp.int32)
    num_shifts = jnp.prod(2 * shift_range + 1, axis=-1).astype(jnp.int32)
    return shift_range, num_shifts


class IsNvalchemiNaiveParams(Protocol):
    """Protocol for parameters required by ``NvalchemiNaiveNeighborList``."""

    @property
    def max_neighbors(self) -> int: ...
    @property
    def avg_edges(self) -> int: ...
    @property
    def max_shifts(self) -> int: ...


class IsNvalchemiCellListParams(Protocol):
    """Protocol for parameters required by ``NvalchemiCellListNeighborList``."""

    @property
    def max_neighbors(self) -> int: ...
    @property
    def avg_edges(self) -> int: ...
    @property
    def max_total_cells(self) -> int: ...


@dataclass
class NvalchemiSelector:
    """Selector that emits the toolkit's neighbor matrix as padded candidates.

    Converts the pipeline's fractional positions back to Cartesian, calls the
    ``naive`` or ``cell_list`` toolkit kernel (single-system) or its batched
    variant (multi-system, atoms grouped by system), and flattens the resulting
    ``(num_rows, max_neighbors)`` matrix into ``(key, query)`` candidate pairs.
    Toolkit integer shifts are used directly as the candidates' fractional
    shifts. No image replication is needed: the kernel enumerates periodic images.

    Attributes:
        cutoff: Static Cartesian cutoff for toolkit candidate generation.
        max_neighbors: Capacity for the matrix's per-row neighbor width; grown
            against the returned ``num_neighbors`` when a row overflows.
        method: Toolkit algorithm, ``"naive"`` (O(N^2)) or ``"cell_list"`` (O(N)).
        max_total_cells: Capacity for the cell-grid buffer (``cell_list``); grown
            in-trace to the promoted grid product so the toolkit never coarsens.
        max_shifts: Capacity for the per-system periodic-shift launch dimension
            (``naive``); grown in-trace to the shift count.
    """

    cutoff: float = field(static=True)
    max_neighbors: Capacity[int]
    method: NvalchemiMethod = field(static=True)
    max_total_cells: Capacity[int]
    max_shifts: Capacity[int]

    @jit
    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        if ctx.queries is not None:
            raise NotImplementedError(
                "Nvalchemi neighbor lists do not support bipartite queries; "
                "the toolkit has no bipartite-graph kernel."
            )
        nv = _import_nvalchemi()
        keys = ctx.keys
        n_atoms = keys.size
        frac = keys.data.positions
        cell = ctx.systems.data.cell
        frames = ctx.systems.map_data(lambda s: s.cell.frame.materialize())
        real_pos = frames[keys.data.system].to_real(frac)
        cell_vectors = frames.data.vectors  # (num_systems, 3, 3), rows = lattice vecs
        num_systems = ctx.systems.size
        periodic = jnp.asarray(cell.periodic, dtype=bool)
        pbc = jnp.broadcast_to(periodic, (num_systems, 3))
        perp = jnp.broadcast_to(cell.perpendicular_lengths, (num_systems, 3))

        nbr_atoms, center_atoms, shifts = self._run(
            nv,
            real_pos,
            keys.data.system.indices,
            cell_vectors,
            pbc,
            perp,
            ctx.queried_keys,
            n_atoms,
            num_systems,
        )

        # The pipeline's query side carries the affected subset (so
        # QueriedKeysDedupMask keys off the unaffected endpoint), so centers go
        # on the query side and their neighbors on the key side. The toolkit
        # shift ``s`` satisfies ``r = pos[neighbor] - pos[center] + s @ cell``;
        # with key=neighbor, query=center the pipeline shift is ``f = -s``.
        max_nbr = nbr_atoms.shape[1]
        center_flat = jnp.repeat(center_atoms.astype(jnp.int32), max_nbr)
        neighbor_flat = nbr_atoms.reshape(-1).astype(jnp.int32)
        shift_flat = -shifts.reshape(-1, 3).astype(real_pos.dtype)

        candidates = Candidates(
            key_idx=Index(keys.keys, neighbor_flat),
            query_idx=Index(keys.keys, center_flat),
        )
        # is_minimum_image: True where the toolkit shift is the minimum-image
        # shift (needed so ExclusionMask keeps replicated copies of excluded
        # pairs). Padding neighbors gather clamped positions but are dropped by
        # InBoundsMask, so their flag is irrelevant.
        key_pos = frac.at[neighbor_flat].get(mode="clip")
        query_pos = frac[center_flat]
        mic = cell.minimum_image_shifts(key_pos - query_pos)
        is_minimum_image = (jnp.abs(shift_flat - mic) < 0.5).all(axis=-1)
        return candidates_to_batch(candidates, shift_flat, is_minimum_image)

    def _run(
        self,
        nv: ModuleType,
        real_pos: Array,
        system_ids: Array,
        cell_vectors: Array,
        pbc: Array,
        perp: Array,
        queried_keys: Array | None,
        n_atoms: int,
        num_systems: int,
    ) -> tuple[Array, Array, Array]:
        """Run the toolkit kernel; return ``(nbr_atoms, center_atoms, shifts)``.

        ``nbr_atoms`` is the ``(num_rows, max_neighbors)`` matrix of neighbor atom
        ids (padding ``n_atoms``), ``center_atoms`` is each row's central atom id,
        and ``shifts`` is the matching ``(num_rows, max_neighbors, 3)`` integer
        shift matrix, all in the original (unsorted) atom index space. Grid/shift
        sizes are computed in-trace and their capacities grown against the demand;
        ``max_neighbors`` is grown against the returned per-row counts.
        """
        target_indices = (
            None if queried_keys is None else queried_keys.astype(jnp.int32)
        )
        if num_systems == 1:
            if self.method == "cell_list":
                _, promoted, radius = _toolkit_cell_grid(perp, pbc, self.cutoff)
                self.max_total_cells.generate_assertion(
                    jnp.prod(promoted, axis=-1).max()
                )
                nbr, counts, shifts = nv.cell_list(
                    real_pos,
                    self.cutoff,
                    cell_vectors,
                    pbc,
                    max_neighbors=self.max_neighbors.size,
                    max_total_cells=self.max_total_cells.size,
                    neighbor_search_radius=radius[0],
                    strategy="atom_centric",
                    target_indices=target_indices,
                )
            else:
                shift_range, num_shifts = _toolkit_shifts(perp, pbc, self.cutoff)
                self.max_shifts.generate_assertion(num_shifts.max())
                nbr, counts, shifts = nv.naive_neighbor_list(
                    real_pos,
                    self.cutoff,
                    cell=cell_vectors,
                    pbc=pbc,
                    max_neighbors=self.max_neighbors.size,
                    target_indices=target_indices,
                    shift_range_per_dimension=shift_range,
                    num_shifts_per_system=num_shifts,
                    max_shifts_per_system=self.max_shifts.size,
                )
            self.max_neighbors.generate_assertion(jnp.max(counts))
            centers = (
                jnp.arange(n_atoms, dtype=jnp.int32)
                if target_indices is None
                else target_indices
            )
            return nbr, centers, shifts

        # Multi-system: the toolkit's batch path requires system-contiguous
        # atoms (batch_ptr = cumulative per-system counts). Sort by system,
        # run, then map neighbor/center ids back to the original ordering.
        order = jnp.argsort(system_ids)
        # Inverse permutation by scatter (inverse[order[k]] = k); a second
        # argsort hits an int32/int64 accumulator mismatch in XLA under x64.
        inverse = (
            jnp.zeros_like(order)
            .at[order]
            .set(jnp.arange(order.shape[0], dtype=order.dtype))
        )
        atoms_per_system = jnp.bincount(system_ids, length=num_systems)
        batch_ptr = jnp.concatenate(
            [jnp.zeros(1, jnp.int32), jnp.cumsum(atoms_per_system).astype(jnp.int32)]
        )
        batch_idx = system_ids[order].astype(jnp.int32)
        sorted_targets = (
            None
            if target_indices is None
            else inverse[target_indices].astype(jnp.int32)
        )
        if self.method == "cell_list":
            natural, promoted, _ = _toolkit_cell_grid(perp, pbc, self.cutoff)
            self.max_total_cells.generate_assertion(
                (jnp.prod(natural, axis=-1) * num_systems).max()
            )
            # batch_cell_list exposes no neighbor_search_radius; its radius is 1,
            # correct only on the un-promoted grid (>=4 cells/axis). Fail loudly
            # rather than silently drop neighbors when promotion would occur.
            runtime_assert(
                (promoted == natural).all(),
                "Multi-system nvalchemi cell_list requires at least 4 cells per "
                "axis (cutoff <= perpendicular_length / 4); use the naive method "
                "for smaller boxes.",
                exception_type=ValueError,
            )
            nbr_sorted, counts, shifts = nv.batch_cell_list(
                real_pos[order],
                self.cutoff,
                cell_vectors,
                pbc,
                batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=self.max_neighbors.size,
                max_total_cells=self.max_total_cells.size,
                strategy="atom_centric",
                target_indices=sorted_targets,
            )
        else:
            shift_range, num_shifts = _toolkit_shifts(perp, pbc, self.cutoff)
            self.max_shifts.generate_assertion(num_shifts.max())
            nbr_sorted, counts, shifts = nv.batch_naive_neighbor_list(
                real_pos[order],
                self.cutoff,
                cell=cell_vectors,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=self.max_neighbors.size,
                target_indices=sorted_targets,
                shift_range_per_dimension=shift_range,
                num_shifts_per_system=num_shifts,
                max_shifts_per_system=self.max_shifts.size,
                max_atoms_per_system=n_atoms,
            )
        self.max_neighbors.generate_assertion(jnp.max(counts))
        # Sorted->original: padding (n_atoms) maps to itself via the appended row.
        order_pad = jnp.concatenate([order, jnp.array([n_atoms], order.dtype)])
        nbr_atoms = order_pad[nbr_sorted]
        centers = order if target_indices is None else target_indices
        return nbr_atoms, centers, shifts


def _run_pipeline(
    selector: NvalchemiSelector,
    keys: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoff: float,
    avg_edges: Capacity[int],
    queries: Table[ParticleId, NeighborListPoints] | None,
    queried_keys: Index[ParticleId] | None,
) -> Edges[Literal[2]]:
    """Build the standard self-graph pair pipeline around ``selector`` and run it.

    Args:
        selector: The toolkit-backed candidate selector to drive the pipeline.
        keys: Self-graph/output particle table.
        systems: Indexed system data with cell information.
        cutoff: The single Cartesian cutoff, applied per system by the mask.
        avg_edges: Capacity for the compacted edge array (per query particle).
        queries: Optional bipartite query table (rejected by the selector).
        queried_keys: Optional affected ``keys`` subset for self-graph updates.

    Returns:
        Compacted pair ``Edges`` indexing ``keys``.
    """
    query_size = (
        queried_keys.size
        if queried_keys is not None
        else (queries.size if queries is not None else keys.size)
    )
    cutoffs = systems.map_data(lambda _s: jnp.full((systems.size,), cutoff))
    pipeline = Pipeline[Literal[2]](
        selector=selector,
        masks=(
            InBoundsMask(),
            InclusionMatchMask(),
            QueriedKeysDedupMask(),
            DistanceCutoffMask(cutoffs=cutoffs),
            ExclusionMask(),
        ),
        compactor=ReduceCompactor(avg_edges=avg_edges.multiply(query_size)),
        postprocessors=(MirrorPairEdges(),),
    )
    if queries is not None:
        return pipeline(keys, systems, queries=queries)
    return pipeline(keys, systems, queried_keys=queried_keys)


@dataclass
class NvalchemiCellListNeighborList(NeighborList[Literal[2]]):
    """O(N) cell-list neighbor list backed by ``nvalchemiops``.

    Wraps the toolkit's ``cell_list`` kernel (``atom_centric`` strategy) as a
    candidate generator. The toolkit builds a padded neighbor matrix; the kUPS
    pipeline masks drop padding and apply the cutoff, exclusions, and
    inclusion-segment separation. Bipartite ``queries`` are rejected;
    ``queried_keys`` map onto the toolkit's compact ``target_indices``.

    Multi-system batches require at least 4 cells per axis (``cutoff <=
    perpendicular_length / 4``); smaller boxes raise (the batched kernel cannot
    take a corrected search radius) -- use the naive method there.

    Attributes:
        cutoff: Static Cartesian cutoff for toolkit candidate generation.
        max_neighbors: Capacity for the toolkit matrix's per-row neighbor width.
        avg_edges: Capacity for the final compacted edge array.
        max_total_cells: Capacity for the cell-grid buffer, grown in-trace to the
            grid the toolkit builds; no host-side estimate is needed to jit.

    Example:
        ```python
        nl = NvalchemiCellListNeighborList.from_state(state, cutoffs)
        edges = nl(particles, systems)
        ```
    """

    cutoff: float = field(static=True)
    max_neighbors: Capacity[int]
    avg_edges: Capacity[int]
    max_total_cells: Capacity[int]

    @classmethod
    def new[S](
        cls,
        state: S,
        lens: Lens[S, IsNvalchemiCellListParams],
        cutoffs: Table[SystemId, Array],
    ) -> NvalchemiCellListNeighborList:
        """Construct from a ``state`` and a lens to its toolkit parameters.

        Args:
            state: Object exposing the cell-list toolkit parameters via ``lens``.
            lens: Lens focusing the ``IsNvalchemiCellListParams`` parameters.
            cutoffs: Per-system cutoffs; the maximum becomes the static cutoff.

        Returns:
            A configured ``NvalchemiCellListNeighborList``.
        """
        params = lens.get(state)
        return cls(
            cutoff=float(jnp.max(jnp.asarray(cutoffs.data))),
            max_neighbors=LensCapacity(
                params.max_neighbors, lens.focus(lambda x: x.max_neighbors)
            ),
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            max_total_cells=LensCapacity(
                params.max_total_cells, lens.focus(lambda x: x.max_total_cells), base=1
            ),
        )

    @classmethod
    def from_state(
        cls,
        state: IsNeighborListState[IsNvalchemiCellListParams],
        cutoffs: Table[SystemId, Array],
    ) -> NvalchemiCellListNeighborList:
        """Construct from a state exposing ``neighborlist_params``."""
        return cls.new(state, lens(lambda s: s.neighborlist_params), cutoffs)

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
        return _run_pipeline(
            NvalchemiSelector(
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                method="cell_list",
                max_total_cells=self.max_total_cells,
                max_shifts=FixedCapacity(1),
            ),
            keys,
            systems,
            self.cutoff,
            self.avg_edges,
            queries,
            queried_keys,
        )


@dataclass
class NvalchemiNaiveNeighborList(NeighborList[Literal[2]]):
    """O(N^2) naive neighbor list backed by ``nvalchemiops``.

    Wraps the toolkit's ``naive_neighbor_list`` kernel as a candidate generator.
    Mirrors ``NvalchemiCellListNeighborList`` without the cell-hash capacity;
    suitable for small systems or large cutoff/box ratios.

    Attributes:
        cutoff: Static Cartesian cutoff for toolkit candidate generation.
        max_neighbors: Capacity for the toolkit matrix's per-row neighbor width.
        avg_edges: Capacity for the final compacted edge array.
        max_shifts: Capacity for the per-system periodic-shift launch dimension,
            grown in-trace to the shift count; no host-side estimate is needed.

    Example:
        ```python
        nl = NvalchemiNaiveNeighborList.from_state(state, cutoffs)
        edges = nl(particles, systems)
        ```
    """

    cutoff: float = field(static=True)
    max_neighbors: Capacity[int]
    avg_edges: Capacity[int]
    max_shifts: Capacity[int]

    @classmethod
    def new[S](
        cls,
        state: S,
        lens: Lens[S, IsNvalchemiNaiveParams],
        cutoffs: Table[SystemId, Array],
    ) -> NvalchemiNaiveNeighborList:
        """Construct from a ``state`` and a lens to its toolkit parameters.

        Args:
            state: Object exposing the naive toolkit parameters via ``lens``.
            lens: Lens focusing the ``IsNvalchemiNaiveParams`` parameters.
            cutoffs: Per-system cutoffs; the maximum becomes the static cutoff.

        Returns:
            A configured ``NvalchemiNaiveNeighborList``.
        """
        params = lens.get(state)
        return cls(
            cutoff=float(jnp.max(jnp.asarray(cutoffs.data))),
            max_neighbors=LensCapacity(
                params.max_neighbors, lens.focus(lambda x: x.max_neighbors)
            ),
            avg_edges=LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            max_shifts=LensCapacity(
                params.max_shifts, lens.focus(lambda x: x.max_shifts), base=1
            ),
        )

    @classmethod
    def from_state(
        cls,
        state: IsNeighborListState[IsNvalchemiNaiveParams],
        cutoffs: Table[SystemId, Array],
    ) -> NvalchemiNaiveNeighborList:
        """Construct from a state exposing ``neighborlist_params``."""
        return cls.new(state, lens(lambda s: s.neighborlist_params), cutoffs)

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
        return _run_pipeline(
            NvalchemiSelector(
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                method="naive",
                max_total_cells=FixedCapacity(1),
                max_shifts=self.max_shifts,
            ),
            keys,
            systems,
            self.cutoff,
            self.avg_edges,
            queries,
            queried_keys,
        )
