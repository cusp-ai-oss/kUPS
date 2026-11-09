"""kUPS on NVIDIA ALCHEMI (nvalchemi) ops: a neighbor list and a Lennard-Jones potential.

``NvalchemiLennardJones`` replaces kUPS's whole LJ evaluation (neighbor list, edges,
pair energies, autodiff) with nvalchemi's cell list and fused LJ kernel.

``NvalchemiNeighborList`` keeps kUPS's LJ and swaps only the neighbor search. It is a
``SelectableNeighborList``: its selector runs
``nvalchemiops.jax`` (batched) cell list at a static cutoff covering every system, and
kUPS's own masks (bounds, inclusion, per-system cutoff, exclusion) and compactor run on
the result as for the library neighbor lists. Bipartite (``queries``) and
``queried_keys`` calls fall back to kUPS's cell-list selector.

``use_backend`` chooses how nvalchemi's Warp kernels launch: ``"stock"`` keeps
``warp.jax_kernel`` (an XLA FFI callback into Python per launch), ``"jax-ptx"`` swaps in
``jax_ptx_warp.jax_kernel`` (native launches). It must run before ``nvalchemiops.jax`` is
first imported; this module imports nvalchemi lazily, at trace time.

Restrictions: cutoffs within the minimum-image regime (``cutoff <= perp / 2``), as every
candidate is flagged as its pair's minimum image.
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Literal, overload

import jax.numpy as jnp
from jax import Array

from kups.config import index_dtype
from kups.core.capacity import Capacity, LensCapacity
from kups.core.cell import Cell
from kups.core.data import Index, Table
from kups.core.lens import Lens, lens
from kups.core.neighborlist import CellListNeighborList, CellListSelector, Edges
from kups.core.neighborlist.common import Candidates, candidates_to_batch
from kups.core.neighborlist.pipeline import build_radius_graph
from kups.core.patch import IdPatch, WithPatch
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, jit

Backend = Literal["stock", "jax-ptx"]


def use_backend(backend: Backend) -> None:
    """Launch nvalchemi's Warp kernels with ``backend`` (before importing nvalchemiops.jax)."""
    if any(m.startswith("nvalchemiops.jax") for m in sys.modules):
        raise RuntimeError("use_backend must run before nvalchemiops.jax is imported")
    if backend == "jax-ptx":
        import jax_ptx_warp
        import warp

        warp.jax_kernel = functools.partial(jax_ptx_warp.jax_kernel, graph_capturable=True)


@functools.cache
def _fix_static_arrays() -> None:
    """Work around an nvalchemi 0.5.0 bug, with either backend: its jitted kernel-call
    wrapper marks every argument not typed ``wp.array`` static, which includes the
    generically annotated arrays of the batched ``gather`` kernel, so jitted
    ``batch_cell_list`` fails ("non-hashable static argument")."""
    import importlib

    import jax
    from warp._src import types as wpt

    def cached(kernel_call: Any) -> Any:
        static = tuple(i for i, a in enumerate(kernel_call.kernel.adj.args)
                       if not wpt.is_array(a.type))
        return jax.jit(kernel_call, static_argnums=static, static_argnames=("launch_dims",))

    # By module path: the package re-exports functions named like these modules.
    for name in ("_registration", "_cell_grid", "cell_list", "batch_cell_list"):
        module = importlib.import_module(f"nvalchemiops.jax.neighbors.{name}")
        module._cached_jax_kernel_call = cached  # type: ignore[attr-defined]


def _neighbor_matrix(positions: Array, system: Array, cell: Cell[Any], cutoff: float,
                     max_neighbors: Capacity[int], max_total_cells: int,
                     ) -> tuple[Array, Array, Array]:
    """nvalchemi's full neighbor matrix ``(nbr, count, shift)`` of Cartesian ``positions``
    in systems ``system`` (rows of ``cell``): row ``i`` lists ``count[i]`` neighbors ``j``
    (padded with ``len(positions)``) with ``x_j - x_i + shift @ vectors`` within
    ``cutoff``. Asserts (with kUPS's capacity fix) that every row fits."""
    from nvalchemiops.jax.neighbors import batch_cell_list, cell_list

    _fix_static_arrays()
    n_sys = cell.vectors.shape[0]
    pbc = jnp.broadcast_to(jnp.asarray(cell.periodic), (n_sys, 3))
    common: dict[str, Any] = dict(cell=cell.vectors, pbc=pbc,
                                  max_neighbors=max_neighbors.size,
                                  max_total_cells=max_total_cells)
    if n_sys == 1:
        nbr, count, shift = cell_list(positions, cutoff, **common)
    else:
        counts = jnp.zeros(n_sys, jnp.int32).at[system].add(1)
        ptr = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(counts, dtype=jnp.int32)])
        nbr, count, shift = batch_cell_list(positions, cutoff,
                                            batch_idx=system.astype(jnp.int32),
                                            batch_ptr=ptr, **common)
    max_neighbors.generate_assertion(count.max())
    return nbr, count, shift


@dataclass
class NvalchemiSelector:
    """Candidates within ``cutoff`` from nvalchemi's cell list (full self-graphs)."""

    max_neighbors: Capacity[int]
    fallback: CellListSelector
    cutoff: float = field(static=True)
    max_total_cells: int = field(static=True)

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        if ctx.queries is not None or ctx.queried_keys is not None:
            return self.fallback(ctx)
        keys, cell = ctx.keys, ctx.systems.data.cell
        n, n_sys = keys.size, ctx.systems.size
        system = jnp.minimum(keys.data.system.indices_in(ctx.systems.keys), n_sys - 1)
        # ctx positions are fractional; nvalchemi takes Cartesian rows @ lattice vectors.
        positions = jnp.einsum("ni,nij->nj", keys.data.positions, cell.vectors[system])
        nbr, count, shift = _neighbor_matrix(positions, system, cell, self.cutoff,
                                             self.max_neighbors, self.max_total_cells)
        width = self.max_neighbors.size
        slot = jnp.arange(width)[None, :] < count[:, None]
        rows = jnp.where(slot, jnp.arange(n)[:, None], n).ravel().astype(index_dtype())
        cols = jnp.where(slot, nbr, n).ravel().astype(index_dtype())
        # nvalchemi: r_ij = x_j - x_i + S @ vectors; kUPS: x_query - x_key + s @ vectors.
        candidates = Candidates(Index(keys.keys, rows, _cls=keys.cls),
                                Index(keys.keys, cols, _cls=keys.cls))
        shifts = shift.reshape(-1, 3).astype(keys.data.positions.dtype)
        return candidates_to_batch(candidates, shifts, jnp.ones(rows.shape, bool))


@dataclass
class NvalchemiNeighborList:
    """kUPS radius graph with nvalchemi's cell list as the candidate search.

    Attributes:
        cutoffs: Per-system cutoffs (kUPS masks the candidates to these).
        avg_edges: Edge capacity per particle; also the width of nvalchemi's
            neighbor matrix (its rows hold at most this many neighbors, and an
            overflow grows it through kUPS's capacity fix).
        fallback: kUPS cell list for bipartite and ``queried_keys`` calls.
        cutoff: Static search cutoff, at least every per-system cutoff.
        max_total_cells: Static cell capacity of nvalchemi's grid (all systems).
    """

    cutoffs: Table[SystemId, Array]
    avg_edges: Capacity[int]
    fallback: CellListNeighborList
    cutoff: float = field(static=True)
    max_total_cells: int = field(static=True)

    @classmethod
    def new[S](cls, state: S, lens: Lens[S, IsUniversalNeighborlistParams],
               cutoffs: Table[SystemId, Array], *, cutoff: float,
               max_total_cells: int) -> NvalchemiNeighborList:
        params = lens.get(state)
        return cls(
            cutoffs,
            LensCapacity(params.avg_edges, lens.focus(lambda x: x.avg_edges)),
            CellListNeighborList.new(state, lens, cutoffs),
            cutoff,
            max_total_cells,
        )

    @classmethod
    def from_state(cls, state: IsNeighborListState[IsUniversalNeighborlistParams],
                   cutoffs: Table[SystemId, Array], *, cutoff: float,
                   max_total_cells: int) -> NvalchemiNeighborList:
        return cls.new(state, lens(lambda s: s.neighborlist_params), cutoffs,
                       cutoff=cutoff, max_total_cells=max_total_cells)

    def selector(self, query_size: int,
                 systems: Table[SystemId, NeighborListSystems]) -> NvalchemiSelector:
        return NvalchemiSelector(self.avg_edges, self.fallback.selector(query_size, systems),
                                 self.cutoff, self.max_total_cells)

    @overload
    def __call__(self, keys: Table[ParticleId, NeighborListPoints],
                 systems: Table[SystemId, NeighborListSystems], *,
                 queries: Table[ParticleId, NeighborListPoints],
                 ) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]: ...
    @overload
    def __call__(self, keys: Table[ParticleId, NeighborListPoints],
                 systems: Table[SystemId, NeighborListSystems], *,
                 queried_keys: Index[ParticleId] | None = None,
                 ) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]: ...
    @jit
    def __call__(self, keys: Table[ParticleId, NeighborListPoints],
                 systems: Table[SystemId, NeighborListSystems], *,
                 queries: Table[ParticleId, NeighborListPoints] | None = None,
                 queried_keys: Index[ParticleId] | None = None,
                 ) -> WithPatch[Edges[Literal[2]], IdPatch[Any]]:
        edges = build_radius_graph(self, keys, systems, queries=queries,
                                   queried_keys=queried_keys)
        return WithPatch(edges, IdPatch[Any]())


@functools.cache
def _lj_kernel(dtype: Any) -> Any:
    """nvalchemi's batched LJ energy/forces/virial kernel as a JAX function (through
    whichever ``warp.jax_kernel`` ``use_backend`` left), its accumulated outputs passed
    in zeroed and updated in place."""
    import numpy as np
    import warp
    from nvalchemiops.interactions import lj

    t = warp.float64 if np.dtype(dtype) == np.float64 else warp.float32
    return warp.jax_kernel(lj._batch_lj_energy_forces_virial_matrix_kernel_overload[t],
                           num_outputs=3,
                           in_out_argnames=["atomic_energies", "atomic_forces", "virial"])


@dataclass
class NvalchemiLennardJones:
    """Single-species, plainly truncated Lennard-Jones (kUPS's ``lennard_jones_energy``
    for one label) evaluated by nvalchemi: its cell list, then one fused kernel for
    energies, forces and virials. A kUPS ``Potential`` with ``PositionsAndCell``
    gradients, so ``make_md_propagator`` takes it in place of kUPS's graph LJ.

    Gradients come from the kernel outputs, not autodiff: ``∂E/∂x = -F``, and the cell
    gradient with atoms pinned is ``∂E/∂h = h⁻ᵀ (P - W)`` per system, with ``h``'s rows the
    lattice vectors, ``P = Σᵢ xᵢ ⊗ Fᵢ`` and ``W`` nvalchemi's full-list virial
    ``Σ_edges ½ rᵢⱼ ⊗ Fᵢⱼ`` (``rᵢⱼ = xᵢ - xⱼ - Sᵢⱼ h``): ``∂E/∂h = Σ_edges ½ Sᵢⱼ ⊗ Fᵢⱼ``, and
    ``W = P - hᵀ ∂E/∂h`` since each pair's two edges carry opposite forces.
    """

    epsilon: float = field(static=True)
    sigma: float = field(static=True)
    cutoff: float = field(static=True)
    max_total_cells: int = field(static=True)

    def __call__(self, state: Any, patch: Any = None, *,
                 include_compensate: bool = False) -> Any:
        from kups.core.patch import IdPatch, WithPatch

        particles, systems = state.particles, state.systems
        positions = particles.data.positions
        system = jnp.minimum(particles.data.system.indices_in(systems.keys),
                             systems.size - 1)
        width = LensCapacity(state.neighborlist_params.avg_edges,
                             lens(lambda s: s.neighborlist_params.avg_edges))
        nbr, count, shift = _neighbor_matrix(positions, system, systems.data.cell,
                                             self.cutoff, width, self.max_total_cells)
        energies, forces, d_cell = _lj(systems.data.cell, positions, positions, system,
                                       nbr, shift, count, self.epsilon, self.sigma,
                                       self.cutoff)
        out = _potential_out(particles, systems, energies, forces, d_cell)
        return WithPatch(_compensated(out) if include_compensate else out, IdPatch())


def _lj(cell: Cell[Any], current: Array, kernel_positions: Array, system: Array,
        nbr: Array, shift: Array, count: Array, epsilon: float, sigma: float,
        cutoff: float) -> tuple[Array, Array, Any]:
    """Per-system energies, forces and the cell gradient (a ``Cell``) from nvalchemi's
    LJ kernel run on ``kernel_positions`` with the neighbor matrix ``nbr`` (rows padded
    with ``len(nbr)``) and its image shifts. ``kernel_positions`` may differ from the
    particles' ``current`` positions by whole cell vectors (unwrapped since a skin
    list's build): taking ``P`` over ``current`` makes the cell gradient the one at
    ``current``, as the wrap terms ``Σ wᵢ ⊗ Fᵢ`` cancel."""
    import jax

    n, n_sys = current.shape[0], cell.vectors.shape[0]
    dtype = current.dtype
    scalar = lambda v: jnp.full((1,), v, dtype)  # noqa: E731
    energies, forces, virial = _lj_kernel(dtype)(
        kernel_positions, cell.vectors, nbr, shift, count, system.astype(jnp.int32),
        scalar(epsilon), scalar(sigma), scalar(cutoff), scalar(0.0),
        False, n, jnp.zeros(n, dtype), jnp.zeros((n, 3), dtype),
        jnp.zeros((n_sys, 9), dtype))
    total = jax.ops.segment_sum(energies, system, n_sys)
    pinned = jax.ops.segment_sum(current[:, :, None] * forces[:, None, :], system, n_sys)
    inv_t = jnp.swapaxes(cell.inverse_vectors, -1, -2)
    d_vectors = inv_t @ (pinned - virial.reshape(n_sys, 3, 3))
    _, pull = jax.vjp(lambda c: c.vectors, cell)
    (d_cell,) = pull(d_vectors)
    return total, forces, d_cell


def _potential_out(particles: Any, systems: Any, energies: Array, forces: Array,
                   d_cell: Any) -> Any:
    from kups.core.potential import EMPTY, PotentialOut
    from kups.potential.common.geometry import PositionsAndCell

    return PotentialOut(systems.set_data(energies),
                        PositionsAndCell(particles.set_data(-forces),
                                         systems.set_data(d_cell)),
                        EMPTY)


def _compensated(out: Any) -> Any:
    from kups.core.utils.kahan import KahanSummand

    return KahanSummand.init(out)


@dataclass
class NvalchemiRowsLennardJones:
    """nvalchemi's fused LJ kernel on kUPS's skin list: the state's skin list (rebuilt
    by kUPS when its skin is used up) supplies the pairs as neighbor rows, padded to a
    neighbor matrix, and the kernel applies the cutoff. Same physics, gradients and
    restrictions as ``NvalchemiLennardJones``; it isolates the pair kernel from the
    neighbor search.
    """

    epsilon: float = field(static=True)
    sigma: float = field(static=True)
    cutoff: float = field(static=True)

    def __call__(self, state: Any, patch: Any = None, *,
                 include_compensate: bool = False) -> Any:
        from kups.core.neighborlist.verlet import read_skin_list, wraps_since_build
        from kups.core.patch import WithPatch

        particles, systems = state.particles, state.systems
        n = particles.size
        listed = read_skin_list(state, lens(lambda s: s.skin_list), particles, systems)
        rows = listed.data.rows
        system = particles.data.system
        ids = jnp.minimum(system.indices_in(systems.keys), systems.size - 1)
        frames = systems.map_data(lambda s: s.cell.frame.materialize())[system]
        positions = particles.data.positions
        wraps = wraps_since_build(listed.data.reference, frames.to_fractional(positions),
                                  system, systems)
        unwrapped = positions + frames.to_real(wraps)
        # The rows (in the skin list's spatial order), padded to a neighbor matrix.
        width = LensCapacity(state.neighborlist_params.avg_edges,
                             lens(lambda s: s.neighborlist_params.avg_edges))
        count = (rows.offsets[1:] - rows.offsets[:-1]).astype(jnp.int32)
        width.generate_assertion(count.max())
        valid = jnp.arange(width.size)[None, :] < count[:, None]
        slot = jnp.where(valid, rows.offsets[:-1, None] + jnp.arange(width.size), 0)
        nbr = jnp.where(valid, rows.neighbors[slot], n).astype(jnp.int32)
        shift = jnp.where(valid[..., None], rows.shifts[slot], 0).astype(jnp.int32)
        order = rows.order
        energies, forces, d_cell = _lj(systems.data.cell, positions[order],
                                       unwrapped[order], ids[order], nbr, shift, count,
                                       self.epsilon, self.sigma, self.cutoff)
        out = _potential_out(particles, systems, energies, forces[rows.rank], d_cell)
        return WithPatch(_compensated(out) if include_compensate else out, listed.patch)
