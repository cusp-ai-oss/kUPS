# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pair energies over a skin list's neighbor rows.

[`RowPairEnergy`][kups.potential.common.rows.RowPairEnergy] sums a pair function over
a skin list's [`NeighborRows`][kups.core.neighborlist.rows.NeighborRows]. Rows are
symmetric, so a particle's gradient and its share of the cell gradient are sums over
its own row: one pass computes them with the energy. In float32 on CUDA GPUs of
compute capability 8.0+, a Pallas kernel runs one program per row.
"""

from __future__ import annotations

import functools
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from kups.core.assertion import runtime_assert
from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import View, lens
from kups.core.neighborlist.rows import NeighborRows
from kups.core.neighborlist.skin import SkinList, wraps_since_build
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import Energy
from kups.core.typing import (
    DTypeLike,
    HasCell,
    HasPositionsAndSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import (
    dataclass,
    field,
    jit,
    non_differentiable,
    triton_available,
)
from kups.core.utils.segment import segment_sum
from kups.potential.common.graph import POINTCLOUD_GEOMETRY, PointCloud

# One warp walks a row 32 slots per step: 32-64 equal on an L4, rows hold ~90 slots
# at 9.5 Å in argon.
BLOCK, WARPS = 32, 1
# Positions and cells split into multiples of 1/8 Å, exact in float32 below 2**21 Å,
# and remainders, so pair vectors keep their dtype's precision at any coordinate.
COARSE = 8.0


class RowPairFunction(Protocol):
    """Pair energy at squared distances ``r2`` from per-pair coefficients.

    Elementwise ``jax.numpy`` only: it also runs inside a Pallas kernel, which reads
    coefficients from species-pair tables instead of calling a ``PairKernel``. Its
    derivative in ``r2`` comes from ``jax.jvp``.
    """

    def __call__(self, r2: Array, coefficients: tuple[Array, ...], /) -> Array: ...


class RowSpecies[Params, Part](Protocol):
    """Row of the coefficient tables for each particle, ``(n,)``."""

    def __call__(self, parameters: Params, particles: Part, /) -> Array: ...


@dataclass
class RowPotentialInput[
    Params,
    Part: HasPositionsAndSystemIndex,
    Sys: HasCell[AnyPeriodicity],
]:
    """Input of a potential over a skin list's neighbor rows.

    Attributes:
        parameters: Potential parameters.
        cloud: Particles and systems; positions and cells get gradients.
        skin_list: Skin list whose rows hold the pairs.
    """

    parameters: Params
    cloud: PointCloud[Part, Sys]
    skin_list: SkinList


ROW_GEOMETRY = lens(lambda inp: inp.cloud, cls=RowPotentialInput).nest(
    POINTCLOUD_GEOMETRY
)
"""Geometry adapter of row-potential inputs."""


@dataclass
class RowInputConstructor[
    State,
    Params,
    Part: HasPositionsAndSystemIndex,
    Sys: HasCell[AnyPeriodicity],
]:
    """Whole-state inputs of a row potential; proposals are not supported.

    Attributes:
        particles: View of the particles.
        systems: View of the systems.
        skin_list: View of the skin list.
        parameter_view: View of the parameters.
    """

    particles: View[State, Table[ParticleId, Part]] = field(static=True)
    systems: View[State, Table[SystemId, Sys]] = field(static=True)
    skin_list: View[State, SkinList] = field(static=True)
    parameter_view: View[State, Params] = field(static=True)

    def __call__(
        self, state: State, patch: Patch[State] | None, old_input: bool = False
    ) -> RowPotentialInput[Params, Part, Sys]:
        if patch is not None:
            raise ValueError(
                "Row potentials evaluate whole states; use a graph potential for "
                "proposals."
            )
        cloud = PointCloud(self.particles(state), self.systems(state))
        return RowPotentialInput(
            self.parameter_view(state), cloud, self.skin_list(state)
        )


@dataclass
class RowPairEnergy[Params, Part: HasPositionsAndSystemIndex]:
    """Per-system energies of a pair function over a skin list's rows.

    Each pair within its system's cutoff adds ``u(r²)`` once. Particles are moved
    back by the cell vectors they were wrapped by since the build, so the rows'
    image shifts hold; autodiff carries this into the cell gradient. Gradients in
    positions and cell vectors come from the same pass as the energy; the tables
    are not differentiable.

    Attributes:
        function: Pair energy in ``r²``.
        tables: Species-pair coefficient tables, each symmetric.
        species: Table row of each particle.
        cutoffs: Cutoffs (Å) per system, or one for all; at most the skin list's.
        dtype: Precision of the pair terms. ``float32`` is much faster on GPUs with
            slow double precision and runs a Pallas kernel on CUDA GPUs of compute
            capability 8.0+; pairs within about 2e-7 of the cutoff (relative) may
            then fall on either side. Each row is summed in it; system totals use
            the wider of it and the inputs' precision, and energies come out in
            that of the positions and cells.
    """

    function: RowPairFunction = field(static=True)
    tables: View[Params, tuple[Array, ...]] = field(static=True)
    species: RowSpecies[Params, Part] = field(static=True)
    cutoffs: View[Params, Table[SystemId, Array]] = field(static=True)
    dtype: DTypeLike = field(static=True, default=jnp.float64, kw_only=True)

    @jit
    def __call__(
        self, inp: RowPotentialInput[Params, Part, HasCell[AnyPeriodicity]]
    ) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
        params, skin_list = inp.parameters, inp.skin_list
        particles, systems = inp.cloud.particles, inp.cloud.systems
        cutoffs = Table.broadcast_to(self.cutoffs(params), systems).data
        listed = Table.broadcast_to(skin_list.cutoffs, systems).data
        runtime_assert(
            jnp.all(cutoffs <= listed),
            "Cutoffs exceed the skin list's; create it at the largest cutoff.",
        )
        tables = non_differentiable(self.tables(params))
        runtime_assert(
            jnp.all(jnp.asarray([jnp.all(t == t.T) for t in tables])),
            "Row pair tables must be symmetric.",
        )
        system = particles.data.system
        ids = system.indices_in(systems.keys)
        frames = systems.map_data(lambda s: s.cell.frame.materialize())[system]
        positions = particles.data.positions
        fractional = frames.to_fractional(jax.lax.stop_gradient(positions))
        wraps = wraps_since_build(skin_list.reference, fractional, system, systems)
        # inactive particles have no rows; zero wraps keep NaN out of the cell gradient
        wraps = jax.lax.stop_gradient(
            jnp.where((ids < systems.size)[:, None], wraps, 0)
        )
        vectors = systems.data.cell.vectors
        energies = _energies(
            self.function,
            jnp.dtype(self.dtype),
            positions + frames.to_real(wraps),
            vectors,
            tables,
            self.species(params, particles.data),
            ids,
            cutoffs,
            skin_list.rows,
        )
        dtype = jnp.result_type(positions, vectors)
        return WithPatch(systems.set_data(energies.astype(dtype)), IdPatch[Any]())


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _energies(
    function: RowPairFunction,
    dtype: jnp.dtype,
    positions: Array,
    vectors: Array,
    tables: tuple[Array, ...],
    species: Array,
    system: Array,
    cutoffs: Array,
    rows: NeighborRows,
) -> Array:
    terms = _system_terms(
        function, dtype, positions, vectors, tables, species, system, cutoffs, rows
    )
    return terms[0]


@_energies.defjvp
def _energies_jvp(
    function: RowPairFunction,
    dtype: jnp.dtype,
    primals: tuple[Array, Array, tuple[Array, ...], Array, Array, Array, NeighborRows],
    tangents: tuple[Array, Array, tuple[Array, ...], Array, Array, Array, NeighborRows],
) -> tuple[Array, Array]:
    # tables, species, system, cutoffs and rows are constants
    system = primals[4]
    energies, gradients, cell_gradients = _system_terms(function, dtype, *primals)
    tangent = segment_sum((gradients * tangents[0]).sum(-1), system, len(energies))
    return energies, tangent + (cell_gradients * tangents[1]).sum((-2, -1))


def _system_terms(
    function: RowPairFunction,
    dtype: jnp.dtype,
    positions: Array,
    vectors: Array,
    tables: tuple[Array, ...],
    species: Array,
    system: Array,
    cutoffs: Array,
    rows: NeighborRows,
) -> tuple[Array, Array, Array]:
    """Energy and cell gradient of each system, gradient of each particle."""
    args = (function, positions, vectors, tables, species, system, cutoffs, rows)
    if dtype == jnp.float32 and _use_kernel() and len(rows.order):
        terms = _rows_pallas(*args)
    else:
        terms = _rows_xla(*args, dtype)
    # system totals in the wider dtype
    sum_dtype = jnp.result_type(dtype, positions, vectors, *tables)
    energy, gradient, cell = (term.astype(sum_dtype) for term in terms)
    row_system = system[rows.order]
    return (
        segment_sum(energy, row_system, len(vectors)),
        gradient[rows.rank],
        segment_sum(cell, row_system, len(vectors)).reshape(-1, 3, 3),
    )


def _use_kernel() -> bool:
    """Whether float32 pair terms run in the Pallas kernel (interpreted off Triton)."""
    return triton_available()


def _rows_xla(
    function: RowPairFunction,
    positions: Array,
    vectors: Array,
    tables: tuple[Array, ...],
    species: Array,
    system: Array,
    cutoffs: Array,
    rows: NeighborRows,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """Per row: half its pair energies, their gradient in the row's particle and in
    its system's cell vectors, flattened to 9."""
    n = len(rows.order)
    # padding slots name row n, out of bounds: the fills give them a zero cell and
    # cutoff, so none is inside, and the per-row sums drop them instead of adding
    # them all into one row
    row = rows.centers
    slot_system = system[rows.order].at[row].get(mode="fill", fill_value=len(vectors))
    n_ = rows.shifts.astype(dtype)

    def part(y: Array, h: Array) -> Array:
        """``y_j - y_i + n h`` of one part of the split."""
        h = h.at[slot_system].get(mode="fill", fill_value=0)
        r = y.at[rows.neighbors].get(mode="fill", fill_value=0)
        r -= y.at[row].get(mode="fill", fill_value=0)
        # explicit products: a batched matmul would not fuse
        return r + n_[:, :1] * h[:, 0] + n_[:, 1:2] * h[:, 1] + n_[:, 2:] * h[:, 2]

    coarse, fine = _split(positions[rows.order], dtype)
    vectors_coarse, vectors_fine = _split(vectors, dtype)
    r = part(coarse, vectors_coarse) + part(fine, vectors_fine)
    r2 = (r**2).sum(-1)
    rc2 = (cutoffs**2).astype(dtype).at[slot_system].get(mode="fill", fill_value=0)
    inside = r2 < rc2
    r = jnp.where(inside[:, None], r, 0.0)
    row_species = species[rows.order]
    left = row_species.at[row].get(mode="clip")
    right = row_species.at[rows.neighbors].get(mode="clip")
    coefficients = tuple(table.astype(dtype)[left, right] for table in tables)
    u, du = jax.jvp(
        lambda t: function(t, coefficients),
        (jnp.where(inside, r2, 1.0),),
        (jnp.ones_like(r2),),
    )
    u, du = jnp.where(inside, u, 0.0), jnp.where(inside, du, 0.0)
    # flat: XLA can lay (n, 3, 3) sums out with n minor, and the scatter then
    # writes each slot's nine terms to nine places far apart
    cell = (du[:, None, None] * n_[:, :, None] * r[:, None, :]).reshape(-1, 9)
    return (
        segment_sum(u / 2, row, n),
        segment_sum(-2 * du[:, None] * r, row, n),
        segment_sum(cell, row, n),
    )


def _rows_pallas(
    function: RowPairFunction,
    positions: Array,
    vectors: Array,
    tables: tuple[Array, ...],
    species: Array,
    system: Array,
    cutoffs: Array,
    rows: NeighborRows,
) -> tuple[Array, Array, Array]:
    """``_rows_xla`` in float32 as a Pallas kernel, one program per row."""
    n, n_sys = len(rows.order), len(vectors)
    n_species = max((table.shape[0] for table in tables), default=1)
    coarse, fine = _split(positions[rows.order], jnp.float32)
    cells = jnp.concatenate(_split(vectors.reshape(n_sys, 9), jnp.float32), axis=1)
    kernel = functools.partial(
        _row_kernel, function=function, n_tables=len(tables), n_species=n_species
    )
    out = jax.ShapeDtypeStruct((n,), jnp.float32)
    energy, *terms = pl.pallas_call(
        kernel,
        out_shape=(out,) * (1 + 3 + 9),  # energy, gradient, cell gradient
        grid=(n,),
        compiler_params=plgpu.CompilerParams(num_warps=WARPS, num_stages=1),
        interpret=not triton_available(),
    )(
        *coarse.T,
        *fine.T,
        jnp.clip(system[rows.order], 0, n_sys - 1).astype(jnp.int32),
        jnp.clip(species[rows.order], 0, n_species - 1).astype(jnp.int32),
        rows.offsets,
        rows.neighbors,
        rows.shifts.reshape(-1),
        cells,
        (cutoffs**2).astype(jnp.float32),
        *(table.astype(jnp.float32).ravel() for table in tables),
    )
    return (
        energy,
        jnp.stack(terms[:3], -1),
        jnp.stack(terms[3:], -1),
    )


def _split(x: Array, dtype: DTypeLike) -> tuple[Array, Array]:
    """``x`` as a multiple of ``1 / COARSE``, exact in ``dtype``, and a remainder."""
    coarse = jnp.round(x * COARSE) / COARSE
    return coarse.astype(dtype), (x - coarse).astype(dtype)


def _row_kernel(
    cx_ref: jax.Ref, cy_ref: jax.Ref, cz_ref: jax.Ref, fx_ref: jax.Ref, fy_ref: jax.Ref, fz_ref: jax.Ref,
    system_ref: jax.Ref, species_ref: jax.Ref, offsets_ref: jax.Ref, neighbors_ref: jax.Ref,
    shifts_ref: jax.Ref, cell_ref: jax.Ref, rc2_ref: jax.Ref,
    *refs: jax.Ref,
    function: RowPairFunction, n_tables: int, n_species: int,
) -> None:  # fmt: skip
    """Energy, gradient and cell-gradient terms of row ``pl.program_id(0)``.

    Per slot, ``c = (yc_j - yc_i) + n h_c`` is exact and ``f = (yf_j - yf_i) + n h_f``
    rounds; ``r = c + f``. Masked lanes take a safe ``r²`` and contribute zero.
    """
    table_refs, out_refs = refs[:n_tables], refs[n_tables:]
    coarse, fine = (cx_ref, cy_ref, cz_ref), (fx_ref, fy_ref, fz_ref)
    i = pl.program_id(0)
    s = system_ref[i]
    hc, hf = [cell_ref[s, k] for k in range(9)], [cell_ref[s, 9 + k] for k in range(9)]
    yc, yf = [ref[i] for ref in coarse], [ref[i] for ref in fine]
    pair_row = species_ref[i] * n_species
    start, end = offsets_ref[i], offsets_ref[i + 1]

    def chunk_body(chunk: Array, acc: tuple[Array, ...]) -> tuple[Array, ...]:
        slot = start + chunk * BLOCK + jnp.arange(BLOCK, dtype=jnp.int32)
        valid = slot < end
        j = plgpu.load(neighbors_ref.at[slot], mask=valid, other=0)
        n = [
            plgpu.load(shifts_ref.at[3 * slot + a], mask=valid, other=0).astype(
                jnp.float32
            )
            for a in range(3)
        ]

        def component(b: int) -> Array:
            c = plgpu.load(coarse[b].at[j], mask=valid, other=0.0) - yc[b]
            f = plgpu.load(fine[b].at[j], mask=valid, other=0.0) - yf[b]
            c += n[0] * hc[b] + n[1] * hc[3 + b] + n[2] * hc[6 + b]
            f += n[0] * hf[b] + n[1] * hf[3 + b] + n[2] * hf[6 + b]
            return c + f

        r = [component(b) for b in range(3)]
        r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
        inside = valid & (r2 < rc2_ref[s])
        pair = pair_row + plgpu.load(species_ref.at[j], mask=valid, other=0)
        coefficients = tuple(
            plgpu.load(ref.at[pair], mask=valid, other=0.0) for ref in table_refs
        )
        u, du = jax.jvp(
            lambda t: function(t, coefficients),
            (jnp.where(inside, r2, 1.0),),
            (jnp.ones_like(r2),),
        )
        u, du = jnp.where(inside, u, 0.0), jnp.where(inside, du, 0.0)
        terms = (
            u / 2,
            *(-2 * du * r[b] for b in range(3)),
            *(du * n[a] * r[b] for a in range(3) for b in range(3)),
        )
        return tuple(total + jnp.sum(term) for total, term in zip(acc, terms))

    n_chunks = (end - start + BLOCK - 1) // BLOCK
    zero = jnp.zeros((), jnp.float32)
    totals = jax.lax.fori_loop(0, n_chunks, chunk_body, (zero,) * len(out_refs))
    for ref, total in zip(out_refs, totals):
        ref[i] = total
