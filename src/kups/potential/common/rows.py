# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pair energies over a skin list's neighbor rows.

[`RowPairEnergy`][kups.potential.common.rows.RowPairEnergy] sums a pair function over
a skin list's [`NeighborRows`][kups.core.neighborlist.rows.NeighborRows]. Rows are
symmetric, so a particle's gradient and its share of the cell gradient are sums over
its own row: one pass computes them with the energy.
"""

from __future__ import annotations

import functools
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import View
from kups.core.neighborlist.rows import NeighborRows
from kups.core.neighborlist.verlet import SkinList, wraps_since_build
from kups.core.patch import IdPatch, WithPatch
from kups.core.potential import Energy
from kups.core.typing import HasCell, HasPositionsAndSystemIndex, SystemId
from kups.core.utils.jax import dataclass, field, jit, non_differentiable
from kups.core.utils.segment import segment_sum
from kups.potential.common.graph import PointCloud


class RowPairFunction(Protocol):
    """Pair energy at squared distances ``r2`` from per-pair coefficients.

    Elementwise ``jax.numpy`` only, with coefficients from species-pair tables
    instead of a ``PairKernel``'s parameters, so that a kernel can evaluate it per
    row. Its derivative in ``r2`` comes from ``jax.jvp``.
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
    """

    function: RowPairFunction = field(static=True)
    tables: View[Params, tuple[Array, ...]] = field(static=True)
    species: RowSpecies[Params, Part] = field(static=True)
    cutoffs: View[Params, Table[SystemId, Array]] = field(static=True)

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
        tables = self.tables(params)
        runtime_assert(
            jnp.all(jnp.asarray([jnp.all(t == t.T) for t in tables])),
            "Row pair tables must be symmetric.",
        )
        tables = non_differentiable(tables)
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
        energies = _energies(
            self.function,
            positions + frames.to_real(wraps),
            systems.data.cell.vectors,
            tables,
            self.species(params, particles.data),
            ids,
            cutoffs,
            skin_list.rows,
        )
        return WithPatch(systems.set_data(energies), IdPatch[Any]())


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _energies(
    function: RowPairFunction,
    positions: Array,
    vectors: Array,
    tables: tuple[Array, ...],
    species: Array,
    system: Array,
    cutoffs: Array,
    rows: NeighborRows,
) -> Array:
    terms = _system_terms(
        function, positions, vectors, tables, species, system, cutoffs, rows
    )
    return terms[0]


@_energies.defjvp
def _energies_jvp(
    function: RowPairFunction,
    primals: tuple[Array, Array, tuple[Array, ...], Array, Array, Array, NeighborRows],
    tangents: tuple[Array, Array, tuple[Array, ...], Array, Array, Array, NeighborRows],
) -> tuple[Array, Array]:
    # tables, species, system, cutoffs and rows are constants
    system = primals[4]
    energies, gradients, cell_gradients = _system_terms(function, *primals)
    tangent = segment_sum((gradients * tangents[0]).sum(-1), system, len(energies))
    return energies, tangent + (cell_gradients * tangents[1]).sum((-2, -1))


def _system_terms(
    function: RowPairFunction,
    positions: Array,
    vectors: Array,
    tables: tuple[Array, ...],
    species: Array,
    system: Array,
    cutoffs: Array,
    rows: NeighborRows,
) -> tuple[Array, Array, Array]:
    """Energy and cell gradient of each system, gradient of each particle."""
    energy, gradient, cell = _row_terms(
        function, positions, vectors, tables, species, system, cutoffs, rows
    )
    row_system = system[rows.order]
    return (
        segment_sum(energy, row_system, len(vectors)),
        gradient[rows.rank],
        segment_sum(cell, row_system, len(vectors)).reshape(-1, 3, 3),
    )


def _row_terms(
    function: RowPairFunction,
    positions: Array,
    vectors: Array,
    tables: tuple[Array, ...],
    species: Array,
    system: Array,
    cutoffs: Array,
    rows: NeighborRows,
) -> tuple[Array, Array, Array]:
    """Per row: half its pair energies, their gradient in the row's particle and in
    its system's cell vectors, flattened to 9."""
    n = len(rows.order)
    # padding slots name row n, out of bounds: the fills give them a zero cell and
    # cutoff, so none is inside, and the per-row sums drop them instead of adding
    # them all into one row
    row = rows.centers
    slot_system = system[rows.order].at[row].get(mode="fill", fill_value=len(vectors))
    n_ = rows.shifts.astype(positions.dtype)
    y = positions[rows.order]
    h = vectors.at[slot_system].get(mode="fill", fill_value=0)
    # r = y_j - y_i + n h, with explicit products: a batched matmul would not fuse
    r = y.at[rows.neighbors].get(mode="fill", fill_value=0)
    r -= y.at[row].get(mode="fill", fill_value=0)
    r += n_[:, :1] * h[:, 0] + n_[:, 1:2] * h[:, 1] + n_[:, 2:] * h[:, 2]
    r2 = (r**2).sum(-1)
    rc2 = (cutoffs**2).at[slot_system].get(mode="fill", fill_value=0)
    inside = r2 < rc2
    r = jnp.where(inside[:, None], r, 0.0)
    row_species = species[rows.order]
    left = row_species.at[row].get(mode="clip")
    right = row_species.at[rows.neighbors].get(mode="clip")
    coefficients = tuple(table[left, right] for table in tables)
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
