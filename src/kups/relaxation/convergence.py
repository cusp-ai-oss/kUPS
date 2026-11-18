# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system convergence criteria for gradient-based relaxation.

ASE's ``fmax`` criterion per system: a system converges when its largest
DOF-gradient norm is below the tolerance. The helpers take ``PositionsAndCell``
gradients and a ``PositionsAndCellIndex`` that assigns each particle row and
cell to its system, so each system in a batch is judged as if it ran alone.
Cell gradients are forces too: cell vectors are lengths, and a log-deformation
with a length ``cell_factor`` is one. ASE compares ``virial / n_atoms`` in eV
with forces in eV/Å, which is ``cell_factor = n_atoms × 1 Å``.

ASE's cell filters append the ``3x3`` cell DOFs to the forces as three extra
atoms, one per row, and compare each row's length with ``fmax``. kUPS deforms
the cell from the right (``vectors = base @ D``) where ASE deforms from the left
(``cell = orig_cell @ F.T``), so a kUPS ``3x3`` cell-gradient leaf is ASE's
transpose: ASE's rows are its columns, and those column lengths are what count.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from kups.core import units
from kups.core.assertion import runtime_assert
from kups.core.cell import Cell
from kups.core.data.table import Table
from kups.core.geometry import (
    PositionsAndCell,
    PositionsAndCellIndex,
)
from kups.core.typing import SystemId


def _cell_leaf_norms(leaf: units.Force) -> units.Force:
    """ASE-style magnitudes of one cell-gradient leaf.

    A ``(..., 3, 3)`` leaf (the matrix-log deformation the relaxation builders
    use) yields the length of each column, which is ASE's row. Packed
    parameterisations such as ``tril`` or ``lengths`` have no ASE row layout and
    count per entry.
    """
    if leaf.shape[-2:] == (3, 3):
        return units.norm(leaf, axis=-2)
    return units.absolute(leaf)


def _cell_row_max(cell: Cell[Any]) -> units.Force:
    """Largest ASE-style cell-gradient row of each cell, across all its leaves."""
    norms = map(_cell_leaf_norms, units.tree_leaves(cell))
    rows = [units.max(n, axis=tuple(range(1, n.ndim))) for n in norms]
    return units.max(units.stack(rows, axis=-1), axis=-1)


def _inf_if_nonfinite(x: units.Force) -> units.Force:
    """``x`` with NaN and ``±inf`` raised to ``inf``; a segment max may drop NaNs."""
    return units.apply(lambda m: jnp.where(jnp.isfinite(m), m, jnp.inf), x)


def fmax_per_system(
    gradients: PositionsAndCell[units.Force],
    index_prefix: PositionsAndCellIndex,
    *,
    include_cell: bool = True,
) -> Table[SystemId, units.Force]:
    """Largest DOF-gradient norm of each system (ASE ``fmax``).

    Non-finite gradients count as ``inf``. Rows with an out-of-bounds system
    index (padding) are ignored.

    Args:
        gradients: DOF gradients ``∂E/∂u`` as the relaxation filter reports them.
        index_prefix: ``PositionsAndCellIndex(particles.data.system, systems.index)``.
        include_cell: Whether the cell DOF gradients take part in the maximum.

    Returns:
        Maximum per system, shape ``(n_systems,)``. A system without particles
        is ``-inf`` when cell gradients are excluded.
    """
    positions = units.norm(gradients.positions.data, axis=-1)
    result = index_prefix.positions.max_over(_inf_if_nonfinite(positions))
    if include_cell:
        cell = _cell_row_max(gradients.cell.data)
        cell_max = index_prefix.cell.max_over(_inf_if_nonfinite(cell))
        result = result.map_data(lambda p: units.maximum(p, cell_max.data))
    return result


def converged_per_system(
    gradients: PositionsAndCell[units.Force],
    index_prefix: PositionsAndCellIndex,
    tolerance: units.Force,
    *,
    include_cell: bool = True,
) -> Table[SystemId, Array]:
    """Flag systems whose :func:`fmax_per_system` is below ``tolerance``.

    A non-finite gradient that takes part in the maximum fails a runtime
    assertion. ``as_result_function`` and ``with_runtime_assertions`` collect
    it; a plain call, jitted or not, only reports the system as not converged.
    """
    maximum = fmax_per_system(gradients, index_prefix, include_cell=include_cell)
    runtime_assert(
        ~jnp.any(jnp.isposinf(units.magnitudes(maximum.data))),
        "Non-finite relaxation gradients.",
    )
    return maximum.map_data(lambda m: m < tolerance)
