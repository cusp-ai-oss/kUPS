# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system convergence criteria for gradient-based relaxation.

ASE's ``fmax`` criterion per system: a system converges when its largest
DOF-gradient norm is below the tolerance. The helpers take ``PositionsAndCell``
gradients and a ``PositionsAndCellIndex`` that assigns each particle row and
cell to its system, so each system in a batch is judged as if it ran alone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.data.table import Table
from kups.core.typing import SystemId
from kups.core.utils.segmented_tree import tree_segment_max
from kups.potential.common.geometry import PositionsAndCell, PositionsAndCellIndex


def _gradient_norm_per_row(gradients: PositionsAndCell) -> PositionsAndCell:
    """Per-row gradient magnitudes: ``|∂E/∂r_i|`` per particle, ``|∂E/∂h|`` per cell entry."""
    return PositionsAndCell(
        gradients.positions.map_data(lambda g: jnp.linalg.norm(g, axis=-1)),
        gradients.cell.map_data(lambda c: jax.tree.map(jnp.abs, c)),
    )


def fmax_per_system(
    gradients: PositionsAndCell,
    index_prefix: PositionsAndCellIndex,
    *,
    include_cell: bool = True,
) -> Table[SystemId, Array]:
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
    # Segment max may drop NaNs, so map non-finite norms to inf first.
    norms = jax.tree.map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.inf),
        _gradient_norm_per_row(gradients),
    )
    result = index_prefix.positions.max_over(norms.positions.data)
    if include_cell:
        cell_max = tree_segment_max(norms.cell, index_prefix.cell)
        result = result.map_data(
            lambda positions: jnp.maximum(positions, cell_max.data)
        )
    return result


def converged_per_system(
    gradients: PositionsAndCell,
    index_prefix: PositionsAndCellIndex,
    tolerance: float,
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
        ~jnp.any(jnp.isposinf(maximum.data)), "Non-finite relaxation gradients."
    )
    return maximum.map_data(lambda m: m < tolerance)
