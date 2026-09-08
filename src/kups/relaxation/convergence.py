# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system convergence criteria for gradient-based relaxation.

ASE's ``fmax`` criterion generalised to a batch: every system converges on its
own, judged by the largest norm of any of its DOF gradients. The helpers take
the optimizer's ``PositionsAndCell`` gradient payload together with the
matching ``index_prefix`` (per-particle ``system`` index and systems index)
that the transforms in :mod:`kups.relaxation.transforms` use, so a batched run
judges each system exactly as a single-system run would.
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


def dof_norm_per_row(gradients: PositionsAndCell) -> PositionsAndCell:
    """Per-row gradient magnitudes: ``|∂E/∂r_i|`` per particle, ``|∂E/∂h|`` per cell entry."""
    return PositionsAndCell(
        gradients.positions.map_data(lambda g: jnp.linalg.norm(g, axis=-1)),
        gradients.cell.map_data(lambda c: jax.tree.map(jnp.abs, c)),
    )


def max_dof_per_system(
    gradients: PositionsAndCell,
    index_prefix: PositionsAndCellIndex,
    *,
    include_cell: bool = True,
) -> Table[SystemId, Array]:
    """Largest DOF-gradient norm of each system (ASE ``fmax``).

    Args:
        gradients: DOF gradients ``∂E/∂u`` as the relaxation filter reports them.
        index_prefix: ``PositionsAndCellIndex(particles.data.system, systems.index)``.
        include_cell: Whether the cell DOF gradients take part in the maximum.

    Returns:
        ``Table[SystemId, Array]`` of shape ``(n_systems,)``. A system without
        particles evaluates to ``-inf`` when cell gradients are excluded.
    """
    # Segment maxima can discard NaNs. Promote them to +inf before reducing,
    # so invalid active rows cannot masquerade as convergence; OOB rows remain ignored.
    norms = jax.tree.map(
        lambda x: jnp.where(jnp.isfinite(x), x, jnp.inf), dof_norm_per_row(gradients)
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
    """``Table[SystemId, bool]``: systems whose :func:`max_dof_per_system` is below ``tolerance``."""
    maximum = max_dof_per_system(gradients, index_prefix, include_cell=include_cell)
    runtime_assert(
        ~jnp.any(jnp.isposinf(maximum.data)), "Non-finite relaxation gradients."
    )
    return maximum.map_data(lambda m: m < tolerance)
