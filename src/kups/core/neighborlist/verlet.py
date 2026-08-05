# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet neighbor list with a skin.

Builds a *conservative* neighbor list at ``cutoff + skin`` once, stores its edges
in the simulation state, and reuses them for many steps via
[`RefineCutoffNeighborList`][kups.core.neighborlist.refine.RefineCutoffNeighborList]
(which re-masks to the true cutoff and recomputes minimum-image shifts for the
current — possibly deformed — cell). The expensive build then runs only when the
margin bookkeeping below demands it, amortizing it over the rebuild window.

## Completeness bound

A pair absent from the stored skin list had build-time minimum-image distance
``> r_build`` (and every periodic image of it was farther still). Decompose the
change since the build into the affine cell deformation ``F = h_ref⁻¹ h_now``
(row convention: pair vectors map as ``d ↦ d @ F``, and so do the image lattice
vectors) and the per-atom *non-affine* residual ``u_i = x_i - x_i_ref @ F``. Then
every image of a non-listed pair now sits at distance at least
``σ_min(F) * r_build - 2 * max|u|``, with ``σ_min`` the smallest singular value
of ``F``. The stored list therefore still contains every pair within ``cutoff``
while

    2 * max|u| + r_build * max(0, 1 - σ_min(F))  <=  r_build - cutoff

The left side is the *consumed* margin and the right side the *budget*
(:func:`skin_margin` computes both; the difference is the *headroom*). Because
``σ_min`` sees the full deformation, pure shear consumes margin like any other
strain — the trigger is not blind to off-diagonal cell moves. Residuals are
minimum-image wrapped in the current cell (honoring per-axis periodicity), so an
atom crossing a boundary along a sheared lattice vector is undone exactly; only
a genuine non-affine drift beyond half a cell between rebuilds (impossible in
practice, since rebuilds fire at skin scale) would be under-measured.

## Single-image clamp

Refine-based reuse can only represent one periodic image per stored pair, so the
build radius is clamped to half the smallest perpendicular length over periodic
axes (:func:`effective_build_radii`). A cell that compresses mid-run therefore
degrades to a thinner effective skin — more frequent rebuilds — instead of
failing. Only when the *cutoff itself* no longer fits (no skin can help; the
refine path would drop images) does the rebuild raise, telling the user to set
``verlet_skin = 0``.

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity, Cell

# Above this ratio of build radius to a cell's perpendicular length, a build needs
# more than one periodic image per pair and the refine-based reuse path is incomplete.
_SINGLE_IMAGE_LIMIT = 0.5


def effective_build_radii(
    cutoffs: Array, skin: float, cell: Cell[AnyPeriodicity]
) -> Array:
    """Per-system build radius: ``cutoff + skin`` clamped to the single-image limit.

    The refine-based reuse path collapses each stored pair to one minimum image,
    so the build must not replicate images: the radius is capped at half the
    cell's smallest perpendicular length over periodic axes (no cap in vacuum).

    Args:
        cutoffs: true cutoffs (Å), ``(n_sys,)``.
        skin: requested skin width (Å).
        cell: ``(n_sys,)``-batched cell the build runs in.

    Returns:
        Build radii (Å), ``(n_sys,)``. ``radii - cutoffs`` is the effective skin.
    """
    perp = cell.perpendicular_lengths
    limit = _SINGLE_IMAGE_LIMIT * jnp.min(
        jnp.where(jnp.array(cell.periodic), perp, jnp.inf), axis=-1
    )
    return jnp.minimum(cutoffs + skin, limit)


def skin_margin(
    positions: Array,
    reference_positions: Array,
    cell_now: Cell[AnyPeriodicity],
    cell_ref: Cell[AnyPeriodicity],
    system: Array,
    cutoffs: Array,
    skin: float,
) -> tuple[Array, Array]:
    """Consumed and budgeted completeness margin of the stored skin list.

    Implements the bound from the module docstring: the deformation term uses
    the smallest singular value of ``F = h_ref⁻¹ h_now`` per system, the motion
    term the largest minimum-image *non-affine* residual per system. Pairs
    never span systems, so the accounting is fully per system — one hot system
    neither charges nor rebuilds the others.

    Args:
        positions: current cartesian positions, ``(N, 3)``.
        reference_positions: positions at the last rebuild, ``(N, 3)``.
        cell_now: current cells, ``(n_sys,)``-batched.
        cell_ref: cells at the last rebuild, ``(n_sys,)``-batched.
        system: particle → system index array, ``(N,)``.
        cutoffs: true cutoffs (Å), ``(n_sys,)``.
        skin: requested skin width (Å).

    Returns:
        ``(consumed, budget)``, both ``(n_sys,)``. The stored list is complete
        while ``consumed <= budget`` in every system; ``budget - consumed`` is
        the headroom.
    """
    deform = cell_ref.inverse_vectors @ cell_now.vectors  # d_now = d_ref @ F
    co_moved = jnp.einsum("ni,nij->nj", reference_positions, deform[system])
    residual = cell_now[system].wrap(positions - co_moved)
    u_max = jax.ops.segment_max(
        jnp.linalg.norm(residual, axis=-1), system, num_segments=cutoffs.shape[0]
    )
    u_max = jnp.maximum(u_max, 0.0)  # empty segments reduce to -inf
    gram = deform @ jnp.swapaxes(deform, -1, -2)
    sigma_min = jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram)[..., 0], 0.0))
    r_build = effective_build_radii(cutoffs, skin, cell_ref)
    consumed = 2.0 * u_max + r_build * jnp.maximum(0.0, 1.0 - sigma_min)
    return consumed, r_build - cutoffs
