# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet neighbor list with a skin: the margin accounting.

A Verlet-skin scheme builds one conservative neighbor list at an enlarged
radius ``r_build ≈ cutoff + skin``, stores its edges, and reuses them over many
steps via
[`RefineCutoffNeighborList`][kups.core.neighborlist.refine.RefineCutoffNeighborList],
amortizing the expensive build over the rebuild window. This module holds the
pure geometry underneath such a scheme:
[`skin_margin`][kups.core.neighborlist.verlet.skin_margin] decides how long the
stored list remains complete, and
[`effective_build_radii`][kups.core.neighborlist.verlet.effective_build_radii]
keeps the build radius inside the single-image regime that edge reuse requires.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.neighborlist.types import NeighborListSystems
from kups.core.typing import HasPositionsAndSystemIndex, ParticleId, SystemId
from kups.core.utils.jax import dataclass


def effective_build_radii(
    cutoffs: Array, skin: ArrayLike, cell: Cell[AnyPeriodicity]
) -> Array:
    """Per-system build radius: ``cutoff + skin``, clamped to a single image.

    Reusing stored edges keeps exactly one periodic image per pair, so the
    build radius must stay below half the cell's smallest perpendicular length
    on every periodic axis — beyond that, second images enter the build sphere
    and the reuse path would drop them. A cell that compresses mid-run thus
    degrades to a thinner effective skin (more frequent rebuilds) instead of an
    incomplete list. No clamp applies in vacuum.

    Args:
        cutoffs: True cutoffs (Å), ``(n_sys,)``.
        skin: Requested skin width (Å).
        cell: ``(n_sys,)``-batched cell the build runs in.

    Returns:
        Build radii (Å), ``(n_sys,)``. ``radii - cutoffs`` is the effective skin.
    """
    perp = cell.perpendicular_lengths
    limit = 0.5 * jnp.min(jnp.where(jnp.array(cell.periodic), perp, jnp.inf), axis=-1)
    return jnp.minimum(cutoffs + skin, limit)


@dataclass
class SkinReference:
    """Geometry snapshot taken when the skin list was built.

    [`skin_margin`][kups.core.neighborlist.verlet.skin_margin] measures the
    drift of the current geometry relative to this snapshot. The arrays must
    not alias the live position/cell buffers (donated jitted steps would then
    receive the same buffer twice).

    Attributes:
        positions: Cartesian positions at the build, ``(N, 3)``.
        cell: ``(n_sys,)``-batched cell at the build.
    """

    positions: Array
    cell: Cell[AnyPeriodicity]


@dataclass
class SkinMargin:
    """Per-system completeness accounting of a stored skin list.

    Attributes:
        consumed: Worst-case distance (Å) by which atom motion and cell
            deformation since the build can have pulled a non-listed pair
            inward, ``(n_sys,)``.
        budget: Distance (Å) such a pair had to spare at build time — the
            effective skin ``r_build - cutoff``, ``(n_sys,)``.
    """

    consumed: Array
    budget: Array

    @property
    def headroom(self) -> Array:
        """``budget - consumed``; the stored list is complete while ``>= 0``."""
        return self.budget - self.consumed


def _sym3_min_eigenvalue(a: Array) -> Array:
    """Smallest eigenvalue of symmetric 3x3 matrices in closed form (Smith 1961).

    ``jnp.linalg.eigvalsh`` dispatches to an iterative GPU solver whose launch
    and sync overhead (~1 ms for a single 3x3) dominates the per-step margin check.
    """
    q = jnp.trace(a, axis1=-2, axis2=-1) / 3
    off = a[..., 0, 1] ** 2 + a[..., 0, 2] ** 2 + a[..., 1, 2] ** 2
    diag = jnp.diagonal(a, axis1=-2, axis2=-1) - q[..., None]
    p = jnp.sqrt((jnp.sum(diag**2, axis=-1) + 2 * off) / 6)
    safe_p = jnp.where(p > 0, p, 1.0)
    b = (a - q[..., None, None] * jnp.eye(3, dtype=a.dtype)) / safe_p[..., None, None]
    det = (
        b[..., 0, 0] * (b[..., 1, 1] * b[..., 2, 2] - b[..., 1, 2] * b[..., 2, 1])
        - b[..., 0, 1] * (b[..., 1, 0] * b[..., 2, 2] - b[..., 1, 2] * b[..., 2, 0])
        + b[..., 0, 2] * (b[..., 1, 0] * b[..., 2, 1] - b[..., 1, 1] * b[..., 2, 0])
    )
    r = jnp.clip(det / 2, -1.0, 1.0)
    smallest = q + 2 * p * jnp.cos(jnp.arccos(r) / 3 + 2 * jnp.pi / 3)
    return jnp.where(p > 0, smallest, q)  # p == 0: a multiple of the identity


def skin_margin(
    particles: Table[ParticleId, HasPositionsAndSystemIndex],
    systems: Table[SystemId, NeighborListSystems],
    reference: SkinReference,
    cutoffs: Table[SystemId, Array],
    skin: ArrayLike,
) -> Table[SystemId, SkinMargin]:
    """How much of the skin list's safety margin the geometry has used up.

    A skin list built at radius ``r_build`` stays complete for the true
    ``cutoff`` as long as no pair that was *outside* ``r_build`` at build time
    has come *inside* ``cutoff`` since. Two things move pairs inward:

    1. **Cell deformation.** Between the build and now the cell changed by the
       linear map ``F = h_ref⁻¹ h_now`` (row-vector convention), which maps
       every build-time pair vector ``d`` — including those to periodic images —
       to ``d @ F``. A linear map cannot shrink any vector by more than its
       smallest singular value: ``|d @ F| >= σ_min(F) |d|`` for all ``d``. So
       the affine part of the motion leaves every non-listed pair at distance
       at least ``σ_min(F) r_build``, an inward move of at most
       ``r_build (1 - σ_min(F))`` — and none at all if the cell only expanded
       (``σ_min >= 1``). Because ``σ_min`` sees the whole map, pure shear
       counts like any other strain, unlike per-axis length ratios.
    2. **Atom motion on top of the deformation.** Each atom's *non-affine*
       displacement is ``u_i = x_i - x_i_ref @ F`` — what remains after riding
       the cell — minimum-image wrapped in the current cell so that a boundary
       crossing (even along a sheared lattice vector) is undone exactly (a
       genuine non-affine drift beyond half a cell would be under-measured, but
       rebuilds fire at skin scale long before that). A pair distance changes
       by at most the two endpoint displacements, ``2 max|u|``.

    The stored list is therefore complete while, per system,

        consumed := 2 max|u| + r_build max(0, 1 - σ_min(F))  <=  r_build - cutoff =: budget

    i.e. while the worst-case inward motion of a non-listed pair (*consumed*)
    has not eaten the extra radius the build added on top of the cutoff
    (*budget*). Pairs never span systems, so the accounting is fully per
    system: one hot system neither charges nor rebuilds the others.

    Args:
        particles: Current particle table (positions and system index).
        systems: Current system table (cells).
        reference: Positions and cell snapshot taken at the last build.
        cutoffs: True cutoffs (Å) per system.
        skin: Requested skin width (Å) the list was built with.

    Returns:
        Per-system [`SkinMargin`][kups.core.neighborlist.verlet.SkinMargin]
        table (``consumed`` and ``budget``, both in Å).
    """
    cell_now = systems.data.cell
    system = particles.data.system
    cutoff_values = Table.broadcast_to(cutoffs, systems).data
    deform = reference.cell.inverse_vectors @ cell_now.vectors  # d_now = d_ref @ F
    # u_i = x_i - x_i_ref @ F, min-image wrapped
    co_moved = jnp.einsum(
        "ni,nij->nj", reference.positions, systems.set_data(deform)[system]
    )
    residual = systems[system].cell.wrap(particles.data.positions - co_moved)
    u_max = particles.data.system.max_over(jnp.linalg.norm(residual, axis=-1)).data
    u_max = jnp.maximum(u_max, 0.0)  # empty segments reduce to -inf
    # σ_min(F) from the smallest eigenvalue of the 3x3 Gram matrix F Fᵀ (the
    # clamp guards tiny negative rounding noise).
    gram = deform @ jnp.swapaxes(deform, -1, -2)
    sigma_min = jnp.sqrt(jnp.maximum(_sym3_min_eigenvalue(gram), 0.0))
    r_build = effective_build_radii(cutoff_values, skin, reference.cell)
    consumed = 2.0 * u_max + r_build * jnp.maximum(0.0, 1.0 - sigma_min)
    return Table(systems.keys, SkinMargin(consumed, r_build - cutoff_values))
