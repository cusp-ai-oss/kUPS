# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet-skin neighbor list: margin accounting, candidate cache, and refresh."""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Index, Table
from kups.core.lens import LambdaLens, Lens
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.types import (
    NeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.typing import (
    ExclusionId,
    HasPositionsAndSystemIndex,
    InclusionId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, skip_post_init_if_disabled, tree_copy


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
class SkinPoints:
    """Particle inputs of a neighbor-list build."""

    positions: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class SkinReference:
    """Particle inputs and cells at the last build.

    [`skin_margin`][kups.core.neighborlist.verlet.skin_margin] measures drift
    against this snapshot. Builders apply inclusion and exclusion masks, so the
    labels are part of the snapshot.

    Attributes:
        particles: Positions and labels at the build.
        cell: Cells at the build.
    """

    particles: Table[ParticleId, SkinPoints]
    cell: Table[SystemId, Cell[AnyPeriodicity]]

    @classmethod
    def new(
        cls,
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
    ) -> SkinReference:
        """Snapshot the inputs of a build.

        Args:
            particles: Particles passed to the builder.
            systems: Systems passed to the builder.

        Returns:
            Reference sharing the input arrays.
        """
        return cls(
            particles.map_data(
                lambda p: SkinPoints(p.positions, p.system, p.inclusion, p.exclusion)
            ),
            systems.map_data(lambda s: s.cell),
        )


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


def skin_margin(
    particles: Table[ParticleId, HasPositionsAndSystemIndex],
    systems: Table[SystemId, NeighborListSystems],
    reference: SkinReference,
    radii: Table[SystemId, Array],
    cutoffs: Table[SystemId, Array],
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
        reference: Particle inputs and cells at the last build.
        radii: Build radii ``r_build`` (Å) per system; zero for no build.
        cutoffs: True cutoffs (Å) per system.

    Returns:
        Per-system [`SkinMargin`][kups.core.neighborlist.verlet.SkinMargin]
        table (``consumed`` and ``budget``, both in Å).
    """
    cells = systems.map_data(lambda s: s.cell)
    deformation = Table.join(cells, reference.cell).map_data(
        lambda pair: pair[1].inverse_vectors @ pair[0].vectors  # d_now = d_ref @ F
    )
    # u_i = x_i - x_i_ref @ F, min-image wrapped
    co_moved = jnp.einsum(
        "ni,nij->nj",
        reference.particles[particles.index].positions,
        deformation[particles.data.system],
    )
    residual = cells[particles.data.system].wrap(particles.data.positions - co_moved)
    displacement = particles.data.system.update_labels(systems.keys).max_over(
        jnp.linalg.norm(residual, axis=-1)
    )
    # Empty systems reduce to -inf.
    u_max = jnp.maximum(Table.broadcast_to(displacement, systems).data, 0.0)
    # σ_min(F) from the smallest eigenvalue of the 3x3 Gram matrix F Fᵀ
    # (cheaper than an SVD; the clamp guards eigvalsh's tiny negative noise).
    f = deformation.data
    gram = f @ jnp.swapaxes(f, -1, -2)
    sigma_min = jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram)[..., 0], 0.0))
    r_build = Table.broadcast_to(radii, systems).data
    cutoff = Table.broadcast_to(cutoffs, systems).data.astype(r_build.dtype)
    consumed = 2.0 * u_max + r_build * jnp.maximum(0.0, 1.0 - sigma_min)
    return Table(systems.keys, SkinMargin(consumed, r_build - cutoff))


@dataclass
class VerletSkinState:
    """The last skin build and the capacities it was built with.

    Attributes:
        params: Capacities of the build. ``edges`` has ``params.avg_edges`` rows
            per particle, the full self-graph output size of the pair builders.
        edges: Candidate pairs within ``radii`` at the build.
        reference: Particle inputs and cells at the build.
        radii: Build radii (Å); zero before the first build.
    """

    params: UniversalNeighborlistParameters
    edges: Edges[Literal[2]]
    reference: SkinReference
    radii: Table[SystemId, Array]

    @skip_post_init_if_disabled
    def __post_init__(self) -> None:
        rows = self.params.avg_edges * self.reference.particles.size
        assert len(self.edges) == rows, (
            f"Skin cache holds {len(self.edges)} edges; its capacities imply {rows}. "
            "Change capacities through SKIN_PARAMS."
        )

    @classmethod
    def new(
        cls,
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        params: UniversalNeighborlistParameters,
    ) -> VerletSkinState:
        """Allocate an unbuilt cache.

        Args:
            particles: Particle table.
            systems: System table.
            params: Capacities of the skin build.

        Returns:
            Cache with out-of-bounds edges and zero radii. The reference is a
            copy, so donating a state that holds both is safe.
        """
        return _unbuilt(params, tree_copy(SkinReference.new(particles, systems)))


def _unbuilt(
    params: UniversalNeighborlistParameters, reference: SkinReference
) -> VerletSkinState:
    particles = reference.particles
    rows = params.avg_edges * particles.size
    # Builders emit shifts in the dtype of the fractional coordinates.
    dtype = jnp.result_type(particles.data.positions, reference.cell.data.vectors)
    return VerletSkinState(
        params,
        Edges(
            Index(particles.keys, jnp.full((rows, 2), particles.size, dtype=int)),
            jnp.zeros((rows, 1, 3), dtype),
        ),
        reference,
        Table(reference.cell.keys, jnp.zeros(reference.cell.size, dtype)),
    )


def _set_skin_params(
    state: VerletSkinState, value: UniversalNeighborlistParameters
) -> VerletSkinState:
    if value == state.params:
        return state
    return _unbuilt(value, state.reference)


SKIN_PARAMS: Lens[VerletSkinState, UniversalNeighborlistParameters] = LambdaLens(
    lambda c: c.params, _set_skin_params
)
"""Capacities of a skin cache. Setting different ones discards the build, which may
have overflowed them."""


def skin_covers(
    cache: VerletSkinState,
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
) -> Array:
    """Whether ``cache.edges`` hold every pair within ``cutoffs`` of this configuration.

    Args:
        cache: Skin cache.
        particles: Current particles.
        systems: Current systems.
        cutoffs: Interaction cutoffs (Å).

    Returns:
        Scalar bool: the labels match the build and every system has headroom.
    """
    built, now = cache.reference.particles.data, particles.data
    same_labels = (
        (built.system.indices == now.system.indices).all()
        & (built.inclusion.indices == now.inclusion.indices).all()
        & (built.exclusion.indices == now.exclusion.indices).all()
    )
    margin = skin_margin(particles, systems, cache.reference, cache.radii, cutoffs)
    return same_labels & (margin.data.headroom >= 0).all()


def refresh_skin(
    cache: VerletSkinState,
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: ArrayLike,
    builder: Callable[[Table[SystemId, Array]], NeighborList[Literal[2]]],
) -> VerletSkinState:
    """Rebuild the cache at single-image radii once it stops covering the geometry.

    Args:
        cache: Skin cache.
        particles: Current particles.
        systems: Current systems.
        cutoffs: Largest interaction cutoffs read through the cache (Å).
        skin: Requested skin width (Å).
        builder: Neighbor list bound to the given build radii, with
            ``cache.params`` as its capacities.

    Returns:
        ``cache`` while it covers the geometry, otherwise a new build. A build
        without skin budget in some system (zero skin, or a cutoff past the
        single-image limit) could never cover another geometry, so it is not
        stored.
    """
    c = Table.broadcast_to(cutoffs, systems).data
    dtype = cache.radii.data.dtype
    radii = Table(
        systems.keys, effective_build_radii(c, skin, systems.data.cell).astype(dtype)
    )
    has_budget = (radii.data > c.astype(dtype)).all()

    def rebuild() -> VerletSkinState:
        return VerletSkinState(
            cache.params,
            builder(radii)(particles, systems),
            SkinReference.new(particles, systems),
            radii,
        )

    stale = ~skin_covers(cache, particles, systems, cutoffs)
    return jax.lax.cond(stale & has_budget, rebuild, lambda: cache)
