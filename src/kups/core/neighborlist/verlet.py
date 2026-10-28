# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet neighbor list with a skin.

A Verlet-skin scheme builds one conservative neighbor list at
``r_build = cutoff + skin``, stores its pairs, and reuses them over many steps,
amortizing the expensive build over the rebuild window.
[`SkinList`][kups.core.neighborlist.verlet.SkinList] holds the pairs as neighbor
rows. Every read goes through
[`read_skin_list`][kups.core.neighborlist.verlet.read_skin_list], which rebuilds the
list when it may miss a pair of the particles it is given and returns the patch that
stores it. [`skin_margin`][kups.core.neighborlist.verlet.skin_margin] bounds how
much of the skin the geometry has used up since the build.
[`SkinNeighborList`][kups.core.neighborlist.verlet.SkinNeighborList] gives a
potential the stored pairs within its own cutoffs as edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import check_assertions, runtime_assert
from kups.core.capacity import LensCapacity
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Index, Table
from kups.core.lens import LambdaLens, Lens, bind, identity_lens, view
from kups.core.neighborlist.adaptive import AdaptiveNeighborList
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    InBoundsMask,
    TouchesQueriedKeysMask,
)
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.rows import NeighborRows, neighbor_rows
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.patch import Accept, Patch, WithPatch
from kups.core.result import as_result_function
from kups.core.typing import (
    HasPositionsAndSystemIndex,
    HasSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import (
    dataclass,
    field,
    jit,
    no_jax_tracing,
    tree_copy,
    tree_where_broadcast_last,
)


@dataclass
class SkinReference:
    """Snapshot of the particles and cells taken when the skin list was built.

    [`skin_margin`][kups.core.neighborlist.verlet.skin_margin] measures the
    drift of the current geometry relative to this snapshot. The arrays must
    not alias the live particle and cell buffers (donated jitted steps would
    then receive the same buffer twice).

    Attributes:
        positions: Cartesian positions at the build, ``(N, 3)``.
        cell: ``(n_sys,)``-batched cell at the build.
        system: Raw system index of each particle at the build, ``(N,)`` int32.
    """

    positions: Array
    cell: Cell[AnyPeriodicity]
    system: Array


@dataclass
class SkinMargin:
    """Per-system completeness accounting of a stored skin list.

    Attributes:
        consumed: Worst-case distance (Å) by which atom motion and cell
            deformation since the build can have pulled a non-listed pair
            inward, ``(n_sys,)``.
        budget: Distance (Å) such a pair had to spare at build time, the skin,
            ``(n_sys,)``.
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
    skin: Table[SystemId, Array],
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
       crossing (even along a sheared lattice vector) is undone exactly (motion
       is read modulo the lattice, and
       [`wraps_since_build`][kups.core.neighborlist.verlet.wraps_since_build]
       counts wraps with the same rounding, so the bound holds however the
       positions changed between reads). A pair distance changes by at most the
       two endpoint displacements, ``2 max|u|``.

    The stored list is therefore complete while, per system,

        consumed := 2 max|u| + r_build max(0, 1 - σ_min(F))  <=  r_build - cutoff =: budget

    i.e. while the worst-case inward motion of a non-listed pair (*consumed*)
    has not eaten the extra radius the build added on top of the cutoff
    (*budget*). Pairs never span systems, so the accounting is fully per
    system: one hot system does not charge the others.

    Args:
        particles: Current particle table (positions and system index).
        systems: Current system table (cells).
        reference: Positions and cell snapshot taken at the last build.
        cutoffs: True cutoffs (Å) per system.
        skin: Skin (Å) per system the list was built with.

    Returns:
        Per-system [`SkinMargin`][kups.core.neighborlist.verlet.SkinMargin]
        table (``consumed`` and ``budget``, both in Å).
    """
    # Materialized first, because a parametrized frame gathered per particle is
    # evaluated once per particle.
    cell_now = systems.data.cell.materialize()
    system = particles.data.system
    cutoff_values = Table.broadcast_to(cutoffs, systems).data
    skin_values = Table.broadcast_to(skin, systems).data
    deform = reference.cell.inverse_vectors @ cell_now.vectors  # d_now = d_ref @ F
    # u_i = x_i - x_i_ref @ F, min-image wrapped
    co_moved = jnp.einsum(
        "ni,nij->nj", reference.positions, systems.set_data(deform)[system]
    )
    residual = systems.set_data(cell_now)[system].wrap(
        particles.data.positions - co_moved
    )
    u_max = particles.data.system.max_over(jnp.linalg.norm(residual, axis=-1)).data
    u_max = jnp.maximum(u_max, 0.0)  # empty segments reduce to -inf
    # σ_min(F) from the smallest eigenvalue of the 3x3 Gram matrix F Fᵀ (the
    # clamp guards tiny negative rounding noise).
    gram = deform @ jnp.swapaxes(deform, -1, -2)
    sigma_min = jnp.sqrt(jnp.maximum(_sym3_min_eigenvalue(gram), 0.0))
    r_build = cutoff_values + skin_values
    consumed = 2.0 * u_max + r_build * jnp.maximum(0.0, 1.0 - sigma_min)
    return Table(systems.keys, SkinMargin(consumed, skin_values))


@dataclass
class SkinList:
    """Pairs within ``cutoffs + skin`` of the particles they were built from.

    Inclusion and exclusion are applied at the build, where an excluded pair drops
    only its minimum image: exact for one-particle exclusion groups, or for cells
    wider than ``2 (cutoffs + skin)``. A read checks positions, cells and systems but
    not groups, so the particles the list is created from and those of every reader
    must carry the same inclusion and exclusion groups. A particle that changes its
    system makes the next read rebuild the whole list, so the list serves dynamics
    whose particles keep their systems, such as MD and relaxation. Read it with
    [`read_skin_list`][kups.core.neighborlist.verlet.read_skin_list].

    Attributes:
        rows: Pairs and their image shifts at the build, one row per particle.
        reference: Particles and cells at the build.
        cutoffs: Cutoffs (Å) within which the list holds every pair, the largest
            of the potentials that read it.
        skin: Distance (Å) added to ``cutoffs`` at each build.
        params: Capacities of the build.
        builder: Builds the pairs, e.g. ``CellListNeighborList.new``.
    """

    rows: NeighborRows
    reference: SkinReference
    cutoffs: Table[SystemId, Array]
    skin: Table[SystemId, Array]
    params: UniversalNeighborlistParameters
    builder: NeighborListFactory[IsUniversalNeighborlistParams] = field(static=True)

    @staticmethod
    @no_jax_tracing
    def create(
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        skin: Table[SystemId, Array],
        *,
        builder: NeighborListFactory[
            IsUniversalNeighborlistParams
        ] = AdaptiveNeighborList.new,
    ) -> SkinList:
        """Build a skin list from the particle and system tables.

        Capacities start from the estimate at ``cutoffs + skin`` and grow until the
        build fits.

        Args:
            particles: Particles to pair, with the inclusion and exclusion groups
                of every reader.
            systems: Systems with their cells.
            cutoffs: Largest cutoff (Å) per system of the potentials that read the
                list.
            skin: Distance (Å) per system added to ``cutoffs``.
            builder: Builds the pairs.

        Returns:
            The skin list.
        """
        cutoffs = Table.broadcast_to(cutoffs, systems)
        skin = Table.broadcast_to(skin, systems)
        params = UniversalNeighborlistParameters.estimate(
            particles.data.system.counts, systems, cutoffs + skin
        )
        return _build_on_host(params, particles, systems, cutoffs, skin, builder)

    def is_complete(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, NeighborListSystems],
    ) -> Array:
        """Whether the list holds every pair of ``particles`` within its cutoffs.

        True while every particle is in the system it had at the build and no
        system has used up its skin. NaN geometry counts as used up.

        Args:
            particles: Particles to pair.
            systems: Their systems.

        Returns:
            Scalar bool.
        """
        reference = self.reference
        margin = skin_margin(particles, systems, reference, self.cutoffs, self.skin)
        same = jnp.all(_system(particles) == reference.system)
        return same & jnp.all(margin.data.headroom >= 0)


class IsSkinListState(IsNeighborListState[IsUniversalNeighborlistParams], Protocol):
    """State with a skin list.

    ``neighborlist_params.avg_edges`` sizes the edges read from the list.
    """

    @property
    def skin_list(self) -> SkinList: ...


@dataclass
class SkinListPatch[State](Patch[State]):
    """Stores the skin list a reader used, unless no system accepts.

    The list covers all systems at once, so it is stored as soon as one system
    accepts. This is safe because every read checks the stored list against the
    particles it is given. With no system accepting, it leaves the state
    unchanged, as Monte Carlo requires of a zero-filled patch.

    Attributes:
        skin_list: The list the reader used.
        lens: Lens to the stored list.
    """

    skin_list: SkinList
    lens: Lens[State, SkinList] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        stored = self.lens.get(state)
        kept = tree_where_broadcast_last(jnp.any(accept.data), self.skin_list, stored)
        return self.lens.set(state, kept)


def read_skin_list[S](
    state: S,
    lens: Lens[S, SkinList],
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> WithPatch[SkinList, SkinListPatch[S]]:
    """The skin list at ``lens``, rebuilt if it may miss a pair, and its patch.

    The returned list holds every pair of ``particles`` within its cutoffs whenever
    the enclosing computation runs with runtime assertions and they pass. A rebuild
    keeps the stored capacities. One that outgrows them fails an assertion and is
    discarded, and the fix of that assertion grows the capacities through ``lens``.

    Args:
        state: State that the patch and capacity fixes apply to.
        lens: Lens from ``state`` to the list.
        particles: Particles to pair, those the list was built from.
        systems: Their systems.

    Returns:
        The list, and the patch that stores it.
    """
    stored = lens.get(state)
    assert particles.size == len(stored.rows.order), "Not the skin list's particles."

    def rebuilt() -> SkinList:
        built = _build(
            state,
            lens.nest(_PARAMS),
            particles,
            systems,
            stored.cutoffs,
            stored.skin,
            stored.builder,
        )
        # A build that outgrew its capacities is discarded.
        return tree_where_broadcast_last(check_assertions(), built, stored)

    # The stored list is the false branch, where XLA reuses its buffers instead of
    # copying the rows at every read.
    current: SkinList = jax.lax.cond(
        ~stored.is_complete(particles, systems), rebuilt, lambda: stored
    )
    return WithPatch(current, SkinListPatch(current, lens))


@dataclass
class SkinNeighborList[State]:
    """Pairs of a state's skin list within ``cutoffs``, as edges.

    Each call reads the list with
    [`read_skin_list`][kups.core.neighborlist.verlet.read_skin_list] and returns the
    patch that stores it. The groups of ``keys`` take effect only when the read
    rebuilds the list. Stored shifts follow each particle's periodic wraps since the
    build, so cells narrower than twice the cutoff work too, within the int8 limit of
    ``NeighborRows``.

    Attributes:
        state: State that holds the list; patches and fixes apply to it.
        lens: Lens to the part of ``state`` with the list and the edge capacity.
        cutoffs: Cutoffs (Å) of the edges this reader returns, at most the list's.
    """

    state: State
    lens: Lens[State, IsSkinListState] = field(static=True)
    cutoffs: Table[SystemId, Array]

    @staticmethod
    def from_state[S](
        state: S, lens: Lens[S, IsSkinListState], cutoffs: Table[SystemId, Array]
    ) -> SkinNeighborList[S]:
        """Neighbor list over the skin list of the state at ``lens``.

        Args:
            state: State that holds the list; patches and fixes apply to it.
            lens: Lens to the part of ``state`` with the list and the edge capacity.
            cutoffs: Cutoffs (Å) of the edges the reader returns, at most the list's.

        Returns:
            The neighbor list.
        """
        return SkinNeighborList(state, lens, cutoffs)

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> WithPatch[Edges[Literal[2]], SkinListPatch[State]]: ...
    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> WithPatch[Edges[Literal[2]], SkinListPatch[State]]: ...
    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> WithPatch[Edges[Literal[2]], SkinListPatch[State]]:
        assert queries is None, "A skin list pairs its own particles; no queries."
        skin_list = self.lens.focus(lambda s: s.skin_list)
        read = read_skin_list(self.state, skin_list, keys, systems)
        cutoffs = Table.broadcast_to(self.cutoffs, systems)
        listed = Table.broadcast_to(read.data.cutoffs, systems)
        runtime_assert(
            jnp.all(cutoffs.data <= listed.data),
            "Cutoffs exceed the skin list's; create it at the largest cutoff.",
        )
        avg_edges = self.lens.focus(lambda s: s.neighborlist_params.avg_edges)
        size = keys.size if queried_keys is None else queried_keys.size
        pipeline = Pipeline[Literal[2]](
            selector=_FollowWraps(read.data.rows, read.data.reference),
            masks=(
                InBoundsMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                TouchesQueriedKeysMask(),
            ),
            compactor=ReduceCompactor(
                LensCapacity(avg_edges.get(self.state), avg_edges).multiply(size)
            ),
        )
        edges = pipeline(keys, systems, queried_keys=queried_keys)
        return WithPatch(edges, read.patch)


def wraps_since_build(
    reference: SkinReference,
    fractional: Array,
    system: Index[SystemId],
    systems: Table[SystemId, NeighborListSystems],
) -> Array:
    """Cell vectors each particle was wrapped back by since a skin list's build.

    A particle at ``x`` is the build's particle at ``x_build + u - w h``, with ``u``
    its motion since; ``u`` stays below half a cell, so ``w`` rounds.

    Args:
        reference: Positions and cells at the build.
        fractional: Current fractional positions, ``(n, 3)``.
        system: System of each particle.
        systems: Current systems.

    Returns:
        ``w``, integer-valued, ``(n, 3)``; zero on open axes.
    """
    built = Table(systems.keys, reference.cell.frame.materialize())[system]
    return systems.data.cell.minimum_image_shifts(
        built.to_fractional(reference.positions) - fractional
    )


@dataclass
class _FollowWraps:
    """Stored rows as pairs, shifted by the cells their particles were wrapped by
    since the build.

    A slot of particle ``i`` with neighbor ``j`` and shift ``s`` is
    ``x_j - x_i + s h`` at the build; with ``w`` from ``wraps_since_build`` the same
    image is now ``s + w_j - w_i``.
    """

    rows: NeighborRows
    reference: SkinReference

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        keys, rows = ctx.keys.data, self.rows
        wraps = ctx.keys.set_data(
            wraps_since_build(self.reference, keys.positions, keys.system, ctx.systems)
        )
        # The particles at both ends of each slot; padding (row n) stays out of
        # bounds.
        particle = jnp.append(rows.order, len(rows.order))
        ends = particle[jnp.stack([rows.centers, rows.neighbors], axis=-1)]
        dtype = keys.system.raw_indices.dtype
        indices = Index(ctx.edge_keys, ends.astype(dtype), _cls=ctx.keys.cls)
        w = wraps[indices]
        shifts = rows.shifts[:, None].astype(keys.positions.dtype)
        shifts += w[:, 1:] - w[:, :1]
        # Inclusion and exclusion were applied at the build, so no mask reads the
        # image flags.
        return CandidateBatch(Edges(indices, shifts), jnp.ones(len(ends), bool))


def _system(particles: Table[ParticleId, HasSystemIndex]) -> Array:
    """Raw system index of each particle, ``(n,)`` int32."""
    return particles.data.system.raw_indices.astype(jnp.int32)


def _regrown(skin_list: SkinList, value: UniversalNeighborlistParameters) -> SkinList:
    """``skin_list`` with capacities ``value``, its rows padded to the slots of a build
    at ``value``.

    A build emits slots in proportion to ``avg_edges``. Slots after ``offsets[-1]``
    are padding, so the padded list holds the same pairs.
    """
    rows, n = skin_list.rows, len(skin_list.rows.order)
    slots = len(rows.centers) // skin_list.params.avg_edges * value.avg_edges
    pad = slots - len(rows.centers)
    padded = NeighborRows(
        rows.order,
        rows.rank,
        rows.offsets,
        jnp.pad(rows.centers, (0, pad), constant_values=n),
        jnp.pad(rows.neighbors, (0, pad), constant_values=n),
        jnp.pad(rows.shifts, ((0, pad), (0, 0))),
    )
    return bind(skin_list, lambda s: (s.rows, s.params)).set((padded, value))


_PARAMS: Lens[SkinList, UniversalNeighborlistParameters] = LambdaLens(
    view(lambda s: s.params, cls=SkinList), _regrown
)
"""Capacities of a skin list; setting them pads its rows."""


def _build[S](
    state: S,
    params: Lens[S, UniversalNeighborlistParameters],
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: Table[SystemId, Array],
    builder: NeighborListFactory[IsUniversalNeighborlistParams],
) -> SkinList:
    """Build a skin list at the capacities ``params`` focuses in ``state``."""
    # Stored materialized, so steps do not evaluate a parametrized frame again.
    cell = systems.data.cell.materialize()
    widths = jnp.where(jnp.asarray(cell.periodic), cell.perpendicular_lengths, jnp.inf)
    # Wraps are counted by rounding, so motion between builds must stay below half
    # a cell; rebuilds keep it below half the skin.
    runtime_assert(
        jnp.all(widths > skin.data[:, None]),
        "Skin list needs cells wider than the skin.",
    )
    # The skin list stores the pairs itself, so the builder's patch is dropped.
    edges = builder(state, params, cutoffs + skin)(particles, systems).data
    # Copies, so a donated state does not hold one buffer twice.
    positions, cell, system = tree_copy(
        (particles.data.positions, cell, _system(particles))
    )
    return SkinList(
        neighbor_rows(edges, particles, systems),
        SkinReference(positions, cell, system),
        *tree_copy((cutoffs, skin)),
        params.get(state),
        builder,
    )


@jit(static_argnames=("builder",))
@as_result_function
def _build_eagerly(
    params: UniversalNeighborlistParameters,
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: Table[SystemId, Array],
    *,
    builder: NeighborListFactory[IsUniversalNeighborlistParams],
) -> SkinList:
    lens = identity_lens(UniversalNeighborlistParameters)
    return _build(params, lens, particles, systems, cutoffs, skin, builder)


def _build_on_host(
    params: UniversalNeighborlistParameters,
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: Table[SystemId, Array],
    builder: NeighborListFactory[IsUniversalNeighborlistParams],
) -> SkinList:
    """Build eagerly, growing ``params`` until the build fits."""
    while True:
        result = _build_eagerly(
            params, particles, systems, cutoffs, skin, builder=builder
        )
        if not result.failed_assertions:
            return result.value
        params = result.fix_or_raise(params)


if TYPE_CHECKING:
    _: NeighborListFactory[IsSkinListState] = SkinNeighborList.from_state

    def _check_reader[S](a: SkinNeighborList[S]) -> None:
        _: NeighborList[Literal[2]] = a
