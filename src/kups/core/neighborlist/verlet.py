# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet neighbor list with a skin: the stored list and its margin accounting.

    skin_list = SkinList.create(particles, systems, cutoffs, skin)
    state = State(particles, systems, neighborlist_params, skin_list)
    potential = make_lennard_jones_from_state(
        lens, parameters=parameters, neighborlist_factory=SkinNeighborList.from_state
    )
    update = SkinListUpdate(lens)
    forces = SequentialPropagator((update, PotentialAsPropagator(potential)))

A [`SkinList`][kups.core.neighborlist.verlet.SkinList] holds the edges of one
search at ``cutoff + skin`` in the state.
[`SkinListUpdate`][kups.core.neighborlist.verlet.SkinListUpdate] searches again
when [`skin_margin`][kups.core.neighborlist.verlet.skin_margin] reports the skin
used up, and
[`SkinNeighborList`][kups.core.neighborlist.verlet.SkinNeighborList] returns the
stored pairs within a cutoff at the current positions.

The margin counts a skin only up to
[`effective_build_radii`][kups.core.neighborlist.verlet.effective_build_radii].
In a cell narrower than twice the cutoff it counts none, and every update
searches.
"""

from __future__ import annotations

from typing import Literal, Protocol, overload, override

import jax
import jax.numpy as jnp
from jax import Array

from kups.core import units
from kups.core.assertion import check_assertions, runtime_assert
from kups.core.capacity import Capacity, LensCapacity
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Index, Table
from kups.core.lens import BaseLens, Lens, identity_lens, lens
from kups.core.neighborlist.adaptive import AdaptiveNeighborList
from kups.core.neighborlist.compact import MaskOnlyCompactor, ReduceCompactor
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.masks import (
    DistanceCutoffMask,
    InBoundsMask,
    TouchesQueriedKeysMask,
)
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.pipeline import Pipeline
from kups.core.neighborlist.types import (
    CandidateBatch,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.propagator import Propagator
from kups.core.result import as_result_function
from kups.core.typing import HasPositionsAndSystemIndex, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, jit, no_jax_tracing


def effective_build_radii(
    cutoffs: units.Length, skin: units.Length, cell: Cell[AnyPeriodicity]
) -> units.Length:
    """Per-system build radius: ``cutoff + skin``, clamped to a single image.

    Reusing stored edges keeps exactly one periodic image per pair, so the
    build radius must stay below half the cell's smallest perpendicular length
    on every periodic axis — beyond that, second images enter the build sphere
    and the reuse path would drop them. A cell that compresses mid-run thus
    degrades to a thinner effective skin (more frequent rebuilds) instead of an
    incomplete list. No clamp applies in vacuum.

    Args:
        cutoffs: True cutoffs, ``(n_sys,)``.
        skin: Requested skin width.
        cell: ``(n_sys,)``-batched cell the build runs in.

    Returns:
        Build radii, ``(n_sys,)``. ``radii - cutoffs`` is the effective skin.
    """
    perp = cell.perpendicular_lengths
    limit = 0.5 * units.min(
        units.where(jnp.array(cell.periodic), perp, jnp.inf * perp.unit), axis=-1
    )
    return units.minimum(cutoffs + skin, limit)


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

    positions: units.Length
    cell: Cell[AnyPeriodicity]


@dataclass
class SkinMargin:
    """Per-system completeness accounting of a stored skin list.

    Attributes:
        consumed: Worst-case distance by which atom motion and cell
            deformation since the build can have pulled a non-listed pair
            inward, ``(n_sys,)``.
        budget: Distance such a pair had to spare at build time — the
            effective skin ``r_build - cutoff``, ``(n_sys,)``.
    """

    consumed: units.Length
    budget: units.Length

    @property
    def headroom(self) -> units.Length:
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
    cutoffs: Table[SystemId, units.Length],
    skin: units.Length,
) -> Table[SystemId, SkinMargin]:
    """How much of the skin list's safety margin the geometry has used up.

    A list built at radius ``r_build`` stays complete while no pair outside
    ``r_build`` at build time has moved inside ``cutoff``. The cell deformation
    ``F = h_ref⁻¹ h_now`` shrinks pair vectors by at most ``σ_min(F)``, moving
    them inward by at most ``r_build (1 - σ_min(F))``; shear counts like any
    other strain. Each atom's non-affine displacement ``u_i = x_i - x_i_ref @ F``,
    minimum-image wrapped, moves a pair by at most ``2 max|u|``. The list is
    complete per system while

        consumed := 2 max|u| + r_build max(0, 1 - σ_min(F))  <=  r_build - cutoff =: budget

    Args:
        particles: Current particle table (positions and system index).
        systems: Current system table (cells).
        reference: Positions and cell snapshot taken at the last build.
        cutoffs: True cutoffs per system.
        skin: Requested skin width the list was built with.

    Returns:
        Per-system [`SkinMargin`][kups.core.neighborlist.verlet.SkinMargin]
        table (``consumed`` and ``budget``, both lengths).
    """
    cell_now = systems.data.cell
    system = particles.data.system
    cutoff_values = Table.broadcast_to(cutoffs, systems).data
    deform = reference.cell.inverse_vectors @ cell_now.vectors  # d_now = d_ref @ F
    # u_i = x_i - x_i_ref @ F, min-image wrapped
    co_moved = units.einsum(
        "ni,nij->nj", reference.positions, systems.set_data(deform)[system]
    )
    residual = systems[system].cell.wrap(particles.data.positions - co_moved)
    u_max = particles.data.system.max_over(units.norm(residual, axis=-1)).data
    # empty segments reduce to -inf
    u_max = units.maximum(u_max, units.zeros_like(u_max))
    # σ_min(F) from the smallest eigenvalue of the 3x3 Gram matrix F Fᵀ (the
    # clamp guards tiny negative rounding noise).
    gram = deform @ jnp.swapaxes(deform, -1, -2)
    sigma_min = jnp.sqrt(jnp.maximum(_sym3_min_eigenvalue(gram), 0.0))
    r_build = effective_build_radii(cutoff_values, skin, reference.cell)
    consumed = 2.0 * u_max + r_build * jnp.maximum(0.0, 1.0 - sigma_min)
    return Table(systems.keys, SkinMargin(consumed, r_build - cutoff_values))


@dataclass
class SkinList:
    """Pairs within ``cutoffs + skin`` of the geometry they were built from.

    Inclusion and exclusion groups are applied at the build, so the particles of
    every reader must carry the groups the list was built with.

    Attributes:
        edges: Pairs and their image shifts at the build, padded out of bounds.
        reference: Positions and cells at the build.
        cutoffs: Largest cutoff per system of the potentials that read the list.
        skin: Distance added to ``cutoffs`` at each build.
        parameters: Capacities of the build.
    """

    edges: Edges[Literal[2]]
    reference: SkinReference
    cutoffs: Table[SystemId, units.Length]
    skin: units.Length
    parameters: UniversalNeighborlistParameters

    @staticmethod
    @no_jax_tracing
    def create(
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, units.Length],
        skin: units.Length,
    ) -> SkinList:
        """Build a skin list, growing the estimated capacities until it fits.

        Args:
            particles: Particles to pair, with the groups of every reader.
            systems: Systems with their cells.
            cutoffs: Largest cutoff per system of the potentials that read the list.
            skin: Distance added to ``cutoffs``.

        Returns:
            The skin list. It shares no array with its arguments.
        """
        cutoffs = Table.broadcast_to(cutoffs, systems)
        radii = cutoffs.set_data(cutoffs.data + skin)
        parameters = UniversalNeighborlistParameters.estimate(
            particles.data.system.counts, systems, radii
        )
        capacities = identity_lens(UniversalNeighborlistParameters)

        @jit
        @as_result_function
        def build(parameters: UniversalNeighborlistParameters) -> Edges[Literal[2]]:
            search = AdaptiveNeighborList.new(parameters, capacities, radii)
            return search(particles, systems)

        result = build(parameters)
        while result.failed_assertions:
            parameters = result.fix_or_raise(parameters)
            result = build(parameters)
        reference = SkinReference(particles.data.positions, systems.data.cell)
        reference, cutoffs, skin = jax.tree.map(jnp.copy, (reference, cutoffs, skin))
        return SkinList(result.value, reference, cutoffs, skin, parameters)

    def is_complete(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, NeighborListSystems],
    ) -> Array:
        """Whether the list holds every pair of ``particles`` within its cutoffs.

        Args:
            particles: Particles to pair, those the list was built from.
            systems: Their systems.

        Returns:
            Scalar bool, false for NaN geometry.
        """
        margin = skin_margin(
            particles, systems, self.reference, self.cutoffs, self.skin
        ).data
        return jnp.all(margin.consumed <= margin.budget)

    def rebuilt[State](
        self,
        state: State,
        lens: Lens[State, SkinList],
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
    ) -> SkinList:
        """This list built from ``particles`` with its stored capacities.

        A build that outgrows the capacities fails an assertion and returns the
        stored list. The fix of that assertion grows the capacities and the
        stored edges through ``lens``.

        Args:
            state: State that capacity fixes apply to.
            lens: Lens from ``state`` to this list.
            particles: Particles to pair.
            systems: Their systems.

        Returns:
            The list of the new geometry, or this list.
        """
        radii = self.cutoffs.set_data(self.cutoffs.data + self.skin)
        build = AdaptiveNeighborList.new(state, _Capacities(lens), radii)
        built = (
            build(particles, systems),
            SkinReference(particles.data.positions, systems.data.cell),
        )
        fits = check_assertions()

        def select(new: Array, old: Array) -> Array:
            return jnp.where(fits, new, old)

        edges, reference = jax.tree.map(select, built, (self.edges, self.reference))
        return SkinList(edges, reference, self.cutoffs, self.skin, self.parameters)


@dataclass
class _Capacities[State](BaseLens[State, UniversalNeighborlistParameters]):
    """Lens to a skin list's capacities whose ``set`` pads the stored edges.

    Attributes:
        skin_list: Lens to the list.
    """

    skin_list: Lens[State, SkinList] = field(static=True)

    @override
    def get(self, state: State) -> UniversalNeighborlistParameters:
        return self.skin_list.get(state).parameters

    @override
    def set(self, state: State, value: UniversalNeighborlistParameters) -> State:
        stored = self.skin_list.get(state)
        n = stored.reference.positions.shape[0]
        grown = (value.avg_edges - stored.parameters.avg_edges) * n
        indices = stored.edges.indices
        raw = jnp.pad(indices.raw_indices, ((0, grown), (0, 0)), constant_values=n)
        shifts = jnp.pad(stored.edges.shifts, ((0, grown), (0, 0), (0, 0)))
        edges = Edges(indices.with_raw_indices(raw), shifts)
        return self.skin_list.set(
            state,
            SkinList(edges, stored.reference, stored.cutoffs, stored.skin, value),
        )


class IsSkinListState(IsNeighborListState[IsUniversalNeighborlistParams], Protocol):
    """State with particles, systems and the skin list that pairs them.

    ``neighborlist_params.avg_edges`` sizes the edges read from the list.
    """

    @property
    def particles(self) -> Table[ParticleId, NeighborListPoints]: ...
    @property
    def systems(self) -> Table[SystemId, NeighborListSystems]: ...
    @property
    def skin_list(self) -> SkinList: ...


@dataclass
class SkinListUpdate[State](Propagator[State]):
    """Rebuild a state's skin list if it may miss a pair of the current geometry.

    Run it before every evaluation that reads the list with
    [`SkinNeighborList`][kups.core.neighborlist.verlet.SkinNeighborList].

    Attributes:
        state: Lens to the part of the state with particles, systems and the list.
    """

    state: Lens[State, IsSkinListState] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        sub = self.state.get(state)
        lens = self.state.focus(lambda s: s.skin_list)
        stored = sub.skin_list

        def rebuilt() -> SkinList:
            return stored.rebuilt(state, lens, sub.particles, sub.systems)

        def kept() -> SkinList:
            return stored

        # The stored list is the false branch: XLA hands that operand through,
        # and copies the list at every call when it is the true branch.
        used_up = ~stored.is_complete(sub.particles, sub.systems)
        return lens.set(state, jax.lax.cond(used_up, rebuilt, kept))


@dataclass
class _FollowWraps:
    """Stored pairs, with shifts that follow the wraps of their particles.

    A particle at ``x`` is the build's particle at ``x + w h``, ``w`` integer:
    its motion since the build stays below half a cell while the list is
    complete. A pair's shift changes by the difference of its particles' ``w``.

    Attributes:
        skin_list: The stored list.
    """

    skin_list: SkinList

    def __call__(self, ctx: PipelineContext) -> CandidateBatch[Literal[2]]:
        stored = self.skin_list.edges
        reference = self.skin_list.reference
        indices = stored.indices.update_labels(ctx.edge_keys)
        system = ctx.keys.data.system
        built = Table(ctx.systems.keys, reference.cell.frame.materialize())[system]
        wraps = ctx.systems.data.cell.minimum_image_shifts(
            built.to_fractional(reference.positions) - ctx.keys.data.positions
        )
        w = Table(ctx.keys.keys, wraps.astype(stored.shifts.dtype))[indices]
        shifts = stored.shifts + (w[:, 1:] - w[:, :1])
        return CandidateBatch(
            Edges(indices, shifts), jnp.ones((len(stored),), dtype=bool)
        )


@dataclass
class SkinNeighborList:
    """Pairs of a skin list within ``cutoffs`` at the current positions.

    The edges fit ``avg_edges`` per particle, as those of a search. Stored
    edges that already fit keep their order and are masked in place; longer
    ones are compacted.

    Attributes:
        skin_list: The list to read, complete for the particles of each call.
        cutoffs: Cutoffs of the edges this reader returns, at most the list's.
        avg_edges: Capacity of the returned edges per particle.
    """

    skin_list: SkinList
    cutoffs: Table[SystemId, units.Length]
    avg_edges: Capacity[int]

    @staticmethod
    def from_state(
        state: IsSkinListState, cutoffs: Table[SystemId, units.Length]
    ) -> SkinNeighborList:
        """Neighbor list over the skin list of ``state``.

        Args:
            state: State that holds the list and the edge capacity.
            cutoffs: Cutoffs of the edges the reader returns.

        Returns:
            The neighbor list.
        """
        avg_edges = lens(lambda s: s.neighborlist_params.avg_edges, cls=IsSkinListState)
        return SkinNeighborList(
            state.skin_list, cutoffs, LensCapacity(avg_edges.get(state), avg_edges)
        )

    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints],
    ) -> Edges[Literal[2]]: ...
    @overload
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]: ...
    @jit
    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> Edges[Literal[2]]:
        assert queries is None, "A skin list pairs its own particles; no queries."
        cutoffs = Table.broadcast_to(self.cutoffs, systems)
        listed = Table.broadcast_to(self.skin_list.cutoffs, systems)
        runtime_assert(
            jnp.all(cutoffs.data <= listed.data),
            "Cutoffs exceed the skin list's; create it at the largest cutoff.",
        )
        size = keys.size if queried_keys is None else queried_keys.size
        capacity = self.avg_edges.multiply(size)
        fits = len(self.skin_list.edges) <= capacity.size
        pipeline = Pipeline[Literal[2]](
            selector=_FollowWraps(self.skin_list),
            masks=(
                InBoundsMask(),
                DistanceCutoffMask(cutoffs),
                TouchesQueriedKeysMask(),
            ),
            compactor=MaskOnlyCompactor() if fits else ReduceCompactor(capacity),
        )
        return pipeline(keys, systems, queried_keys=queried_keys)
