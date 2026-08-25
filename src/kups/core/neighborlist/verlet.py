# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet neighbor list with a skin.

A Verlet-skin scheme builds one conservative neighbor list at an enlarged
radius ``r_build ≈ cutoff + skin``, stores its edges, and reuses them over many
steps, amortizing the expensive build over the rebuild window. The pieces:

- [`skin_margin`][kups.core.neighborlist.verlet.skin_margin] — the pure
  completeness bound deciding how long the stored list can be trusted
  (derivation in its docstring);
- [`effective_build_radii`][kups.core.neighborlist.verlet.effective_build_radii]
  — the single-image clamp on the build radius;
- [`VerletSkinState`][kups.core.neighborlist.verlet.VerletSkinState] — the
  bookkeeping group carried on the simulation state, seeded from the particle
  and system tables via
  [`VerletSkinState.seed`][kups.core.neighborlist.verlet.VerletSkinState.seed];
- [`skin_neighborlist`][kups.core.neighborlist.verlet.skin_neighborlist] — the
  cheap per-step reuse path, a
  [`NeighborListFactory`][kups.core.neighborlist.types.NeighborListFactory]
  for potential adapters;
- [`VerletSkinPropagator`][kups.core.neighborlist.verlet.VerletSkinPropagator]
  — wraps a dynamics step with the on-device rebuild, the margin trigger, and
  the hard correctness backstop (details in its docstring).
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol, Self, override

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from kups.core.assertion import runtime_assert
from kups.core.capacity import LensCapacity
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.lens import BaseLens, Lens, bind, identity_lens, lens
from kups.core.neighborlist.adaptive import AdaptiveNeighborList
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.refine import RefineCutoffNeighborList
from kups.core.neighborlist.types import (
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.propagator import Propagator
from kups.core.typing import (
    HasPositionsAndSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field


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


def estimate_skin_params(
    particles_per_system: Table[SystemId, Array],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: float,
) -> UniversalNeighborlistParameters:
    """Capacities sized consistently for the ``cutoff + skin`` sphere.

    Re-estimates *all* capacities at the enlarged radius (never hand-scale a
    single one — mismatched candidate/image buffers trigger a pathological
    dense-build path). Estimated at the unclamped radius, so a clamped build
    can only need less.

    Args:
        particles_per_system: Particle count per system.
        systems: System table (cells) the builds will run in.
        cutoffs: True cutoffs (Å) per system.
        skin: Requested skin width (Å).

    Returns:
        Capacity hints for the skin build.
    """
    return UniversalNeighborlistParameters.estimate(
        particles_per_system, systems, cutoffs.map_data(lambda c: c + skin)
    )


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
    system = particles.data.system.indices
    cutoff_values = Table.broadcast_to(cutoffs, systems).data
    deform = reference.cell.inverse_vectors @ cell_now.vectors  # d_now = d_ref @ F
    # u_i = x_i - x_i_ref @ F, min-image wrapped
    co_moved = jnp.einsum("ni,nij->nj", reference.positions, deform[system])
    residual = cell_now[system].wrap(particles.data.positions - co_moved)
    u_max = particles.data.system.max_over(jnp.linalg.norm(residual, axis=-1)).data
    u_max = jnp.maximum(u_max, 0.0)  # empty segments reduce to -inf
    # σ_min(F) from the smallest eigenvalue of the 3x3 Gram matrix F Fᵀ
    # (cheaper than an SVD; the clamp guards eigvalsh's tiny negative noise).
    gram = deform @ jnp.swapaxes(deform, -1, -2)
    sigma_min = jnp.sqrt(jnp.maximum(jnp.linalg.eigvalsh(gram)[..., 0], 0.0))
    r_build = effective_build_radii(cutoff_values, skin, reference.cell)
    consumed = 2.0 * u_max + r_build * jnp.maximum(0.0, 1.0 - sigma_min)
    return Table(systems.keys, SkinMargin(consumed, r_build - cutoff_values))


class BackingNeighborListFactory(Protocol):
    """Constructs the neighbor list that runs the expensive skin build.

    Matches the ``new`` classmethods of the concrete implementations — pass
    e.g. [`AdaptiveNeighborList.new`][kups.core.neighborlist.adaptive.AdaptiveNeighborList.new]
    (the default),
    [`CellListNeighborList.new`][kups.core.neighborlist.cell_list.CellListNeighborList.new],
    or
    [`DenseNearestNeighborList.new`][kups.core.neighborlist.dense.DenseNearestNeighborList.new].
    The lens focuses the *skin* build capacities on ``state`` so the backing
    list shares them and capacity fixes grow them in place.
    """

    def __call__[S](
        self,
        state: S,
        lens: Lens[S, IsUniversalNeighborlistParams],
        cutoffs: Table[SystemId, Array],
    ) -> NeighborList[Literal[2]]: ...


def _fit_edges_like(
    edges: Edges[Literal[2]],
    target: Edges[Literal[2]],
    out_of_bounds_value: int,
) -> Edges[Literal[2]]:
    """Pad (with out-of-bounds rows) or truncate stored edges to ``target``'s shapes.

    A capacity fix grows the static build shape between compilations, leaving
    the stored edges leaf at the old shape; the traced step requires the carried
    edges to match the shape a fresh build would have. Padded rows use the
    out-of-bounds index, which every consumer already discards
    (``InBoundsMask``).

    Args:
        edges: Stored edges to refit.
        target: An ``jax.eval_shape`` result of the build — an ``Edges`` pytree
            with ``ShapeDtypeStruct`` leaves.
        out_of_bounds_value: Index value marking padded rows as invalid
            (the particle-table size).

    Returns:
        ``edges`` with every leaf padded or truncated to ``target``'s shape.
    """

    def fit(leaf: Array, like: jax.ShapeDtypeStruct) -> Array:
        if leaf.shape == like.shape:
            return leaf
        n = like.shape[0]
        if leaf.shape[0] >= n:
            return leaf[:n]
        fill = out_of_bounds_value if jnp.issubdtype(leaf.dtype, jnp.integer) else 0
        pad = jnp.full((n - leaf.shape[0], *leaf.shape[1:]), fill, leaf.dtype)
        return jnp.concatenate([leaf, pad], axis=0)

    return jax.tree.map(fit, edges, target)


def _build_edges[Holder](
    holder: Holder,
    params_lens: Lens[Holder, UniversalNeighborlistParameters],
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: float,
    neighborlist: BackingNeighborListFactory,
) -> Edges[Literal[2]]:
    """One conservative build at the clamped ``cutoff + skin`` radius."""
    radii = effective_build_radii(
        Table.broadcast_to(cutoffs, systems).data, skin, systems.data.cell
    )
    skin_nl = neighborlist(holder, params_lens, Table(systems.keys, radii))
    return skin_nl(particles, systems)


@dataclass
class VerletSkinState:
    """The Verlet-skin bookkeeping a simulation state carries between steps.

    Groups the stored skin list with everything needed to reuse and rebuild
    it, so a state opts into the skin path with a single
    ``verlet_skin: VerletSkinState | None`` field. Seed it from the particle
    and system tables with
    [`seed`][kups.core.neighborlist.verlet.VerletSkinState.seed] before
    constructing the full state.

    Attributes:
        neighborlist_params: Capacity hints for the skin *build* (distinct
            from the state's own ``neighborlist_params``, which size the
            refine output).
        edges: The stored conservative skin list.
        reference: Positions and cell snapshot taken at the last build.
        should_rebuild: Scalar bool; requests an on-device rebuild at the
            start of the next step.
        headroom: Last measured per-system margin headroom (Å), ``(n_sys,)``.
    """

    neighborlist_params: UniversalNeighborlistParameters
    edges: Edges[Literal[2]]
    reference: SkinReference
    should_rebuild: Array
    headroom: Array

    @classmethod
    def seed(
        cls,
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        skin: float,
        params: UniversalNeighborlistParameters | None = None,
        neighborlist: BackingNeighborListFactory = AdaptiveNeighborList.new,
    ) -> Self:
        """Seed the group from the particle and system tables.

        Runs one eager build to give ``edges`` concrete shapes, and refits
        those edges to the capacity-implied shape: on overflow an eager build
        silently outgrows the static ``params`` (its capacity assertions
        cannot surface outside a traced propagator), which would otherwise
        desynchronize the carried edges from the shape a traced rebuild
        produces. For the same reason the seeded content is not trusted —
        ``should_rebuild=True`` schedules an assertion-covered rebuild for the
        first traced step.

        Args:
            particles: Particle table the skin list is built over.
            systems: System table (cells) the build runs in.
            cutoffs: True cutoffs (Å) per system.
            skin: Requested skin width (Å).
            params: Build capacities; estimated from the tables when ``None``.
            neighborlist: Backing implementation performing the build; must
                match the one the enclosing
                [`VerletSkinPropagator`][kups.core.neighborlist.verlet.VerletSkinPropagator]
                uses, so the seeded shapes equal the traced rebuild's.

        Returns:
            A fully populated group, ready to be placed on the state.
        """
        if params is None:
            params = estimate_skin_params(
                particles.data.system.counts, systems, cutoffs, skin
            )
        params_lens = identity_lens(UniversalNeighborlistParameters)

        def build(
            p: Table[ParticleId, NeighborListPoints],
            s: Table[SystemId, NeighborListSystems],
        ) -> Edges[Literal[2]]:
            return _build_edges(params, params_lens, p, s, cutoffs, skin, neighborlist)

        target = jax.eval_shape(build, particles, systems)
        edges = _fit_edges_like(build(particles, systems), target, particles.size)
        # Deep-copied reference: it must not alias the live positions/cell
        # buffers, or a donated jitted step would receive the same buffer twice.
        return cls(
            neighborlist_params=params,
            edges=edges,
            reference=SkinReference(
                jnp.copy(particles.data.positions),
                jax.tree.map(jnp.copy, systems.data.cell),
            ),
            should_rebuild=jnp.array(True),
            headroom=jnp.zeros(systems.size),
        )


class IsVerletState(IsNeighborListState[IsUniversalNeighborlistParams], Protocol):
    """State that carries a Verlet-skin group alongside the standard tables.

    ``verlet_skin`` is ``None`` while the skin path is disabled; the functions
    and the propagator in this module require it to be populated (see
    [`VerletSkinState.seed`][kups.core.neighborlist.verlet.VerletSkinState.seed]).
    """

    @property
    def particles(self) -> Table[ParticleId, NeighborListPoints]: ...
    @property
    def systems(self) -> Table[SystemId, NeighborListSystems]: ...
    @property
    def verlet_skin(self) -> VerletSkinState | None: ...


@dataclass
class _SkinParamsLens[State: IsVerletState](
    BaseLens[State, UniversalNeighborlistParameters]
):
    """Lens to the skin build capacities whose ``set`` also refits the edges.

    [`LensCapacityFix`][kups.core.capacity.LensCapacityFix] writes grown
    capacities through this lens, so the stored edges leaf is re-padded in the
    same repair — the retraced step then sees carried edges whose static shape
    matches a fresh build's. It travels as a static parameter of the capacity
    assertion, so it must stay hashable. The build's output shapes depend only
    on the capacities and table sizes, never on the radii, so the shape probe
    can use dummy cutoffs; it must, however, use the same backing
    ``neighborlist`` as the real build.
    """

    neighborlist: BackingNeighborListFactory = field(static=True)

    @override
    def get(self, state: State) -> UniversalNeighborlistParameters:
        assert state.verlet_skin is not None, "Verlet-skin state is not seeded."
        return state.verlet_skin.neighborlist_params

    @override
    def set(self, state: State, value: UniversalNeighborlistParameters) -> State:
        skin_state = state.verlet_skin
        assert skin_state is not None, "Verlet-skin state is not seeded."
        skin_state = dataclasses.replace(skin_state, neighborlist_params=value)
        state = bind(state).focus(lambda s: s.verlet_skin).set(skin_state)
        dummy_cutoffs = Table(state.systems.keys, jnp.ones(state.systems.size))
        target = jax.eval_shape(
            lambda s: build_skin_edges(s, dummy_cutoffs, 0.0, self.neighborlist),
            state,
        )
        edges = _fit_edges_like(skin_state.edges, target, state.particles.size)
        return (
            bind(state)
            .focus(lambda s: s.verlet_skin)
            .set(dataclasses.replace(skin_state, edges=edges))
        )


def build_skin_edges[State: IsVerletState](
    state: State,
    cutoffs: Table[SystemId, Array],
    skin: float,
    neighborlist: BackingNeighborListFactory = AdaptiveNeighborList.new,
) -> Edges[Literal[2]]:
    """One conservative build at the clamped ``cutoff + skin`` radius.

    Capacities come from ``state.verlet_skin.neighborlist_params`` via
    [`LensCapacity`][kups.core.capacity.LensCapacity], so an overflow inside a
    traced propagator is auto-resized instead of raising; the capacity fix
    also refits ``state.verlet_skin.edges`` to the grown build shape (see
    ``_SkinParamsLens``). Outside a trace the capacity assertions cannot
    surface, which is why
    [`VerletSkinState.seed`][kups.core.neighborlist.verlet.VerletSkinState.seed]
    schedules an assertion-covered rebuild for the first traced step.

    Args:
        state: State carrying a seeded ``verlet_skin`` group.
        cutoffs: True cutoffs (Å) per system.
        skin: Requested skin width (Å).
        neighborlist: Backing implementation performing the build.

    Returns:
        Conservative skin edges for the current positions and cells.
    """
    assert state.verlet_skin is not None, "Verlet-skin state is not seeded."
    return _build_edges(
        state,
        _SkinParamsLens[State](neighborlist),
        state.particles,
        state.systems,
        cutoffs,
        skin,
        neighborlist,
    )


def skin_neighborlist[State: IsVerletState](
    state: State, cutoffs: Table[SystemId, Array]
) -> RefineCutoffNeighborList:
    """The cheap per-step reuse path: refine stored skin edges to the true cutoff.

    A [`NeighborListFactory`][kups.core.neighborlist.types.NeighborListFactory]:
    pass it as ``neighborlist_factory=`` to a potential adapter so every force
    eval reads ``state.verlet_skin.edges`` instead of rebuilding. The
    refinement re-masks to ``cutoffs`` and recomputes minimum-image shifts for
    the current (possibly deformed) cell.

    Args:
        state: State carrying a seeded ``verlet_skin`` group.
        cutoffs: True cutoffs (Å) per system to re-mask to.

    Returns:
        A [`RefineCutoffNeighborList`][kups.core.neighborlist.refine.RefineCutoffNeighborList]
        over the stored skin edges, its output capacity lens-backed by the
        state's own ``neighborlist_params``.
    """
    assert state.verlet_skin is not None, "Verlet-skin state is not seeded."
    return RefineCutoffNeighborList(
        candidates=state.verlet_skin.edges,
        avg_edges=LensCapacity(
            state.neighborlist_params.avg_edges,
            lens(lambda s: s.neighborlist_params).focus(lambda p: p.avg_edges),
        ),
        cutoffs=cutoffs,
    )


def _request_rebuild[State: IsVerletState](state: State, deficit: Array) -> State:
    """Fix for an exhausted margin: request a rebuild and let the block replay.

    The rebuild itself must run in-trace (under the ``lax.cond`` in
    [`VerletSkinPropagator`][kups.core.neighborlist.verlet.VerletSkinPropagator])
    where its capacity assertions surface, so the fix only flips the request
    bit. A rebuild resets the consumed margin to zero, so the replay succeeds
    unless a *single step* consumed more than some system's whole budget —
    then no rebuild schedule can help and the configuration itself is at
    fault. ``deficit`` (single-step consumption minus budget, per system)
    combines coherently under the fused loop's elementwise-maximum merge of
    fix args.
    """
    if bool((deficit > 0.0).any()):
        worst = int(jnp.argmax(deficit))
        raise ValueError(
            "verlet_skin cannot absorb a single step of motion: one step consumed "
            f"{float(deficit[worst]):.3g} Å more than system {worst}'s "
            "whole margin budget. Increase verlet_skin, reduce the time step, or "
            "set verlet_skin = 0."
        )
    skin_state = state.verlet_skin
    assert skin_state is not None, "Verlet-skin state is not seeded."
    return (
        bind(state)
        .focus(lambda s: s.verlet_skin)
        .set(dataclasses.replace(skin_state, should_rebuild=jnp.array(True)))
    )


@dataclass
class VerletSkinPropagator[State: IsVerletState]:
    """One dynamics step under Verlet-skin neighbor-list maintenance.

    Wraps ``step`` — whose potential must be built with
    ``neighborlist_factory=``[`skin_neighborlist`][kups.core.neighborlist.verlet.skin_neighborlist]
    — and runs, in order: (1) an on-device rebuild of the stored list behind
    ``lax.cond`` when ``should_rebuild`` is set, so the same step's force eval
    sees the fresh list and blocked stepping stays fused; (2) the wrapped
    ``step``; (3) the margin trigger and hard backstop below.

    ## Rebuild scheduling and the hard backstop

    The end-of-step trigger sets ``should_rebuild`` by extrapolating one step
    of margin consumption per system, so the *next* step rebuilds before the
    margin runs out. It also hard-asserts that every system's headroom is
    still non-negative — i.e. this step's force evaluation could not have
    missed a pair. A step that outruns the margin is reverted by the enclosing
    [`ResetOnErrorPropagator`][kups.core.propagator.ResetOnErrorPropagator];
    the exhausted margin is a function of the reverted state, so subsequent
    fused iterations fail and revert the same way and the block's output
    stalls at the last valid step while the failure stays recorded.
    [`propagate_and_fix`][kups.core.propagator.propagate_and_fix] then applies
    the assertion's fix — flip ``should_rebuild`` — and re-dispatches, so
    stepping resumes from the last valid state with a rebuild. (A fused
    iteration after the failure that satisfies its own margin — e.g. under
    fresh thermostat noise — commits soundly: the assertion *is* the validity
    condition for its step.) The trigger formula is thereby a performance
    heuristic, not a correctness proof: over-firing costs a rebuild,
    under-firing costs re-running a few steps. Only a single step that
    consumes more than a system's whole budget — unabsorbable by any rebuild
    schedule — escalates, with a configuration hint. The measurement point
    (end of step) trails the force evaluation by at most the intra-step
    position update, the standard LAMMPS-style slack the extrapolation
    absorbs.

    The rebuild request travels as the ``should_rebuild`` bit — rather than
    the fix assigning a fresh edge list directly — because the build must run
    *in-trace*: only there do its capacity assertions surface and auto-grow
    the buffers, while an eager build inside the fix would outgrow the static
    params without recording the growth, and refitting its output back to the
    params-implied shape would silently drop edges.

    Attributes:
        step: The wrapped dynamics step (e.g. one MD or relaxation step).
        cutoffs: True cutoffs (Å) per system.
        skin: Requested skin width (Å).
        neighborlist: Backing implementation performing the expensive build.
    """

    step: Propagator[State] = field(static=True)
    cutoffs: Table[SystemId, Array]
    skin: float = field(static=True)
    neighborlist: BackingNeighborListFactory = field(
        static=True, default=AdaptiveNeighborList.new
    )

    def __call__(self, key: Array, state: State) -> State:
        """Advance one step, maintaining the stored skin list.

        ``key`` is passed to the wrapped ``step`` unchanged — the skin
        machinery consumes no randomness — so trajectories match a skinless
        run key for key.

        Args:
            key: JAX PRNG key for the wrapped step.
            state: State carrying a seeded ``verlet_skin`` group.

        Returns:
            The stepped state with updated skin bookkeeping.
        """
        skin_state = state.verlet_skin
        assert skin_state is not None, "Verlet-skin state is not seeded."
        state = jax.lax.cond(
            skin_state.should_rebuild, self._rebuild, lambda s: s, state
        )
        return self._trigger(self.step(key, state))

    def _rebuild(self, state: State) -> State:
        """Fresh build: resets the reference geometry and the full headroom.

        The build radius is clamped to the single-image limit
        ([`effective_build_radii`][kups.core.neighborlist.verlet.effective_build_radii]);
        the assertion fires only when the *cutoff itself* exceeds the limit,
        i.e. when no skin at all would be representable.
        """
        skin_state = state.verlet_skin
        assert skin_state is not None, "Verlet-skin state is not seeded."
        cell = state.systems.data.cell
        cutoffs = Table.broadcast_to(self.cutoffs, state.systems).data
        radii = effective_build_radii(cutoffs, self.skin, cell)
        runtime_assert(
            (radii > cutoffs).all(),
            "Verlet-skin reuse needs the cutoff below half the cell's smallest "
            "perpendicular length on every periodic axis, else the refine path "
            "drops periodic images (build radius {radii}, cutoffs {cutoffs}). "
            "Enlarge the cell or set verlet_skin = 0.",
            fmt_args={"radii": radii, "cutoffs": cutoffs},
        )
        edges = build_skin_edges(state, self.cutoffs, self.skin, self.neighborlist)
        return (
            bind(state)
            .focus(lambda s: s.verlet_skin)
            .set(
                dataclasses.replace(
                    skin_state,
                    edges=edges,
                    reference=SkinReference(state.particles.data.positions, cell),
                    should_rebuild=jnp.array(False),
                    headroom=radii - cutoffs,
                )
            )
        )

    def _trigger(self, state: State) -> State:
        """End-of-step margin measurement: backstop assert + rebuild schedule.

        Cheap — O(N) residuals plus a per-system 3×3 spectral bound, no
        neighbor build.
        """
        skin_state = state.verlet_skin
        assert skin_state is not None, "Verlet-skin state is not seeded."
        margin = skin_margin(
            state.particles,
            state.systems,
            skin_state.reference,
            self.cutoffs,
            self.skin,
        ).data
        headroom = margin.headroom
        single_step = jnp.maximum(skin_state.headroom - headroom, 0.0)
        runtime_assert(
            (headroom >= 0.0).all(),
            "Verlet-skin margin exhausted mid-step (headroom {headroom} Å): the "
            "stored neighbor list may be incomplete for this step's forces.",
            fmt_args={"headroom": headroom},
            fix_fn=_request_rebuild,
            fix_args=single_step - margin.budget,
        )
        flag = (headroom - single_step < 0.0).any()
        return (
            bind(state)
            .focus(lambda s: s.verlet_skin)
            .set(
                dataclasses.replace(skin_state, should_rebuild=flag, headroom=headroom)
            )
        )
