# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet neighbor list with a skin.

A Verlet-skin scheme builds one conservative neighbor list at an enlarged
radius ``r_build ≈ cutoff + skin``, stores its edges, and reuses them over many
steps via
[`RefineCutoffNeighborList`][kups.core.neighborlist.refine.RefineCutoffNeighborList],
amortizing the expensive build over the rebuild window. This module holds the
pure geometry underneath such a scheme
([`skin_margin`][kups.core.neighborlist.verlet.skin_margin] decides how long
the stored list remains complete,
[`effective_build_radii`][kups.core.neighborlist.verlet.effective_build_radii]
keeps the build radius inside the single-image regime that edge reuse requires)
plus the build, reuse and rebuild machinery on top of it.

## Rebuild scheduling and the hard backstop

[`RebuildSkinStep`][kups.core.neighborlist.verlet.RebuildSkinStep] rebuilds
on-device behind ``lax.cond`` on ``state.should_rebuild``, so it fuses into
blocked stepping. [`TriggerStep`][kups.core.neighborlist.verlet.TriggerStep]
runs at the end of every MD step: it sets the flag by extrapolating one step of
margin consumption per system (so the *next* step rebuilds before the margin
runs out), and it hard-asserts that every system's headroom is still
non-negative — i.e. this step's force evaluation could not have missed a pair.
A step that outruns the margin is reverted by
[`ResetOnErrorPropagator`][kups.core.propagator.ResetOnErrorPropagator]; the
exhausted margin is a function of the reverted state, so subsequent fused
iterations fail and revert the same way and the block's output stalls at the
last valid step while the failure stays recorded.
[`propagate_and_fix`][kups.core.propagator.propagate_and_fix] then applies the
assertion's fix — flip ``should_rebuild`` — and re-dispatches, so stepping
resumes from the last valid state with a rebuild. (A fused iteration after the
failure that satisfies its own margin — e.g. under fresh thermostat noise —
commits soundly: the assertion *is* the validity condition for its step.) The
trigger formula is thereby a performance heuristic, not a correctness proof:
over-firing costs a rebuild, under-firing costs re-running a few steps. Only a
single step that consumes more than a system's whole budget — unabsorbable by
any rebuild schedule — escalates, with a configuration hint. The measurement
point (end of step) trails the force evaluation by at most the intra-step
position update, the standard LAMMPS-style slack the extrapolation absorbs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Protocol, override

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from kups.core.assertion import runtime_assert
from kups.core.capacity import LensCapacity
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.lens import BaseLens, bind, lens
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.refine import RefineCutoffNeighborList
from kups.core.neighborlist.types import (
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborListSystems,
)
from kups.core.typing import (
    HasPositionsAndSystemIndex,
    IsState,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field


class IsVerletState(
    IsState[Any, Any], IsNeighborListState[IsUniversalNeighborlistParams], Protocol
):
    """State carrying the stored skin list and its rebuild bookkeeping.

    All Verlet fields are ``None`` while the skin path is disabled; the
    functions and steps in this module require them to be populated.
    """

    @property
    def skin_neighborlist_params(self) -> IsUniversalNeighborlistParams | None: ...
    @property
    def stored_skin_edges(self) -> Edges[Literal[2]] | None: ...
    @property
    def reference_positions(self) -> Array | None: ...
    @property
    def reference_cell(self) -> Cell[AnyPeriodicity] | None: ...
    @property
    def should_rebuild(self) -> Array | None: ...
    @property
    def skin_headroom(self) -> Array | None: ...


def _skin_cutoffs(
    cutoffs: Table[SystemId, Array], skin: float
) -> Table[SystemId, Array]:
    """Per-system ``cutoff + skin`` table (the unclamped build radius)."""
    return cutoffs.map_data(lambda c: c + skin)


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

    Re-estimates *all* capacities at the enlarged radius (never hand-scale a single
    one — mismatched candidate/image buffers trigger a pathological dense-build path).
    Estimated at the unclamped radius, so a clamped build can only need less.
    """
    return UniversalNeighborlistParameters.estimate(
        particles_per_system, systems, _skin_cutoffs(cutoffs, skin)
    )


def _fit_edges_like(
    edges: Edges[Literal[2]], target: Any, invalid_index: int
) -> Edges[Literal[2]]:
    """Pad (with invalid rows) or truncate stored edges to ``target``'s shapes.

    A capacity fix grows the static build shape between compilations, leaving
    the stored edges leaf at the old shape; the traced step requires the carried
    edges to match the shape a fresh build would have. Padded rows use an
    out-of-bounds index, which every consumer already discards
    (``InBoundsMask``).
    """

    def fit(leaf: Array, like: jax.ShapeDtypeStruct) -> Array:
        if leaf.shape == like.shape:
            return leaf
        n = like.shape[0]
        if leaf.shape[0] >= n:
            return leaf[:n]
        fill = invalid_index if jnp.issubdtype(leaf.dtype, jnp.integer) else 0
        pad = jnp.full((n - leaf.shape[0], *leaf.shape[1:]), fill, leaf.dtype)
        return jnp.concatenate([leaf, pad], axis=0)

    return jax.tree.map(fit, edges, target)


@dataclass
class _SkinParamsLens[State: IsVerletState](BaseLens[State, Any]):
    """Lens to ``skin_neighborlist_params`` whose ``set`` also refits the stored
    edges to the new build shape.

    [`LensCapacityFix`][kups.core.capacity.LensCapacityFix] writes grown
    capacities through this lens, so the stored edges leaf is re-padded in the
    same repair — the retraced step then sees carried edges whose static shape
    matches a fresh build's. Fieldless: it travels as a static parameter of the
    capacity assertion, so it must stay hashable. The build's output shapes
    depend only on the capacities and table sizes, never on the radii, so the
    shape probe can use dummy cutoffs.
    """

    @override
    def get(self, state: State) -> Any:
        return state.skin_neighborlist_params

    @override
    def set(self, state: State, value: Any) -> State:
        state = bind(state).focus(lambda s: s.skin_neighborlist_params).set(value)
        if state.stored_skin_edges is None:
            return state
        dummy_cutoffs = Table(state.systems.keys, jnp.ones(len(state.systems.keys)))
        target = jax.eval_shape(lambda s: build_skin_edges(s, dummy_cutoffs, 0.0), state)
        return (
            bind(state)
            .focus(lambda s: s.stored_skin_edges)
            .set(_fit_edges_like(state.stored_skin_edges, target, state.particles.size))
        )


def build_skin_edges[State: IsVerletState](
    state: State, cutoffs: Table[SystemId, Array], skin: float
) -> Edges[Literal[2]]:
    """One conservative dense build at the clamped ``cutoff + skin`` radius.

    Capacities come from ``state.skin_neighborlist_params`` via
    [`LensCapacity`][kups.core.capacity.LensCapacity], so an overflow inside a
    traced propagator is auto-resized instead of raising; the capacity fix also
    refits ``state.stored_skin_edges`` to the grown build shape (see
    ``_SkinParamsLens``). Outside a traced propagator the capacity assertions do
    not surface — callers seeding a state eagerly must set
    ``should_rebuild=True`` so the first traced step rebuilds under assertion
    coverage. Used by
    [`RebuildSkinStep`][kups.core.neighborlist.verlet.RebuildSkinStep].
    """
    cell = state.systems.data.cell
    radii = effective_build_radii(
        Table.broadcast_to(cutoffs, state.systems).data, skin, cell
    )
    skin_nl = DenseNearestNeighborList.new(
        state,
        _SkinParamsLens[State](),
        Table(state.systems.keys, radii),
    )
    return skin_nl(state.particles, state.systems)


def skin_neighborlist[State: IsVerletState](
    state: State, cutoffs: Table[SystemId, Array]
) -> RefineCutoffNeighborList:
    """The cheap per-step reuse path: refine stored skin edges to the true cutoff.

    A [`NeighborListFactory`][kups.core.neighborlist.types.NeighborListFactory]:
    pass it as ``neighborlist_factory=`` to a potential adapter so every force
    eval reads ``state.stored_skin_edges`` instead of rebuilding. The refinement
    re-masks to ``cutoffs`` and recomputes minimum-image shifts for the current
    (possibly deformed) cell.
    """
    assert state.stored_skin_edges is not None, "Verlet skin state is not initialized."
    return RefineCutoffNeighborList(
        candidates=state.stored_skin_edges,
        avg_edges=LensCapacity(
            state.neighborlist_params.avg_edges,
            lens(lambda s: s.neighborlist_params).focus(lambda p: p.avg_edges),
        ),
        cutoffs=cutoffs,
    )


def seed_verlet_state[State: IsVerletState](
    state: State,
    cutoffs: Table[SystemId, Array],
    skin: float,
    params: IsUniversalNeighborlistParams | None = None,
) -> State:
    """Populate every Verlet-skin field on ``state``.

    Takes the build capacities from ``params`` if given, else from capacities
    already on the state, else estimates them. Runs one eager build to give
    ``stored_skin_edges`` concrete shapes, and refits those edges
    to the capacity-implied shape: on overflow an eager build silently outgrows
    the static params (its capacity assertions cannot surface outside a traced
    propagator), which would otherwise desynchronize the carried edges from the
    shape a traced rebuild produces. For the same reason the seeded content is
    not trusted — ``should_rebuild=True`` schedules an assertion-covered
    rebuild for the first traced step.

    Reference fields are deep copies: they must not alias the live
    positions/cell buffers, or the donated jitted step would receive the same
    buffer twice.
    """
    if params is None:
        params = state.skin_neighborlist_params
    if params is None:
        counts = state.particles.data.system.counts
        params = estimate_skin_params(
            Table(state.systems.keys, counts.data), state.systems, cutoffs, skin
        )
    state = dataclasses.replace(  # pyrefly: ignore [bad-specialization]
        state,
        skin_neighborlist_params=params,
        reference_positions=jnp.copy(state.particles.data.positions),
        reference_cell=jax.tree.map(jnp.copy, state.systems.data.cell),
        should_rebuild=jnp.array(True),
        skin_headroom=jnp.zeros(len(state.systems.keys)),
    )
    state = dataclasses.replace(  # pyrefly: ignore [bad-specialization]
        state,
        stored_skin_edges=build_skin_edges(state, cutoffs, skin),
    )
    return _SkinParamsLens[State]().set(state, params)


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


def _flip_rebuild_flag[State: IsVerletState](
    state: State, args: dict[str, Array]
) -> State:
    """Fix for an exhausted margin: request a rebuild and let the block replay.

    A rebuild resets the consumed margin to zero, so the replay succeeds unless
    a *single step* consumed more than some system's whole budget — then no
    rebuild schedule can help and the configuration itself is at fault.
    ``deficit`` (single-step consumption minus budget, per system) combines
    coherently under the fused loop's elementwise-maximum merge of fix args.
    """
    if bool((args["deficit"] > 0.0).any()):
        worst = int(jnp.argmax(args["deficit"]))
        raise ValueError(
            "verlet_skin cannot absorb a single step of motion: one step consumed "
            f"{float(args['deficit'][worst]):.3g} Å more than system {worst}'s "
            "whole margin budget. Increase verlet_skin, reduce the time step, or "
            "set verlet_skin = 0."
        )
    return bind(state).focus(lambda s: s.should_rebuild).set(jnp.array(True))


@dataclass
class RebuildSkinStep[State: IsVerletState]:
    """Propagator step that rebuilds the stored skin list when
    ``state.should_rebuild`` is set (identity otherwise), via ``lax.cond`` so it
    stays on device and fuses into blocked stepping. Run at step start, so the
    same step's force eval sees the fresh list.

    The build radius is clamped to the single-image limit
    (:func:`effective_build_radii`); the rebuild only raises when the *cutoff*
    itself exceeds the limit, i.e. no skin at all would be representable.
    """

    cutoffs: Table[SystemId, Array]
    skin: float = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        assert state.stored_skin_edges is not None, (
            "Verlet skin state is not initialized."
        )

        def rebuild(state: State) -> State:
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
            edges = build_skin_edges(state, self.cutoffs, self.skin)
            return (
                bind(state)
                .focus(
                    lambda s: (
                        s.stored_skin_edges,
                        s.reference_positions,
                        s.reference_cell,
                        s.should_rebuild,
                        s.skin_headroom,
                    )
                )
                .set(
                    (
                        edges,
                        state.particles.data.positions,
                        cell,
                        jnp.array(False),
                        radii - cutoffs,
                    )
                )
            )

        return jax.lax.cond(state.should_rebuild, rebuild, lambda s: s, state)


@dataclass
class TriggerStep[State: IsVerletState]:
    """Propagator step that maintains the rebuild schedule and its backstop.

    Runs at the end of every MD step (cheap: O(N) residuals plus a per-system
    3×3 spectral bound, no neighbor build). It

    1. hard-asserts every system's margin headroom is still non-negative —
       this step's force eval could not have missed a pair. On failure the step
       is reverted by the enclosing
       [`ResetOnErrorPropagator`][kups.core.propagator.ResetOnErrorPropagator]
       and the attached fix flips ``should_rebuild`` so
       [`propagate_and_fix`][kups.core.propagator.propagate_and_fix] re-runs
       with a rebuild (see the module docstring);
    2. sets ``should_rebuild`` for the *next* step by extrapolating one step of
       margin consumption from ``state.skin_headroom``, per system;
    3. records the new per-system headroom.
    """

    cutoffs: Table[SystemId, Array]
    skin: float = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        del key
        assert state.reference_positions is not None
        assert state.reference_cell is not None
        assert state.skin_headroom is not None
        margin = skin_margin(
            state.particles,
            state.systems,
            SkinReference(state.reference_positions, state.reference_cell),
            self.cutoffs,
            self.skin,
        ).data
        headroom = margin.headroom
        single_step = jnp.maximum(state.skin_headroom - headroom, 0.0)
        runtime_assert(
            (headroom >= 0.0).all(),
            "Verlet-skin margin exhausted mid-step (headroom {headroom} Å): the "
            "stored neighbor list may be incomplete for this step's forces.",
            fmt_args={"headroom": headroom},
            fix_fn=_flip_rebuild_flag,
            fix_args={"deficit": single_step - margin.budget},
        )
        flag = (headroom - single_step < 0.0).any()
        return (
            bind(state)
            .focus(lambda s: (s.should_rebuild, s.skin_headroom))
            .set((flag, headroom))
        )
