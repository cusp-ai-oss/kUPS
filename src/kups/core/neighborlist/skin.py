# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet skin list: neighbor pairs kept on the state across steps.

[`SkinList`][kups.core.neighborlist.skin.SkinList] holds the pairs within
``cutoff + skin`` of its last build as neighbor rows,
[`SkinNeighborList`][kups.core.neighborlist.skin.SkinNeighborList] gives a potential
those within its own cutoff as edges, and
[`SkinListPropagator`][kups.core.neighborlist.skin.SkinListPropagator] rebuilds the
list once [`skin_margin`][kups.core.neighborlist.verlet.skin_margin] says a missing
pair could be within the cutoff.
"""

from __future__ import annotations

from typing import Literal, Protocol, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import check_assertions, runtime_assert
from kups.core.capacity import Capacity, LensCapacity
from kups.core.data import Index, Table
from kups.core.lens import LambdaLens, Lens, View, identity_lens, lens
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
    NeighborListBuilder,
    NeighborListPoints,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.neighborlist.verlet import SkinReference, skin_margin
from kups.core.propagator import Propagator
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import (
    dataclass,
    field,
    jit,
    no_jax_tracing,
    tree_copy,
    tree_where_broadcast_last,
)


@dataclass
class SkinList:
    """Pairs within ``cutoffs + skin`` of the geometry they were built at.

    Inclusion and exclusion are applied at the build, where an excluded pair drops
    only its minimum image: exact for one-particle exclusion groups, or for cells
    wider than ``2 (cutoffs + skin)``.

    Attributes:
        rows: Pairs and their image shifts at the build, one row per particle.
        reference: Positions and cells at the build.
        cutoffs: Cutoffs (Å) the list serves.
        skin: Distance (Å) added to ``cutoffs`` at each build.
        params: Capacities of the build.
        builder: Builds the pairs, e.g. ``CellListNeighborList.new``.
    """

    rows: NeighborRows
    reference: SkinReference
    cutoffs: Table[SystemId, Array]
    skin: Table[SystemId, Array]
    params: UniversalNeighborlistParameters
    builder: NeighborListBuilder = field(static=True)

    @staticmethod
    @no_jax_tracing
    def create(
        particles: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        skin: Table[SystemId, Array],
        *,
        builder: NeighborListBuilder = AdaptiveNeighborList.new,
    ) -> SkinList:
        """Build a skin list from the particle and system tables.

        Capacities start from the estimate at ``cutoffs + skin`` and grow until the
        build fits.

        Args:
            particles: Particles to pair.
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


class IsSkinListState(IsNeighborListState[IsUniversalNeighborlistParams], Protocol):
    """State with a skin list; ``neighborlist_params.avg_edges`` sizes the pairs
    given to potentials."""

    @property
    def skin_list(self) -> SkinList: ...


@dataclass
class SkinNeighborList:
    """Pairs of a skin list within ``cutoffs`` at the current geometry, as edges.

    Stored shifts follow each particle's periodic wraps since the build, so cells
    narrower than twice the cutoff work too, within the int8 limit of
    ``NeighborRows``. Inclusion and exclusion are those of the build (``SkinList``).

    Attributes:
        skin_list: The stored list; its cutoffs must cover ``cutoffs``.
        avg_edges: Capacity of the returned edges per particle.
        cutoffs: Cutoffs (Å) per system.
    """

    skin_list: SkinList
    avg_edges: Capacity[int]
    cutoffs: Table[SystemId, Array]

    @classmethod
    def from_state(
        cls, state: IsSkinListState, cutoffs: Table[SystemId, Array]
    ) -> SkinNeighborList:
        """Neighbor list over ``state.skin_list``, sized by ``state.neighborlist_params``.

        Args:
            state: State with a skin list and neighbor-list capacities.
            cutoffs: Cutoffs (Å) of the potential.

        Returns:
            The neighbor list.
        """
        avg_edges = LensCapacity(
            state.neighborlist_params.avg_edges,
            lens(lambda s: s.neighborlist_params.avg_edges),
        )
        return SkinNeighborList(state.skin_list, avg_edges, cutoffs)

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
        pipeline = Pipeline[Literal[2]](
            selector=_FollowWraps(self.skin_list.rows, self.skin_list.reference),
            masks=(
                InBoundsMask(),
                DistanceCutoffMask(cutoffs=cutoffs),
                TouchesQueriedKeysMask(),
            ),
            compactor=ReduceCompactor(avg_edges=self.avg_edges.multiply(size)),
        )
        return pipeline(keys, systems, queried_keys=queried_keys)


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
        # particles at both ends of each slot; padding (row n) stays out of bounds
        particle = jnp.append(rows.order, len(rows.order))
        ends = particle[jnp.stack([rows.centers, rows.neighbors], axis=-1)]
        dtype = keys.system.raw_indices.dtype
        indices = Index(ctx.edge_keys, ends.astype(dtype), _cls=ctx.keys.cls)
        w = wraps[indices]
        shifts = rows.shifts[:, None].astype(keys.positions.dtype)
        shifts += w[:, 1:] - w[:, :1]
        # inclusion and exclusion were applied at the build, so no mask reads the
        # image flags
        return CandidateBatch(Edges(indices, shifts), jnp.ones(len(ends), bool))


@dataclass
class SkinListPropagator[State](Propagator[State]):
    """Rebuild the whole list once any system has used up its skin.

    Run it before each force evaluation. A rebuild that outgrows its capacities keeps
    the stored list; the capacity fix then rebuilds it on the host.

    Attributes:
        skin_list: Lens to the list.
        particles: View of the particles it pairs.
        systems: View of their systems.
    """

    skin_list: Lens[State, SkinList] = field(static=True)
    particles: View[State, Table[ParticleId, NeighborListPoints]] = field(static=True)
    systems: View[State, Table[SystemId, NeighborListSystems]] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        stored = self.skin_list.get(state)
        margin = skin_margin(
            self.particles(state),
            self.systems(state),
            stored.reference,
            stored.cutoffs,
            stored.skin,
        )
        built = jax.lax.cond(
            jnp.any(margin.data.headroom < 0),
            lambda: self._rebuilt(state),
            lambda: stored,
        )
        return self.skin_list.set(state, built)

    def _rebuilt(self, state: State) -> SkinList:
        stored = self.skin_list.get(state)
        # the capacity fix sets these through _regrown, which rebuilds on the host
        params = LambdaLens(self._params, self._regrown)
        built = _build(
            state,
            params,
            self.particles(state),
            self.systems(state),
            stored.cutoffs,
            stored.skin,
            stored.builder,
        )
        # a build that outgrew its capacities keeps the stored list
        return tree_where_broadcast_last(check_assertions(), built, stored)

    def _params(self, state: State) -> UniversalNeighborlistParameters:
        return self.skin_list.get(state).params

    def _regrown(self, state: State, value: UniversalNeighborlistParameters) -> State:
        stored = self.skin_list.get(state)
        if value == stored.params:
            return state
        built = _build_on_host(
            value,
            self.particles(state),
            self.systems(state),
            stored.cutoffs,
            stored.skin,
            stored.builder,
        )
        return self.skin_list.set(state, built)


def _build[S](
    state: S,
    params: Lens[S, UniversalNeighborlistParameters],
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: Table[SystemId, Array],
    builder: NeighborListBuilder,
) -> SkinList:
    # stored materialized, so steps do not re-evaluate a parametrized frame
    cell = systems.data.cell.materialize()
    widths = jnp.where(jnp.asarray(cell.periodic), cell.perpendicular_lengths, jnp.inf)
    # wraps are counted by rounding, so motion between builds must stay below half
    # a cell; rebuilds keep it below half the skin
    runtime_assert(
        jnp.all(widths > skin.data[:, None]),
        "Skin list needs cells wider than the skin.",
    )
    edges = builder(state, params, cutoffs + skin)(particles, systems)
    # copies: a donated state must not hold one buffer twice
    reference = SkinReference(*tree_copy((particles.data.positions, cell)))
    return SkinList(
        neighbor_rows(edges, particles, systems),
        reference,
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
    builder: NeighborListBuilder,
) -> SkinList:
    lens = identity_lens(UniversalNeighborlistParameters)
    return _build(params, lens, particles, systems, cutoffs, skin, builder)


def _build_on_host(
    params: UniversalNeighborlistParameters,
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: Table[SystemId, Array],
    builder: NeighborListBuilder,
) -> SkinList:
    """Build eagerly, growing ``params`` until the build fits."""
    while True:
        result = _build_eagerly(
            params, particles, systems, cutoffs, skin, builder=builder
        )
        if not result.failed_assertions:
            return result.value
        params = result.fix_or_raise(params)
