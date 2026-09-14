# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Chunked pair evaluation using the neighbor pipeline and standard sum composers.

Add ``PairEnergy`` terms before constructing ``FusedNeighborEnergy``. The sum
shares one spatial table and geometry pass, while each term retains its own
cutoff and inclusion/exclusion policy. ``PotentialFromEnergy`` provides the
same differentiation, compensated caching, and patches as graph potentials.
Periodic images expand within each chunk using the neighbor-list image and
exclusion rules.
"""

from __future__ import annotations

from dataclasses import replace
from operator import itemgetter
from typing import Any, Callable, Self

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table, WithIndices
from kups.core.lens import Lens, View, lens
from kups.core.neighborlist.cell_table import (
    CellRows,
    CellTable,
    CellTableParameters,
    CellTableUpdatePatch,
    build_cell_table,
    cell_candidates,
)
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY,
    EMPTY_LENS,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    SummedPotential,
    empty_patch_idx_view,
    sum_potentials,
)
from kups.core.typing import HasCell, ParticleId, SystemId
from kups.core.utils.jax import dataclass, field, jit, tree_map
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.energy import (
    FullSumComposer,
    InputConstructor,
    LocalSumComposer,
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    POINTCLOUD_GEOMETRY,
    GraphInputConstructor,
    GraphPairEnergy,
    IsRadiusGraphPoints,
    PointCloud,
)
from kups.potential.common.pair import PairBatch, PairData, PairEnergySum, PairTerm


def sum_chunks[Rows](
    consume: Callable[[Rows], Array],
    rows: Rows,
    chunk_size: int,
    *,
    active: Callable[[Rows], Array] | None = None,
) -> Array:
    """Sum a consumer over equally sized chunks of an already padded pytree.

    Only the reduced output crosses chunk boundaries; reverse mode recomputes
    each chunk instead of retaining its intermediates.

    If ``active`` returns False for a whole chunk, its contribution is zero
    and the consumer is skipped. It must return a scalar boolean.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    # Reuse the consumer trace for shape discovery and chunk evaluation.
    consume = jax.jit(consume)
    chunks = jax.tree.map(
        lambda x: x.reshape(x.shape[0] // chunk_size, chunk_size, *x.shape[1:]), rows
    )

    sample = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct((chunk_size, *x.shape[1:]), x.dtype), rows
    )
    with jax.disable_jit(False):
        output = jax.eval_shape(consume, sample)
    initial = jnp.zeros(output.shape, output.dtype)
    if jax.tree.leaves(rows)[0].shape[0] == 0:
        return initial

    def evaluate(chunk: Rows) -> Array:
        with jax.disable_jit(False):
            if active is None:
                return consume(chunk)
            return jax.lax.cond(active(chunk), consume, lambda _: initial, chunk)

    if jax.tree.leaves(rows)[0].shape[0] == chunk_size:
        return evaluate(rows)

    @jax.checkpoint
    def body(total: Array, chunk: Rows) -> tuple[Array, None]:
        return total + evaluate(chunk), None

    # MCMC temporarily disables jit to preserve identities when composing
    # patches. That also makes scan expand into a Python loop during tracing,
    # duplicating the pair kernel for every chunk. Keep this traversal rolled
    # regardless of the caller's identity-preserving context.
    with jax.disable_jit(False):
        total, _ = jax.lax.scan(body, initial, chunks)
    return total


@dataclass
class FusedPotentialInput[Params, Part: IsRadiusGraphPoints, Feat]:
    """Parameters and geometry for a full fused evaluation."""

    parameters: Params
    cloud: PointCloud[Part, HasCell[AnyPeriodicity]]


@dataclass
class FusedLocalInput[Params, Part: IsRadiusGraphPoints, Feat](
    FusedPotentialInput[Params, Part, Feat]
):
    """Incident pairs against an initialized table, excluding ``removed``.

    One query table evaluates its incident energy. Two tables evaluate new
    minus old, including intra-query pairs once and no old-new cross pairs.
    Query vocabularies may differ from the cloud's; they are matched by key.
    """

    queries: tuple[Table[ParticleId, Part], ...]
    removed: Index[ParticleId]
    cell_table: CellTable[Feat]

    def __post_init__(self) -> None:
        if len(self.queries) not in (1, 2):
            raise ValueError("Local evaluation requires one query or an old/new pair")


def _local_difference[Params, Part: IsRadiusGraphPoints, Feat](
    previous: FusedPotentialInput[Params, Part, Feat],
    proposed: FusedPotentialInput[Params, Part, Feat],
) -> FusedLocalInput[Params, Part, Feat]:
    assert isinstance(previous, FusedLocalInput)
    assert isinstance(proposed, FusedLocalInput)
    return replace(proposed, queries=previous.queries + proposed.queries)


FUSED_GEOMETRY = lens(lambda inp: inp.cloud, cls=FusedPotentialInput).nest(
    POINTCLOUD_GEOMETRY
)
"""Standard geometry lens for full fused evaluations."""


@dataclass
class FusedNeighborEnergy[State, Params, Part: IsRadiusGraphPoints, Feat]:
    """Evaluate a pair term or additive sum over spatially sorted chunks.

    ``pair`` defines the interaction and ``layout`` configures cell storage
    and traversal. A persistent ``cell_table_lens``
    enables acceptance-conditional updates for local proposals. Full calls
    rebuild from the current cloud so positions and cell remain differentiable.

    ``max_queries_per_system`` bounds active rows in each old/new local query
    (asserted). It keeps query-pair storage proportional to the batch size.
    Without a bound, the total query count is used conservatively. Rigid MCMC
    supplies the largest motif size automatically.
    """

    pair: PairTerm[Params, Part, Feat] = field(static=True)
    layout: CellTableParameters = field(static=True)
    cell_table_lens: Lens[State, CellTable[Feat]] | None = field(
        static=True, default=None
    )
    max_queries_per_system: int | None = field(static=True, default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.max_queries_per_system is not None and self.max_queries_per_system < 1:
            raise ValueError("max_queries_per_system must be positive")

    @jit
    def build_cell_table(
        self,
        parameters: Params,
        cloud: PointCloud[Part, HasCell[AnyPeriodicity]],
    ) -> CellTable[Feat]:
        return build_cell_table(
            cloud.particles,
            cloud.systems,
            self.pair.cutoffs(parameters),
            self.pair.features(cloud.particles.data),
            self.layout,
        )

    def _energies(
        self,
        inp: FusedPotentialInput[Params, Part, Feat],
        rows: CellRows[Feat],
        slots: Array,
        keys: CellRows[Feat],
        index: Array,
        excluded: Array,
    ) -> Array:
        batch, ctx = cell_candidates(
            keys,
            rows,
            index,
            inp.cloud.systems,
            self.pair.cutoffs(inp.parameters),
            self.layout.max_images_per_pair,
        )
        if self.layout.max_images_per_pair > 1:
            pairs = PairBatch.from_candidates(batch, ctx)
            key_index = batch.key_idx.indices
            query_index = batch.query_idx.indices
            keep = pairs.valid & (
                (key_index != slots[query_index]) | ~batch.is_minimum_image
            )
            keep &= ~excluded[key_index]
            energies = self.pair.evaluate(
                inp.parameters,
                tree_map(itemgetter(query_index), rows.data),
                tree_map(itemgetter(key_index), keys.data),
                pairs._replace(valid=keep),
            )
            return jax.ops.segment_sum(
                energies, query_index, num_segments=rows.frac.shape[0]
            )
        pairs = PairBatch.from_candidates(batch, ctx, query_lanes=index.shape[1])
        pairs = tree_map(lambda x: x.reshape(*index.shape, *x.shape[1:]), pairs)
        keep = pairs.valid & (index != slots[:, None])
        keep &= ~excluded[index]
        pairs = pairs._replace(valid=keep)
        left = tree_map(lambda x: x[:, None], rows.data)
        right = tree_map(lambda x: x[index], keys.data)
        return self.pair.evaluate(inp.parameters, left, right, pairs).sum(-1)

    def _sum_keys(
        self,
        inp: FusedPotentialInput[Params, Part, Feat],
        table: CellTable[Feat],
        rows: CellRows[Feat],
        slots: Array,
        excluded: Array,
    ) -> Array:
        def consume(index: Array) -> Array:
            return self._energies(inp, rows, slots, table.rows, index, excluded)

        chunk_size = self.layout.key_chunk_size
        if self.layout.key_layout == "slots":
            n = table.sentinel_slot
            chunk_size = chunk_size or max(n, 1)
            index = jnp.minimum(jnp.arange(-(-n // chunk_size) * chunk_size), n)
            return sum_chunks(
                lambda keys: consume(
                    jnp.broadcast_to(keys, (rows.frac.shape[0], chunk_size))
                ),
                index,
                chunk_size,
                active=lambda keys: jnp.any(
                    table.rows.cell[keys] != table.sentinel_cell
                ),
            )
        if chunk_size is None or table.cells.shape[1] <= chunk_size:
            return consume(table.candidates(rows.cell))
        return sum_chunks(
            lambda block: consume(block[0]),
            table.candidate_chunks(rows.cell, chunk_size),
            1,
            active=lambda block: jnp.any(block != table.sentinel_slot),
        )

    def _sum_environment(
        self,
        inp: FusedPotentialInput[Params, Part, Feat],
        table: CellTable[Feat],
        rows: CellRows[Feat],
        slots: Array,
        excluded: Array,
        weights: Array,
        query_chunk_size: int,
    ) -> Array:
        def consume(chunk: tuple[CellRows[Feat], Array, Array]) -> Array:
            rows, slots, weights = chunk
            energies = self._sum_keys(inp, table, rows, slots, excluded)
            return jax.ops.segment_sum(
                energies * weights,
                rows.system.indices,
                num_segments=inp.cloud.systems.size,
            )

        n = rows.frac.shape[0]
        chunk_size = (
            min(n, self.layout.chunk_size, query_chunk_size)
            if n
            else self.layout.chunk_size
        )
        padding = -n % chunk_size
        if padding:
            rows = rows.pad(padding, sentinel_cell=table.sentinel_cell)
            slots = jnp.pad(slots, (0, padding), constant_values=table.sentinel_slot)
            weights = jnp.pad(weights, (0, padding))
        return sum_chunks(
            consume,
            (rows, slots, weights),
            chunk_size,
            active=lambda chunk: jnp.any(chunk[0].inclusion.valid_mask),
        )

    def _full(self, inp: FusedPotentialInput[Params, Part, Feat]) -> Array:
        table = self.build_cell_table(inp.parameters, inp.cloud)
        n = table.sentinel_slot
        rows = jax.tree.map(lambda x: x[:n], table.rows)
        return (
            self._sum_environment(
                inp,
                table,
                rows,
                jnp.arange(n),
                jnp.zeros(n + 1, bool),
                jnp.ones(n, dtype=jnp.int8),
                self.layout.chunk_size,
            )
            / 2
        )

    def _local(
        self,
        inp: FusedLocalInput[Params, Part, Feat],
        table: CellTable[Feat],
    ) -> tuple[Array, CellRows[Feat], Array]:
        queries = inp.queries
        parts = [
            table.bin_rows(q, inp.cloud.systems, self.pair.features(q.data))
            for q in queries
        ]
        rows = CellRows.concatenate(*parts)
        weights = jnp.concatenate(
            [
                jnp.full(q.size, -1 if len(queries) == 2 and i == 0 else 1, jnp.int8)
                for i, q in enumerate(queries)
            ]
        )
        removed_rows = inp.removed.indices_in(inp.cloud.particles.keys)
        slots = table.slot_of_row.at[removed_rows].get(
            mode="fill",
            fill_value=table.sentinel_slot,
        )
        removed = jnp.zeros(table.sentinel_slot + 1, bool).at[slots].set(True)
        n = rows.frac.shape[0]
        environment = self._sum_environment(
            inp,
            table,
            rows,
            jnp.full((n,), table.sentinel_slot),
            removed,
            weights,
            # Separate halves let CPU skip empty work; GPU favors larger chunks.
            n
            if jax.default_backend() in {"gpu", "cuda", "rocm"}
            else max(q.size for q in queries),
        )
        # Keep old and new query pairs separate: no old-new cross interactions.
        phase = jnp.concatenate(
            [jnp.full(q.size, i, dtype=int) for i, q in enumerate(queries)]
        )
        n_systems = inp.cloud.systems.size
        system = Index(
            tuple(range(len(queries) * n_systems)),
            rows.system.indices + phase * n_systems,
        ).apply_mask(rows.inclusion.valid_mask)
        width = min(max(q.size for q in queries), self.max_queries_per_system or n)
        runtime_assert(
            (system.counts.data <= width).all(),
            "max_queries_per_system exceeded in fused local evaluation.",
        )
        query_index = system.where_rectangular(system, width)
        query_index = jnp.where(system.valid_mask[:, None], query_index, n)
        pairs = self._energies(
            inp,
            rows,
            jnp.arange(n),
            rows,
            query_index,
            jnp.zeros(n + 1, bool),
        )
        values = environment + jax.ops.segment_sum(
            pairs * weights / 2,
            rows.system.indices,
            num_segments=inp.cloud.systems.size,
        )
        return values, parts[-1], slots

    @jit
    def __call__(
        self,
        inp: FusedPotentialInput[Params, Part, Feat],
    ) -> WithPatch[Table[SystemId, Energy], Patch[State]]:
        keys = inp.cloud.systems.keys
        if not isinstance(inp, FusedLocalInput):
            return WithPatch(Table(keys, self._full(inp)), IdPatch[State]())
        table = inp.cell_table
        patch: Patch[State] = IdPatch[State]()
        values, rows, slots = self._local(inp, table)
        if self.cell_table_lens is not None:
            patch = CellTableUpdatePatch(
                slots,
                inp.queries[-1].data.system,
                rows,
                self.cell_table_lens,
            )
        return WithPatch(Table(keys, values), patch)


@dataclass
class FusedInputConstructor[
    State,
    Ptch: Patch[Any],
    P: IsRadiusGraphPoints,
    S: HasCell[AnyPeriodicity],
    Params,
    Feat,
](InputConstructor[State, FusedPotentialInput[Params, P, Feat], Ptch]):
    """Construct inputs for the same full/local sum plans used by graph potentials.

    With no probe, apply a proposal and rebuild the full cloud. With a probe,
    restrict to its incident pairs and let ``LocalSumComposer`` handle old/new
    weights and the cached total. ``old_input`` selects the current state
    in either case.
    """

    particles: View[State, Table[ParticleId, P]] = field(static=True)
    systems: View[State, Table[SystemId, S]] = field(static=True)
    parameter_view: View[State, Params] = field(static=True)
    probe: Probe[State, Ptch, WithIndices[ParticleId, P]] | None = field(static=True)
    cell_table: View[State, CellTable[Feat]] = field(static=True)

    def __call__(
        self,
        state: State,
        patch: Ptch | None,
        old_input: bool = False,
    ) -> FusedPotentialInput[Params, P, Feat]:
        if patch is not None and self.probe is None:
            if not old_input:
                systems = self.systems(state)
                state = patch(state, systems.set_data(jnp.ones(systems.size, bool)))
            patch = None
        params = self.parameter_view(state)
        particles = self.particles(state)
        cloud = PointCloud(particles, self.systems(state))
        if patch is None:
            return FusedPotentialInput(params, cloud)
        assert self.probe is not None
        update = self.probe(state, patch)
        old = particles.subset(update.indices)
        # Padded proposal targets are inactive regardless of their payload.
        data = lens(lambda p: p.inclusion, cls=type(update.data)).apply(
            update.data, lambda index: index.apply_mask(update.indices.valid_mask)
        )
        new = Table(old.keys, data, _cls=old.cls)
        return FusedLocalInput(
            params,
            cloud,
            (old if old_input else new,),
            update.indices,
            self.cell_table(state),
        )


def make_fused_potential[
    State,
    Ptch: Patch[Any],
    P: IsRadiusGraphPoints,
    Params,
    Feat,
    Gradients,
    Hessians,
](
    engine: FusedNeighborEnergy[State, Params, P, Feat],
    particles_view: View[State, Table[ParticleId, P]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, Params],
    probe: Probe[State, Ptch, WithIndices[ParticleId, P]] | None,
    gradient_lens: Lens[FusedPotentialInput[Params, P, Feat], Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Compose a fused engine into a potential with incremental updates.

    Args:
        engine: Pair term or sum with a cell-table layout.
        particles_view: View extracting the particle table.
        systems_view: View extracting the system table.
        parameter_view: View extracting the potential parameters.
        probe: Changed particles of a proposal; ``None`` for full recomputation.
        gradient_lens: Differentiation target on the fused input.
        hessian_lens: Gradients selected for the Hessian.
        hessian_idx_view: Hessian index structure.
        patch_idx_view: Cached output index structure; supplied together with
            ``cache_lens``.
        cache_lens: Lens to an initialized potential output cache. Required
            with a probe; optional for full recomputation.

    Returns:
        The composed potential.
    """
    if (cache_lens is None) != (patch_idx_view is None):
        raise ValueError("cache_lens and patch_idx_view must be provided together")
    if probe is not None and cache_lens is None:
        raise ValueError("A particle probe requires an initialized potential cache")

    def build_table(state: State) -> CellTable[Feat]:
        return engine.build_cell_table(
            parameter_view(state),
            PointCloud(particles_view(state), systems_view(state)),
        )

    constructor = FusedInputConstructor(
        particles=particles_view,
        systems=systems_view,
        parameter_view=parameter_view,
        probe=probe,
        cell_table=engine.cell_table_lens or build_table,
    )
    composer = (
        LocalSumComposer(
            constructor,
            difference=_local_difference,
        )
        if probe is not None
        else FullSumComposer(constructor)
    )
    return PotentialFromEnergy(
        energy_fn=engine,
        composer=composer,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=cache_lens,
        patch_idx_view=patch_idx_view,
    )


@dataclass
class FusedPotentialCache[Feat]:
    """Persistent neighbor table, combined energy, and the layout used to build it."""

    table: CellTable[Feat]
    energy: KahanSummand[PotentialOut[EmptyType, EmptyType]]
    layout: CellTableParameters = field(static=True)

    @classmethod
    def create[Params, Part: IsRadiusGraphPoints](
        cls,
        pair: PairTerm[Params, Part, Feat],
        parameters: Params,
        cloud: PointCloud[Part, HasCell[AnyPeriodicity]],
        layout: CellTableParameters | None = None,
    ) -> Self:
        """Build a shared table and empty energy cache before creating a state."""
        if layout is None:
            backend = jax.default_backend()
            on_cpu = backend == "cpu"
            on_gpu = backend in {"gpu", "cuda", "rocm"}
            layout = CellTableParameters.estimate(
                cloud.particles,
                cloud.systems,
                pair.cutoffs(parameters),
                key_chunk_size="auto" if on_cpu or on_gpu else None,
                key_layout="auto" if on_cpu else "cells",
            )
            if on_gpu:
                # Increase work per GPU launch within the candidate-image budget.
                width = (
                    layout.stencil_width
                    * min(
                        layout.cell_capacity,
                        layout.key_chunk_size or layout.cell_capacity,
                    )
                    * layout.max_images_per_pair
                )
                limit = min(
                    512,
                    max(1, cloud.particles.size),
                    max(1, 524_288 // width),
                )
                layout = replace(layout, chunk_size=1 << (limit.bit_length() - 1))
        table = FusedNeighborEnergy(pair, layout).build_cell_table(parameters, cloud)
        energy = KahanSummand.init(
            PotentialOut(
                cloud.systems.set_data(jnp.zeros(cloud.systems.size)), EMPTY, EMPTY
            )
        )
        return cls(table, energy, layout)


def fuse_pair_potentials[State, Part: IsRadiusGraphPoints, Ptch: Patch[Any]](
    potential: Potential[State, EmptyType, EmptyType, Ptch],
    state: State,
    particles: View[State, Table[ParticleId, Part]],
    systems: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    cache: Lens[State, FusedPotentialCache[tuple[PairData, ...]]],
    probe: Probe[State, Ptch, WithIndices[ParticleId, Part]] | None = None,
    *,
    max_queries_per_system: int | None = None,
) -> Potential[State, EmptyType, EmptyType, Ptch]:
    """Replace graph pair terms with one cached, energy-only fused evaluator.

    Collects ``GraphPairEnergy`` terms recursively from summed potentials;
    each keeps its own parameters, cutoff and group masks. Other terms,
    including scaled exclusion corrections, retain their evaluators and caches.

    ``cache`` must select a ``FusedPotentialCache`` initialized with the same
    pair terms, in collection order, before creating ``state``. Its layout
    and table are reused by the returned potential. ``probe`` selects changed
    particles for incremental updates; without one, proposals recompute in full.
    """
    terms: list[PairTerm[tuple[PairData, ...], Part, PairData]] = []
    parameters: list[View[State, PairData]] = []
    remainder: list[Potential[State, EmptyType, EmptyType, Ptch]] = []

    def collect(component: Potential[State, EmptyType, EmptyType, Ptch]) -> None:
        if isinstance(component, SummedPotential):
            for child in component.potentials:
                collect(child)
        elif isinstance(component, PotentialFromEnergy) and isinstance(
            component.energy_fn, GraphPairEnergy
        ):
            composer = component.composer
            if not isinstance(composer, (LocalSumComposer, FullSumComposer)) or (
                not isinstance(composer.constructor, GraphInputConstructor)
            ):
                raise ValueError("Pair fusion requires a graph input constructor")
            terms.append(
                component.energy_fn.pair.packed().with_parameters(
                    itemgetter(len(terms))
                )
            )
            parameter_view = composer.constructor.parameter_view
            parameters.append(lambda s, view=parameter_view: PairData.pack(view(s)))
        else:
            remainder.append(component)

    collect(potential)
    if not terms:
        raise ValueError("No graph pair energies found in the potential")
    engine = FusedNeighborEnergy(
        PairEnergySum(tuple(terms)),
        cache(state).layout,
        cache.focus(lambda x: x.table),
        max_queries_per_system=max_queries_per_system,
    )
    fused = make_fused_potential(
        engine,
        particles,
        systems,
        lambda s: tuple(view(s) for view in parameters),
        probe,
        EMPTY_LENS,
        EMPTY_LENS,
        EMPTY_LENS,
        lambda s: empty_patch_idx_view(PointCloud(particles(s), systems(s))),
        cache.focus(lambda x: x.energy),
    )
    return sum_potentials(fused, *remainder)
