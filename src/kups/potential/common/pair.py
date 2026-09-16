# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Composable pair energies, independent of neighbor traversal.

Add terms before constructing an evaluator to share candidates, feature
gathers, geometry, and reduction. Each term keeps its own cutoff and masks.
"""

from __future__ import annotations

from dataclasses import dataclass as plain_dataclass
from typing import Literal, NamedTuple, Protocol, Self, cast

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import View
from kups.core.neighborlist.masks import ExclusionMask, InBoundsMask, InclusionMatchMask
from kups.core.neighborlist.types import CandidateBatch, PipelineContext
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field


class PairBatch(NamedTuple):
    """Geometry and independent masks for a batch of candidate pairs.

    ``valid`` checks active endpoints and matching systems. Inclusion and
    exclusion policies are kept separate so several consumers can share the
    same candidates. Vectors point from query to key. Invalid geometry is
    sanitized before a consumer evaluates singular pair kernels.
    """

    rij: Array
    r2: Array
    system: Index[SystemId]
    valid: Array
    inclusion: Array
    exclusion: Array

    @classmethod
    def from_candidates(
        cls,
        batch: CandidateBatch[Literal[2]],
        ctx: PipelineContext,
        *,
        query_lanes: int | None = None,
    ) -> Self:
        """Prepare pair geometry and masks from any selector's candidate batch.

        ``query_lanes`` declares a regular block of candidates for each query, in
        query-table order. This shares one cell matrix across a whole lane block
        and broadcasts query coordinates so their gradients reduce along lanes.
        """
        keys = ctx.keys[batch.key_idx]
        queries = (
            ctx.edge_query_table[batch.query_idx]
            if query_lanes is None
            else jax.tree.map(
                lambda x: jnp.repeat(x, query_lanes, axis=0),
                ctx.edge_query_table.data,
            )
        )
        key_system, query_system = Index.match(keys.system, queries.system)
        valid = InBoundsMask()(batch, ctx) & (key_system == query_system)
        delta = keys.positions - queries.positions - batch.edges.shifts[:, 0]
        # Avoid inf - inf propagating NaNs into cell/position derivatives.
        delta = jnp.where(valid[:, None], delta, 0.0)
        frames = ctx.systems.map_data(lambda s: s.cell.frame.materialize())
        if query_lanes is None:
            rij = frames[keys.system].to_real(delta)
        else:
            query_table = ctx.edge_query_table
            delta = delta.reshape(query_table.size, query_lanes, 3)
            # Keep the frame's diagonal/triangular structure and share it across
            # lanes. A dense batched matmul can become a separate GPU BLAS call,
            # materializing pair vectors instead of fusing with the pair kernel.
            query_frames = jax.tree.map(
                lambda x: x[:, None], frames[query_table.data.system]
            )
            rij = query_frames.to_real(delta).reshape(-1, 3)
        # Unroll the short contraction to avoid a reduction over each pair's
        # coordinates. XLA can still materialize distances before the energy sum.
        r2 = rij[:, 0] ** 2
        for axis in range(1, rij.shape[-1]):
            r2 += rij[:, axis] ** 2
        return cls(
            rij,
            r2,
            keys.system,
            valid,
            InclusionMatchMask()(batch, ctx),
            ExclusionMask()(batch, ctx),
        )


class PairKernel[Params, Feat](Protocol):
    """Numerical energy formula evaluated on already selected pairs.

    A kernel takes parameters, the two endpoints' features and their geometry,
    and returns one energy per pair. For example, the Lennard-Jones kernel uses
    species labels to look up mixing parameters and evaluates the 12-6 formula.
    It leaves neighbor selection, cutoffs, masks and summation to its callers.
    In particular, it does not halve energies to account for directed edges.

    Use JAX-compatible array operations and support broadcasting over the pair
    axes: the same kernel handles flat graph edges and rectangular query/key
    blocks. ``PairEnergy`` replaces masked pairs' geometry with finite, nonzero
    values before calling the kernel, then sets their returned energies to zero.

    Type Parameters:
        Params: The term's parameter bundle, such as mixing tables or screening
            constants.
        Feat: Per-particle feature pytree selected by ``PairTerm.features``, such
            as species indices or charges, gathered for each endpoint.
    """

    def __call__(
        self,
        parameters: Params,
        features_i: Feat,
        features_j: Feat,
        rij: Array,
        r2: Array,
        system: Index[SystemId],
        /,
    ) -> Array:
        """Evaluate the interaction without reducing its pair axes.

        Args:
            parameters: Parameters for this interaction.
            features_i: Left-endpoint features, broadcastable over the pair axes.
            features_j: Right-endpoint features, with the same pytree structure.
            rij: Displacement from left to right, including the periodic shift,
                with shape ``(*pair_shape, 3)``.
            r2: Squared distances, with shape ``pair_shape``.
            system: System ids for selecting per-system parameters.

        Returns:
            Energies with shape ``pair_shape``, in the potential's energy units.
        """
        ...


class PairTerm[Params, Part, Feat](Protocol):
    """Complete pair interaction interface consumed by neighbor evaluators.

    A term selects particle features, declares the search cutoff, and evaluates
    candidate pairs with its own cutoff and mask policy. ``PairEnergy`` implements
    this interface for one ``PairKernel``. ``PairEnergySum`` combines terms: its
    search cutoff is their maximum, while evaluation retains each term's cutoff.
    Its inclusion/exclusion flags describe masks required by every contribution,
    allowing neighbor selection to apply those shared masks before compaction.

    The evaluator constructs neighbors and periodic geometry, gathers endpoint
    features, and reduces the returned pair energies into system totals.
    ``GraphPairEnergy`` derives a graph evaluator from a ``PairEnergy``;
    ``FusedNeighborEnergy`` accepts any ``PairTerm`` for shared full/local evaluation.

    Type Parameters:
        Params: Parameters supplied to the term's cutoff and energy functions.
        Part: Particle data presented to the feature selector.
        Feat: Selected per-particle feature pytree. Feature arrays preserve the
            leading particle axis; ``evaluate`` receives gathered pair endpoints.
    """

    @property
    def features(self) -> View[Part, Feat]:
        """Select the payload needed by the kernel, e.g. labels or charges.

        Evaluators may cache these per-particle features in their cell table.
        """
        ...

    @property
    def cutoffs(self) -> View[Params, Table[SystemId, Array]]:
        """Select per-system search radii; a singleton table applies to all systems."""
        ...

    @property
    def inclusion(self) -> bool:
        """Whether every contribution requires matching inclusion groups."""
        ...

    @property
    def exclusion(self) -> bool:
        """Whether every contribution excludes same-group closest images."""
        ...

    def evaluate(
        self,
        parameters: Params,
        left: Feat,
        right: Feat,
        pairs: PairBatch,
    ) -> Array:
        """Return pair energies after applying this term's cutoff and masks.

        ``left`` and ``right`` contain endpoint features broadcastable over
        ``pairs.r2.shape``. The result has that shape, with zero contributions
        for invalid or filtered pairs. Geometry and candidate masks come from
        ``PairBatch``; the term decides which inclusion/exclusion masks apply.
        Summation and directed-edge counting remain the evaluator's responsibility.
        """
        ...


@jax.tree_util.register_pytree_node_class
@plain_dataclass(frozen=True)
class PairData:
    """Array leaves and their structure at the heterogeneous pair-sum boundary.

    Kernels retain their concrete parameter and feature types. Packing lets a
    dynamic sum store them uniformly without discarding index vocabularies or
    other static pytree metadata. The pytree registration exposes the original
    structure so tree operations can still align and concatenate Index nodes.
    """

    arrays: tuple[Array, ...]
    structure: jax.tree_util.PyTreeDef

    @classmethod
    def pack[Data](cls, data: Data) -> Self:
        arrays, structure = jax.tree.flatten(data)
        return cls(tuple(arrays), structure)

    def unpack(self) -> object:
        return jax.tree.unflatten(self.structure, self.arrays)

    def tree_flatten(self) -> tuple[tuple[object], None]:
        return (self.unpack(),), None

    @classmethod
    def tree_unflatten(cls, auxiliary: None, children: tuple[object]) -> Self:
        return cls.pack(children[0])


@dataclass
class _PackedPairTerm[Params, Part, Feat]:
    term: PairTerm[Params, Part, Feat] = field(static=True)

    def features(self, particles: Part, /) -> PairData:
        return PairData.pack(self.term.features(particles))

    @property
    def cutoffs(self) -> View[Params, Table[SystemId, Array]]:
        return self.term.cutoffs

    @property
    def inclusion(self) -> bool:
        return self.term.inclusion

    @property
    def exclusion(self) -> bool:
        return self.term.exclusion

    def evaluate(
        self, parameters: Params, left: PairData, right: PairData, pairs: PairBatch
    ) -> Array:
        # These trees were produced by this term's feature selector above.
        return self.term.evaluate(
            parameters, cast(Feat, left.unpack()), cast(Feat, right.unpack()), pairs
        )


def _add[Params, Part, Left, Right](
    a: PairTerm[Params, Part, Left],
    b: PairTerm[Params, Part, Right],
) -> PairEnergySum[Params, Part]:
    left = a.terms if isinstance(a, PairEnergySum) else (_PackedPairTerm(a),)
    right = b.terms if isinstance(b, PairEnergySum) else (_PackedPairTerm(b),)
    return PairEnergySum((*left, *right))


@dataclass
class PairEnergy[Params, Part, Feat]:
    """A pair kernel with feature, cutoff, and neighbor-mask policies.

    ``with_parameters(view)`` binds a term to a field of shared parameters.
    For example, ``lj.with_parameters(lambda p: p.lj) +
    coulomb.with_parameters(lambda p: p.ewald)`` uses one neighbor traversal
    at the maximum cutoff, retaining each term's individual cutoff.
    Terms use a shared particle type; ``with_particles(view)`` adapts each
    feature selector to that type before adding them.

    Set ``inclusion=False`` to interact across inclusion groups within a
    system, and ``exclusion=False`` to include same-group pairs. Exclusions
    apply to the closest image; other periodic copies remain eligible.
    Zero-shift self pairs and inactive particles are always excluded.
    """

    kernel: PairKernel[Params, Feat] = field(static=True)
    features: View[Part, Feat] = field(static=True)
    cutoffs: View[Params, Table[SystemId, Array]] = field(static=True)
    inclusion: bool = field(static=True, default=True)
    exclusion: bool = field(static=True, default=True)

    def __add__[Other](
        self,
        other: PairTerm[Params, Part, Other],
    ) -> PairEnergySum[Params, Part]:
        return _add(self, other)

    def evaluate(
        self,
        parameters: Params,
        left: Feat,
        right: Feat,
        pairs: PairBatch,
    ) -> Array:
        cutoffs = self.cutoffs(parameters)
        cutoff = cutoffs.data[0] if cutoffs.size == 1 else cutoffs[pairs.system]
        keep = pairs.valid & (pairs.r2 < cutoff**2)
        if self.inclusion:
            keep &= pairs.inclusion
        if self.exclusion:
            keep &= pairs.exclusion
        rij = jnp.where(keep[..., None], pairs.rij, 1.0)
        r2 = jnp.where(keep, pairs.r2, 3.0)
        return jnp.where(
            keep,
            self.kernel(
                parameters,
                left,
                right,
                rij,
                r2,
                pairs.system,
            ),
            0.0,
        )

    def with_particles[Outer](
        self, view: View[Outer, Part]
    ) -> PairEnergy[Params, Outer, Feat]:
        """Select this term's particle interface from a shared particle type."""
        return PairEnergy(
            self.kernel,
            lambda p: self.features(view(p)),
            self.cutoffs,
            self.inclusion,
            self.exclusion,
        )

    def with_parameters[Outer](
        self,
        view: View[Outer, Params],
    ) -> PairEnergy[Outer, Part, Feat]:
        """Read this term's parameters from a shared parameter bundle."""

        def kernel(
            parameters: Outer,
            left: Feat,
            right: Feat,
            rij: Array,
            r2: Array,
            system: Index[SystemId],
            /,
        ) -> Array:
            return self.kernel(view(parameters), left, right, rij, r2, system)

        return PairEnergy(
            kernel,
            self.features,
            lambda p: self.cutoffs(view(p)),
            self.inclusion,
            self.exclusion,
        )

    def packed(self) -> PairEnergy[PairData, Part, PairData]:
        """Adapt concrete parameters and features for dynamic potential fusion."""

        def kernel(
            parameters: PairData,
            left: PairData,
            right: PairData,
            rij: Array,
            r2: Array,
            system: Index[SystemId],
            /,
        ) -> Array:
            return self.kernel(
                cast(Params, parameters.unpack()),
                cast(Feat, left.unpack()),
                cast(Feat, right.unpack()),
                rij,
                r2,
                system,
            )

        return PairEnergy(
            kernel,
            lambda p: PairData.pack(self.features(p)),
            lambda p: self.cutoffs(cast(Params, p.unpack())),
            self.inclusion,
            self.exclusion,
        )


@dataclass
class PairEnergySum[Params, Part]:
    """A flat sum of pair terms evaluated on the same candidate batch.

    Use ``+`` to pack distinct feature types into the shared ``PairData``
    representation automatically.
    """

    terms: tuple[PairTerm[Params, Part, PairData], ...] = field(static=True)

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("At least one pair energy is required")

    def __add__[Other](
        self,
        other: PairTerm[Params, Part, Other],
    ) -> PairEnergySum[Params, Part]:
        return _add(self, other)

    def features(self, particles: Part, /) -> tuple[PairData, ...]:
        return tuple(term.features(particles) for term in self.terms)

    @property
    def inclusion(self) -> bool:
        return all(term.inclusion for term in self.terms)

    @property
    def exclusion(self) -> bool:
        return all(term.exclusion for term in self.terms)

    def cutoffs(self, parameters: Params, /) -> Table[SystemId, Array]:
        tables = [term.cutoffs(parameters) for term in self.terms]
        # A singleton cutoff broadcasts to every system, as elsewhere in kUPS.
        target = max(tables, key=lambda t: t.size)
        return target.set_data(
            jnp.stack([Table.broadcast_to(t, target).data for t in tables]).max(axis=0)
        )

    def evaluate(
        self,
        parameters: Params,
        left: tuple[PairData, ...],
        right: tuple[PairData, ...],
        pairs: PairBatch,
    ) -> Array:
        values = [
            term.evaluate(parameters, a, b, pairs)
            for term, a, b in zip(self.terms, left, right, strict=True)
        ]
        return sum(values[1:], values[0])
