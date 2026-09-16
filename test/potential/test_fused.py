# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Literal, Protocol, TypedDict, Unpack, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.assertion import runtime_assert
from kups.core.capacity import FixedCapacity
from kups.core.cell import AnyPeriodicity, Cell, TriclinicFrame
from kups.core.data import Index, Table, WithIndices
from kups.core.lens import Lens, View, identity_lens, lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.neighborlist.cell_list import CellListNeighborList
from kups.core.neighborlist.cell_table import CellTable, CellTableParameters
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.patch import Accept, Probe
from kups.core.potential import (
    EMPTY,
    EMPTY_LENS,
    EmptyType,
    Potential,
    PotentialOut,
    ScaledPotential,
    SummedPotential,
    empty_patch_idx_view,
    sum_potentials,
)
from kups.core.result import as_result_function
from kups.core.typing import (
    ExclusionId,
    HasCell,
    InclusionId,
    Label,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass
from kups.core.utils.kahan import KahanSummand
from kups.potential.classical.ewald import (
    EwaldParameters,
    ewald_short_range_energy,
    ewald_short_range_pair,
)
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    lennard_jones_energy,
    lennard_jones_pair,
)
from kups.potential.common.energy import (
    EnergyFunction,
    LocalSumComposer,
    PotentialFromEnergy,
)
from kups.potential.common.fused import (
    FUSED_GEOMETRY,
    FusedFullInput,
    FusedInputConstructor,
    FusedLocalInput,
    FusedNeighborEnergy,
    FusedPotentialCache,
    fuse_pair_potentials,
    make_fused_potential,
    sum_chunks,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph, PointCloud
from kups.potential.common.pair import (
    PairBatch,
    PairData,
    PairEnergy,
    PairEnergySum,
    PairTerm,
)

_LABELS = (Label("A"), Label("B"))
_PARAMS = CellTableParameters(chunk_size=8, max_cells_per_system=64, cell_capacity=16)


@dataclass
class _Points:
    positions: jax.Array
    labels: Index[Label]
    charges: jax.Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class _System:
    cell: Cell[AnyPeriodicity]


@dataclass
class _MiniState:
    particles: Table[ParticleId, _Points]
    systems: Table[SystemId, HasCell[AnyPeriodicity]]
    cache: KahanSummand[PotentialOut[EmptyType, EmptyType]]

    @property
    def neighborlist_params(self) -> UniversalNeighborlistParameters:
        return UniversalNeighborlistParameters(768, 384, 768, 768)


@dataclass
class _CachedState[Feat](_MiniState):
    cell_table: CellTable[Feat]


class _Parameters(Protocol):
    @property
    def cutoff(self) -> Table[SystemId, jax.Array]: ...


class _PointChanges(TypedDict, total=False):
    positions: jax.Array
    labels: Index[Label]
    charges: jax.Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class _Move:
    """Proposal patch carrying the changed rows."""

    indices: Index[ParticleId]
    data: _Points

    def __call__[State: _MiniState](self, state: State, accept: Accept) -> State:
        return dataclasses.replace(
            state, particles=state.particles.update_if(accept, self.indices, self.data)
        )


def _probe(state: _MiniState, move: _Move) -> WithIndices[ParticleId, _Points]:
    del state
    return WithIndices(move.indices, move.data)


def _make_state(
    key: jax.Array,
    counts: tuple[int, ...],
    box_lengths: tuple[float, ...],
    *,
    n_inactive: int = 0,
    molecule_size: int = 3,
) -> _MiniState:
    """Random particles; the last ``molecule_size`` active atoms of each system form a
    molecule (shared exclusion group); ``n_inactive`` trailing rows per system are
    inactive (out-of-bounds inclusion, zeroed positions)."""
    keys = jax.random.split(key, 2 * len(counts))
    n_systems = len(counts)
    positions: list[jax.Array] = []
    charges: list[jax.Array] = []
    system_ids: list[np.ndarray] = []
    exclusion: list[np.ndarray] = []
    inclusion: list[np.ndarray] = []
    group = 0
    for i, (count, length) in enumerate(zip(counts, box_lengths)):
        total = count + n_inactive
        pos = jax.random.uniform(keys[2 * i], (total, 3)) * length
        first = count - molecule_size
        pos = pos.at[first + 1 : count].set(
            pos[first] + 0.4 * pos[first + 1 : count] / length
        )
        pos = pos.at[count:].set(0.0)
        positions.append(pos)
        q = jax.random.normal(keys[2 * i + 1], (total,)) * 0.5
        charges.append(jnp.where(jnp.arange(total) < count, q, 0.0))
        system_ids.append(np.full((total,), i))
        excl = np.arange(total) + group
        excl[first:count] = group + first
        exclusion.append(excl)
        inclusion.append(np.where(np.arange(total) < count, i, n_systems))
        group += total
    n = sum(counts) + n_inactive * n_systems
    particles = Table.arange(
        _Points(
            positions=jnp.concatenate(positions),
            labels=Index(_LABELS, jnp.arange(n) % 2),
            charges=jnp.concatenate(charges),
            system=Index(
                tuple(SystemId(i) for i in range(n_systems)),
                jnp.asarray(np.concatenate(system_ids)),
            ),
            inclusion=Index(
                tuple(InclusionId(i) for i in range(n_systems)),
                jnp.asarray(np.concatenate(inclusion)),
            ),
            exclusion=Index.new(np.concatenate(exclusion), label=ExclusionId),
        ),
        label=ParticleId,
    )
    lattices = jnp.stack([jnp.eye(3) * length for length in box_lengths])
    systems = Table.arange(
        _System(
            cell=Cell(TriclinicFrame.from_matrix(lattices), periodic=(True, True, True))
        ),
        label=SystemId,
    )
    zero = Table.arange(jnp.zeros(n_systems), label=SystemId)
    return _MiniState(
        particles, systems, KahanSummand.init(PotentialOut(zero, EMPTY, EMPTY))
    )


def _lj_params(n_systems: int, cutoff: float) -> LennardJonesParameters:
    return LennardJonesParameters.from_lorentz_berthelot_mixing(
        labels=_LABELS,
        sigma=jnp.array([1.8, 2.2]),
        epsilon=jnp.array([25.0, 60.0]),
        cutoff=Table.arange(jnp.full((n_systems,), cutoff), label=SystemId),
    )


def _ewald_params(n_systems: int, cutoff: float) -> EwaldParameters:
    return EwaldParameters(
        alpha=Table.arange(jnp.full((n_systems,), 0.3), label=SystemId),
        cutoff=Table.arange(jnp.full((n_systems,), cutoff), label=SystemId),
        k_max=Table.arange(jnp.zeros(n_systems), label=SystemId),
        reciprocal_lattice_shifts=Table.arange(
            jnp.zeros((n_systems, 1, 3), dtype=int), label=SystemId
        ),
    )


def _neighborlist(params: _Parameters) -> CellListNeighborList:
    return CellListNeighborList(
        avg_candidates=FixedCapacity(768),
        avg_edges=FixedCapacity(384),
        cells=FixedCapacity(128),
        avg_image_candidates=FixedCapacity(768),
        cutoffs=params.cutoff,
    )


def _graph_full[Params: _Parameters](
    state: _MiniState,
    params: Params,
    energy_fn: EnergyFunction[
        _MiniState,
        GraphPotentialInput[Params, _Points, HasCell[AnyPeriodicity], Literal[2]],
    ],
) -> jax.Array:
    """Ground truth: materialized-edge full evaluation, per-system energies."""
    edges = _neighborlist(params)(state.particles, state.systems)
    graph = HyperGraph(state.particles, state.systems, edges)
    return energy_fn(GraphPotentialInput(params, graph)).data.data


def _full_input[State, Params, Feat](
    engine: FusedNeighborEnergy[State, Params, _Points, Feat],
    params: Params,
    cloud: PointCloud[_Points, HasCell[AnyPeriodicity]],
) -> FusedFullInput[Params, _Points, Feat]:
    from kups.core.neighborlist.adaptive import (
        AdaptiveNeighborList,
        NeighborListCandidate,
        dense_cost,
    )

    n = max(1, cloud.particles.size)
    neighbors = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n),
        avg_edges=FixedCapacity(n * engine.layout.max_images_per_pair),
        avg_image_candidates=FixedCapacity(n * engine.layout.max_images_per_pair),
        cutoffs=engine.pair.cutoffs(params),
    )
    adaptive = AdaptiveNeighborList((NeighborListCandidate(neighbors, dense_cost),))
    return FusedFullInput(
        params,
        cloud,
        adaptive.pair_candidates(
            cloud.particles,
            cloud.systems,
            inclusion=engine.pair.inclusion,
            exclusion=engine.pair.exclusion,
        ),
    )


def _with_points[State: _MiniState](
    state: State, **changes: Unpack[_PointChanges]
) -> State:
    points = dataclasses.replace(state.particles.data, **changes)
    return dataclasses.replace(state, particles=Table(state.particles.keys, points))


def _per_particle_exclusion(state: _MiniState) -> _MiniState:
    """Ewald real-space convention: every particle its own exclusion group."""
    exclusion = Index.arange(state.particles.size, label=ExclusionId)
    return _with_points(state, exclusion=exclusion)


_LJ_PAIR: PairEnergy[LennardJonesParameters, _Points, Index[Label]] = (
    lennard_jones_pair.with_particles(identity_lens(_Points))
)
_EWALD_PAIR: PairEnergy[EwaldParameters, _Points, jax.Array] = (
    ewald_short_range_pair.with_particles(identity_lens(_Points))
)

_LJ: FusedNeighborEnergy[_MiniState, LennardJonesParameters, _Points, Index[Label]] = (
    FusedNeighborEnergy(_LJ_PAIR, _PARAMS)
)
_EWALD: FusedNeighborEnergy[_MiniState, EwaldParameters, _Points, jax.Array] = (
    FusedNeighborEnergy(_EWALD_PAIR, _PARAMS)
)


def _potential[State: _MiniState, Params, Feat](
    engine: FusedNeighborEnergy[State, Params, _Points, Feat],
    params: Params,
    probe: Probe[State, _Move, WithIndices[ParticleId, _Points]] | None = None,
    *,
    patch_idx_view: View[State, PotentialOut[EmptyType, EmptyType]] | None = None,
    cache_lens: Lens[State, KahanSummand[PotentialOut[EmptyType, EmptyType]]]
    | None = None,
) -> Potential[State, EmptyType, EmptyType, _Move]:
    return make_fused_potential(
        engine,
        lambda s: s.particles,
        lambda s: s.systems,
        lambda s: params,
        probe,
        EMPTY_LENS,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        cache_lens=cache_lens,
    )


type _PairParameters = tuple[LennardJonesParameters, EwaldParameters]
type _KeyLayout = Literal["cells", "slots"]


def _patch_indices(state: _MiniState) -> PotentialOut[EmptyType, EmptyType]:
    return empty_patch_idx_view(state)


class TestPotentialFusion:
    def test_automatic_gpu_layout_limits_padding(self, monkeypatch):
        state = _make_state(jax.random.key(32), (8, 6), (12.0, 10.0))
        params = _lj_params(2, 3.0)
        cloud = PointCloud(state.particles, state.systems)
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
        cache = FusedPotentialCache.create(_LJ_PAIR, params, cloud)
        assert cache.table.rows.frac.shape[0] < 2 * state.particles.size

    def test_nested_sum_and_remaining_terms_with_mixed_acceptance(self):
        from kups.application.potential.classical.lennard_jones import (
            make_lennard_jones_from_state,
        )
        from kups.core.lens import identity_lens
        from kups.potential.common.energy import FullSumComposer

        state = _make_state(jax.random.key(31), (8, 6), (12.0, 10.0))
        params = _lj_params(2, 3.0)
        params = dataclasses.replace(params, sigma=params.sigma * 0.25)
        other = dataclasses.replace(
            params, cutoff=params.cutoff.set_data(jnp.full(2, 3.5))
        )

        def make(parameters: LennardJonesParameters):
            return make_lennard_jones_from_state(
                identity_lens(_CachedState[tuple[PairData, ...]]),
                parameters=parameters,
                neighborlist_factory=lambda state, cutoffs: _neighborlist(parameters),
            )

        first, second = make(params), make(other)
        assert isinstance(first, PotentialFromEnergy)
        assert isinstance(first.composer, LocalSumComposer)
        retained = ScaledPotential(
            dataclasses.replace(
                first, composer=FullSumComposer(first.composer.constructor)
            ),
            0.25,
        )
        original = sum_potentials(first, sum_potentials(retained, second))
        pair: PairEnergySum[
            tuple[LennardJonesParameters, LennardJonesParameters], _Points
        ] = _LJ_PAIR.with_parameters(lambda p: p[0]) + _LJ_PAIR.with_parameters(
            lambda p: p[1]
        )
        cache = FusedPotentialCache.create(
            pair,
            (params, other),
            PointCloud(state.particles, state.systems),
            _PARAMS,
        )
        state = _CachedState(state.particles, state.systems, cache.energy, cache.table)
        potential = fuse_pair_potentials(
            original,
            state,
            lambda s: s.particles,
            lambda s: s.systems,
            lens(lambda s: FusedPotentialCache(s.cell_table, s.cache, _PARAMS)),
            probe=_probe,
            max_queries_per_system=1,
        )
        assert isinstance(potential, SummedPotential)
        assert potential.potentials[1:] == (retained,)
        accept_all = state.systems.set_data(jnp.ones(2, bool))
        state = potential(state).patch(state, accept_all)
        for accepted in ([True, False], [False, True]):
            move = _move(
                state,
                jnp.array([0, 8]),
                state.particles.data.positions[jnp.array([0, 8])] + 0.15,
            )
            proposed = move(state, accept_all)
            result = potential(state, move)
            npt.assert_allclose(
                result.data.total_energies.data,
                original(proposed).data.total_energies.data,
                rtol=1e-10,
                atol=1e-10,
            )
            accept = state.systems.set_data(jnp.array(accepted))
            state = move(result.patch(state, accept), accept)
            npt.assert_allclose(
                potential(state).data.total_energies.data,
                original(state).data.total_energies.data,
                rtol=1e-10,
                atol=1e-10,
            )


class TestFullEvaluation:
    @pytest.mark.parametrize(
        "local,cache,indices",
        [
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_incomplete_cache_rejected_at_construction(
        self, local: bool, cache: bool, indices: bool
    ):
        with pytest.raises(ValueError, match="cache"):
            _potential(
                _LJ,
                _lj_params(1, 3.0),
                _probe if local else None,
                patch_idx_view=_patch_indices if indices else None,
                cache_lens=lens(lambda s: s.cache, cls=_MiniState) if cache else None,
            )

    @pytest.mark.parametrize("old_input", [False, True])
    def test_unprobed_constructor_selects_old_or_proposed_input(self, old_input: bool):
        state = _make_state(jax.random.key(32), (4,), (12.0,))
        move = _move(state, jnp.array([0]), state.particles.data.positions[:1] + 0.5)
        constructor = FusedInputConstructor(
            particles=lambda s: s.particles,
            systems=lambda s: s.systems,
            parameter_view=lambda s: s.particles.data.positions.sum(),
            pair=_LJ_PAIR.with_parameters(lambda _: _lj_params(1, 3.0)),
            probe=None,
            cell_table=lambda s: _LJ.build_cell_table(
                _lj_params(s.systems.size, 3.0), PointCloud(s.particles, s.systems)
            ),
        )
        inp = constructor(state, move, old_input=old_input)
        expected = state.particles.data.positions
        if not old_input:
            expected = expected.at[0].add(0.5)
        npt.assert_allclose(inp.cloud.particles.data.positions, expected)
        npt.assert_allclose(inp.parameters, expected.sum())

    def test_lj_matches_graph_with_inactive_rows(self):
        state = _make_state(jax.random.PRNGKey(0), (24, 16), (12.0, 10.0), n_inactive=4)
        params = _lj_params(2, 3.0)
        pot = _potential(_LJ, params)
        ref = _graph_full(state, params, lennard_jones_energy)
        npt.assert_allclose(pot(state).data.total_energies.data, ref, rtol=1e-10)

    def test_ewald_sr_matches_graph(self):
        state = _per_particle_exclusion(
            _make_state(jax.random.PRNGKey(1), (20, 14), (12.0, 10.0), n_inactive=3)
        )
        params = _ewald_params(2, 3.5)
        pot = _potential(_EWALD, params)
        ref = _graph_full(state, params, ewald_short_range_energy)
        npt.assert_allclose(pot(state).data.total_energies.data, ref, rtol=1e-10)

    def test_autodiff_gradients_match_graph(self):
        state = _make_state(jax.random.PRNGKey(2), (24, 16), (12.0, 10.0))
        params = _lj_params(2, 3.0)
        edges = _neighborlist(params)(state.particles, state.systems)

        def moved(positions: jax.Array) -> Table[ParticleId, _Points]:
            return _with_points(state, positions=positions).particles

        def graph_energy(positions: jax.Array) -> jax.Array:
            graph = HyperGraph(moved(positions), state.systems, edges)
            return lennard_jones_energy(
                GraphPotentialInput(params, graph)
            ).data.data.sum()

        def fused_energy(positions: jax.Array) -> jax.Array:
            inp = _full_input(_LJ, params, PointCloud(moved(positions), state.systems))
            return _LJ(inp).data.data.sum()

        x = state.particles.data.positions
        npt.assert_allclose(
            jax.grad(fused_energy)(x), jax.grad(graph_energy)(x), rtol=1e-8, atol=1e-12
        )

    def test_small_box_stencil_dedup(self):
        # Two bins per axis: the stencil wraps onto duplicates, which must not
        # double-count pairs; also exercises the estimated stencil width.
        state = _make_state(jax.random.PRNGKey(3), (12,), (5.0,))
        params = _lj_params(1, 2.4)
        estimated = CellTableParameters.estimate(
            state.particles, state.systems, params.cutoff
        )
        assert estimated.stencil_width == 8
        pot = _potential(FusedNeighborEnergy(_LJ_PAIR, estimated), params)
        ref = _graph_full(state, params, lennard_jones_energy)
        npt.assert_allclose(pot(state).data.total_energies.data, ref, rtol=1e-10)


class TestPeriodicImages:
    @staticmethod
    def _setup(
        periodic: AnyPeriodicity = (True, True, True),
        key_layout: _KeyLayout = "cells",
        key_chunk_size: int | None = None,
    ):
        state = _make_state(jax.random.key(91), (6, 5), (4.0, 10.0), n_inactive=1)
        vectors = jnp.array(
            [[[4.0, 0.0, 0.0], [1.4, 5.0, 0.0], [0.6, 0.3, 6.0]], jnp.eye(3) * 10.0]
        )
        system = state.particles.data.system.indices
        fractions = (
            state.particles.data.positions / jnp.array([4.0, 10.0])[system, None]
        )
        state = _with_points(
            state,
            positions=jnp.einsum("ni,nij->nj", fractions, vectors[system]),
            charges=state.particles.data.charges.at[jnp.array([6, 12])].set(
                jnp.array([0.2, -0.3])
            ),
        )
        state = dataclasses.replace(
            state,
            systems=state.systems.set_data(
                _System(Cell(TriclinicFrame.from_matrix(vectors), periodic=periodic))
            ),
        )
        lj = _lj_params(2, 4.4)
        lj = dataclasses.replace(
            lj, sigma=lj.sigma * 0.2, cutoff=lj.cutoff.set_data(jnp.array([4.4, 3.5]))
        )
        ew = _ewald_params(2, 5.7)
        ew = dataclasses.replace(ew, cutoff=ew.cutoff.set_data(jnp.array([3.0, 5.7])))
        params = (lj, ew)
        pair: PairEnergySum[_PairParameters, _Points] = _LJ_PAIR.with_parameters(
            lambda p: p[0]
        ) + (_EWALD_PAIR.with_parameters(lambda p: p[1]))
        layout = CellTableParameters.estimate(
            state.particles,
            state.systems,
            pair.cutoffs(params),
            chunk_size=4,
            occupancy_headroom=4,
            key_layout=key_layout,
            key_chunk_size=key_chunk_size,
        )
        return state, params, FusedNeighborEnergy(pair, layout)

    @staticmethod
    def _reference(
        state: _MiniState, params: _PairParameters, images: int
    ) -> jax.Array:
        def evaluate[Params: _Parameters](
            points: _MiniState,
            parameters: Params,
            energy: EnergyFunction[
                _MiniState,
                GraphPotentialInput[
                    Params, _Points, HasCell[AnyPeriodicity], Literal[2]
                ],
            ],
        ) -> jax.Array:
            n = points.particles.size
            neighborlist = DenseNearestNeighborList(
                avg_candidates=FixedCapacity(n),
                avg_edges=FixedCapacity(n * images),
                avg_image_candidates=FixedCapacity(n * images),
                cutoffs=parameters.cutoff,
            )
            edges = neighborlist(points.particles, points.systems)
            graph = HyperGraph(points.particles, points.systems, edges)
            return energy(GraphPotentialInput(parameters, graph)).data.data

        return evaluate(state, params[0], lennard_jones_energy) + evaluate(
            _per_particle_exclusion(state), params[1], ewald_short_range_energy
        )

    @pytest.mark.parametrize("periodic", [(True, True, True), (False, True, True)])
    def test_full_mixed_image_counts_and_exclusions(
        self,
        periodic: AnyPeriodicity,
    ):
        state, params, engine = self._setup(periodic)
        assert engine.layout.max_images_per_pair > 1

        @jax.jit
        @as_result_function
        def compare(state: _MiniState):
            actual = engine(
                _full_input(engine, params, PointCloud(state.particles, state.systems))
            ).data.data
            expected = self._reference(state, params, engine.layout.max_images_per_pair)
            return actual, expected

        result = compare(state)
        result.raise_assertion()
        npt.assert_allclose(*result.value, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("kind", ["lj", "ewald"])
    def test_single_particle_retains_nonzero_self_images(
        self, kind: Literal["lj", "ewald"]
    ):
        from kups.potential.classical.ewald import TO_STANDARD_UNITS

        state = _make_state(jax.random.key(92), (1,), (4.0,), molecule_size=1)

        def evaluate[Params: _Parameters, Feat](
            parameters: Params, pair: PairTerm[Params, _Points, Feat]
        ) -> jax.Array:
            layout = CellTableParameters.estimate(
                state.particles, state.systems, parameters.cutoff
            )
            engine = FusedNeighborEnergy(pair, layout)
            result = jax.jit(as_result_function(engine))(
                _full_input(
                    engine,
                    parameters,
                    PointCloud(state.particles, state.systems),
                )
            )
            result.raise_assertion()
            return result.value.data.data

        if kind == "lj":
            lj = _lj_params(1, 4.4)
            c6 = (lj.sigma[0, 0] / 4.0) ** 6
            expected = 3 * 4 * lj.epsilon[0, 0] * (c6**2 - c6)
            actual = evaluate(lj, _LJ_PAIR)
        else:
            ew = _ewald_params(1, 4.4)
            charge = state.particles.data.charges[0]
            expected = 3 * TO_STANDARD_UNITS * charge**2 * (1 - jax.lax.erf(1.2)) / 4
            actual = evaluate(ew, _EWALD_PAIR)
        # Six nearest self-images, halved for directed pairs; the zero shift is absent.
        npt.assert_allclose(actual, [expected], rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize(
        "key_layout,key_chunk_size", [("cells", None), ("cells", 2), ("slots", 4)]
    )
    def test_persistent_updates_with_periodic_images(
        self, key_layout: _KeyLayout, key_chunk_size: int | None
    ):
        state, params, engine = self._setup(
            key_layout=key_layout, key_chunk_size=key_chunk_size
        )
        engine = dataclasses.replace(
            engine,
            cell_table_lens=lens(lambda s: s.cell_table),
            max_queries_per_system=3,
        )
        reference = jax.jit(
            as_result_function(
                lambda s: self._reference(s, params, engine.layout.max_images_per_pair)
            )
        )
        initial = reference(state)
        initial.raise_assertion()
        state = _with_cache(
            state,
            initial.value,
            engine.build_cell_table(params, PointCloud(state.particles, state.systems)),
        )
        potential = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        evaluate = jax.jit(as_result_function(potential))
        accept_all = state.systems.set_data(jnp.ones(2, bool))
        for step, accepted in enumerate(
            ([True, False], [True, True], [False, True], [True, True])
        ):
            if step == 0:
                rows = jnp.array([3, 4, 5, 9, 10, 11])
                move = _move(
                    state,
                    rows,
                    state.particles.data.positions[rows] + jnp.array([4.2, 0.3, -0.2]),
                )
            elif step == 1:
                move = _move(
                    state,
                    jnp.array([6, 12]),
                    jnp.array([[1.1, 2.2, 3.3], [8.2, 1.2, 4.2]]),
                )
            elif step == 2:
                rows = jnp.array([6, 12])
                move = _move(
                    state, rows, state.particles.data.positions[rows], active=False
                )
                move = dataclasses.replace(
                    move,
                    data=dataclasses.replace(
                        move.data, system=Index(state.systems.keys, jnp.full(2, 2))
                    ),
                )
            else:
                rows = jnp.array([3, 10])
                move = _move(state, rows, state.particles.data.positions[rows] + 0.2)
            result = evaluate(state, move)
            result.raise_assertion()
            expected = reference(move(state, accept_all))
            expected.raise_assertion()
            npt.assert_allclose(
                result.value.data.total_energies.data,
                expected.value,
                rtol=1e-9,
                atol=1e-9,
            )
            accept = state.systems.set_data(jnp.array(accepted))
            committed = as_result_function(
                lambda s: move(result.value.patch(s, accept), accept)
            )(state)
            committed.raise_assertion()
            state = committed.value
            expected = reference(state)
            expected.raise_assertion()
            npt.assert_allclose(
                state.cache.total.total_energies.data,
                expected.value,
                rtol=1e-9,
                atol=1e-9,
            )

    def test_position_and_cell_derivatives_with_images(self):
        state, params, engine = self._setup()

        def energy(positions: jax.Array, vectors: jax.Array, fused: bool):
            moved = _with_points(state, positions=positions)
            moved = dataclasses.replace(
                moved,
                systems=state.systems.set_data(
                    _System(
                        Cell(
                            TriclinicFrame.from_matrix(vectors),
                            periodic=(True, True, True),
                        )
                    )
                ),
            )
            if fused:
                values = engine(
                    _full_input(
                        engine, params, PointCloud(moved.particles, moved.systems)
                    )
                ).data.data
            else:
                values = self._reference(
                    moved, params, engine.layout.max_images_per_pair
                )
            return values.sum()

        actual = jax.jit(
            as_result_function(jax.grad(lambda x, h: energy(x, h, True), (0, 1)))
        )(state.particles.data.positions, state.systems.data.cell.vectors)
        expected = jax.jit(
            as_result_function(jax.grad(lambda x, h: energy(x, h, False), (0, 1)))
        )(state.particles.data.positions, state.systems.data.cell.vectors)
        actual.raise_assertion()
        expected.raise_assertion()
        for a, b in zip(actual.value, expected.value):
            assert jnp.isfinite(a).all()
            npt.assert_allclose(a, b, rtol=1e-9, atol=1e-9)

    def test_image_capacity_is_checked(self):
        state, params, engine = self._setup()
        engine = dataclasses.replace(
            engine,
            layout=dataclasses.replace(engine.layout, max_images_per_pair=1),
        )
        result = jax.jit(as_result_function(engine.build_cell_table))(
            params, PointCloud(state.particles, state.systems)
        )
        with pytest.raises(AssertionError, match="max_images_per_pair exceeded"):
            result.raise_assertion()


def _molecule_rows(state: _MiniState, system: int, size: int = 3) -> jax.Array:
    p = state.particles.data
    active = p.inclusion.indices < p.inclusion.num_labels
    return jnp.nonzero(active & (p.system.indices == system))[0][-size:]


def _move(
    state: _MiniState, rows: jax.Array, positions: jax.Array, active: bool = True
) -> _Move:
    p = state.particles.data
    n_systems = p.inclusion.num_labels
    inclusion = jnp.where(active, p.system.indices[rows], n_systems)
    return _Move(
        indices=Index(state.particles.keys, rows),
        data=_Points(
            positions=positions,
            labels=p.labels[rows],
            charges=p.charges[rows],
            system=p.system[rows],
            inclusion=Index(p.inclusion.keys, inclusion),
            exclusion=p.exclusion[rows],
        ),
    )


@overload
def _with_cache[Feat](
    state: _MiniState, energies: jax.Array, table: CellTable[Feat]
) -> _CachedState[Feat]: ...
@overload
def _with_cache(state: _MiniState, energies: jax.Array) -> _MiniState: ...
def _with_cache[Feat](
    state: _MiniState, energies: jax.Array, table: CellTable[Feat] | None = None
) -> _MiniState:
    cache = KahanSummand.init(
        PotentialOut(Table.arange(energies, label=SystemId), EMPTY, EMPTY)
    )
    if table is None:
        return dataclasses.replace(state, cache=cache)
    return _CachedState(state.particles, state.systems, cache, table)


class TestDeltaEvaluation:
    """Fused delta vs graph full recomputation: the cache is seeded with the graph
    energy of the current state and the fused (cache + delta) total must equal the
    graph energy of the proposed state. One molecule per system moves."""

    def _check[Params: _Parameters, Feat](
        self,
        state: _MiniState,
        move: _Move,
        params: Params,
        energy_fn: EnergyFunction[
            _MiniState,
            GraphPotentialInput[Params, _Points, HasCell[AnyPeriodicity], Literal[2]],
        ],
        engine: FusedNeighborEnergy[_MiniState, Params, _Points, Feat],
    ) -> None:
        before = _graph_full(state, params, energy_fn)
        after = _graph_full(
            move(state, Table.arange(jnp.array([True, True]), label=SystemId)),
            params,
            energy_fn,
        )
        state = _with_cache(state, before)
        pot = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        npt.assert_allclose(
            pot(state, move).data.total_energies.data, after, rtol=1e-9, atol=1e-12
        )

    def test_lj_translation_both_systems(self):
        state = _make_state(jax.random.PRNGKey(4), (24, 16), (12.0, 10.0), n_inactive=2)
        rows = jnp.concatenate([_molecule_rows(state, 0), _molecule_rows(state, 1)])
        move = _move(
            state,
            rows,
            state.particles.data.positions[rows] + jnp.array([1.1, -0.6, 0.4]),
        )
        self._check(state, move, _lj_params(2, 3.0), lennard_jones_energy, _LJ)

    def test_ewald_sr_nonrigid_perturbation(self):
        # Non-rigid move: the intra-query pair term must reproduce the changed
        # intramolecular contributions (per-particle exclusion groups).
        state = _per_particle_exclusion(
            _make_state(jax.random.PRNGKey(5), (20, 14), (12.0, 10.0), n_inactive=2)
        )
        rows = jnp.concatenate([_molecule_rows(state, 0), _molecule_rows(state, 1)])
        new = state.particles.data.positions[rows] + jnp.array([0.9, 0.3, -0.5])
        new = new + 0.2 * jax.random.normal(jax.random.PRNGKey(6), (rows.shape[0], 3))
        self._check(
            state,
            _move(state, rows, new),
            _ewald_params(2, 3.5),
            ewald_short_range_energy,
            _EWALD,
        )

    def test_lj_insertion_and_deletion(self):
        state = _make_state(jax.random.PRNGKey(7), (18, 12), (12.0, 10.0), n_inactive=3)
        params = _lj_params(2, 3.0)
        p = state.particles.data
        inactive = jnp.nonzero(p.inclusion.indices >= 2)[0][:2]
        insert = _move(state, inactive, jnp.array([[3.1, 4.2, 5.3], [3.6, 4.2, 5.3]]))
        self._check(state, insert, params, lennard_jones_energy, _LJ)
        rows = _molecule_rows(state, 1)
        delete = _move(state, rows, p.positions[rows], active=False)
        self._check(state, delete, params, lennard_jones_energy, _LJ)

    def test_ewald_sr_insertion(self):
        state = _per_particle_exclusion(
            _make_state(jax.random.PRNGKey(8), (18, 12), (12.0, 10.0), n_inactive=3)
        )
        p = state.particles.data
        inactive = jnp.nonzero(p.inclusion.indices >= 2)[0][:3]
        charges = p.charges.at[inactive].set(jnp.array([0.4, -0.25, -0.15]))
        state = _with_points(state, charges=charges)
        new = jnp.array([[3.1, 4.2, 5.3], [4.0, 4.2, 5.3], [3.5, 4.9, 5.3]])
        self._check(
            state,
            _move(state, inactive, new),
            _ewald_params(2, 3.5),
            ewald_short_range_energy,
            _EWALD,
        )


class TestPersistentCellTable:
    """Multi-move chain through a persistent table with mixed accept/reject: after
    applying each composed patch (cache + table) and the move, the committed cache
    must equal a graph full recomputation of the committed particles."""

    @pytest.mark.parametrize("valid_target", [False, True])
    def test_padded_proposal_targets_ignore_active_payload(self, valid_target):
        state = _make_state(jax.random.key(31), (16,), (6.0,))
        params = _lj_params(1, 3.0)
        engine = FusedNeighborEnergy(_LJ_PAIR, _PARAMS, lens(lambda s: s.cell_table))
        table = engine.build_cell_table(
            params, PointCloud(state.particles, state.systems)
        )
        state = _with_cache(
            state, _graph_full(state, params, lennard_jones_energy), table
        )
        potential = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        n = state.particles.size
        targets = Index(
            state.particles.keys, jnp.array([0 if valid_target else n, n, n])
        )
        data = state.particles[Index(state.particles.keys, jnp.zeros(3, dtype=int))]
        move = _Move(targets, dataclasses.replace(data, positions=data.positions + 0.2))
        accept = state.systems.set_data(jnp.array([True]))
        result = potential(state, move)
        expected = _graph_full(move(state, accept), params, lennard_jones_energy)
        npt.assert_allclose(result.data.total_energies.data, expected, rtol=1e-10)
        updated = result.patch(state, accept)
        for before, after in zip(
            jax.tree.leaves(table.rows),
            jax.tree.leaves(updated.cell_table.rows),
            strict=True,
        ):
            npt.assert_array_equal(before[-1], after[-1])
        if not valid_target:
            for before, after in zip(
                jax.tree.leaves(table), jax.tree.leaves(updated.cell_table), strict=True
            ):
                npt.assert_array_equal(before, after)

    @pytest.mark.parametrize(
        "key_layout,key_chunk_size", [("cells", None), ("cells", 1), ("slots", 8)]
    )
    def test_move_chain_matches_graph(
        self, key_layout: _KeyLayout, key_chunk_size: int | None
    ):
        state = _make_state(
            jax.random.PRNGKey(11), (24, 16), (12.0, 10.0), n_inactive=3
        )
        params = _lj_params(2, 3.0)
        engine = FusedNeighborEnergy(
            _LJ_PAIR,
            dataclasses.replace(
                _PARAMS, key_layout=key_layout, key_chunk_size=key_chunk_size
            ),
            lens(lambda s: s.cell_table),
        )
        engine = dataclasses.replace(engine, max_queries_per_system=3)
        table = engine.build_cell_table(
            params, PointCloud(state.particles, state.systems)
        )
        state = _with_cache(
            state, _graph_full(state, params, lennard_jones_energy), table
        )
        pot = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        inactive = jnp.nonzero(state.particles.data.inclusion.indices >= 2)[0]

        def translation(state: _MiniState, key: jax.Array) -> _Move:
            rows = jnp.concatenate([_molecule_rows(state, 0), _molecule_rows(state, 1)])
            shift = jax.random.uniform(key, (1, 3)) * 2.0 - 1.0
            return _move(state, rows, state.particles.data.positions[rows] + shift)

        def insertion(state: _MiniState, key: jax.Array) -> _Move:
            base = jax.random.uniform(key, (1, 3)) * 8.0 + 1.0
            offsets = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.55, 0.0]])
            return _move(state, inactive[:3], base + offsets)

        def deletion(state: _MiniState, key: jax.Array) -> _Move:
            del key
            rows = _molecule_rows(state, 1)
            return _move(
                state, rows, state.particles.data.positions[rows], active=False
            )

        moves = [translation, insertion, translation, deletion, translation, insertion]
        accepts = [
            [True, True],
            [True, False],
            [False, True],
            [True, True],
            [False, False],
            [True, True],
        ]
        for step, (make_move, accepted, key) in enumerate(
            zip(moves, accepts, jax.random.split(jax.random.PRNGKey(12), len(moves)))
        ):
            move = make_move(state, key)
            result = pot(state, move)
            accept = Table.arange(jnp.array(accepted), label=SystemId)
            state = move(result.patch(state, accept), accept)
            committed = _graph_full(state, params, lennard_jones_energy)
            npt.assert_allclose(
                state.cache.total.total_energies.data,
                committed,
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"step {step}",
            )


class TestAdditivePairEnergy:
    def test_sum_with_shared_group_masks(self):
        state = _make_state(jax.random.key(84), (12, 9), (10.0, 12.0), n_inactive=2)
        params = _lj_params(2, 3.0)
        engine: FusedNeighborEnergy[
            _MiniState, LennardJonesParameters, _Points, tuple[PairData, ...]
        ] = FusedNeighborEnergy(_LJ_PAIR + _LJ_PAIR, _PARAMS)
        actual = _potential(engine, params)(state).data.total_energies.data
        expected = 2 * _graph_full(state, params, lennard_jones_energy)
        npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    @staticmethod
    def _engine[State: _MiniState](
        table_lens: Lens[State, CellTable[tuple[PairData, ...]]] | None = None,
    ) -> FusedNeighborEnergy[State, _PairParameters, _Points, tuple[PairData, ...]]:
        pair: PairEnergySum[_PairParameters, _Points] = _LJ_PAIR.with_parameters(
            lambda p: p[0]
        ) + _EWALD_PAIR.with_parameters(lambda p: p[1])
        return FusedNeighborEnergy(pair, _PARAMS, table_lens)

    @staticmethod
    def _reference(state: _MiniState, params: _PairParameters) -> jax.Array:
        return _graph_full(state, params[0], lennard_jones_energy) + _graph_full(
            _per_particle_exclusion(state), params[1], ewald_short_range_energy
        )

    def test_bounded_local_queries_with_uneven_system_counts(self):
        state = _make_state(jax.random.key(32), (8, 8, 8, 8), (12.0,) * 4, n_inactive=2)
        params = (_lj_params(4, 3.0), _ewald_params(4, 3.5))
        engine = dataclasses.replace(self._engine(), max_queries_per_system=2)
        state = _with_cache(state, self._reference(state, params))
        pot = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        # Unsorted queries, unequal active counts per system, and inactive rows
        # from several systems. Two distinct groups interact within systems 0/3.
        ids = jnp.array([30, 0, 11, 20, 1, 31, 8, 18, 28, 38])
        positions = jnp.array(
            [
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
                [4.0, 3.0, 3.0],
                [3.0, 4.0, 3.0],
                [5.0, 3.0, 3.0],
                [5.0, 3.0, 3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        move = _Move(
            Index(state.particles.keys, ids),
            dataclasses.replace(
                state.particles[Index(state.particles.keys, ids)],
                positions=positions,
            ),
        )
        proposed = move(state, state.systems.set_data(jnp.ones(4, bool)))
        result = as_result_function(lambda: pot(state, move))()
        result.raise_assertion()
        npt.assert_allclose(
            result.value.data.total_energies.data,
            self._reference(proposed, params),
            rtol=1e-9,
            atol=1e-9,
        )
        too_small = dataclasses.replace(engine, max_queries_per_system=1)
        cloud = PointCloud(state.particles, state.systems)
        inp = FusedLocalInput(
            params,
            cloud,
            (state.particles.subset(move.indices),),
            move.indices,
            engine.build_cell_table(params, cloud),
        )
        with pytest.raises(AssertionError, match="max_queries_per_system"):
            as_result_function(lambda: too_small(inp))().raise_assertion()

    def test_local_sum_aligns_feature_vocabularies(self):
        state = _make_state(jax.random.key(44), (8,), (12.0,))
        state = _with_points(
            state,
            labels=Index((_LABELS[0],), jnp.zeros(state.particles.size, int)),
        )
        params = (_lj_params(1, 3.0), _ewald_params(1, 3.5))
        engine = self._engine()
        state = _with_cache(state, self._reference(state, params))
        potential = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        move = _move(state, jnp.array([0]), state.particles.data.positions[:1] + 0.2)
        move = dataclasses.replace(
            move,
            data=dataclasses.replace(
                move.data, labels=Index((_LABELS[1],), jnp.zeros(1, int))
            ),
        )
        proposed = move(state, state.systems.set_data(jnp.ones(1, bool)))
        npt.assert_allclose(
            potential(state, move).data.total_energies.data,
            self._reference(proposed, params),
            rtol=1e-9,
            atol=1e-10,
        )

    def test_sum_with_distinct_cutoffs_and_exclusions(self):
        state = _make_state(jax.random.key(21), (16, 12), (12.0, 10.0), n_inactive=2)
        # LJ has one broadcast cutoff; Ewald has two distinct system cutoffs.
        lj = _lj_params(1, 2.8)
        ew = dataclasses.replace(
            _ewald_params(2, 3.5),
            cutoff=Table.arange(jnp.array([3.5, 4.0]), label=SystemId),
        )
        engine = self._engine()
        npt.assert_allclose(engine.pair.cutoffs((lj, ew)).data, [3.5, 4.0])
        result = _potential(engine, (lj, ew))(state)
        npt.assert_allclose(
            result.data.total_energies.data,
            self._reference(state, (lj, ew)),
            rtol=1e-10,
        )
        # Composition remains flat when appending additional terms.
        assert isinstance(engine.pair, PairEnergySum)
        triple = engine.pair + engine.pair.terms[0]
        assert len(triple.terms) == 3
        doubled_lj = dataclasses.replace(engine, pair=triple)
        npt.assert_allclose(
            _potential(doubled_lj, (lj, ew))(state).data.total_energies.data,
            self._reference(state, (lj, ew))
            + _graph_full(state, lj, lennard_jones_energy),
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        "key_layout,key_chunk_size", [("cells", None), ("cells", 4), ("slots", 8)]
    )
    def test_position_and_cell_derivatives(
        self, key_layout: _KeyLayout, key_chunk_size: int | None
    ):
        state = _make_state(jax.random.key(22), (14, 10), (12.0, 10.0), n_inactive=2)
        params = (_lj_params(2, 3.0), _ewald_params(2, 3.5))
        # Keep random close contacts from magnifying cancellation in cell derivatives.
        params = (
            dataclasses.replace(params[0], sigma=params[0].sigma * 0.25),
            params[1],
        )
        engine = dataclasses.replace(
            self._engine(),
            layout=dataclasses.replace(
                _PARAMS, key_layout=key_layout, key_chunk_size=key_chunk_size
            ),
        )
        edges_lj = _neighborlist(params[0])(state.particles, state.systems)
        atomic = _per_particle_exclusion(state)
        edges_ew = _neighborlist(params[1])(atomic.particles, state.systems)

        def energy(positions: jax.Array, vectors: jax.Array, fused: bool):
            moved = _with_points(state, positions=positions)
            systems: Table[SystemId, HasCell[AnyPeriodicity]] = Table(
                state.systems.keys,
                _System(
                    Cell(
                        TriclinicFrame.from_matrix(vectors), periodic=(True, True, True)
                    )
                ),
            )
            if fused:
                return engine(
                    _full_input(engine, params, PointCloud(moved.particles, systems))
                ).data.data.sum()
            lj = lennard_jones_energy(
                GraphPotentialInput(
                    params[0], HyperGraph(moved.particles, systems, edges_lj)
                )
            ).data.data.sum()
            ew = ewald_short_range_energy(
                GraphPotentialInput(
                    params[1],
                    HyperGraph(
                        _per_particle_exclusion(moved).particles, systems, edges_ew
                    ),
                )
            ).data.data.sum()
            return lj + ew

        x = state.particles.data.positions
        vectors = state.systems.data.cell.vectors
        for fused, reference in zip(
            jax.grad(lambda x, h: energy(x, h, True), (0, 1))(x, vectors),
            jax.grad(lambda x, h: energy(x, h, False), (0, 1))(x, vectors),
        ):
            assert jnp.isfinite(fused).all()
            npt.assert_allclose(fused, reference, rtol=1e-9, atol=1e-9)

    def test_persistent_sum_multiple_groups_and_mixed_acceptance(self):
        state = _make_state(jax.random.key(23), (16, 12), (12.0, 10.0), n_inactive=2)
        params = (_lj_params(2, 3.0), _ewald_params(2, 3.5))
        engine = self._engine(lens(lambda s: s.cell_table))
        state = _with_cache(
            state,
            self._reference(state, params),
            engine.build_cell_table(params, PointCloud(state.particles, state.systems)),
        )
        potential = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        # Multiple exclusion groups change together, including one inactive row.
        rows = jnp.array([0, 1, 16, 18, 19, 30])
        for step, accepted in enumerate(([True, False], [False, True], [True, True])):
            positions = (
                jnp.array(
                    [
                        [2.0, 3.0, 4.0],
                        [4.0, 3.0, 4.0],
                        [3.0, 5.0, 4.0],
                        [2.0, 4.0, 5.0],
                        [4.0, 4.0, 5.0],
                        [3.0, 6.0, 5.0],
                    ]
                )
                + step * 0.2
            )
            move = _move(state, rows, positions)
            result = potential(state, move)
            proposed = move(state, state.systems.set_data(jnp.ones(2, bool)))
            npt.assert_allclose(
                result.data.total_energies.data,
                self._reference(proposed, params),
                rtol=1e-9,
                atol=1e-9,
            )
            accept = state.systems.set_data(jnp.array(accepted))
            state = move(result.patch(state, accept), accept)
            npt.assert_allclose(
                state.cache.total.total_energies.data,
                self._reference(state, params),
                rtol=1e-9,
                atol=1e-9,
            )

    def test_full_recomputation_with_patch_without_probe(self):
        state = _make_state(jax.random.key(24), (16, 12), (12.0, 10.0))
        params = (_lj_params(2, 3.0), _ewald_params(2, 3.5))
        move = _move(
            state, jnp.array([0, 16]), jnp.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
        )
        proposed = move(state, state.systems.set_data(jnp.ones(2, bool)))
        npt.assert_allclose(
            _potential(self._engine(), params)(state, move).data.total_energies.data,
            self._reference(proposed, params),
            rtol=1e-10,
        )

    def test_shared_inclusion_does_not_connect_different_systems(self):
        from kups.core.typing import InclusionId

        state = _make_state(jax.random.key(25), (5, 5), (12.0, 10.0))
        state = _with_points(
            state, inclusion=Index((InclusionId(0),), jnp.zeros(10, int))
        )
        params = (_lj_params(2, 3.0), _ewald_params(2, 3.5))
        engine = self._engine()
        state = _with_cache(
            state, _potential(engine, params)(state).data.total_energies.data
        )
        pot = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        rows = jnp.array([0, 5])
        # Nearby coordinates in independent systems must not add an intra-query edge.
        move = _move(state, rows, jnp.array([[3.0, 3.0, 3.0], [3.1, 3.0, 3.0]]))
        move = dataclasses.replace(
            move,
            data=dataclasses.replace(
                move.data, inclusion=Index((InclusionId(0),), jnp.zeros(2, int))
            ),
        )
        proposed = move(state, state.systems.set_data(jnp.ones(2, bool)))
        npt.assert_allclose(
            pot(state, move).data.total_energies.data,
            _potential(engine, params)(proposed).data.total_energies.data,
            rtol=1e-9,
            atol=1e-9,
        )

    def test_empty_cloud(self):
        state = _make_state(jax.random.key(26), (4, 4), (12.0, 10.0))
        empty = state.particles.subset(Index(state.particles.keys, jnp.array([], int)))
        state = dataclasses.replace(state, particles=empty)
        params = (_lj_params(2, 3.0), _ewald_params(2, 3.5))
        npt.assert_array_equal(
            _potential(self._engine(), params)(state).data.total_energies.data,
            jnp.zeros(2),
        )

    def test_generic_selector_can_be_consumed_in_chunks(self):
        from kups.core.neighborlist.all_dense import AllDenseSelector
        from kups.core.neighborlist.pipeline import _prepare

        state = _make_state(jax.random.key(27), (8, 4), (12.0, 10.0), n_inactive=2)
        params = (_lj_params(2, 3.0), _ewald_params(2, 3.5))
        pair = self._engine().pair
        ctx = _prepare(state.particles, None, state.systems, None)
        features = pair.features(state.particles.data)
        selector = AllDenseSelector(
            pair.cutoffs(params), FixedCapacity(state.particles.size * 4)
        )

        def consume(indices: jax.Array):
            local = dataclasses.replace(ctx, queried_keys=indices)
            batch = selector(local)
            pairs = PairBatch.from_candidates(batch, local)
            pairs = pairs._replace(
                valid=pairs.valid & (batch.key_idx.indices != batch.query_idx.indices)
            )
            left = jax.tree.map(lambda x: x[batch.query_idx.indices], features)
            right = jax.tree.map(lambda x: x[batch.key_idx.indices], features)
            energies = pair.evaluate(params, left, right, pairs)
            return jax.ops.segment_sum(energies, pairs.system.indices, num_segments=2)

        result = (
            jax.jit(lambda: sum_chunks(consume, jnp.arange(state.particles.size), 4))()
            / 2
        )
        npt.assert_allclose(result, self._reference(state, params), rtol=1e-10)

    def test_fused_gradient_matches_application_geometry_lens(self):
        from kups.application.potential.classical.lennard_jones import (
            make_lennard_jones_from_state,
        )
        from kups.application.potential.filter import POSITIONS_AND_CELL
        from kups.core.lens import identity_lens

        state = _make_state(jax.random.key(28), (8, 4), (12.0, 10.0), n_inactive=2)
        params = _lj_params(2, 3.0)
        params = dataclasses.replace(params, sigma=params.sigma * 0.25)
        fused = make_fused_potential(
            _LJ,
            lambda s: s.particles,
            lambda s: s.systems,
            lambda s: params,
            None,
            FUSED_GEOMETRY.nest(POSITIONS_AND_CELL),
            EMPTY_LENS,
            EMPTY_LENS,
        )(state).data
        graph = make_lennard_jones_from_state(
            identity_lens(_MiniState),
            neighborlist_factory=lambda state, cutoffs: _neighborlist(params),
            parameters=params,
            gradient=POSITIONS_AND_CELL,
        )(state).data
        for a, b in zip(jax.tree.leaves(fused), jax.tree.leaves(graph)):
            npt.assert_allclose(a, b, rtol=1e-9, atol=1e-8)

    def test_query_inclusion_vocabulary_is_independent(self):
        state = _make_state(jax.random.key(29), (8,), (12.0,))
        params = _ewald_params(1, 3.5)
        removed = Index(state.particles.keys, jnp.array([0]))
        old = state.particles.subset(removed)
        new = dataclasses.replace(
            old,
            data=dataclasses.replace(
                old.data,
                positions=jnp.array([[3.0, 4.0, 5.0]]),
                inclusion=Index((InclusionId(0), InclusionId(1)), jnp.array([1])),
            ),
        )
        proposal = _with_points(
            state,
            positions=state.particles.data.positions.at[0].set(new.data.positions[0]),
            inclusion=Index(new.data.inclusion.keys, jnp.zeros(8, int).at[0].set(1)),
        )
        # The new group is active, even though it is absent from the key vocabulary.
        result = _EWALD(
            FusedLocalInput(
                params,
                PointCloud(state.particles, state.systems),
                (old, new),
                removed,
                _EWALD.build_cell_table(
                    params, PointCloud(state.particles, state.systems)
                ),
            )
        ).data.data
        before = _EWALD(
            _full_input(_EWALD, params, PointCloud(state.particles, state.systems))
        ).data.data
        after = _EWALD(
            _full_input(
                _EWALD, params, PointCloud(proposal.particles, proposal.systems)
            )
        ).data.data
        npt.assert_allclose(result + before, after, rtol=1e-10, atol=1e-12)

    def test_buffered_deletion_uses_old_system_for_acceptance(self):
        state = _make_state(jax.random.key(30), (16, 12), (12.0, 10.0), n_inactive=2)
        params = (_lj_params(2, 3.0), _ewald_params(2, 3.5))
        engine = self._engine(lens(lambda s: s.cell_table))
        state = _with_cache(
            state,
            self._reference(state, params),
            engine.build_cell_table(params, PointCloud(state.particles, state.systems)),
        )
        potential = _potential(
            engine,
            params,
            _probe,
            patch_idx_view=_patch_indices,
            cache_lens=lens(lambda s: s.cache),
        )
        deleted = jnp.concatenate([_molecule_rows(state, 0), _molecule_rows(state, 1)])
        positions = state.particles.data.positions[deleted]
        deletion = _move(state, deleted, positions, active=False)
        deletion = dataclasses.replace(
            deletion,
            data=dataclasses.replace(
                deletion.data,
                system=Index(
                    state.systems.keys, jnp.full(deleted.size, state.systems.size)
                ),
            ),
        )
        accept = state.systems.set_data(jnp.array([False, True]))
        result = potential(state, deletion)
        state = deletion(result.patch(state, accept), accept)
        npt.assert_allclose(
            state.cache.total.total_energies.data,
            self._reference(state, params),
            rtol=1e-9,
            atol=1e-9,
        )
        # A subsequent nearby move must not see the deleted molecule in the table.
        move = _move(state, jnp.array([0, 18]), positions[jnp.array([0, 3])] + 0.6)
        result = potential(state, move)
        proposed = move(state, state.systems.set_data(jnp.ones(2, bool)))
        npt.assert_allclose(
            result.data.total_energies.data,
            self._reference(proposed, params),
            rtol=1e-9,
            atol=1e-9,
        )


class TestSumChunks:
    @pytest.mark.parametrize("disable_jit", [False, True])
    def test_shape_discovery_reuses_consumer_trace(self, disable_jit):
        traced = []

        def consume(chunk):
            traced.append(chunk.shape)
            return jnp.sum(chunk**2)

        def evaluate(rows):
            with jax.disable_jit(disable_jit):
                return sum_chunks(consume, rows, 4)

        rows = jnp.arange(12.0)
        result = jax.jit(evaluate)(rows)
        npt.assert_allclose(result, jnp.sum(rows**2))
        assert traced == [(4,)]

    def test_chunk_tracing_does_not_grow_with_number_of_chunks(self):
        def evaluate(rows: jax.Array):
            # MCMC uses this context while constructing and applying patches.
            with jax.disable_jit():
                return sum_chunks(lambda chunk: jnp.sum(chunk**2), rows, 4)

        programs = [
            jax.make_jaxpr(evaluate)(jnp.arange(n, dtype=float)) for n in (8, 800)
        ]
        assert len(programs[0].jaxpr.eqns) == len(programs[1].jaxpr.eqns)
        for program in programs:
            assert any(eqn.primitive.name == "scan" for eqn in program.jaxpr.eqns)
        rows = jnp.arange(12, dtype=float)
        npt.assert_allclose(jax.jit(evaluate)(rows), jnp.sum(rows**2))
        npt.assert_allclose(jax.jit(jax.grad(evaluate))(rows), 2 * rows)

    @pytest.mark.parametrize("disable_jit", [False, True])
    @pytest.mark.parametrize("n_chunks", [1, 3])
    def test_inactive_chunks_skip_consumer(self, disable_jit: bool, n_chunks: int):
        visited: list[float] = []

        def consume(chunk: tuple[jax.Array, jax.Array]):
            values, _ = chunk
            jax.debug.callback(lambda x: visited.append(float(x)), values[0])
            runtime_assert(jnp.all(values >= 0), "negative active row")
            return jnp.sum(values**2).astype(jnp.float32)

        def evaluate(values: jax.Array, active: jax.Array):
            with jax.disable_jit(disable_jit):
                return sum_chunks(
                    consume,
                    (values, active),
                    2,
                    active=lambda chunk: jnp.any(chunk[1]),
                )

        flags = jnp.arange(2 * n_chunks) >= 2 * (n_chunks - 1)
        values = jnp.where(flags, jnp.arange(2 * n_chunks, dtype=jnp.float64), -1.0)
        run = jax.jit(as_result_function(evaluate))
        result = run(values, flags)
        result.raise_assertion()
        jax.effects_barrier()
        assert visited == [float(values[-2])]
        assert result.value.dtype == jnp.float32
        npt.assert_allclose(result.value, jnp.sum(values[-2:] ** 2))
        visited.clear()
        empty = run(values, jnp.zeros_like(flags))
        empty.raise_assertion()
        jax.effects_barrier()
        assert visited == []
        assert empty.value == 0
        npt.assert_allclose(
            jax.jit(jax.grad(lambda v: run(v, flags).value))(values),
            jnp.where(flags, 2 * values, 0),
        )
        with pytest.raises(AssertionError, match="negative active row"):
            run(-jnp.ones_like(values), flags).raise_assertion()

    @pytest.mark.parametrize("disable_jit", [False, True])
    def test_chunked_selector_assertions_survive_checkpointing(self, disable_jit: bool):
        def consume(rows: jax.Array):
            runtime_assert(jnp.all(rows >= 0), "negative chunk entry")
            return jnp.sum(rows**2)

        def evaluate(rows: jax.Array):
            with jax.disable_jit(disable_jit):
                return sum_chunks(consume, rows, 2)

        run = jax.jit(as_result_function(evaluate))
        valid = run(jnp.array([1.0, 2.0, 3.0, 4.0]))
        valid.raise_assertion()
        npt.assert_allclose(valid.value, 30.0)
        # A failure in a later chunk must propagate out through remat and scan.
        with pytest.raises(AssertionError, match="negative chunk entry"):
            run(jnp.array([1.0, 2.0, -3.0, 4.0])).raise_assertion()
        empty = run(jnp.zeros(0))
        empty.raise_assertion()
        npt.assert_allclose(empty.value, 0.0)
