# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.mcmc import (
    MCMCGroup,
    MCMCParticles,
    MCMCSystems,
)
from kups.application.mcmc.data import MotifParticles, RunConfig
from kups.application.simulations.mcmc_rigid import (
    MCMCState,
    MCMCStateUpdate,
    _probe,
    make_propagator,
)
from kups.core.data import Table, WithCache, WithIndices
from kups.core.data.buffered import Buffered
from kups.core.data.index import Index
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.parameter_scheduler import (
    AcceptanceHistory,
    Correlation,
    ParameterSchedulerState,
)
from kups.core.potential import EMPTY, PotentialOut
from kups.core.typing import (
    GroupId,
    Label,
    MotifId,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import TriclinicUnitCell
from kups.mcmc.moves import (
    ExchangeChanges,
    ExchangeGroupData,
    ExchangeParticleData,
    ParticlePositionChanges,
    exchange_changes_from_position_changes,
)
from kups.potential.classical.ewald import EwaldCache, EwaldParameters
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
)


@pytest.fixture(autouse=True, scope="module")
def clear_cache():
    """Override class-scoped clear_cache: clear once per module, not per class."""
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()


L = 15.0  # box side (Ang)
N_MAX = 3  # max molecules (2 real + 1 empty slot for exchange)
PI = tuple(ParticleId(i) for i in range(N_MAX))  # particle index labels


def _build_state() -> MCMCState:
    positions = jnp.array(
        [
            [2.0, 2.0, 2.0],  # molecule 0
            [9.0, 9.0, 9.0],  # molecule 1
            [0.0, 0.0, 0.0],  # empty slot
        ]
    )
    sys_ids = jnp.zeros((N_MAX,), dtype=int)
    grp_ids = jnp.arange(N_MAX, dtype=int)

    particles = Buffered.arange(
        MCMCParticles(
            positions=positions,
            masses=jnp.full((N_MAX,), 40.0),
            atomic_numbers=jnp.full((N_MAX,), 18),
            charges=jnp.zeros((N_MAX,)),
            labels=Index.new([Label("Ar")] * N_MAX),
            system=Index.integer(sys_ids, label=SystemId, max_count=N_MAX),
            group=Index.integer(grp_ids, label=GroupId, max_count=1),
            motif=Index.integer(jnp.zeros((N_MAX,), dtype=int), label=MotifParticleId),
        ),
        label=ParticleId,
    )
    groups = Buffered.arange(
        MCMCGroup(
            system=Index.integer(sys_ids, label=SystemId, max_count=N_MAX),
            motif=Index.integer(jnp.zeros((N_MAX,), dtype=int), label=MotifId),
        ),
        label=GroupId,
    )
    motifs = Table.arange(
        MotifParticles(
            positions=jnp.zeros((1, 3)),
            masses=jnp.ones(1),
            atomic_numbers=jnp.array([18]),
            charges=jnp.zeros(1),
            labels=Index.new([Label("Ar")]),
            motif=Index.integer(jnp.zeros(1, dtype=int), label=MotifId, max_count=1),
        ),
        label=MotifParticleId,
    )
    systems = Table.arange(
        MCMCSystems(
            unitcell=TriclinicUnitCell.from_matrix(jnp.eye(3)[None] * L),
            temperature=jnp.array([300.0]),
            potential_energy=jnp.array([0.0]),
            log_fugacity=jnp.array([[0.0]]),  # (n_sys, n_motifs)
        ),
        label=SystemId,
    )
    lj_params = WithCache(
        GlobalTailCorrectedLennardJonesParameters(
            labels=(Label("Ar"),),
            sigma=jnp.array([[3.4]]),
            epsilon=jnp.array([[1.0]]),
            cutoff=Table((SystemId(0),), jnp.array([8.0])),
            tail_corrected=jnp.array([[True]]),
        ),
        PotentialOut(Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY),
    )
    ewald_params = WithCache(
        EwaldParameters(
            alpha=Table((SystemId(0),), jnp.array([0.0])),
            cutoff=Table((SystemId(0),), jnp.array([0.0])),  # disabled
            reciprocal_lattice_shifts=Table(
                (SystemId(0),), jnp.zeros((1, 1, 3), dtype=int)
            ),
        ),
        EwaldCache(
            structure_factor=jnp.zeros((1, 1, 2)),
            short_range=PotentialOut(
                Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
            ),
            long_range=PotentialOut(
                Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
            ),
            self_interaction=PotentialOut(
                Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
            ),
            exclusion=PotentialOut(
                Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
            ),
        ),
    )
    move_params = ParameterSchedulerState(
        value=jnp.array([1.0]),
        multiplicity=jnp.array([1.1]),
        target=jnp.array([0.5]),
        tolerance=jnp.array([0.05]),
        correlation=Correlation.NEGATIVE,
        bounds=(jnp.zeros((1,)), jnp.array([L / 2])),
        history=AcceptanceHistory(
            values=jnp.zeros((1, 10)),
            index=jnp.zeros((1,), dtype=int),
        ),
    )
    return MCMCState(
        particles=particles,
        groups=groups,
        motifs=motifs,
        systems=systems,
        neighborlist_params=UniversalNeighborlistParameters(
            avg_edges=N_MAX**2,
            avg_candidates=N_MAX**2,
            avg_image_candidates=N_MAX**2,
            cells=N_MAX**2,
        ),
        lj_parameters=lj_params,
        ewald_parameters=ewald_params,
        translation_params=Table.arange(move_params, label=SystemId),
        rotation_params=Table.arange(move_params, label=SystemId),
        reinsertion_params=Table.arange(move_params, label=SystemId),
        exchange_params=Table.arange(move_params, label=SystemId),
    )


@pytest.fixture(scope="module")
def state() -> MCMCState:
    return _build_state()


def _movement_patch(key, state, changes):
    """Convert position changes to exchange format and build update."""
    proposal = exchange_changes_from_position_changes(
        changes,
        state.particles,
        state.groups,
        state.particles.data.system.num_labels,
    )
    return MCMCStateUpdate.from_changes(key, state, proposal)


@pytest.fixture(scope="module")
def movement_update_pid0(state):
    """Cached _movement_patch result for particle 0 (reused by many tests)."""
    return _movement_patch(
        jax.random.key(0),
        state,
        ParticlePositionChanges(
            particle_ids=Index(PI, jnp.array([0])),
            new_positions=jnp.array([[3.0, 3.0, 3.0]]),
        ),
    )


@pytest.fixture(scope="module")
def movement_update_newpos(state):
    """Cached _movement_patch result with new_pos=[3,4,5] for particle 0."""
    new_pos = jnp.array([[3.0, 4.0, 5.0]])
    update = _movement_patch(
        jax.random.key(0),
        state,
        ParticlePositionChanges(
            particle_ids=Index(PI, jnp.array([0])), new_positions=new_pos
        ),
    )
    return update, new_pos


def _make_exchange_proposal(particle_ids, new_positions, group_ids):
    n = particle_ids.shape[0]
    gi = tuple(GroupId(i) for i in range(N_MAX))
    particle_data = ExchangeParticleData(
        new_positions=new_positions,
        group=Index.integer(group_ids, n=N_MAX, label=GroupId, max_count=1),
        system=Index.integer(
            jnp.zeros_like(group_ids), label=SystemId, max_count=N_MAX
        ),
        motif=Index.integer(jnp.zeros((n,), dtype=int), label=MotifParticleId),
    )
    group_data = ExchangeGroupData(
        motif=Index.integer(jnp.zeros((n,), dtype=int), label=MotifId),
        system=Index.integer(
            jnp.zeros((n,), dtype=int), label=SystemId, max_count=N_MAX
        ),
    )
    p_idx = Index(PI, particle_ids)
    g_idx = Index(gi, group_ids)
    return ExchangeChanges(
        particles=WithIndices(p_idx, Buffered.arange(particle_data, label=ParticleId)),
        groups=WithIndices(g_idx, Buffered.arange(group_data, label=GroupId)),
    )


@pytest.fixture(scope="module")
def exchange_update_default(state):
    """Cached _exchange_patch result for particle 2, group 2, pos=[5,5,5]."""
    proposal = _make_exchange_proposal(
        jnp.array([2]), jnp.array([[5.0, 5.0, 5.0]]), jnp.array([2])
    )
    return MCMCStateUpdate.from_changes(jax.random.key(0), state, proposal)


class TestProbe:
    def test_probe_is_identity(self, state, movement_update_pid0):
        result = _probe(state, movement_update_pid0)
        assert result is movement_update_pid0


class TestPatchFn:
    def test_new_positions_stored(self, movement_update_newpos):
        update, new_pos = movement_update_newpos
        npt.assert_allclose(update._particles.data.data.positions, new_pos)

    def test_indices_stored(self, state):
        ids = Index(PI, jnp.array([1]))
        update = _movement_patch(
            jax.random.key(0),
            state,
            ParticlePositionChanges(
                particle_ids=ids, new_positions=jnp.array([[5.0, 5.0, 5.0]])
            ),
        )
        npt.assert_array_equal(update._particles.indices.indices, jnp.array([1]))

    def test_non_particle_fields_from_motif(self, movement_update_pid0):
        npt.assert_array_equal(
            movement_update_pid0._particles.data.data.labels.indices, jnp.array([0])
        )
        # Masses now come from motif data (1.0), not original particles (40.0)
        npt.assert_allclose(
            movement_update_pid0._particles.data.data.masses, jnp.array([1.0])
        )

    def test_group_changes_single(self, movement_update_pid0):
        assert movement_update_pid0.groups.indices.indices.shape == (1,)

    def test_edges_have_valid_indices(self, movement_update_pid0):
        after_idx = movement_update_pid0.edges_after.indices.indices
        before_idx = movement_update_pid0.edges_before.indices.indices
        assert after_idx.shape[-1] == 2
        assert before_idx.shape[-1] == 2

    def test_neighborlists_return_correct_type(self, movement_update_pid0):
        from kups.core.neighborlist import RefineMaskNeighborList

        assert isinstance(
            movement_update_pid0.neighborlist_after, RefineMaskNeighborList
        )
        assert isinstance(
            movement_update_pid0.neighborlist_before, RefineMaskNeighborList
        )


class TestExchPatchFn:
    def test_new_positions_stored(self, state):
        new_pos = jnp.array([[5.0, 6.0, 7.0]])
        proposal = _make_exchange_proposal(jnp.array([2]), new_pos, jnp.array([2]))
        update = MCMCStateUpdate.from_changes(jax.random.key(0), state, proposal)
        npt.assert_allclose(update._particles.data.data.positions, new_pos)

    def test_motif_properties_used(self, exchange_update_default):
        npt.assert_array_equal(
            exchange_update_default._particles.data.data.labels.indices, jnp.array([0])
        )
        # Mass comes from motifs (1.0), not from the particle fixture (40.0)
        npt.assert_allclose(
            exchange_update_default._particles.data.data.masses, jnp.array([1.0])
        )

    def test_group_indices_stored(self, exchange_update_default):
        npt.assert_array_equal(
            exchange_update_default.groups.indices.indices, jnp.array([2])
        )

    def test_returns_mcmc_state_update(self, exchange_update_default):
        assert isinstance(exchange_update_default, MCMCStateUpdate)


class TestMCMCStateUpdate:
    def test_accept_updates_position(self, state, movement_update_newpos):
        update, new_pos = movement_update_newpos
        new_state = update(
            state, state.systems.set_data(jnp.ones(len(state.systems), dtype=bool))
        )
        npt.assert_allclose(new_state.particles.data.positions[0], new_pos[0])

    def test_accept_leaves_other_particles(self, state, movement_update_newpos):
        update, _ = movement_update_newpos
        new_state = update(
            state, state.systems.set_data(jnp.ones(len(state.systems), dtype=bool))
        )
        npt.assert_allclose(
            new_state.particles.data.positions[1], state.particles.data.positions[1]
        )

    def test_reject_leaves_all_positions(self, state, movement_update_newpos):
        update, _ = movement_update_newpos
        new_state = update(
            state, state.systems.set_data(jnp.zeros(len(state.systems), dtype=bool))
        )
        npt.assert_allclose(
            new_state.particles.data.positions, state.particles.data.positions
        )


class TestMakePropagator:
    def test_creates_propagator(self):
        config = RunConfig(
            out_file="/tmp/test.h5",
            num_cycles=1,
            num_warmup_cycles=0,
            min_cycle_length=10,
        )
        potential, propagator = make_propagator(config)
        assert callable(propagator)
        assert callable(potential)
