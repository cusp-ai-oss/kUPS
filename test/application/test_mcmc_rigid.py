# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.mcmc import (
    MCMCGroup,
    MCMCParticles,
    MCMCSystems,
)
from kups.application.mcmc.analysis import analyze_mcmc_file
from kups.application.mcmc.data import (
    AdsorbateConfig,
    BlockingSphereConfig,
    HostConfig,
    MotifParticles,
    RunConfig,
)
from kups.application.mcmc.logging import MCMCLoggedData, MCMCStepData
from kups.application.mcmc.rigid_body_composition import make_rigid_body_composition
from kups.application.potential.classical.blocking import (
    make_blocking_spheres_from_state,
)
from kups.application.potential.classical.ewald import make_ewald_from_state
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
    make_lennard_jones_tail_correction_from_state,
)
from kups.application.simulations.mcmc_rigid import (
    Config,
    EwaldConfig,
    LJConfig,
    MCMCState,
    MCMCStateUpdate,
    _make_potential,
    _probe,
    init_cell_tables,
    init_state,
    make_guest_stress,
    make_propagator,
    run,
)
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Table, WithCache, WithIndices
from kups.core.data.buffered import Buffered
from kups.core.data.index import Index
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.parameter_scheduler import (
    AcceptanceHistory,
    Correlation,
    ParameterSchedulerState,
)
from kups.core.potential import (
    EMPTY,
    PotentialAsPropagator,
    PotentialOut,
    ScaledPotential,
    sum_potentials,
)
from kups.core.storage import HDF5StorageReader
from kups.core.typing import (
    GroupId,
    Label,
    MotifId,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.utils.kahan import KahanSummand
from kups.mcmc.moves import (
    ExchangeChanges,
    ExchangeGroupData,
    ExchangeParticleData,
    ParticlePositionChanges,
    delete_random_motif,
    exchange_changes_from_position_changes,
    insert_random_motif,
    propose_group_rotation,
    propose_group_translation,
    propose_reinsertion,
)
from kups.potential.classical.blocking import BlockingSpheresParameters
from kups.potential.classical.ewald import EwaldCache, EwaldParameters
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
)
from kups.potential.common.graph import PointCloud

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
            cell=PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * L)),
            temperature=jnp.array([300.0]),
            potential_energy=KahanSummand.init(jnp.array([0.0])),
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
        KahanSummand.init(
            PotentialOut(Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY)
        ),
    )
    ewald_params = WithCache(
        EwaldParameters(
            alpha=Table((SystemId(0),), jnp.array([0.0])),
            cutoff=Table((SystemId(0),), jnp.array([0.0])),  # disabled
            k_max=Table((SystemId(0),), jnp.zeros(1)),
            reciprocal_lattice_shifts=Table(
                (SystemId(0),), jnp.zeros((1, 1, 3), dtype=int)
            ),
        ),
        EwaldCache(
            structure_factor=KahanSummand.init(jnp.zeros((1, 1, 2))),
            short_range=KahanSummand.init(
                PotentialOut(
                    Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
                )
            ),
            long_range=KahanSummand.init(
                PotentialOut(
                    Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
                )
            ),
            self_interaction=KahanSummand.init(
                PotentialOut(
                    Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
                )
            ),
            exclusion=KahanSummand.init(
                PotentialOut(
                    Table.arange(jnp.zeros((1,)), label=SystemId), EMPTY, EMPTY
                )
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
    cell_tables = init_cell_tables(
        particles, motifs, systems, lj_params.data, ewald_params.data
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
        blocking_spheres_parameters=BlockingSpheresParameters(
            radii=jnp.zeros((0,)),
            positions=jnp.zeros((0, 3)),
            system=Index.arange(0, label=SystemId),
            motif=Index.arange(0, label=MotifId),
        ),
        blocking_spheres_neighborlist_params=UniversalNeighborlistParameters(
            avg_edges=0,
            avg_candidates=0,
            avg_image_candidates=0,
            cells=0,
        ),
        translation_params=Table.arange(move_params, label=SystemId),
        rotation_params=Table.arange(move_params, label=SystemId),
        reinsertion_params=Table.arange(move_params, label=SystemId),
        exchange_params=Table.arange(move_params, label=SystemId),
        cell_tables=cell_tables,
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
    def test_creates_propagator(self, state):
        config = RunConfig(
            out_file="/tmp/test.h5",
            num_cycles=1,
            num_warmup_cycles=0,
            min_cycle_length=10,
        )
        potential, propagator = make_propagator(state, config)
        assert callable(propagator)
        assert callable(potential)


class TestMakeGuestStress:
    def test_evaluates_through_cache_lens(self, state: MCMCState):
        """Guest stress applies its cache patch and yields a physical tensor."""
        # Applying the patch writes back through the cache lens, which only
        # resolves if every leaf of its focus function is a settable path.
        stress = make_guest_stress(state)(jax.random.key(0), state).data.potential
        assert jnp.isfinite(stress).all()
        assert jnp.abs(stress).max() > 0
        npt.assert_allclose(stress, jnp.swapaxes(stress, -1, -2), atol=1e-12)

    def test_tail_correction_stays_out_of_configurational_stress(
        self, state: MCMCState
    ):
        """The analytical tail reaches the stress only via ``tail_correction``.

        ``U_tail`` depends on the cell through the volume, so differentiating it
        would add ``(U_tail / V) * I`` to the configurational term on top of the
        closed-form correction that ``analyze_mcmc`` already sums in.
        """

        def stress(tail_corrected: bool):
            toggled = bind(state, lambda x: x.lj_parameters.data.tail_corrected).set(
                jnp.full((1, 1), tail_corrected, dtype=bool)
            )
            return make_guest_stress(toggled)(jax.random.key(0), toggled).data

        on, off = stress(True), stress(False)
        npt.assert_array_equal(on.potential, off.potential)
        # Guard: without this the test would also pass if the tail vanished.
        assert jnp.all(jnp.diagonal(on.tail_correction[0]) < 0)
        npt.assert_array_equal(off.tail_correction, jnp.zeros_like(off.tail_correction))


# --- End-to-end smoke tests for the rigid-body MCMC entry point ---

_BOX = 14.0  # box side (Å); cutoff 5.0 < box/2 for minimum image

# Hand-written empty-box CIF with one pseudo "X" host atom (no LJ params),
# i.e. CO2 adsorbing into vacuum; small enough to compile fast.
_EMPTY_CIF = f"""data_cell
_cell_length_a    {_BOX}
_cell_length_b    {_BOX}
_cell_length_c    {_BOX}
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
X1 X 0.0 0.0 0.0
"""


def _tmp_file(suffix: str, content: str | None = None) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    if content is not None:
        f.write(content)
    f.close()
    return f.name


def _co2() -> AdsorbateConfig:
    # Three-site charged CO2 (TraPPE): exercises Ewald + rigid rotation.
    return AdsorbateConfig(
        critical_temperature=303.75,
        critical_pressure=7.84e6,
        acentric_factor=0.22394,
        positions=((0.0, 0.0, 0.0), (-1.16, 0.0, 0.0), (1.16, 0.0, 0.0)),
        symbols=("C_co2", "O_co2", "O_co2"),
        charges=(0.7, -0.35, -0.35),
    )


def _config(*, exchange_prob: float, init_adsorbates: tuple[int, ...]) -> Config:
    return Config(
        adsorbates=(_co2(),),
        hosts=(
            HostConfig(
                cif_file=_tmp_file(".cif", _EMPTY_CIF),
                pressure=1e4,
                temperature=298.15,
                init_adsorbates=init_adsorbates,
                cell_replication=1,
            ),
        ),
        run=RunConfig(
            out_file=_tmp_file(".h5"),
            num_cycles=2,
            num_warmup_cycles=0,
            min_cycle_length=1,
            exchange_prob=exchange_prob,
            seed=42,
        ),
        lj=LJConfig(
            cutoff=5.0,
            tail_correction=True,
            mixing_rule="lorentz_berthelot",
            # (sigma [Å], epsilon [eV]); X1 is the host pseudo-type (no LJ).
            parameters={
                "O_co2": (3.05, 0.0068077),
                "C_co2": (2.8, 0.0023267),
                "X1": (None, None),
            },
        ),
        ewald=EwaldConfig(real_cutoff=5.0, precision=1e-3),
        max_num_adsorbates=4,
    )


def _assert_readable(out_file: str) -> None:
    results = analyze_mcmc_file(out_file, n_blocks=2)
    assert len(results) == 1
    result = next(iter(results.values()))
    assert jnp.isfinite(result.energy.mean).all().item()
    assert jnp.isfinite(result.loading.mean).all().item()
    assert (result.loading.mean >= 0.0).all().item()


def _batched_config(n_hosts: int, init_adsorbates: tuple[int, ...]) -> Config:
    """A multi-host (batched) CO2 config; each host is an independent system."""
    base = _config(exchange_prob=0.5, init_adsorbates=init_adsorbates)
    host = base.hosts[0]
    return base.model_copy(update={"hosts": tuple(host for _ in range(n_hosts))})


class TestRequiredLJTypes:
    def test_keeps_empty_guest_templates_and_drops_unused_types(self):
        config = _config(exchange_prob=0.5, init_adsorbates=(0,))
        config = config.model_copy(
            update={
                "lj": config.lj.model_copy(
                    update={
                        "parameters": {**config.lj.parameters, "unused": (100.0, 200.0)}
                    }
                )
            }
        )
        state = init_state(jax.random.key(91), config)
        parameters = state.lj_parameters.data
        assert set(parameters.labels) == {"X1", "C_co2", "O_co2"}
        assert int(state.groups.num_occupied) == 0
        reference = GlobalTailCorrectedLennardJonesParameters.from_dict(
            config.lj.cutoff, config.lj.parameters, config.lj.mixing_rule
        )
        indices = jnp.array(
            [reference.labels.index(label) for label in parameters.labels]
        )
        for actual, full in (
            (parameters.sigma, reference.sigma),
            (parameters.epsilon, reference.epsilon),
        ):
            npt.assert_array_equal(actual, full[indices[:, None], indices])

    def test_missing_guest_parameters_fail_at_initialization(self):
        config = _config(exchange_prob=0.5, init_adsorbates=(0,))
        config = config.model_copy(
            update={
                "lj": config.lj.model_copy(
                    update={
                        "parameters": {
                            label: value
                            for label, value in config.lj.parameters.items()
                            if label != "O_co2"
                        }
                    }
                )
            }
        )
        with pytest.raises(ValueError, match="Missing Lennard-Jones parameters.*O_co2"):
            init_state(jax.random.key(92), config)


class TestRigidCorrections:
    @pytest.fixture(scope="class")
    def mixture(self) -> MCMCState:
        config = _batched_config(2, (1, 2))
        second = _co2().model_copy(
            update={
                "symbols": ("C_alt", "O_alt", "O_alt"),
                "charges": (0.5, -0.2, -0.2),
            }
        )
        config = config.model_copy(
            update={
                "adsorbates": (_co2(), second),
                "hosts": tuple(
                    host.model_copy(
                        update={
                            "adsorbate_composition": (0.5, 0.5),
                            "adsorbate_interaction": ((0.0, 0.0), (0.0, 0.0)),
                            "blocking_spheres": ((), ()),
                        }
                    )
                    for host in config.hosts
                ),
                "lj": config.lj.model_copy(
                    update={
                        "parameters": {
                            **config.lj.parameters,
                            "X1": (2.0, 0.01),
                            "C_alt": (3.0, 0.004),
                            "O_alt": (2.5, 0.008),
                        }
                    }
                ),
            }
        )
        state = init_state(jax.random.key(93), config)
        state = bind(state, lambda s: s.systems.data.cell).set(
            PeriodicCell(
                TriclinicFrame.from_matrix(
                    state.systems.data.cell.vectors.at[1].multiply(1.2)
                )
            )
        )
        # A charged, interacting fixed host tests the constant and cross terms.
        host = ~state.particles.data.group.valid_mask & state.particles.occupation
        return bind(state, lambda s: s.particles.data.charges).apply(
            lambda q: jnp.where(host, 0.3, q)
        )

    @staticmethod
    def _potentials(state: MCMCState):
        sl = identity_lens(MCMCState)
        composition = make_rigid_body_composition(
            state,
            PointCloud(state.particles, state.systems),
            state.motifs,
            sl.focus(lambda s: s.groups),
        )
        prepared_ewald = make_ewald_from_state(
            sl, _probe, include_exclusion_mask=True, composition=composition
        )
        actual = (
            make_lennard_jones_tail_correction_from_state(sl, composition=composition),
            prepared_ewald.self_interaction,
            ScaledPotential(prepared_ewald.exclusion_correction, -1),
        )
        ewald = make_ewald_from_state(sl, include_exclusion_mask=True)
        reference = (
            make_lennard_jones_tail_correction_from_state(sl),
            ewald.self_interaction,
            ScaledPotential(ewald.exclusion_correction, -1),
        )
        return tuple(map(jax.jit, actual)), tuple(map(jax.jit, reference))

    @pytest.mark.parametrize("term", ["tail", "ewald"])
    def test_composition_rejects_derivatives(self, mixture: MCMCState, term: str):
        """Fixed coefficients must never silently replace forces or cell derivatives."""
        from kups.application.potential.filter import POSITIONS_AND_CELL

        sl = identity_lens(MCMCState)
        composition = make_rigid_body_composition(
            mixture,
            PointCloud(mixture.particles, mixture.systems),
            mixture.motifs,
            sl.focus(lambda s: s.groups),
        )
        factory = (
            make_lennard_jones_tail_correction_from_state
            if term == "tail"
            else make_ewald_from_state
        )
        with pytest.raises(ValueError, match="energy-only"):
            factory(sl, gradient=POSITIONS_AND_CELL, composition=composition)

    def test_fixed_excluded_molecules_retain_their_energy(self, mixture: MCMCState):
        """A fixed molecule with exclusions needs the ordinary geometric term."""
        sl = identity_lens(MCMCState)
        composition = make_rigid_body_composition(
            mixture,
            PointCloud(mixture.particles, mixture.systems),
            mixture.motifs,
            sl.focus(lambda s: s.groups),
        )

        def fixed_counts(state, patch, old_input=False):
            values = jnp.zeros(
                (len(state.systems), 1 + state.motifs.data.motif.num_labels)
            )
            return state.systems.set_data(values.at[:, 0].set(1))

        composition = bind(composition, lambda c: (c.fixed_system, c.counts)).set(
            (mixture.particles.data.system, fixed_counts)
        )
        actual = (
            make_ewald_from_state(
                sl, include_exclusion_mask=True, composition=composition
            )
            .exclusion_correction(mixture)
            .data.total_energies.data
        )
        expected = (
            make_ewald_from_state(sl, include_exclusion_mask=True)
            .exclusion_correction(mixture)
            .data.total_energies.data
        )
        assert jnp.any(jnp.abs(expected) > 0)
        npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("move", ["translation", "rotation", "reinsertion"])
    def test_rigid_moves_leave_corrections_unchanged(
        self, mixture: MCMCState, move: str
    ):
        actual, reference = self._potentials(mixture)
        args = (jax.random.key(94), mixture.particles, mixture.groups, mixture.systems)
        if move == "reinsertion":
            changes = propose_reinsertion(*args, mixture.move_capacity)
        else:
            propose = (
                propose_group_translation
                if move == "translation"
                else propose_group_rotation
            )
            changes = propose(
                *args, mixture.systems.set_data(jnp.ones(2)), mixture.move_capacity
            )
        update = _movement_patch(jax.random.key(95), mixture, changes)
        moved = update(mixture, mixture.systems.set_data(jnp.array([True, True])))
        for candidate, full in zip(actual, reference, strict=True):
            old = candidate(mixture).data.total_energies.data
            proposed = candidate(mixture, update).data.total_energies.data
            npt.assert_array_equal(proposed, old)
            npt.assert_allclose(
                proposed, full(moved).data.total_energies.data, rtol=1e-12, atol=1e-12
            )

    def test_exchange_and_rejection_match_full_recomputation(self, mixture: MCMCState):
        actual, reference = self._potentials(mixture)
        state = mixture
        for candidate in actual:
            state = PotentialAsPropagator(candidate)(jax.random.key(96), state)
        for i in range(10):
            key = jax.random.key(100 + i)
            args = (key, state.motifs, state.particles, state.groups)
            proposal = (
                insert_random_motif(
                    *args, state.systems.map_data(lambda s: s.cell), state.move_capacity
                )
                if i < 3
                else delete_random_motif(*args, state.move_capacity)
            )
            update = MCMCStateUpdate.from_changes(key, state, proposal)
            proposed = update(state, state.systems.set_data(jnp.array([True, True])))
            accept = state.systems.set_data(jnp.array([True, i % 2 == 0]))
            outputs = [candidate(state, update) for candidate in actual]
            for out, full in zip(outputs, reference, strict=True):
                npt.assert_allclose(
                    out.data.total_energies.data,
                    full(proposed).data.total_energies.data,
                    rtol=1e-12,
                    atol=1e-12,
                )
            state = update(state, accept)
            for out in outputs:
                state = out.patch(state, accept)
            npt.assert_allclose(
                state.ewald_parameters.cache.self_interaction.total.total_energies.data,
                reference[1](state).data.total_energies.data,
                atol=1e-12,
            )
            npt.assert_allclose(
                state.ewald_parameters.cache.exclusion.total.total_energies.data,
                reference[2](state).data.total_energies.data,
                atol=1e-12,
            )
        assert int(state.groups.data.system.counts.data[0]) == 0

    @pytest.mark.parametrize("rescale", (False, True))
    def test_prepared_reciprocal_exchange_and_rejection(
        self, mixture: MCMCState, rescale: bool
    ) -> None:
        sl = identity_lens(MCMCState)
        reference = jax.jit(make_ewald_from_state(sl).long_range)
        initial_energy = reference(mixture).data.total_energies.data
        if rescale:
            # The generic potential must read changing cells and parameters;
            # rigid preparation uses their values when the potential is built.
            mixture = bind(mixture, lambda s: s.systems.data.cell).set(
                PeriodicCell(
                    TriclinicFrame.from_matrix(mixture.systems.data.cell.vectors * 1.1)
                )
            )
            mixture = bind(mixture, lambda s: s.ewald_parameters.data.alpha).apply(
                lambda alpha: alpha * 0.9
            )
            assert not jnp.allclose(
                reference(mixture).data.total_energies.data, initial_energy
            )
        composition = make_rigid_body_composition(
            mixture,
            PointCloud(mixture.particles, mixture.systems),
            mixture.motifs,
            sl.focus(lambda s: s.groups),
        )
        candidate = jax.jit(
            make_ewald_from_state(sl, _probe, composition=composition).long_range
        )
        state = PotentialAsPropagator(candidate)(jax.random.key(301), mixture)
        for i in range(6):
            key = jax.random.key(302 + i)
            args = (key, state.motifs, state.particles, state.groups)
            proposal = (
                insert_random_motif(
                    *args, state.systems.map_data(lambda s: s.cell), state.move_capacity
                )
                if i < 3
                else delete_random_motif(*args, state.move_capacity)
            )
            update = MCMCStateUpdate.from_changes(key, state, proposal)
            proposed = update(state, state.systems.set_data(jnp.array([True, True])))
            result = candidate(state, update)
            npt.assert_allclose(
                result.data.total_energies.data,
                reference(proposed).data.total_energies.data,
                rtol=1e-12,
                atol=1e-12,
            )
            accept = state.systems.set_data(jnp.array([True, i % 2 == 0]))
            state = result.patch(update(state, accept), accept)
            npt.assert_allclose(
                state.ewald_parameters.cache.long_range.total.total_energies.data,
                reference(state).data.total_energies.data,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_prepared_reciprocal_recompiles_after_donation(
        self, mixture: MCMCState
    ) -> None:
        state = jax.tree.map(jnp.copy, mixture)
        sl = identity_lens(MCMCState)
        composition = make_rigid_body_composition(
            state,
            PointCloud(state.particles, state.systems),
            state.motifs,
            sl.focus(lambda s: s.groups),
        )
        propagate = PotentialAsPropagator(
            make_ewald_from_state(sl, _probe, composition=composition)
        )

        def evaluate(keys: jax.Array, current: MCMCState) -> MCMCState:
            return propagate(keys[0], current)

        compiled = jax.jit(evaluate, donate_argnums=(1,))
        for size in (8, 3):
            state = compiled(jax.random.split(jax.random.key(1), size), state)
            jax.block_until_ready(state)
        npt.assert_allclose(
            state.ewald_parameters.cache.long_range.total.total_energies.data,
            make_ewald_from_state(sl).long_range(state).data.total_energies.data,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_large_template_retains_geometric_periodic_exclusions(self) -> None:
        config = _config(exchange_prob=0.5, init_adsorbates=(1,))
        template = _co2().model_copy(
            update={"positions": ((0.0, 0.0, 0.0), (-5.0, 0.0, 0.0), (5.0, 0.0, 0.0))}
        )
        config = config.model_copy(update={"adsorbates": (template,)})
        state = init_state(jax.random.key(120), config)
        rows = jnp.flatnonzero(
            state.particles.data.group.valid_mask & state.particles.occupation
        )
        positions = jnp.asarray(template.positions) + 7.0
        state = (
            bind(state, lambda s: s.particles.data.positions).at(rows).set(positions)
        )
        sl = identity_lens(MCMCState)
        ewald = make_ewald_from_state(sl, include_exclusion_mask=True)
        reference = jax.jit(
            sum_potentials(
                make_lennard_jones_from_state(sl),
                make_lennard_jones_tail_correction_from_state(sl),
                ewald,
            )
        )
        candidate = jax.jit(_make_potential(state))
        rotation = (
            jnp.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0**0.5]])
            / 2.0**0.5
        )
        rotated = (
            bind(state, lambda s: s.particles.data.positions)
            .at(rows)
            .set((positions - 7.0) @ rotation + 7.0)
        )
        old_exclusion = ewald.exclusion_correction(state).data.total_energies.data
        new_exclusion = ewald.exclusion_correction(rotated).data.total_energies.data
        assert float(jnp.abs(new_exclusion - old_exclusion).max()) > 0.1
        for configuration in (state, rotated):
            npt.assert_allclose(
                candidate(configuration).data.total_energies.data,
                reference(configuration).data.total_energies.data,
                rtol=1e-12,
                atol=1e-12,
            )


class TestInitStateBlockingSpheres:
    """Sizing the blocking-sphere neighbor list must cover sphere-free systems too."""

    @staticmethod
    def _mixed_batch() -> Config:
        """Two-host config where only the first host defines a blocking sphere."""
        base = _batched_config(2, init_adsorbates=(1,))
        blocked = base.hosts[0].model_copy(
            update={
                "blocking_spheres": (
                    (BlockingSphereConfig(center=(0.0, 0.0, 0.0), radius=2.0),),
                )
            }
        )
        return base.model_copy(update={"hosts": (blocked, base.hosts[1])})

    def test_spheres_on_one_host_of_a_batch(self):
        state = init_state(jax.random.key(0), self._mixed_batch())
        assert state.has_blocking_spheres
        assert state.blocking_spheres_parameters.radii.shape == (1,)
        # Capacities are estimated from the batch-wide max radius, so they are populated
        # for every system rather than only the one owning the sphere.
        params = state.blocking_spheres_neighborlist_params
        assert params.avg_candidates > 0 and params.cells > 0

    def test_no_spheres_leaves_capacities_empty(self):
        state = init_state(jax.random.key(0), _batched_config(2, init_adsorbates=(1,)))
        assert not state.has_blocking_spheres
        assert state.blocking_spheres_neighborlist_params == (
            UniversalNeighborlistParameters(0, 0, 0, 0)
        )


class TestExchangeEnergyConsistency:
    """Incremental GCMC exchange energy must equal a full recomputation.

    The propagator accepts/rejects moves with an incrementally updated energy,
    which must match a full evaluation of the resulting configuration. On an
    insertion/deletion the molecule's intramolecular real-space (short-range)
    Ewald energy must be added/removed, not only its intermolecular pairs.
    Checked here for a deletion on a batched (multi-system) charged system.
    """

    @staticmethod
    def _full_potential():
        sl = identity_lens(MCMCState)
        return sum_potentials(
            make_lennard_jones_from_state(sl, _probe),
            make_lennard_jones_tail_correction_from_state(sl),
            make_ewald_from_state(sl, _probe, include_exclusion_mask=True),
        )

    def test_batched_deletion_matches_full(self):
        config = _batched_config(n_hosts=4, init_adsorbates=(2,))
        state = init_state(jax.random.key(0), config)
        pot = jax.jit(self._full_potential())

        # Populate caches with a full evaluation.
        prop = jax.jit(PotentialAsPropagator(pot))
        state = prop(jax.random.key(1), state)

        # Force a deletion of one molecule per system via the real move machinery.
        proposal = jax.jit(delete_random_motif)(
            jax.random.key(2),
            state.motifs,
            state.particles,
            state.groups,
            state.move_capacity,
        )
        update = jax.jit(MCMCStateUpdate.from_changes)(
            jax.random.key(3), state, proposal
        )

        # Energy the propagator would use to accept/reject (incremental).
        out = pot(state, patch=update)
        e_incr = out.data.total_energies.data

        # Apply the deletion and recompute the energy from scratch.
        accept = Table(state.systems.keys, jnp.ones(len(state.systems), dtype=bool))
        new_state = out.patch(update(state, accept), accept)
        n_after = new_state.groups.data.system.counts.data
        assert int(n_after.max()) < 2, "deletion did not reduce molecule count"
        e_full = pot(new_state).data.total_energies.data
        npt.assert_allclose(e_incr, e_full, atol=1e-3)

    def test_batched_insertion_matches_full(self):
        config = _batched_config(n_hosts=4, init_adsorbates=(2,))
        state = init_state(jax.random.key(0), config)
        pot = jax.jit(self._full_potential())

        # Populate caches with a full evaluation.
        state = PotentialAsPropagator(pot)(jax.random.key(1), state)

        # Force an insertion of one molecule per system via the real move machinery.
        proposal = jax.jit(insert_random_motif)(
            jax.random.key(2),
            state.motifs,
            state.particles,
            state.groups,
            state.systems.map_data(lambda s: s.cell),
            state.move_capacity,
        )
        update = jax.jit(MCMCStateUpdate.from_changes)(
            jax.random.key(3), state, proposal
        )

        # Energy the propagator would use to accept/reject (incremental).
        out = pot(state, patch=update)
        e_incr = out.data.total_energies.data

        # Apply the insertion and recompute the energy from scratch.
        accept = Table(state.systems.keys, jnp.ones(len(state.systems), dtype=bool))
        new_state = out.patch(update(state, accept), accept)
        n_after = new_state.groups.data.system.counts.data
        assert int(n_after.max()) > 2, "insertion did not add a molecule"
        e_full = pot(new_state).data.total_energies.data
        npt.assert_allclose(e_incr, e_full, rtol=1e-5, atol=1e-3)


def test_reinsertion_onto_colliding_particle_matches_full(state: MCMCState):
    """Reinsertion must include newly formed close-contact pairs.

    Molecule 1 (isolated at ``[9, 9, 9]``) is reinserted next to
    particle 0 (at ``[2, 2, 2]``); dropping that close-contact pair would
    make the incremental delta disagree with a full evaluation.
    """
    pot = make_lennard_jones_from_state(identity_lens(MCMCState), _probe)
    state = PotentialAsPropagator(pot)(jax.random.key(0), state)

    proposal = _make_exchange_proposal(
        jnp.array([1]), jnp.array([[4.0, 2.0, 2.0]]), jnp.array([1])
    )
    update = MCMCStateUpdate.from_changes(jax.random.key(1), state, proposal)

    out = pot(state, patch=update)
    accept = Table(state.systems.keys, jnp.ones(len(state.systems), dtype=bool))
    new_state = out.patch(update(state, accept), accept)
    e_full = pot(new_state).data.total_energies.data
    assert float(e_full[0]) > 1.0, "reinserted molecule must contact particle 0"
    npt.assert_allclose(out.data.total_energies.data, e_full, rtol=1e-6, atol=1e-6)


class TestBlockingSpheresPatch:
    """A proposed move must be evaluated against the blocking spheres.

    ``movement_update_pid0`` moves particle 0 from ``[2, 2, 2]`` onto the sphere
    centre, so the patched configuration is blocked while the current one is not.
    """

    @staticmethod
    def _with_sphere(state: MCMCState) -> MCMCState:
        return bind(state, lambda x: x.blocking_spheres_parameters).set(
            BlockingSpheresParameters(
                radii=jnp.array([1.0]),
                positions=jnp.array([[3.0, 3.0, 3.0]]),
                system=Index.new([SystemId(0)]).populate_max_count(),
                motif=Index.new([MotifId(0)]).populate_max_count(),
            )
        )

    def test_probed_move_into_sphere_is_blocked(
        self, state: MCMCState, movement_update_pid0: MCMCStateUpdate
    ):
        pot = make_blocking_spheres_from_state(identity_lens(MCMCState), _probe)
        blocked = self._with_sphere(state)
        assert pot(blocked).data.total_energies.data[0] == 0.0
        assert jnp.isinf(pot(blocked, movement_update_pid0).data.total_energies.data[0])

    def test_probed_move_outside_sphere_is_free(
        self,
        state: MCMCState,
        movement_update_newpos: tuple[MCMCStateUpdate, jax.Array],
    ):
        pot = make_blocking_spheres_from_state(identity_lens(MCMCState), _probe)
        update, _ = movement_update_newpos
        assert pot(self._with_sphere(state), update).data.total_energies.data[0] == 0.0

    def test_probe_narrows_the_query(
        self, state: MCMCState, movement_update_pid0: MCMCStateUpdate
    ):
        """The probed plan carries only the moved particle, not the whole table."""
        blocked = self._with_sphere(state)
        sl = identity_lens(MCMCState)
        ((probed, _),) = make_blocking_spheres_from_state(sl, _probe).composer(
            blocked, movement_update_pid0
        )
        ((full, _),) = make_blocking_spheres_from_state(sl).composer(
            blocked, movement_update_pid0
        )
        assert len(probed.particles) == 1
        assert len(full.particles) == len(state.particles)

    def test_unprobed_move_into_sphere_is_blocked(
        self, state: MCMCState, movement_update_pid0: MCMCStateUpdate
    ):
        pot = make_blocking_spheres_from_state(identity_lens(MCMCState))
        blocked = self._with_sphere(state)
        assert jnp.isinf(pot(blocked, movement_update_pid0).data.total_energies.data[0])


class TestRunNVT:
    """Canonical (fixed-N) MCMC: exchange disabled, host pre-loaded."""

    @pytest.fixture(scope="class")
    def run_result(self) -> tuple[MCMCState, str]:
        config = _config(exchange_prob=0.0, init_adsorbates=(2,))
        return run(config), str(config.run.out_file)

    def test_loading_is_conserved(self, run_result):
        state, _ = run_result
        # exchange_prob=0 keeps the molecule count fixed at the initial loading.
        assert int(state.groups.data.system.counts.data[0]) == 2

    def test_analyzer_reads_back_physical_outputs(self, run_result):
        _, out_file = run_result
        _assert_readable(out_file)


class TestRunGCMC:
    """Grand-canonical (µVT) MCMC with insertions/deletions and initial loading."""

    @pytest.fixture(scope="class")
    def run_result(self) -> tuple[MCMCState, str]:
        config = _config(exchange_prob=0.5, init_adsorbates=(2,))
        return run(config), str(config.run.out_file)

    def test_state_has_finite_energy(self, run_result):
        state, _ = run_result
        assert jnp.isfinite(state.systems.data.potential_energy.total[0]).item()

    def test_analyzer_reads_back_physical_outputs(self, run_result):
        _, out_file = run_result
        _assert_readable(out_file)


@pytest.mark.parametrize("compute_stress", [False, True])
def test_blocked_gcmc_preserves_every_saved_frame(
    tmp_path: Path, compute_stress: bool
) -> None:
    config = _config(exchange_prob=0.5, init_adsorbates=(2,))
    config = config.model_copy(update={"compute_stress": compute_stress})
    frames: list[MCMCStepData] = []
    for block_size in (1, 4):
        output = tmp_path / f"cycles-{block_size}.h5"
        blocked = config.model_copy(
            update={
                "run": config.run.model_copy(
                    update={
                        "out_file": output,
                        "num_warmup_cycles": 3,
                        "num_cycles": 11,
                        "cycles_per_call": block_size,
                    }
                )
            }
        )
        run(blocked)
        with HDF5StorageReader[MCMCLoggedData[MCMCStepData]](output) as reader:
            sample = reader.focus_group(lambda cfg: cfg.per_step)[:]
            assert sample.systems.data.potential_energy.shape[0] == 11
            assert reader.file.attrs["actual_steps"] == 11
            frames.append(sample)
    for expected, actual in zip(
        jax.tree.leaves(frames[0]), jax.tree.leaves(frames[1]), strict=True
    ):
        npt.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


class TestFusedPairIntegration:
    def test_empty_and_occupied_systems_match_without_noop_guard(self):
        from kups.core.propagator import propagate_and_fix
        from kups.core.result import as_result_function

        config = _batched_config(n_hosts=2, init_adsorbates=(1,))
        config = config.model_copy(
            update={
                "hosts": (
                    config.hosts[0].model_copy(update={"init_adsorbates": (0,)}),
                    config.hosts[1],
                )
            }
        )
        state = init_state(jax.random.key(81), config)
        initializer, guarded = make_propagator(
            state, config.run.model_copy(update={"min_cycle_length": 4})
        )
        mc = guarded.propagator.propagator
        assert mc.is_noop is not None
        baseline = replace(
            guarded,
            propagator=replace(
                guarded.propagator, propagator=replace(mc, is_noop=None)
            ),
        )
        state = propagate_and_fix(
            jax.jit(as_result_function(initializer)), jax.random.key(82), state
        )
        npt.assert_array_equal(state.groups.data.system.counts.data, [0, 1])
        reference = candidate = state
        baseline_step = jax.jit(as_result_function(baseline))
        guarded_step = jax.jit(as_result_function(guarded))
        for key in jax.random.split(jax.random.key(83), 8):
            reference = propagate_and_fix(baseline_step, key, reference)
            candidate = propagate_and_fix(guarded_step, key, candidate)
            npt.assert_array_equal(
                candidate.particles.occupation, reference.particles.occupation
            )
            npt.assert_array_equal(
                candidate.groups.occupation, reference.groups.occupation
            )
            npt.assert_allclose(
                candidate.particles.data.positions,
                reference.particles.data.positions,
                rtol=0,
                atol=1e-10,
            )
            npt.assert_allclose(
                candidate.systems.data.potential_energy.total,
                reference.systems.data.potential_energy.total,
                rtol=1e-9,
                atol=1e-10,
            )
            for name in ("translation", "rotation", "reinsertion", "exchange"):
                left = getattr(candidate, name + "_params")
                right = getattr(reference, name + "_params")
                for a, b in zip(
                    jax.tree.leaves(left), jax.tree.leaves(right), strict=True
                ):
                    npt.assert_array_equal(a, b)

    @pytest.mark.parametrize("charged", [False, True])
    @pytest.mark.parametrize("cutoff", [5.0, 8.0])
    def test_cycles_match_full_recomputation(self, charged, cutoff):
        from kups.core.propagator import propagate_and_fix
        from kups.core.result import as_result_function

        config = _config(exchange_prob=0.5, init_adsorbates=(2,))
        config = config.model_copy(
            update={
                "lj": config.lj.model_copy(update={"cutoff": cutoff}),
                "ewald": config.ewald.model_copy(update={"real_cutoff": cutoff}),
            }
        )
        if not charged:
            adsorbate = config.adsorbates[0].model_copy(
                update={"charges": (0.0, 0.0, 0.0)}
            )
            config = config.model_copy(update={"adsorbates": (adsorbate,)})
        state = init_state(jax.random.key(71), config)
        initialize, cycle = make_propagator(state, config.run)
        reference = _make_potential(state)

        def apply(propagator, key, state):
            return propagate_and_fix(as_result_function(propagator), key, state)

        state = apply(initialize, jax.random.key(72), state)
        for key in jax.random.split(jax.random.key(73), 3):
            npt.assert_allclose(
                state.systems.data.potential_energy.total,
                reference(state).data.total_energies.data,
                rtol=1e-9,
                atol=1e-10,
            )
            state = apply(cycle, key, state)
        npt.assert_allclose(
            state.systems.data.potential_energy.total,
            reference(state).data.total_energies.data,
            rtol=1e-9,
            atol=1e-10,
        )
