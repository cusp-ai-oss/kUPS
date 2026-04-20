# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body grand-canonical Monte Carlo simulation entry point."""

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp
import rich
from jax import Array
from nanoargs.cli import NanoArgs
from pydantic import BaseModel

from kups.application.mcmc import (
    MCMCGroup,
    MCMCParticles,
    MCMCSystems,
    RunConfig,
    run_mcmc,
)
from kups.application.mcmc.analysis import analyze_mcmc_file
from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
    MotifParticles,
    StressResult,
    mcmc_state_from_config,
)
from kups.application.mcmc.logging import make_mcmc_logged_data
from kups.core.capacity import Capacity, FixedCapacity
from kups.core.data import Buffered, Table, WithCache, WithIndices
from kups.core.data.buffered import add_buffers, system_view
from kups.core.data.index import unify_keys_by_cls
from kups.core.lens import bind, identity_lens, lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    Edges,
    NearestNeighborList,
    RefineMaskNeighborList,
    UniversalNeighborlistParameters,
    neighborlist_changes,
)
from kups.core.parameter_scheduler import ParameterSchedulerState
from kups.core.patch import Accept
from kups.core.potential import (
    EMPTY,
    CachedPotential,
    EmptyType,
    PotentialAsPropagator,
    PotentialOut,
    sum_potentials,
)
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    ResetOnErrorPropagator,
    StateProperty,
    propagate_and_fix,
)
from kups.core.result import as_result_function
from kups.core.typing import (
    GroupId,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import (
    dataclass,
    key_chain,
    tree_map,
)
from kups.mcmc.moves import (
    ExchangeChanges,
    make_gcmc_mcmc_propagator,
)
from kups.mcmc.probability import make_muvt_probability_ratio
from kups.observables.pressure import ideal_gas_pressure
from kups.observables.stress import molecular_virial_stress_from_state
from kups.potential.classical.ewald import (
    EwaldCache,
    EwaldParameters,
    make_ewald_from_state,
)
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    MixingRule,
    global_lennard_jones_tail_correction_pressure_from_state,
    make_lennard_jones_from_state,
    make_lennard_jones_tail_correction_from_state,
)
from kups.potential.common.energy import (
    PositionAndUnitCell,
    position_and_unitcell_idx_view,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class LJConfig(BaseModel):
    """Lennard-Jones potential configuration."""

    cutoff: float
    parameters: dict[str, tuple[float | None, float | None]]
    tail_correction: bool
    mixing_rule: MixingRule


class EwaldConfig(BaseModel):
    """Ewald summation configuration."""

    real_cutoff: float
    precision: float


class Config(BaseModel):
    """Top-level configuration for rigid-body MCMC simulations."""

    adsorbates: tuple[AdsorbateConfig, ...]
    hosts: tuple[HostConfig, ...]
    run: RunConfig
    lj: LJConfig
    ewald: EwaldConfig
    max_num_adsorbates: int
    compute_stress: bool = False


@dataclass
class MCMCState:
    """Full state for a rigid-body grand-canonical MCMC simulation.

    Holds buffered particle/group arrays, motif templates, system
    thermodynamic data, neighbor lists, potential parameters with
    caches, and per-move adaptive step-size schedulers.
    """

    particles: Buffered[ParticleId, MCMCParticles]
    groups: Buffered[GroupId, MCMCGroup]
    motifs: Table[MotifParticleId, MotifParticles]
    systems: Table[SystemId, MCMCSystems]
    neighborlist_params: UniversalNeighborlistParameters
    lj_parameters: WithCache[
        GlobalTailCorrectedLennardJonesParameters, PotentialOut[EmptyType, EmptyType]
    ]
    ewald_parameters: WithCache[EwaldParameters, EwaldCache[EmptyType, EmptyType]]
    translation_params: Table[SystemId, ParameterSchedulerState]
    rotation_params: Table[SystemId, ParameterSchedulerState]
    reinsertion_params: Table[SystemId, ParameterSchedulerState]
    exchange_params: Table[SystemId, ParameterSchedulerState]

    @property
    def max_cutoff(self) -> Table[SystemId, Array]:
        """Per-system maximum cutoff across LJ and Ewald potentials."""
        return Table(
            self.systems.keys,
            jnp.maximum(
                self.lj_parameters.data.cutoff.data,
                self.ewald_parameters.data.cutoff.data,
            ),
        )

    @property
    def move_capacity(self) -> Capacity[int]:
        """Maximum number of particles per motif (move buffer size)."""
        motif_size = self.motifs.data.motif.max_count
        assert motif_size is not None
        return FixedCapacity(motif_size * len(self.systems))

    @property
    def neighborlist(self) -> NearestNeighborList:
        return DenseNearestNeighborList.from_state(self)

    @property
    def guest_only(self) -> MCMCState:
        return bind(self, lambda x: x.particles.data).apply(MCMCParticles.guest_only)


@dataclass
class MCMCStateUpdate:
    """Proposed MCMC state change with pre-computed neighbor lists.

    Stores the proposed particle and group modifications together with
    neighbor-list edges computed *before* and *after* the change, so
    that energy differences can be evaluated without rebuilding the
    full neighbor list.

    Calling an instance applies the update conditionally on ``accept``.
    """

    _particles: WithIndices[ParticleId, Buffered[ParticleId, MCMCParticles]]
    groups: WithIndices[GroupId, Buffered[GroupId, MCMCGroup]]
    edges_after: Edges[Literal[2]]
    edges_before: Edges[Literal[2]]

    @staticmethod
    def from_changes(
        key: Array,
        state: MCMCState,
        proposal: ExchangeChanges,
    ) -> MCMCStateUpdate:
        """Build an update from exchange changes."""
        p_data = proposal.particles.data.data
        g_data = proposal.groups.data.data

        motif_data = state.motifs[p_data.motif]
        new_particles = Buffered(
            proposal.particles.data.keys,
            MCMCParticles(
                positions=p_data.new_positions,
                masses=motif_data.masses,
                atomic_numbers=motif_data.atomic_numbers,
                charges=motif_data.charges,
                labels=motif_data.labels,
                system=p_data.system,
                group=p_data.group,
                motif=p_data.motif,
            ),
            system_view,
        )
        new_groups = Buffered(
            proposal.groups.data.keys,
            MCMCGroup(g_data.system, g_data.motif),
            system_view,
        )
        particle_changes = WithIndices(proposal.particles.indices, new_particles)
        group_changes = WithIndices(proposal.groups.indices, new_groups)

        result = neighborlist_changes(
            state.neighborlist,
            state.particles,
            particle_changes,
            state.systems,
            state.max_cutoff,
            compaction=1.0,
        )
        return MCMCStateUpdate(
            _particles=particle_changes,
            groups=group_changes,
            edges_after=result.added,
            edges_before=result.removed,
        )

    def __call__(self, state: MCMCState, accept: Accept) -> MCMCState:
        """Apply the update to ``state``, conditional on ``accept``."""
        acc = Table.broadcast_to(accept, state.systems)
        new_groups = state.groups.update_if(
            acc, self.groups.indices, self.groups.data.data
        )
        new_particles = state.particles.update_if(
            acc, self._particles.indices, self._particles.data.data
        )
        return bind(state, lambda x: (x.particles, x.groups)).set(
            (new_particles, new_groups)
        )

    @property
    def particles(self):
        """Proposed particle data (without the buffer wrapper)."""
        return self._particles.map_data(lambda x: x.data)

    @property
    def neighborlist_before(self):
        """Neighbor list for the *current* (pre-move) configuration."""
        return RefineMaskNeighborList(self.edges_before)

    @property
    def neighborlist_after(self):
        """Neighbor list for the *proposed* (post-move) configuration."""
        return RefineMaskNeighborList(self.edges_after)


def _probe(state: MCMCState, update: MCMCStateUpdate) -> MCMCStateUpdate:
    return update


def make_propagator(
    config: RunConfig,
) -> tuple[Propagator[MCMCState], Propagator[MCMCState]]:
    state_lens = identity_lens(MCMCState)
    potential = sum_potentials(
        make_ewald_from_state(state_lens, _probe, include_exclusion_mask=True),
        make_lennard_jones_from_state(state_lens, _probe),
        make_lennard_jones_tail_correction_from_state(state_lens),
    )
    potential, probability_ratio = make_muvt_probability_ratio(state_lens, potential)
    propagator = make_gcmc_mcmc_propagator(
        state_lens,
        MCMCStateUpdate.from_changes,
        probability_ratio,
        translation_weight=config.translation_prob,
        rotation_weight=config.rotation_prob,
        reinsertion_weight=config.reinsertion_prob,
        exchange_weight=config.exchange_prob,
    )
    propagator = LoopPropagator(
        propagator,
        lambda x: jnp.maximum(
            x.groups.data.system.counts.data.max(), config.min_cycle_length
        ),
    )
    init_prop = ResetOnErrorPropagator(PotentialAsPropagator(potential))
    return init_prop, ResetOnErrorPropagator(propagator)


def init_state(key: Array, config: Config) -> MCMCState:
    """Initialize the full MCMC state from configuration."""
    chain = key_chain(key)
    ps: list[Table[ParticleId, MCMCParticles]] = []
    gs: list[Table[GroupId, MCMCGroup]] = []
    ss: list[Table[SystemId, MCMCSystems]] = []
    motifs: Table[MotifParticleId, MotifParticles] | None = None
    for host in config.hosts:
        particles, groups, system, motifs = mcmc_state_from_config(
            next(chain), host, config.adsorbates
        )
        ps.append(particles)
        gs.append(groups)
        ss.append(system)
    assert motifs is not None, "At least one host must be provided."
    particles, groups, system = Table.union(ps, gs, ss)
    n_sys = len(system)
    lj_params = GlobalTailCorrectedLennardJonesParameters.from_dict(
        cutoff=config.lj.cutoff,
        parameters=config.lj.parameters,
        mixing_rule=config.lj.mixing_rule,
        tail_correction=config.lj.tail_correction,
    )
    max_motif_size = motifs.data.motif.max_count
    assert max_motif_size is not None
    particles, groups, motifs, system = unify_keys_by_cls(
        (particles, groups, motifs, system)
    )
    num_buffer_particles = config.max_num_adsorbates * max_motif_size
    particles, groups = add_buffers(
        (particles, num_buffer_particles),
        (groups, config.max_num_adsorbates),
    )

    ewald_params = EwaldParameters.make(
        particles,
        system,
        epsilon_total=config.ewald.precision,
        real_cutoff=config.ewald.real_cutoff,
    )
    n_kvecs = ewald_params.reciprocal_lattice_shifts.data.shape[1]

    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts + num_buffer_particles / n_sys,
        system,
        tree_map(jnp.maximum, lj_params.cutoff, ewald_params.cutoff),
    )
    min_half_box = float(system.data.unitcell.perpendicular_lengths.min() / 2)
    return MCMCState(
        particles=particles,
        groups=groups,
        motifs=motifs,
        systems=system,
        neighborlist_params=neighborlist_params,
        lj_parameters=WithCache(
            lj_params,
            PotentialOut(Table.arange(jnp.zeros(n_sys), label=SystemId), EMPTY, EMPTY),
        ),
        ewald_parameters=WithCache(ewald_params, EwaldCache.make(n_sys, n_kvecs)),
        translation_params=Table.arange(
            ParameterSchedulerState.create(n_sys, upper_bound=min_half_box),
            label=SystemId,
        ),
        rotation_params=Table.arange(
            ParameterSchedulerState.create(n_sys), label=SystemId
        ),
        reinsertion_params=Table.arange(
            ParameterSchedulerState.create(n_sys), label=SystemId
        ),
        exchange_params=Table.arange(
            ParameterSchedulerState.create(n_sys), label=SystemId
        ),
    )


def make_guest_stress() -> StateProperty[
    MCMCState,
    Table[SystemId, StressResult],
]:
    """Build guest-only stress tensor function from MCMCState."""
    state_lens = identity_lens(MCMCState)
    potential = sum_potentials(
        make_ewald_from_state(
            state_lens,
            compute_position_and_unitcell_gradients=True,
            include_exclusion_mask=True,
        ),
        make_lennard_jones_from_state(
            state_lens, compute_position_and_unitcell_gradients=True
        ),
        make_lennard_jones_tail_correction_from_state(
            state_lens, compute_position_and_unitcell_gradients=True
        ),
    )
    potential = CachedPotential(
        potential,
        lens(
            lambda x: PotentialOut(
                x.systems.map_data(lambda s: s.potential_energy),
                PositionAndUnitCell(
                    x.particles.map_data(lambda p: p.position_gradients),
                    x.systems.map_data(lambda s: s.unitcell_gradients),
                ),
                EMPTY,
            )
        ),
        position_and_unitcell_idx_view,
    )
    propagator = PotentialAsPropagator(potential)

    def stress_fn(key: Array, state: MCMCState) -> Table[SystemId, StressResult]:
        chain = key_chain(key)
        # For stress calculations, we only consider the guest particles, so we take a "guest-only view" of the state.
        guest_only_state = state.guest_only
        state_with_derivatives = propagator(next(chain), guest_only_state)
        config_stress = molecular_virial_stress_from_state(
            next(chain), state_with_derivatives
        )
        p_tail = global_lennard_jones_tail_correction_pressure_from_state(
            next(chain), state
        )
        group_counts = state.groups.data.system.counts
        p_ideal = ideal_gas_pressure(group_counts, state.systems)
        return Table(
            state.systems.keys,
            StressResult(
                potential=config_stress.data,
                tail_correction=p_tail.data[:, None, None] * jnp.eye(3),
                ideal_gas=p_ideal.data[:, None, None] * jnp.eye(3),
            ),
        )

    return stress_fn


def run(config: Config) -> MCMCState:
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    state = init_state(next(chain), config)
    init_prop, propagator = make_propagator(config.run)
    state = propagate_and_fix(as_result_function(init_prop), next(chain), state)
    logged_data = make_mcmc_logged_data(
        state, make_guest_stress() if config.compute_stress else None
    )
    return run_mcmc(next(chain), propagator, state, config.run, logged_data)


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_mcmc_file(config.run.out_file))


if __name__ == "__main__":
    main()
