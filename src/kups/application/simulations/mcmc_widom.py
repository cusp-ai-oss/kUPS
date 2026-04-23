# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Plain Widom test-particle insertion simulation entry point.

Runs $NVT$ rigid-body MCMC with periodic Widom ghost insertions. Each cycle
does a displacement-move loop at fixed $N$ followed by a batch of Widom
insertions whose Boltzmann factors $\exp(-\beta\Delta U)$ are accumulated
online into a [WidomStatistics][kups.mcmc.widom.WidomStatistics] running sum.

Post-processing via [finalize_widom][kups.mcmc.widom.finalize_widom] yields
the excess chemical potential $\mu^\mathrm{ex}$, Henry coefficient $K_H$, and
isosteric heat of adsorption $q_\mathrm{st}$ (Vlugt's fluctuation formula).

Multiple hosts in the configuration fan out into parallel batched systems,
making it cheap to sweep $(T, P, \text{host})$ simultaneously.
"""

from __future__ import annotations

import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import rich
from jax import Array
from nanoargs.cli import NanoArgs
from pydantic import BaseModel

from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
    mcmc_state_from_config,
)
from kups.application.simulations.mcmc_rigid import (
    EwaldConfig,
    LJConfig,
    MCMCState,
    MCMCStateUpdate,
)
from kups.application.utils.propagate import (
    run_simulation_cycles,
    run_warmup_cycles,
)
from kups.core.data import Table, WithCache
from kups.core.data.buffered import add_buffers
from kups.core.data.index import unify_keys_by_cls
from kups.core.lens import identity_lens, lens
from kups.core.logging import TqdmLogger
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.parameter_scheduler import ParameterSchedulerState
from kups.core.potential import (
    EMPTY,
    PotentialAsPropagator,
    PotentialOut,
    sum_potentials,
)
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.result import as_result_function
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, key_chain, tree_map
from kups.mcmc.moves import (
    ExchangeMove,
    ParticlePositionChanges,
    exchange_changes_from_position_changes,
    make_displacement_mcmc_propagator,
)
from kups.mcmc.probability import make_muvt_probability_ratio
from kups.mcmc.widom import (
    GhostProbe,
    WidomStatistics,
    finalize_widom,
)
from kups.potential.classical.ewald import (
    EwaldCache,
    EwaldParameters,
    make_ewald_from_state,
)
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    make_lennard_jones_from_state,
    make_lennard_jones_tail_correction_from_state,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class WidomRunConfig(BaseModel):
    """Run-time configuration for a plain Widom simulation."""

    out_file: str
    num_cycles: int
    num_warmup_cycles: int
    num_displacements_per_cycle: int = 20
    num_widom_per_cycle: int = 10
    translation_prob: float = 1 / 3
    rotation_prob: float = 1 / 3
    reinsertion_prob: float = 1 / 3
    seed: int | None = None


class Config(BaseModel):
    """Top-level Widom simulation configuration."""

    adsorbates: tuple[AdsorbateConfig, ...]
    hosts: tuple[HostConfig, ...]
    """Hosts become parallel batched systems; one Widom sweep per host."""
    run: WidomRunConfig
    lj: LJConfig
    ewald: EwaldConfig
    max_num_adsorbates: int


@dataclass
class WidomState(MCMCState):
    """State for the Widom test-particle simulation.

    Inherits all [MCMCState][kups.application.simulations.mcmc_rigid.MCMCState]
    fields and adds one accumulator.

    Attributes:
        widom_statistics: Running sums for the Widom averages
            ($\\langle W\\rangle$, $\\langle UW\\rangle$, $\\langle U\\rangle$).
    """

    widom_statistics: Table[SystemId, WidomStatistics]


def _probe(state: WidomState, update: MCMCStateUpdate) -> MCMCStateUpdate:
    del state
    return update


def init_state(key: Array, config: Config) -> WidomState:
    """Build the batched Widom state via one ``mcmc_state_from_config`` call per host."""
    chain = key_chain(key)
    ps, gs, ss = [], [], []
    motifs = None
    for host in config.hosts:
        p, g, s, m = mcmc_state_from_config(next(chain), host, config.adsorbates)
        ps.append(p)
        gs.append(g)
        ss.append(s)
        motifs = m
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

    return WidomState(
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
        widom_statistics=Table.arange(WidomStatistics.zeros(n_sys), label=SystemId),
    )


def _update_widom_stats(
    state: WidomState, stats: WidomStatistics, ln_alpha: Array
) -> WidomStatistics:
    """Accumulate a ghost-insertion $\\ln\\alpha$ with the current $U$."""
    return stats.update(ln_alpha, state.systems.data.potential_energy)


def make_propagator(
    config: WidomRunConfig,
    *,
    ewald_enabled: bool = True,
) -> tuple[Propagator[WidomState], Propagator[WidomState]]:
    """Build the init / production propagator pair for Widom sampling."""
    state_lens = identity_lens(WidomState)

    ewald_term = (
        (make_ewald_from_state(state_lens, _probe, include_exclusion_mask=True),)
        if ewald_enabled
        else ()
    )
    potential = sum_potentials(
        *ewald_term,
        make_lennard_jones_from_state(state_lens, _probe),
        make_lennard_jones_tail_correction_from_state(state_lens),
    )
    cached_potential, muvt_ratio = make_muvt_probability_ratio(state_lens, potential)
    boltzmann_ratio = muvt_ratio.boltzmann_log_likelihood_ratio

    def displacement_patch_fn(
        key: Array, state: WidomState, proposal: ParticlePositionChanges
    ) -> MCMCStateUpdate:
        n_sys = len(state.systems)
        exchange = exchange_changes_from_position_changes(
            proposal, state.particles, state.groups, n_sys
        )
        return MCMCStateUpdate.from_changes(key, state, exchange)

    nvt_propagator = make_displacement_mcmc_propagator(
        state_lens,
        displacement_patch_fn,
        boltzmann_ratio,
        translation_weight=config.translation_prob,
        rotation_weight=config.rotation_prob,
        reinsertion_weight=config.reinsertion_prob,
    )
    nvt_loop: Propagator[WidomState] = LoopPropagator(
        nvt_propagator, config.num_displacements_per_cycle
    )

    # Widom ghost insertion uses the Boltzmann log-ratio (raw −βΔU) rather
    # than μVT — we want the bare Boltzmann factor, not the fugacity-corrected
    # acceptance ratio used by GCMC.
    exchange = ExchangeMove(
        positions=state_lens.focus(lambda x: x.particles),
        groups=state_lens.focus(lambda x: x.groups),
        motifs=state_lens.focus(lambda x: x.motifs),
        unitcell=state_lens.focus(lambda x: x.systems.map_data(lambda d: d.unitcell)),
        capacity=state_lens.focus(lambda x: x.move_capacity),
    )
    widom_probe = GhostProbe(
        propose_fn=exchange.propose_insertion,
        patch_fn=MCMCStateUpdate.from_changes,
        log_probability_ratio_fn=boltzmann_ratio,
        stat_lens=lens(lambda s: s.widom_statistics.data, cls=WidomState),
        update_fn=_update_widom_stats,
    )
    widom_loop: Propagator[WidomState] = LoopPropagator(
        widom_probe, config.num_widom_per_cycle
    )

    production = SequentialPropagator((ResetOnErrorPropagator(nvt_loop), widom_loop))
    init_prop = ResetOnErrorPropagator(PotentialAsPropagator(cached_potential))
    return init_prop, production


def run(config: Config) -> WidomState:
    """Initialise, warm up, and accumulate Widom statistics."""
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))

    state = init_state(next(chain), config)
    init_prop, propagator = make_propagator(
        config.run, ewald_enabled=config.ewald.enabled
    )
    state = propagate_and_fix(as_result_function(init_prop), next(chain), state)

    # `run_warmup_cycles` / `run_simulation_cycles` wrap the propagator in
    # `jit(..., donate_argnums=(1,))` so XLA can reuse state buffers in-place.
    # Skipping the donation — which a naked Python loop does — leaks O(state
    # size) per cycle because each jit invocation allocates fresh output
    # buffers without freeing the inputs.
    state = run_warmup_cycles(
        next(chain), propagator, state, config.run.num_warmup_cycles
    )
    state = replace(
        state,
        widom_statistics=Table(
            state.widom_statistics.keys, state.widom_statistics.data.reset()
        ),
    )

    logger = TqdmLogger(config.run.num_cycles)
    state = run_simulation_cycles(
        next(chain), propagator, state, config.run.num_cycles, logger
    )
    return state


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    state = run(config)
    temperature = state.systems.data.temperature
    volume = jnp.asarray([float(u.volume) for u in state.systems.data.unitcell[:]])
    result = finalize_widom(state.widom_statistics.data, temperature, volume)
    rich.print("Widom statistics:", state.widom_statistics)
    rich.print("μ_ex [eV]:", result.excess_chemical_potential)
    rich.print("Henry K_H [Å³/eV]:", result.henry_coefficient)
    rich.print("q_st [eV]:", result.heat_of_adsorption)


if __name__ == "__main__":
    main()
