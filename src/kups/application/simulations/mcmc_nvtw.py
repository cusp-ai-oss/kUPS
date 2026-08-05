# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Flat-histogram (TMMC) $NVT\!+\!W$ Monte Carlo simulation entry point.

Builds $N_\mathrm{max} + 1$ parallel NVT simulations (one per macrostate
$N = 0, \ldots, N_\mathrm{max}$) that share a single host framework. Each
cycle runs a loop of rigid-body displacement moves at fixed $N$, then
performs ghost insertion/deletion trials via
[GhostProbe][kups.mcmc.widom.GhostProbe] to accumulate the TMMC collection
matrix and per-macrostate energy cumulants.

Post-processing with
[TMMCSummary][kups.mcmc.flat_histogram.TMMCSummary] reconstructs
$\ln Q_c(N, V, \beta_\mathrm{sim})$ from the C-matrix (eq 7--8 of
Witman 2018) and Taylor-extrapolates it in $\beta$ (eq 9--10) to deliver
adsorption isotherms and isosteric heats over a wide $(T, P)$ range from
a single simulation temperature.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import rich
from jax import Array
from nanoargs.cli import NanoArgs
from pydantic import BaseModel

from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
    MCMCGroup,
    MCMCParticles,
    MCMCSystems,
    place_adsorbates,
    prepare_host,
)
from kups.application.mcmc.logging import make_tmmc_logged_data
from kups.application.potential.classical.blocking import (
    make_blocking_spheres_from_state,
)
from kups.application.potential.classical.ewald import (
    make_ewald_from_state,
)
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
    make_lennard_jones_tail_correction_from_state,
)
from kups.application.simulations.mcmc_rigid import (
    EwaldConfig,
    LJConfig,
    MCMCState,
    MCMCStateUpdate,
)
from kups.application.utils.propagate import (
    make_cycle_function,
    run_simulation_cycles,
    run_warmup_cycles,
)
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table, WithCache
from kups.core.data.buffered import add_buffers
from kups.core.data.index import unify_keys_by_cls
from kups.core.lens import identity_lens, lens
from kups.core.logging import CompositeLogger, TqdmLogger
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
from kups.core.storage import HDF5StorageWriter
from kups.core.typing import GroupId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, key_chain, tree_map
from kups.mcmc.flat_histogram import AdsorbateEOS, TMMCSummary
from kups.mcmc.moves import (
    ExchangeMove,
    ParticlePositionChanges,
    exchange_changes_from_position_changes,
    make_displacement_mcmc_propagator,
)
from kups.mcmc.probability import make_muvt_probability_ratio
from kups.mcmc.widom import (
    EnergyMoments,
    GhostProbe,
    TransitionStatistics,
)
from kups.potential.classical.blocking import (
    BlockingSpheresParameters,
)
from kups.potential.classical.ewald import (
    EwaldCache,
    EwaldParameters,
)
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class NVTWidomRunConfig(BaseModel):
    """Run-time configuration for an $NVT\\!+\\!W$ TMMC simulation."""

    out_file: str
    n_max: int
    """Maximum macrostate particle count. Creates ``n_max + 1`` parallel systems."""
    num_cycles: int
    num_warmup_cycles: int
    num_displacements_per_cycle: int = 20
    """Displacement moves per cycle per system (thermalises at fixed $N$)."""
    num_widom_per_cycle: int = 5
    """Ghost insertion + deletion trials per cycle per macrostate."""
    translation_prob: float = 1 / 3
    rotation_prob: float = 1 / 3
    reinsertion_prob: float = 1 / 3
    seed: int | None = None


class Config(BaseModel):
    """Top-level $NVT\\!+\\!W$ configuration."""

    adsorbates: tuple[AdsorbateConfig, ...]
    host: HostConfig
    """Single host framework, replicated across all macrostates."""
    run: NVTWidomRunConfig
    lj: LJConfig
    ewald: EwaldConfig


@dataclass
class NVTWidomState(MCMCState):
    """State for the $NVT\\!+\\!W$ TMMC simulation.

    Inherits every field from [MCMCState][kups.application.simulations.mcmc_rigid.MCMCState]
    so that existing MCMC propagators, potentials, and neighbor-list machinery
    operate without any adapter layer. Adds three accumulator fields for the
    flat-histogram pipeline:

    Attributes:
        transition_statistics: Per-macrostate TMMC C-matrix sums
            (insertion/deletion acceptances + trial counts).
        energy_moments: Per-macrostate Pébay/Welford running moments of the
            total potential energy.
        macrostate_n: Per-system macrostate particle count, shape
            ``(n_max + 1,)``.
    """

    transition_statistics: Table[SystemId, TransitionStatistics]
    energy_moments: Table[SystemId, EnergyMoments]
    macrostate_n: Array


def _probe(state: NVTWidomState, update: MCMCStateUpdate) -> MCMCStateUpdate:
    del state
    return update


def build_tmmc_state(
    key: Array,
    host: HostConfig,
    adsorbates: tuple[AdsorbateConfig, ...],
    lj: LJConfig,
    ewald: EwaldConfig,
    n_max: int,
) -> NVTWidomState:
    """Build a fully-initialised :class:`NVTWidomState`.

    Parses the host CIF once, fans out a single-system copy per macrostate
    $N \\in \\{0, \\ldots, N_\\mathrm{max}\\}$, unifies them into one batched
    table, adds buffer slots for ghost insertions, and initialises the LJ /
    Ewald / neighbor-list parameters and per-system adaptive-step schedulers.
    """
    assert len(adsorbates) == 1, (
        "TMMC macrostates track a single adsorbate species; got "
        f"{len(adsorbates)} adsorbate configs."
    )
    chain = key_chain(key)
    macrostates = range(n_max + 1)

    # Expensive step (CIF parse, supercell, motifs, Peng-Robinson) — once.
    prepared = prepare_host(host, adsorbates)

    ps: list[Table[ParticleId, MCMCParticles]] = []
    gs: list[Table[GroupId, MCMCGroup]] = []
    ss: list[Table[SystemId, MCMCSystems]] = []
    for n in macrostates:
        p, g, s = place_adsorbates(next(chain), prepared, (n,))
        ps.append(p)
        gs.append(g)
        ss.append(s)

    particles, groups, system = Table.union(ps, gs, ss)
    n_sys = len(system)

    motifs = prepared.motifs
    max_motif_size = motifs.data.motif.max_count
    assert max_motif_size is not None
    particles, groups, motifs, system = unify_keys_by_cls(
        (particles, groups, motifs, system)
    )

    # Buffer: one motif-sized slack per system for ghost insertions, plus
    # absolute headroom for the largest macrostate.
    num_buffer_particles = (n_max + n_sys) * max_motif_size
    num_buffer_groups = n_max + n_sys
    particles, groups = add_buffers(
        (particles, num_buffer_particles),
        (groups, num_buffer_groups),
    )

    lj_params = GlobalTailCorrectedLennardJonesParameters.from_dict(
        cutoff=lj.cutoff,
        parameters=lj.parameters,
        mixing_rule=lj.mixing_rule,
        tail_correction=lj.tail_correction,
    )
    # Each macrostate replicates the same host, so the same spheres apply to
    # every system.
    blocking_spheres = BlockingSpheresParameters.from_data(
        [host.blocking_spheres] * n_sys
    )
    ewald_params = EwaldParameters.make(
        particles,
        system,
        epsilon_total=ewald.precision,
        real_cutoff=ewald.real_cutoff,
    )
    n_kvecs = ewald_params.reciprocal_lattice_shifts.data.shape[1]
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts + num_buffer_particles / n_sys,
        system,
        tree_map(jnp.maximum, lj_params.cutoff, ewald_params.cutoff),
    )
    if blocking_spheres.radii.shape[0] > 0:
        # Systems without spheres have no radius to size from, so use the batch-wide max.
        max_radius = Table((SystemId(0),), blocking_spheres.radii.max(keepdims=True))
        blocking_nlist = UniversalNeighborlistParameters.estimate(
            particles.data.system.counts + num_buffer_particles / n_sys,
            system,
            Table.broadcast_to(max_radius, system),
        )
    else:
        blocking_nlist = UniversalNeighborlistParameters(0, 0, 0, 0)
    min_half_box = float(system.data.cell.perpendicular_lengths.min() / 2)

    return NVTWidomState(
        particles=particles,
        groups=groups,
        motifs=motifs,
        systems=system,
        neighborlist_params=neighborlist_params,
        blocking_spheres_neighborlist_params=blocking_nlist,
        lj_parameters=WithCache(
            lj_params,
            PotentialOut(Table.arange(jnp.zeros(n_sys), label=SystemId), EMPTY, EMPTY),
        ),
        ewald_parameters=WithCache(ewald_params, EwaldCache.make(n_sys, n_kvecs)),
        blocking_spheres_parameters=blocking_spheres,
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
        transition_statistics=Table.arange(
            TransitionStatistics.zeros(n_sys), label=SystemId
        ),
        energy_moments=Table.arange(EnergyMoments.zeros(n_sys), label=SystemId),
        macrostate_n=jnp.asarray(list(macrostates), dtype=jnp.int32),
    )


def init_state(key: Array, config: Config) -> NVTWidomState:
    """Build the batched $NVT\\!+\\!W$ state via :func:`build_tmmc_state`."""
    return build_tmmc_state(
        key,
        config.host,
        config.adsorbates,
        config.lj,
        config.ewald,
        config.run.n_max,
    )


@dataclass
class EnergyMomentsObserver(Propagator[NVTWidomState]):
    """Reads ``state.systems.data.potential_energy`` into a Welford accumulator
    at ``state.energy_moments``."""

    def __call__(self, key: Array, state: NVTWidomState) -> NVTWidomState:
        del key
        energy = state.systems.data.potential_energy
        new_moments = state.energy_moments.data.update(energy)
        return replace(
            state,
            energy_moments=Table(state.energy_moments.keys, new_moments),
        )


def update_insertion_stats(
    _state: Any, stats: TransitionStatistics, ln_alpha: Array
) -> TransitionStatistics:
    r"""Ghost-probe insertion hook: accumulate $\ln\alpha$ into the TMMC C-matrix."""
    return stats.update_insertion(ln_alpha)


def update_deletion_stats(
    state: Any, stats: TransitionStatistics, ln_alpha: Array
) -> TransitionStatistics:
    r"""Ghost-probe deletion hook: $\ln\alpha$ accumulator, with $N=0$ masking via
    ``state.macrostate_n``."""
    return stats.update_deletion(ln_alpha, state.macrostate_n)


def make_propagator(
    state: NVTWidomState,
    config: NVTWidomRunConfig,
) -> tuple[Propagator[NVTWidomState], Propagator[NVTWidomState]]:
    """Build the init / production propagator pair.

    Ewald and blocking spheres are added automatically based on
    ``state.is_charged`` / ``has_blocking_spheres``.

    Returns:
        ``(init, production)`` — the first computes the initial cached
        potential energies (one pass); the second runs one TMMC cycle per
        call: displacement loop + Widom ghost probes + energy-moments
        observer.
    """
    state_lens = identity_lens(NVTWidomState)

    potentials = [
        make_lennard_jones_from_state(state_lens, _probe),
        make_lennard_jones_tail_correction_from_state(state_lens),
    ]
    if state.is_charged:
        potentials.append(
            make_ewald_from_state(state_lens, _probe, include_exclusion_mask=True)
        )
    if state.has_blocking_spheres:
        potentials.append(make_blocking_spheres_from_state(state_lens))
    potential = sum_potentials(*potentials)
    cached_potential, muvt_ratio = make_muvt_probability_ratio(state_lens, potential)

    def displacement_patch_fn(
        key: Array, state: NVTWidomState, proposal: ParticlePositionChanges
    ) -> MCMCStateUpdate:
        exchange = exchange_changes_from_position_changes(
            proposal, state.particles, state.groups
        )
        return MCMCStateUpdate.from_changes(key, state, exchange)

    # NVT translation/rotation/reinsertion loop — no exchange, $N$ is fixed.
    nvt_propagator = make_displacement_mcmc_propagator(
        state_lens,
        displacement_patch_fn,
        muvt_ratio.boltzmann_log_likelihood_ratio,
        translation_weight=config.translation_prob,
        rotation_weight=config.rotation_prob,
        reinsertion_weight=config.reinsertion_prob,
    )
    nvt_loop: Propagator[NVTWidomState] = LoopPropagator(
        nvt_propagator, config.num_displacements_per_cycle
    )

    # Ghost probes reuse kUPS's existing GCMC proposal machinery: the
    # `ExchangeMove` wraps `insert_random_motif` / `delete_random_motif` as
    # `ChangesFn`s over the full state, and the μVT ratio supplies the
    # fugacity-corrected log acceptance. `GhostProbe` runs the probe but
    # discards the resulting patch — state is never modified.
    exchange = ExchangeMove(
        positions=state_lens.focus(lambda x: x.particles),
        groups=state_lens.focus(lambda x: x.groups),
        motifs=state_lens.focus(lambda x: x.motifs),
        cell=state_lens.focus(lambda x: x.systems.map_data(lambda d: d.cell)),
        capacity=state_lens.focus(lambda x: x.move_capacity),
    )
    stat_lens = lens(lambda s: s.transition_statistics.data, cls=NVTWidomState)

    ghost_insertion = GhostProbe(
        propose_fn=exchange.propose_insertion,
        patch_fn=MCMCStateUpdate.from_changes,
        log_probability_ratio_fn=muvt_ratio,
        stat_lens=stat_lens,
        update_fn=update_insertion_stats,
    )
    ghost_deletion = GhostProbe(
        propose_fn=exchange.propose_deletion,
        patch_fn=MCMCStateUpdate.from_changes,
        log_probability_ratio_fn=muvt_ratio,
        stat_lens=stat_lens,
        update_fn=update_deletion_stats,
    )
    energy_observer: Propagator[NVTWidomState] = EnergyMomentsObserver()

    widom_cycle = SequentialPropagator(
        (ghost_insertion, ghost_deletion, energy_observer)
    )
    production = SequentialPropagator(
        (
            ResetOnErrorPropagator(nvt_loop),
            LoopPropagator(widom_cycle, config.num_widom_per_cycle),
        )
    )
    init_prop = ResetOnErrorPropagator(PotentialAsPropagator(cached_potential))
    return init_prop, production


def _reset_accumulators(state: NVTWidomState) -> NVTWidomState:
    """Zero the C-matrix and energy-moments accumulators."""
    return replace(
        state,
        transition_statistics=Table(
            state.transition_statistics.keys,
            state.transition_statistics.data.reset(),
        ),
        energy_moments=Table(
            state.energy_moments.keys, state.energy_moments.data.reset()
        ),
    )


def run(config: Config) -> NVTWidomState:
    """Initialise, warm up, and run the TMMC production loop."""
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))

    state = init_state(next(chain), config)
    init_prop, propagator = make_propagator(state, config.run)
    state = propagate_and_fix(as_result_function(init_prop), next(chain), state)

    # Warmup: thermalise configurations; accumulators are reset afterwards so
    # the extreme transients from random adsorbate placement do not poison
    # the production moments.
    cycle_fn = make_cycle_function(propagator)
    state = run_warmup_cycles(
        next(chain), cycle_fn, state, config.run.num_warmup_cycles
    )
    state = _reset_accumulators(state)

    logged_data = make_tmmc_logged_data(state)
    logger = CompositeLogger(
        HDF5StorageWriter(
            config.run.out_file, logged_data, state, config.run.num_cycles
        ),
        TqdmLogger(config.run.num_cycles),
    )
    return run_simulation_cycles(
        next(chain), cycle_fn, state, config.run.num_cycles, logger
    )


def summarize(config: Config, state: NVTWidomState) -> TMMCSummary:
    r"""Bundle the final accumulators into a :class:`TMMCSummary` for post-processing."""
    adsorbate = config.adsorbates[0]
    stats = state.transition_statistics.data
    beta_sim = 1.0 / (BOLTZMANN_CONSTANT * state.systems.data.temperature[0])
    return TMMCSummary.from_transition_statistics(
        acceptance_insertion=stats.acceptance_insertion,
        acceptance_deletion=stats.acceptance_deletion,
        n_trials_insertion=stats.n_trials_insertion,
        n_trials_deletion=stats.n_trials_deletion,
        cumulants=state.energy_moments.data.finalize(),
        beta_sim=beta_sim,
        log_fugacity_sim=state.systems.data.log_fugacity[0, 0],
        adsorbate=AdsorbateEOS(
            critical_pressure=adsorbate.critical_pressure,
            critical_temperature=adsorbate.critical_temperature,
            acentric_factor=adsorbate.acentric_factor,
        ),
    )


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    state = run(config)
    summary = summarize(config, state)
    rich.print("Macrostate N:", state.macrostate_n)
    rich.print("ln Q_c(N, V, β_sim):", summary.log_partition_fn_sim)


if __name__ == "__main__":
    main()
