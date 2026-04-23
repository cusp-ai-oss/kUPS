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
from kups.core.neighborlist import (
    UniversalNeighborlistParameters,
)
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
from kups.core.typing import GroupId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, key_chain, tree_map
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
    """Ewald summation settings."""


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

    `md_nvtw.init_state` calls this to get the MCMC-typed scaffolding, then
    lifts the ``particles`` and ``systems`` fields to their MD-enabled
    counterparts before constructing its own :class:`MDNVTWidomState`.
    """
    chain = key_chain(key)
    macrostates = range(n_max + 1)

    # Expensive step (CIF parse, supercell, motifs, Peng-Robinson) — once.
    prepared = prepare_host(host, adsorbates)

    ps: list[Table[ParticleId, MCMCParticles]] = []
    gs: list[Table[GroupId, MCMCGroup]] = []
    ss: list[Table[SystemId, MCMCSystems]] = []
    n_species = len(adsorbates)
    for n in macrostates:
        init_ads = tuple(n if i == 0 else 0 for i in range(n_species))
        p, g, s = place_adsorbates(next(chain), prepared, init_ads)
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
    min_half_box = float(system.data.unitcell.perpendicular_lengths.min() / 2)

    return NVTWidomState(
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
class EnergyMomentsObserver[S](Propagator[S]):
    """Reads ``state.systems.data.potential_energy`` into a Welford accumulator
    at ``state.energy_moments``. Shared by mcmc_nvtw and md_nvtw."""

    def __call__(self, key: Array, state: S) -> S:
        del key
        energy = state.systems.data.potential_energy  # type: ignore[attr-defined]
        new_moments = state.energy_moments.data.update(energy)  # type: ignore[attr-defined]
        return replace(
            state,  # type: ignore[type-var]
            energy_moments=Table(state.energy_moments.keys, new_moments),  # type: ignore[attr-defined]
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
    config: NVTWidomRunConfig,
    *,
    ewald_enabled: bool = True,
) -> tuple[Propagator[NVTWidomState], Propagator[NVTWidomState]]:
    """Build the init / production propagator pair.

    Returns:
        ``(init, production)`` — the first computes the initial cached
        potential energies (one pass); the second runs one TMMC cycle per
        call: displacement loop + Widom ghost probes + energy-moments
        observer.
    """
    state_lens = identity_lens(NVTWidomState)

    # No-op probe: we don't exploit incremental updates here — the probe is
    # just the "enable the cache path" flag for LJ/Ewald. Typed locally so it
    # implements the narrow probe protocol the two potentials expect.
    def probe(state: NVTWidomState, update: MCMCStateUpdate) -> MCMCStateUpdate:
        del state
        return update

    # Ewald k-space and real-space sums run unconditionally when included,
    # even if all charges are zero. Skipping the term for neutral adsorbates
    # is a 2-3x per-step speedup without any numerical change.
    ewald_term = (
        (make_ewald_from_state(state_lens, probe, include_exclusion_mask=True),)
        if ewald_enabled
        else ()
    )
    potential = sum_potentials(
        *ewald_term,
        make_lennard_jones_from_state(state_lens, probe),
        make_lennard_jones_tail_correction_from_state(state_lens),
    )
    cached_potential, muvt_ratio = make_muvt_probability_ratio(state_lens, potential)

    nvt_propagator = _nvt_displacement_propagator(
        config, state_lens, cached_potential, muvt_ratio.boltzmann_log_likelihood_ratio
    )

    # Ghost probes reuse kUPS's existing GCMC proposal machinery: the
    # `ExchangeMove` wraps `insert_random_motif` / `delete_random_motif` as
    # `ChangesFn`s over the full state, and `MuVTLogProbabilityRatio` supplies
    # the μVT log acceptance ratio. `GhostProbe` runs the probe but discards
    # the resulting patch — state is never modified.
    exchange = ExchangeMove(
        positions=state_lens.focus(lambda x: x.particles),
        groups=state_lens.focus(lambda x: x.groups),
        motifs=state_lens.focus(lambda x: x.motifs),
        unitcell=state_lens.focus(lambda x: x.systems.map_data(lambda d: d.unitcell)),
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
            ResetOnErrorPropagator(nvt_propagator),
            LoopPropagator(widom_cycle, config.num_widom_per_cycle),
        )
    )
    init_prop = ResetOnErrorPropagator(PotentialAsPropagator(cached_potential))
    return init_prop, production


def _displacement_patch_fn(
    key: Array, state: NVTWidomState, proposal: ParticlePositionChanges
) -> MCMCStateUpdate:
    """Thin adapter: lift a pure displacement proposal to an ``MCMCStateUpdate``.

    `make_displacement_mcmc_propagator` emits ``ParticlePositionChanges``;
    the neighbor-list update pipeline expects ``ExchangeChanges``. The
    conversion is zero-copy (just a re-tagging of particle ids).
    """
    n_sys = len(state.systems)
    exchange = exchange_changes_from_position_changes(
        proposal, state.particles, state.groups, n_sys
    )
    return MCMCStateUpdate.from_changes(key, state, exchange)


def _nvt_displacement_propagator(
    config: NVTWidomRunConfig,
    state_lens,
    cached_potential,
    boltzmann_ratio,
) -> Propagator[NVTWidomState]:
    """NVT translation/rotation/reinsertion loop — no exchange, $N$ is fixed.

    Uses `make_displacement_mcmc_propagator` rather than
    `make_gcmc_mcmc_propagator(..., exchange_weight=0)` so that
    `propose_mixed` does not eagerly evaluate the ExchangeMove proposal
    every step (which `lax.select_n` would then discard).
    """
    propagator = make_displacement_mcmc_propagator(
        state_lens,
        _displacement_patch_fn,
        boltzmann_ratio,
        translation_weight=config.translation_prob,
        rotation_weight=config.rotation_prob,
        reinsertion_weight=config.reinsertion_prob,
    )
    del cached_potential  # energy is re-cached inside the MCMC loop
    return LoopPropagator(propagator, config.num_displacements_per_cycle)


def run(config: Config) -> NVTWidomState:
    """Initialise, warm up, and run the TMMC production loop."""
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))

    state = init_state(next(chain), config)
    init_prop, propagator = make_propagator(
        config.run, ewald_enabled=config.ewald.enabled
    )
    state = propagate_and_fix(as_result_function(init_prop), next(chain), state)

    # Warmup: thermalise configurations; accumulators are reset afterwards so
    # the extreme transients from random adsorbate placement do not poison
    # the production moments. `run_warmup_cycles` wraps the propagator in
    # `jit(..., donate_argnums=(1,))` so XLA can reuse state buffers in-place
    # — critical for memory-bounded execution over long runs.
    state = run_warmup_cycles(
        next(chain), propagator, state, config.run.num_warmup_cycles
    )
    state = _reset_accumulators(state)

    logger = TqdmLogger(config.run.num_cycles)
    state = run_simulation_cycles(
        next(chain), propagator, state, config.run.num_cycles, logger
    )
    return state


def _reset_accumulators(state: NVTWidomState) -> NVTWidomState:
    """Zero the C-matrix and energy-moments accumulators in-place."""
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


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    state = run(config)
    rich.print("Macrostate N:", state.macrostate_n)
    rich.print("Transition statistics:", state.transition_statistics)
    rich.print("Energy moments:", state.energy_moments)


if __name__ == "__main__":
    main()
