# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Widom test-particle insertion entry point.

Each cycle runs a displacement-move loop at fixed $N$ followed by a batch of
Widom ghost insertions; cumulative [WidomStatistics][kups.mcmc.widom.WidomStatistics]
snapshots are logged per cycle to HDF5 and reduced post-hoc via
[analyze_widom_file][kups.application.mcmc.analysis.analyze_widom_file].
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import rich
from jax import Array
from nanoargs.cli import NanoArgs
from pydantic import BaseModel

from kups.application.mcmc.analysis import analyze_widom_file
from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
    MCMCGroup,
    MCMCParticles,
    MCMCSystems,
    mcmc_state_from_config,
)
from kups.application.mcmc.logging import IsWidomState, make_widom_logged_data
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
from kups.core.lens import Lens, bind, identity_lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.parameter_scheduler import ParameterSchedulerState
from kups.core.patch import Patch
from kups.core.potential import (
    EMPTY,
    PotentialAsPropagator,
    PotentialOut,
    sum_potentials,
)
from kups.core.propagator import (
    LogProbabilityRatioFn,
    LoopPropagator,
    PatchFn,
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.result import as_result_function
from kups.core.storage import HDF5StorageWriter
from kups.core.typing import GroupId, ParticleId, SystemId
from kups.core.utils.jax import dataclass, key_chain, tree_map
from kups.mcmc.moves import (
    ExchangeChanges,
    ExchangeMove,
    ParticlePositionChanges,
    exchange_changes_from_position_changes,
    make_displacement_mcmc_propagator,
)
from kups.mcmc.probability import make_muvt_probability_ratio
from kups.mcmc.widom import GhostProbe, WidomStatistics
from kups.potential.classical.blocking import (
    BlockingSpheresParameters,
    make_blocking_spheres_from_state,
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
    ps: list[Table[ParticleId, MCMCParticles]] = []
    gs: list[Table[GroupId, MCMCGroup]] = []
    ss: list[Table[SystemId, MCMCSystems]] = []
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
    blocking_spheres = BlockingSpheresParameters.from_data(
        [host.blocking_spheres for host in config.hosts]
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
    if blocking_spheres.radii.shape[0] > 0:
        blocking_nlist = UniversalNeighborlistParameters.estimate(
            Table.arange(jnp.array([num_buffer_particles / n_sys]), label=SystemId),
            system,
            blocking_spheres.system.max_over(blocking_spheres.radii),
        )
    else:
        blocking_nlist = UniversalNeighborlistParameters(0, 0, 0, 0)
    min_half_box = float(system.data.cell.perpendicular_lengths.min() / 2)

    return WidomState(
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
        widom_statistics=Table.arange(WidomStatistics.zeros(n_sys), label=SystemId),
    )


def _update_widom_stats(
    state: IsWidomState, stats: WidomStatistics, ln_alpha: Array
) -> WidomStatistics:
    r"""Accumulate one ghost insertion.

    The probe uses a bare Boltzmann log-ratio with a zero-move-log insertion
    proposal, so $\Delta U = -k_BT \ln\alpha$ exactly.
    """
    kT = state.systems.data.temperature * BOLTZMANN_CONSTANT
    delta_u = -ln_alpha * kT
    return stats.update(ln_alpha, delta_u)


def make_widom_probe_from_state[S: IsWidomState, Move: Patch[Any]](
    state: Lens[S, IsWidomState],
    patch_fn: PatchFn[S, ExchangeChanges, Move],
    log_probability_ratio_fn: LogProbabilityRatioFn[S, Move],
) -> GhostProbe[S, ExchangeChanges, Move, WidomStatistics]:
    """Plain-Widom probe: ghost insertion + ``WidomStatistics`` accumulator.

    Bundles the ``ExchangeMove`` construction, ``widom_statistics`` stat lens,
    and ``_update_widom_stats`` update callback. ``patch_fn`` and
    ``log_probability_ratio_fn`` are caller-provided (matching the
    :func:`make_displacement_mcmc_propagator` shape) so the same probe works
    with a bare Boltzmann ratio (plain Widom) or a fugacity-corrected ratio
    (Widom-in-GCMC).
    """
    exchange = ExchangeMove(
        positions=state.focus(lambda x: x.particles),
        groups=state.focus(lambda x: x.groups),
        motifs=state.focus(lambda x: x.motifs),
        cell=state.focus(lambda x: x.systems.map_data(lambda d: d.cell)),
        capacity=state.focus(lambda x: x.move_capacity),
    )
    return GhostProbe(
        propose_fn=exchange.propose_insertion,
        patch_fn=patch_fn,
        log_probability_ratio_fn=log_probability_ratio_fn,
        stat_lens=state.focus(lambda x: x.widom_statistics.data),
        update_fn=_update_widom_stats,
    )


def make_propagator(
    state: WidomState,
    config: WidomRunConfig,
) -> tuple[Propagator[WidomState], Propagator[WidomState]]:
    """Init + production propagator pair. Ewald and blocking spheres are
    added automatically based on ``state.is_charged`` / ``has_blocking_spheres``.
    """
    state_lens = identity_lens(WidomState)

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

    # Bare Boltzmann log-ratio (raw -βΔU), not the fugacity-corrected μVT
    # ratio used by GCMC.
    widom_probe = make_widom_probe_from_state(
        state_lens, MCMCStateUpdate.from_changes, boltzmann_ratio
    )
    widom_loop: Propagator[WidomState] = LoopPropagator(
        widom_probe, config.num_widom_per_cycle
    )

    production = ResetOnErrorPropagator(SequentialPropagator((nvt_loop, widom_loop)))
    init_prop = ResetOnErrorPropagator(PotentialAsPropagator(cached_potential))
    return init_prop, production


def run(config: Config) -> WidomState:
    """Initialise, warm up, and accumulate Widom statistics."""
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))

    state = init_state(next(chain), config)
    init_prop, propagator = make_propagator(state, config.run)
    state = propagate_and_fix(as_result_function(init_prop), next(chain), state)

    cycle_fn = make_cycle_function(propagator)
    state = run_warmup_cycles(
        next(chain), cycle_fn, state, config.run.num_warmup_cycles
    )
    state = bind(state, lambda x: x.widom_statistics.data).apply(WidomStatistics.reset)

    logged_data = make_widom_logged_data(state)
    logger = CompositeLogger(
        HDF5StorageWriter(
            config.run.out_file, logged_data, state, config.run.num_cycles
        ),
        TqdmLogger(config.run.num_cycles),
    )
    return run_simulation_cycles(
        next(chain), cycle_fn, state, config.run.num_cycles, logger
    )


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_widom_file(config.run.out_file))


if __name__ == "__main__":
    main()
