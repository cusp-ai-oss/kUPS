# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Hybrid MD + flat-histogram (TMMC) simulation entry point.

Identical in spirit to
[mcmc_nvtw][kups.application.simulations.mcmc_nvtw], but replaces the NVT
Monte Carlo displacement loop with an NVT molecular-dynamics integrator
(BAOAB Langevin). Widom ghost probes and the TMMC / Widom accumulators are
reused without modification — the flat-histogram machinery is agnostic to
how the underlying configurational sampling is done.

Why this works: the Widom ghost probe only reads potential energies and
positions, not momenta. Kinetic contributions to the grand canonical
partition function (the $\\Lambda^{-3N}$ kinetic factor in Witman eq 8)
cancel in the ratio $P_\\mathrm{ins}/P_\\mathrm{del}$ exactly as they do in
a pure-MC run, so $\\ln Q_c(N, V, \\beta)$ reconstruction is unchanged. The
accumulators, `widom.py` primitives, and `flat_histogram.py` post-processing
are all shared with `mcmc_nvtw`.
"""

from __future__ import annotations

import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import optax
import rich
from jax import Array
from nanoargs.cli import NanoArgs
from pydantic import BaseModel

from kups.application.mcmc.data import (
    AdsorbateConfig,
    HostConfig,
    MCMCGroup,
    MCMCParticles,
    MotifParticles,
)
from kups.application.md.data import MDSystems
from kups.application.simulations.mcmc_nvtw import (
    EnergyMomentsObserver,
    build_tmmc_state,
    update_deletion_stats,
    update_insertion_stats,
)
from kups.application.simulations.mcmc_rigid import (
    EwaldConfig,
    LJConfig,
)
from kups.application.utils.propagate import (
    run_simulation_cycles,
    run_warmup_cycles,
)
from kups.core.constants import BOLTZMANN_CONSTANT, FEMTO_SECOND, PASCAL
from kups.core.data import Buffered, Table, WithCache
from kups.core.data.buffered import system_view
from kups.core.data.wrappers import WithIndices
from kups.core.lens import bind as lens_bind
from kups.core.lens import identity_lens, lens
from kups.core.logging import TqdmLogger
from kups.core.neighborlist import (
    Edges,
    RefineMaskNeighborList,
    UniversalNeighborlistParameters,
    neighborlist_changes,
)
from kups.core.parameter_scheduler import ParameterSchedulerState
from kups.core.patch import Accept
from kups.core.potential import PotentialAsPropagator, PotentialOut, sum_potentials
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.result import as_result_function
from kups.core.typing import (
    GroupId,
    MotifParticleId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, key_chain, tree_zeros_like
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
from kups.md.integrators import make_md_step_from_state
from kups.potential.classical.ewald import (
    EwaldCache,
    EwaldParameters,
    make_ewald_from_state,
)
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    make_lennard_jones_from_state,
)
from kups.relaxation.optax import make_optimizer

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


# -- combined particle / system types ----------------------------------


@dataclass
class MDMCMCParticles(MCMCParticles):
    r"""MCMCParticles + ``momenta`` for MD.

    Satisfies both the exchange-move protocols and
    [IsMDState][kups.md.integrators.IsMDState]. The ``exclusion`` index is
    inherited as a property from :class:`MCMCParticles` — derived from the
    group index, so host atoms share an id and LJ skips intra-host pairs.
    """

    momenta: Array = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        super().__post_init__()
        if self.momenta is None:
            object.__setattr__(self, "momenta", jnp.zeros_like(self.positions))

    @property
    def forces(self) -> Array:
        """Negative position gradient, shape ``(n_atoms, 3)``."""
        return -self.position_gradients

    @property
    def velocities(self) -> Array:
        """Momenta divided by masses, shape ``(n_atoms, 3)``."""
        return self.momenta / self.masses[..., None]


@dataclass
class MDMCMCSystems(MDSystems):
    r"""MD systems with an added ``log_fugacity`` for $\mu\text{VT}$ Widom probes.

    All MD thermostat / barostat fields come through from
    :class:`~kups.application.md.data.MDSystems`. The reservoir log-fugacity
    is what the Widom ghost probe's
    [MuVTLogProbabilityRatio][kups.mcmc.probability.MuVTLogProbabilityRatio]
    reads to compute the acceptance factor at simulation conditions.
    """

    log_fugacity: Array

    @property
    def log_activity(self) -> Array:
        """$\\ln(f / k_B T)$, shape ``(n_systems, n_species)``."""
        return (
            self.log_fugacity - jnp.log(self.temperature * BOLTZMANN_CONSTANT)[:, None]
        )


# -- config / state ----------------------------------------------------


class MDNVTWidomRunConfig(BaseModel):
    """Run-time configuration for an MD + TMMC simulation."""

    out_file: str
    n_max: int
    """Maximum macrostate particle count."""
    num_cycles: int
    num_warmup_cycles: int
    """MD-based warmup cycles (run after the MC warmup below)."""
    num_mc_warmup_cycles: int = 200
    """MC displacement-based warmup cycles that run *before* MD.

    Random placement of $N$ adsorbates in a crowded host leaves overlapping
    configurations that BAOAB Langevin can't recover from at any reasonable
    timestep. A short MC displacement/reinsertion loop at fixed $N$ efficiently
    rejects overlapping moves and drives each macrostate to a low-energy,
    non-overlapping configuration before the MD takes over.
    """
    num_mc_displacements_per_warmup_cycle: int = 20
    """MC moves per warmup cycle (translation + rotation + reinsertion)."""
    num_minimization_steps: int = 0
    """Pre-MD FIRE minimization steps on particle positions.

    Defaults to 0 (skip). When the MC warmup is sufficient — which is typical
    once ``num_mc_warmup_cycles`` is reasonably large — the system is already
    at a low-energy, non-overlapping configuration and FIRE's inertial descent
    tends to kick it back out of the minimum rather than improving it (the
    global timestep adapts to whichever system has the largest
    gradient-velocity alignment, which overshoots in the densely-packed
    macrostates). Only enable FIRE if MC warmup cannot reach sub-eV per-system
    energies (e.g. very high density or a small cell).
    """
    minimization_max_step: float = 0.1
    r"""Per-step max atomic displacement in $\text{\AA}$ when FIRE is enabled."""
    num_md_steps_per_cycle: int = 50
    """MD integration steps between ghost-probe batches."""
    num_widom_per_cycle: int = 5
    """Ghost insertion + deletion trials per cycle per macrostate."""
    time_step_fs: float = 1.0
    friction_coefficient_inv_fs: float = 0.01
    translation_prob: float = 1 / 3
    rotation_prob: float = 1 / 3
    reinsertion_prob: float = 1 / 3
    seed: int | None = None


class Config(BaseModel):
    """Top-level configuration for MD + TMMC simulations."""

    adsorbates: tuple[AdsorbateConfig, ...]
    host: HostConfig
    run: MDNVTWidomRunConfig
    lj: LJConfig
    ewald: EwaldConfig


@dataclass
class MDNVTWidomState:
    """State for an MD + TMMC simulation — analogue of
    :class:`~kups.application.simulations.mcmc_nvtw.NVTWidomState` but with
    MD particle/system fields.
    """

    particles: Buffered[ParticleId, MDMCMCParticles]
    groups: Buffered[GroupId, MCMCGroup]
    motifs: Table[MotifParticleId, MotifParticles]
    systems: Table[SystemId, MDMCMCSystems]
    neighborlist_params: UniversalNeighborlistParameters
    lj_parameters: WithCache[
        GlobalTailCorrectedLennardJonesParameters,
        PotentialOut,  # type: ignore[type-arg]
    ]
    ewald_parameters: WithCache[EwaldParameters, EwaldCache]  # type: ignore[type-arg]
    # MC-warmup schedulers: adaptive step widths for translation / rotation /
    # reinsertion moves during the pre-MD equilibration phase.
    translation_params: Table[SystemId, ParameterSchedulerState]
    rotation_params: Table[SystemId, ParameterSchedulerState]
    reinsertion_params: Table[SystemId, ParameterSchedulerState]
    exchange_params: Table[SystemId, ParameterSchedulerState]
    transition_statistics: Table[SystemId, TransitionStatistics]
    energy_moments: Table[SystemId, EnergyMoments]
    macrostate_n: Array
    step: Array  # required by IsMdState for the step counter wrapper

    @property
    def max_cutoff(self) -> Table[SystemId, Array]:
        return Table(
            self.systems.keys,
            jnp.maximum(
                self.lj_parameters.data.cutoff.data,
                self.ewald_parameters.data.cutoff.data,
            ),
        )

    @property
    def move_capacity(self):
        from kups.core.capacity import FixedCapacity

        motif_size = self.motifs.data.motif.max_count
        assert motif_size is not None
        return FixedCapacity(motif_size * len(self.systems))

    @property
    def neighborlist(self):
        from kups.core.neighborlist import DenseNearestNeighborList

        return DenseNearestNeighborList.from_state(self)


@dataclass
class MDNVTWidomStateUpdate:
    """Patch analogue of `MCMCStateUpdate` typed for MD-enabled particles.

    Constructs its `new_particles` as `Buffered[..., MDMCMCParticles]` (rather
    than plain `MCMCParticles`) so that `update_if` against the live state
    sees matching pytree structure when applying a ghost-probe patch (even
    though the ghost patch is discarded, the probe calls
    `CachedPotential(state, patch)` which applies it to an internal copy of
    the state). Mirrors `MCMCStateUpdate` otherwise.
    """

    _particles: WithIndices  # WithIndices[ParticleId, Buffered[..., MDMCMCParticles]]
    groups: WithIndices
    edges_after: Edges
    edges_before: Edges

    @staticmethod
    def from_changes(
        key: Array,
        state: MDNVTWidomState,
        proposal,  # ExchangeChanges
    ) -> MDNVTWidomStateUpdate:
        del key
        p_data = proposal.particles.data.data
        g_data = proposal.groups.data.data
        motif_data = state.motifs[p_data.motif]
        n = len(p_data.motif)
        new_particles = Buffered(
            proposal.particles.data.keys,
            MDMCMCParticles(
                positions=p_data.new_positions,
                masses=motif_data.masses,
                atomic_numbers=motif_data.atomic_numbers,
                charges=motif_data.charges,
                labels=motif_data.labels,
                system=p_data.system,
                group=p_data.group,
                motif=p_data.motif,
                momenta=jnp.zeros((n, 3)),
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
        return MDNVTWidomStateUpdate(
            _particles=particle_changes,
            groups=group_changes,
            edges_after=result.added,
            edges_before=result.removed,
        )

    def __call__(self, state: MDNVTWidomState, accept: Accept) -> MDNVTWidomState:
        acc = Table.broadcast_to(accept, state.systems)
        new_groups = state.groups.update_if(
            acc, self.groups.indices, self.groups.data.data
        )
        new_particles = state.particles.update_if(
            acc, self._particles.indices, self._particles.data.data
        )
        return lens_bind(state, lambda x: (x.particles, x.groups)).set(
            (new_particles, new_groups)
        )

    @property
    def particles(self):
        return self._particles.map_data(lambda x: x.data)

    @property
    def neighborlist_before(self):
        return RefineMaskNeighborList(self.edges_before)

    @property
    def neighborlist_after(self):
        return RefineMaskNeighborList(self.edges_after)


# -- state construction ------------------------------------------------


def _maxwell_boltzmann_momenta(key: Array, masses: Array, temperature: float) -> Array:
    """Sample $p_i \\sim \\mathcal{N}(0, \\sqrt{m_i k_B T})$, zero COM drift."""
    std = jnp.sqrt(masses * temperature * BOLTZMANN_CONSTANT)
    n = masses.shape[0]
    p = jax.random.normal(key, (n, 3)) * std[:, None]
    if n > 0:
        p = p - p.sum(axis=0) / n
    return p


def init_state(key: Array, config: Config) -> MDNVTWidomState:
    """Construct the batched MD + TMMC state.

    Calls :func:`~kups.application.simulations.mcmc_nvtw.build_tmmc_state`
    for the common TMMC scaffolding, then lifts :class:`MCMCParticles` to
    :class:`MDMCMCParticles` (adds Maxwell-Boltzmann momenta) and
    :class:`MCMCSystems` to :class:`MDMCMCSystems` (adds MD thermostat
    fields), and attaches the MD step counter.
    """
    chain = key_chain(key)
    base = build_tmmc_state(
        next(chain),
        config.host,
        config.adsorbates,
        config.lj,
        config.ewald,
        config.run.n_max,
    )

    # Lift MCMCParticles → MDMCMCParticles: add Maxwell-Boltzmann momenta.
    mcmc_p = base.particles.data
    momenta = _maxwell_boltzmann_momenta(
        next(chain), mcmc_p.masses, config.host.temperature
    )
    md_particles_data = MDMCMCParticles(
        positions=mcmc_p.positions,
        masses=mcmc_p.masses,
        atomic_numbers=mcmc_p.atomic_numbers,
        charges=mcmc_p.charges,
        labels=mcmc_p.labels,
        system=mcmc_p.system,
        group=mcmc_p.group,
        motif=mcmc_p.motif,
        momenta=momenta,
    )
    # Buffered zeroes non-viewed leaves (positions, momenta, masses) for
    # unoccupied slots. The BAOAB A-step computes v = p/m = 0/0 = NaN there,
    # but the NaN never reaches a meaningful observable: LJ excludes them via
    # the OOB exclusion index, unit-cell wrap reads an OOB system index, and
    # ghost probes fill real masses before reusing a slot.
    particles = Buffered(base.particles.keys, md_particles_data, base.particles.view)

    # Lift MCMCSystems → MDMCMCSystems: add MD thermostat / barostat fields.
    mcmc_s = base.systems.data
    n_sys = len(base.systems)
    md_systems_data = MDMCMCSystems(
        unitcell=mcmc_s.unitcell,
        temperature=mcmc_s.temperature,
        time_step=jnp.full(n_sys, config.run.time_step_fs * FEMTO_SECOND),
        friction_coefficient=jnp.full(
            n_sys, config.run.friction_coefficient_inv_fs / FEMTO_SECOND
        ),
        thermostat_time_constant=jnp.full(n_sys, 100.0 * FEMTO_SECOND),
        target_pressure=jnp.full(n_sys, 1e5 * PASCAL),
        pressure_coupling_time=jnp.full(n_sys, 1000.0 * FEMTO_SECOND),
        compressibility=jnp.full(n_sys, 4.5e-10 / PASCAL),
        minimum_scale_factor=jnp.full(n_sys, 0.9),
        unitcell_gradients=tree_zeros_like(mcmc_s.unitcell),
        potential_energy=mcmc_s.potential_energy,
        log_fugacity=mcmc_s.log_fugacity,
    )
    systems = Table(base.systems.keys, md_systems_data)

    return MDNVTWidomState(
        particles=particles,
        groups=base.groups,
        motifs=base.motifs,
        systems=systems,
        neighborlist_params=base.neighborlist_params,
        lj_parameters=base.lj_parameters,
        ewald_parameters=base.ewald_parameters,
        translation_params=base.translation_params,
        rotation_params=base.rotation_params,
        reinsertion_params=base.reinsertion_params,
        exchange_params=base.exchange_params,
        transition_statistics=base.transition_statistics,
        energy_moments=base.energy_moments,
        macrostate_n=base.macrostate_n,
        step=jnp.array([0]),
    )


# -- propagators -------------------------------------------------------


def make_propagator(
    config: MDNVTWidomRunConfig,
    *,
    ewald_enabled: bool = True,
):
    """Build the propagator bundle for an MD + TMMC run.

    Returns a dict with:

    - ``init``: populates the MuVT energy cache at $t=0$.
    - ``production``: one TMMC cycle = MD loop + cache refresh + ghost probes.
    - ``mc_warmup``: pure MC displacement loop. Drives random-placed
      configurations to thermally accessible states *before* MD takes over.
    - ``md_potential``: the (gradient-computing) LJ potential used by MD; also
      invoked by the pre-MD energy minimizer.
    """
    state_lens = identity_lens(MDNVTWidomState)

    # No-op probe: we don't exploit incremental updates — the probe is just
    # the "enable the cache path" flag for LJ/Ewald.
    def probe(
        state: MDNVTWidomState, update: MDNVTWidomStateUpdate
    ) -> MDNVTWidomStateUpdate:
        del state
        return update

    # Two LJ instances over the same state: md_potential computes
    # position/unitcell gradients for BAOAB (probe=None → no cache write,
    # matches md_lj's pattern); mc_potential computes energy only and feeds
    # the cached MuVT ratio (probe=identity → cache write with EmptyType
    # gradients, matches mcmc_rigid's pattern). Sharing one cache would
    # mismatch the gradient type.
    md_potential = make_lennard_jones_from_state(
        state_lens, None, compute_position_and_unitcell_gradients=True
    )

    mc_ewald_term = (
        (make_ewald_from_state(state_lens, probe, include_exclusion_mask=True),)
        if ewald_enabled
        else ()
    )
    mc_potential = sum_potentials(
        *mc_ewald_term,
        make_lennard_jones_from_state(state_lens, probe),
    )
    cached_mc_potential, muvt_ratio = make_muvt_probability_ratio(
        state_lens, mc_potential
    )

    # Using `make_md_step_from_state` (not `make_md_propagator`) because the
    # latter hard-wires a cache-lens shape for `LjMdState` that doesn't match
    # our Buffered + MDMCMC state.
    md_force_fn: Propagator[MDNVTWidomState] = PotentialAsPropagator(md_potential)
    md_step: Propagator[MDNVTWidomState] = make_md_step_from_state(
        state_lens,
        md_force_fn,
        "baoab_langevin",  # type: ignore[arg-type]
    )
    md_loop: Propagator[MDNVTWidomState] = LoopPropagator(
        md_step, config.num_md_steps_per_cycle
    )

    # Ghost probes reuse the TMMC pipeline unchanged.
    exchange = ExchangeMove(
        positions=state_lens.focus(lambda x: x.particles),
        groups=state_lens.focus(lambda x: x.groups),
        motifs=state_lens.focus(lambda x: x.motifs),
        unitcell=state_lens.focus(lambda x: x.systems.map_data(lambda d: d.unitcell)),
        capacity=state_lens.focus(lambda x: x.move_capacity),
    )
    stat_lens = lens(lambda s: s.transition_statistics.data, cls=MDNVTWidomState)

    ghost_insertion = GhostProbe(
        propose_fn=exchange.propose_insertion,
        patch_fn=MDNVTWidomStateUpdate.from_changes,
        log_probability_ratio_fn=muvt_ratio,
        stat_lens=stat_lens,
        update_fn=update_insertion_stats,
    )
    ghost_deletion = GhostProbe(
        propose_fn=exchange.propose_deletion,
        patch_fn=MDNVTWidomStateUpdate.from_changes,
        log_probability_ratio_fn=muvt_ratio,
        stat_lens=stat_lens,
        update_fn=update_deletion_stats,
    )
    energy_observer: Propagator[MDNVTWidomState] = EnergyMomentsObserver()

    widom_cycle = SequentialPropagator(
        (ghost_insertion, ghost_deletion, energy_observer)
    )
    # md_potential doesn't touch the MuVT energy cache (probe=None). Re-evaluate
    # via cached_mc_potential after each MD loop so the ghost probes see the
    # current potential energy, not the pre-MD one.
    cache_refresh: Propagator[MDNVTWidomState] = PotentialAsPropagator(
        cached_mc_potential
    )
    production = SequentialPropagator(
        (
            ResetOnErrorPropagator(md_loop),
            cache_refresh,
            LoopPropagator(widom_cycle, config.num_widom_per_cycle),
        )
    )

    # MC warmup propagator: pure displacement moves (translation / rotation /
    # reinsertion) at fixed N, no exchange. Drives each macrostate to a
    # non-overlapping configuration before MD takes over. Uses the same
    # cached_mc_potential / Boltzmann log-ratio as the ghost probes, so the
    # warmup is consistent with the acceptance criteria that TMMC will use.
    boltzmann_ratio = muvt_ratio.boltzmann_log_likelihood_ratio

    def _displacement_patch_fn(
        key: Array, state: MDNVTWidomState, proposal: ParticlePositionChanges
    ) -> MDNVTWidomStateUpdate:
        n_sys_ = len(state.systems)
        ex = exchange_changes_from_position_changes(
            proposal, state.particles, state.groups, n_sys_
        )
        return MDNVTWidomStateUpdate.from_changes(key, state, ex)

    mc_propagator = make_displacement_mcmc_propagator(
        state_lens,
        _displacement_patch_fn,
        boltzmann_ratio,
        translation_weight=config.translation_prob,
        rotation_weight=config.rotation_prob,
        reinsertion_weight=config.reinsertion_prob,
    )
    mc_warmup = LoopPropagator(
        mc_propagator, config.num_mc_displacements_per_warmup_cycle
    )

    # Init: populate the MuVT energy cache; BAOAB computes forces on the first
    # step.
    init_prop = ResetOnErrorPropagator(PotentialAsPropagator(cached_mc_potential))
    return {
        "init": init_prop,
        "production": production,
        "mc_warmup": mc_warmup,
        "md_potential": md_potential,
    }


def _minimize_overlaps(
    state: MDNVTWidomState,
    md_potential,
    n_steps: int,
    max_step: float,
) -> MDNVTWidomState:
    r"""Optional FIRE minimisation on particle positions (no-op if n_steps<=0).

    FIRE + per-step max-step clip + sign-flip chain, same transform chain used
    by :mod:`kups.application.simulations.relax_lj`. Positions are re-wrapped
    into the primary unit cell after each step; lattice vectors are held
    fixed.

    Only enable this when MC warmup leaves overlaps FIRE can still help with;
    on an already-relaxed configuration, FIRE's inertial descent can push the
    system back out of its basin.
    """
    if n_steps <= 0:
        return state

    optimizer = make_optimizer(
        [
            {"transform": "scale_by_fire"},
            {"transform": "max_step_size", "max_step_size": max_step},
            {"transform": "scale", "step_size": -1.0},
        ]
    )
    pos_lens = lens_bind(state, lambda x: x.particles.data.positions)
    positions = state.particles.data.positions
    opt_state = optimizer.init(positions)

    @jax.jit
    def step(positions: Array, opt_state):
        trial_state = pos_lens.set(positions)
        result = md_potential(trial_state)
        grad = result.data.gradients.positions.data
        updates, new_opt_state = optimizer.update(grad, opt_state, positions)
        new_positions: Array = optax.apply_updates(positions, updates)  # type: ignore[assignment]
        return _wrap_positions(new_positions, trial_state), new_opt_state

    for _ in range(n_steps):
        positions, opt_state = step(positions, opt_state)
    return pos_lens.set(positions)


def _wrap_positions(positions: Array, state: MDNVTWidomState) -> Array:
    """Wrap per-particle positions into each system's primary unit cell."""
    sys_idx = state.particles.data.system.indices
    unitcells = state.systems.data.unitcell
    return unitcells[sys_idx].wrap(positions)


def _reset_md_accumulators(state: MDNVTWidomState) -> MDNVTWidomState:
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


def run(config: Config) -> MDNVTWidomState:
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))

    state = init_state(next(chain), config)
    props = make_propagator(config.run, ewald_enabled=config.ewald.enabled)
    state = propagate_and_fix(as_result_function(props["init"]), next(chain), state)

    # Phase 1: MC warmup — resolves gross placement overlaps via rejection.
    if config.run.num_mc_warmup_cycles > 0:
        state = run_warmup_cycles(
            next(chain), props["mc_warmup"], state, config.run.num_mc_warmup_cycles
        )

    # Phase 2 (optional, default off): FIRE minimization. Only useful when MC
    # warmup cannot resolve overlaps on its own; otherwise FIRE's inertial
    # descent tends to kick the system back out of the MC-equilibrated basin.
    state = _minimize_overlaps(
        state,
        props["md_potential"],
        n_steps=config.run.num_minimization_steps,
        max_step=config.run.minimization_max_step,
    )

    # Phase 3: MD warmup — thermalises momenta at the minimized configuration.
    state = run_warmup_cycles(
        next(chain), props["production"], state, config.run.num_warmup_cycles
    )
    state = _reset_md_accumulators(state)

    logger = TqdmLogger(config.run.num_cycles)
    state = run_simulation_cycles(
        next(chain), props["production"], state, config.run.num_cycles, logger
    )
    return state


def main() -> None:
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    state = run(config)
    rich.print("macrostate_n:", state.macrostate_n)
    rich.print("transition_statistics:", state.transition_statistics)
    rich.print("energy_moments:", state.energy_moments)


if __name__ == "__main__":
    main()
