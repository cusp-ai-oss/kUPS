# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Log probability ratio functions for Monte Carlo acceptance criteria.

This module provides acceptance probability calculations for different statistical
ensembles: canonical (NVT) and grand canonical (μVT). These functions compute the
log probability ratios needed for Metropolis-Hastings acceptance/rejection.

Key components:

- **[BoltzmannLogProbabilityRatio][kups.mcmc.probability.BoltzmannLogProbabilityRatio]**: Canonical ensemble energy criterion
- **[LogFugacityRatio][kups.mcmc.probability.LogFugacityRatio]**: Grand canonical chemical potential term
- **[MuVTLogProbabilityRatio][kups.mcmc.probability.MuVTLogProbabilityRatio]**: Combined μVT ensemble acceptance

The acceptance probability in Metropolis-Hastings is:

$$
\\alpha = \\min\\left(1, \\exp(\\log p_{\\text{ratio}})\\right)
$$

where the log probability ratio includes contributions from the target distribution
and proposal probabilities.
"""

from typing import Any, Protocol

import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import Lens, View
from kups.core.patch import ComposedPatch, IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY,
    EMPTY_LENS,
    CachedPotential,
    EmptyType,
    MappedPotential,
    Potential,
    PotentialOut,
)
from kups.core.propagator import LogProbabilityRatio, LogProbabilityRatioFn
from kups.core.typing import (
    GroupId,
    HasLogActivity,
    HasMotifAndSystemIndex,
    HasPotentialEnergy,
    HasTemperature,
    HasUnitCell,
    SystemId,
)
from kups.core.utils.functools import pipe
from kups.core.utils.jax import dataclass, field, jit, vectorize
from kups.core.utils.math import log_factorial_ratio


@dataclass
class BoltzmannLogProbabilityRatio[State, Move: Patch](
    LogProbabilityRatioFn[State, Move]
):
    """Boltzmann acceptance criterion for canonical (NVT) ensemble.

    Computes the log probability ratio based on energy differences:

    $$
    \\log p_{\\text{ratio}} = -\\frac{\\Delta U}{k_B T} = \\frac{U_{\\text{old}} - U_{\\text{new}}}{k_B T}
    $$

    This corresponds to the Metropolis criterion for constant N, V, T simulations.

    Type Parameters:
        State: Simulation state type
        Move: Patch type for state updates

    Attributes:
        temperature: Lens to system temperature [K]
        potential: Cached potential for efficient energy evaluations

    Example:
        ```python
        boltzmann = BoltzmannLogProbabilityRatio(
            temperature=lambda s: s.temperature,
            potential=cached_lj_potential
        )

        # Evaluate acceptance for a proposed move
        result = boltzmann(state, move_patch)
        accept_prob = jnp.exp(result.log_probability_ratio)
        ```
    """

    temperature: View[State, Table[SystemId, Array]] = field(static=True)
    potential: CachedPotential[State, Any, Any, Move] = field(static=True)

    @jit
    def __call__(
        self, state: State, patch: Move
    ) -> WithPatch[LogProbabilityRatio, Patch[State]]:
        old_energies = self.potential.cached_value(state).total_energies
        potential_out = self.potential(state, patch)
        new_energies = potential_out.data.total_energies
        temperature = self.temperature(state)

        log_ratio = (old_energies - new_energies) / (temperature * BOLTZMANN_CONSTANT)
        out_patch = potential_out.patch
        return WithPatch(log_ratio, out_patch)


NVTLogLikelihoodRatio = BoltzmannLogProbabilityRatio
"""Alias for BoltzmannLogProbabilityRatio (canonical ensemble)."""


@dataclass
class LogFugacityRatio[State, Move: Patch](LogProbabilityRatioFn[State, Move]):
    """Chemical potential contribution for grand canonical (μVT) moves.

    Computes the log probability ratio from particle number changes in GCMC:

    $$
    \\log p_{\\text{ratio}} = \\sum_i (N_i^{\\text{new}} - N_i^{\\text{old}}) \\log(f_i V) + \\log\\frac{N_i^{\\text{old}}!}{N_i^{\\text{new}}!}
    $$

    where $f_i$ is the fugacity of species $i$, $V$ is the volume, and $N_i$ is
    the particle count. This implements the acceptance criterion for insertion/deletion
    moves in the grand canonical ensemble.

    Type Parameters:
        State: Simulation state type
        Move: Patch type for state updates

    Attributes:
        log_activity: Lens to log(fugacity/k_B T) per species, shape `(n_systems, n_species)`
        volume: Lens to system volumes, shape `(n_systems,)`
        particle_counts: Lens to current particle counts, shape `(n_systems, n_species)`
        patched_particle_counts: Probe for particle counts after applying patch

    Note:
        Activity is fugacity divided by k_B T: $a = f/(k_B T)$.
        The log_activity should be computed from equation of state (e.g., Peng-Robinson).
    """

    log_activity: View[State, Table[SystemId, Array]] = field(static=True)
    volume: View[State, Table[SystemId, Array]] = field(static=True)
    particle_counts: View[State, Table[SystemId, Array]] = field(static=True)
    patched_particle_counts: Probe[State, Move, Table[SystemId, Array]] = field(
        static=True
    )

    @jit
    def __call__(
        self, state: State, patch: Move
    ) -> WithPatch[LogProbabilityRatio, Patch[State]]:
        log_activity = self.log_activity(state)
        volume = self.volume(state)
        counts = self.particle_counts(state)
        patched_counts = self.patched_particle_counts(state, patch)
        log_activity, volume, counts, patched_counts = Table.broadcast(
            log_activity, volume, counts, patched_counts
        )

        @Table.transform
        def inner(la: Array, vol: Array, c: Array, pc: Array) -> Array:
            ln_fV = la + jnp.log(vol[:, None])
            ln_fact_ratio = log_factorial_ratio(c, pc)
            ln_ratio = (pc - c) * ln_fV + ln_fact_ratio
            ln_ratio = jnp.where(pc == c, 0.0, ln_ratio)
            ln_ratio = ln_ratio.sum(axis=-1)
            return ln_ratio

        return WithPatch(inner(log_activity, volume, counts, patched_counts), IdPatch())


@dataclass
class MuVTLogProbabilityRatio[State, Move: Patch](LogProbabilityRatioFn[State, Move]):
    """Combined acceptance criterion for grand canonical (μVT) ensemble.

    Combines chemical potential and Boltzmann factors for GCMC acceptance:

    $$
    \\log p_{\\text{ratio}} = \\log p_{\\text{fugacity}} + \\log p_{\\text{Boltzmann}}
    $$

    This is the full acceptance criterion for grand canonical Monte Carlo simulations,
    accounting for both particle number fluctuations (fugacity) and energy changes
    (Boltzmann factor).

    Type Parameters:
        State: Simulation state type
        Move: Patch type for state updates

    Attributes:
        log_fugacity_ratio: Chemical potential term for particle insertions/deletions
        boltzmann_log_likelihood_ratio: Energy term for configuration changes

    Example:
        ```python
        muvt = MuVTLogProbabilityRatio(
            log_fugacity_ratio=LogFugacityRatio(...),
            boltzmann_log_likelihood_ratio=BoltzmannLogProbabilityRatio(...)
        )

        # Evaluate GCMC acceptance
        result = muvt(state, exchange_move_patch)
        alpha = jnp.minimum(1.0, jnp.exp(result.log_probability_ratio))
        ```
    """

    log_fugacity_ratio: LogFugacityRatio[State, Move] = field(static=True)
    boltzmann_log_likelihood_ratio: BoltzmannLogProbabilityRatio[State, Move] = field(
        static=True
    )

    @jit
    def __call__(
        self, state: State, patch: Move
    ) -> WithPatch[LogProbabilityRatio, Patch[State]]:
        pot_result = self.boltzmann_log_likelihood_ratio(state, patch)
        fug_result = self.log_fugacity_ratio(state, patch)
        result_ratio = fug_result.data + pot_result.data
        return WithPatch(
            result_ratio, ComposedPatch((fug_result.patch, pot_result.patch))
        )


def motif_counts(groups: Table[GroupId, HasMotifAndSystemIndex]) -> Array:
    """Count the number of motifs of each species per system.

    Accumulates per-group species assignments into a 2D count matrix, where
    entry ``[i, j]`` is the number of groups of species ``j`` in system ``i``.
    Groups whose species index falls outside ``[0, num_motifs)`` are silently
    ignored (``mode="drop"``).

    Args:
        groups: Indexed collection of groups. Each group carries a ``system``
            and ``motif`` Index identifying its system and motif type.

    Returns:
        Integer array of shape ``(n_systems, num_motifs)`` with species counts
        per system.
    """
    num_systems = groups.data.system.num_labels
    num_motifs = groups.data.motif.num_labels

    @jit
    @vectorize(signature="(n),(n)->(k,d)")
    def _f(batch: Array, species: Array):
        return (
            jnp.zeros((num_systems, num_motifs), dtype=int)
            .at[batch, species]
            .add(1, mode="drop")
        )

    return _f(groups.data.system.indices, groups.data.motif.indices)


class IsBoltzmannSystems(HasPotentialEnergy, HasTemperature, Protocol): ...


class IsBoltzmannState(Protocol):
    """State protocol for Boltzmann acceptance probability computation."""

    @property
    def systems(self) -> Table[SystemId, IsBoltzmannSystems]: ...


class IsFugacitySystems(HasLogActivity, HasUnitCell, Protocol): ...


class IsFugacityState(Protocol):
    """State protocol for fugacity acceptance probability computation."""

    @property
    def groups(self) -> Table[GroupId, HasMotifAndSystemIndex]: ...
    @property
    def systems(self) -> Table[SystemId, IsFugacitySystems]: ...


class MuVTSystems(
    HasLogActivity, HasUnitCell, HasTemperature, HasPotentialEnergy, Protocol
): ...


class IsMuVTState(Protocol):
    """State protocol for μVT (grand canonical) acceptance probability computation."""

    @property
    def groups(self) -> Table[GroupId, HasMotifAndSystemIndex]: ...
    @property
    def systems(self) -> Table[SystemId, MuVTSystems]: ...


def make_boltzmann_probability_ratio[State, Move: Patch](
    state: Lens[State, IsBoltzmannState], potential: Potential[State, Any, Any, Move]
) -> tuple[
    CachedPotential[State, EmptyType, EmptyType, Move],
    BoltzmannLogProbabilityRatio[State, Move],
]:
    """Build a Boltzmann acceptance criterion for NVT (canonical) Monte Carlo.

    Wraps ``potential`` in a
    [CachedPotential][kups.core.potential.CachedPotential] so that the previous
    energy is read from ``state.previous_energy`` and the new energy is evaluated
    on demand, then assembles a
    [BoltzmannLogProbabilityRatio][kups.mcmc.probability.BoltzmannLogProbabilityRatio].

    Args:
        state: Lens into the sub-state satisfying
            [IsBoltzmannState][kups.mcmc.probability.IsBoltzmannState] (needs
            ``systems.temperature`` and ``previous_energy``).
        potential: Potential to evaluate on the proposed configuration.

    Returns:
        [BoltzmannLogProbabilityRatio][kups.mcmc.probability.BoltzmannLogProbabilityRatio]
        ready to be called with ``(state, patch)``.
    """
    potential = CachedPotential(
        MappedPotential(potential, EMPTY_LENS, EMPTY_LENS),
        state.focus(
            lambda x: PotentialOut(
                x.systems.map_data(lambda x: x.potential_energy), EMPTY, EMPTY
            )
        ),
        state.focus(lambda x: PotentialOut(Index.new(x.systems.keys), EMPTY, EMPTY)),  # type: ignore
    )
    return potential, BoltzmannLogProbabilityRatio(
        state.focus(lambda x: x.systems.map_data(lambda s: s.temperature)),
        potential=potential,
    )


def make_fugacity_probability_ratio[State, Move: Patch](
    state: Lens[State, IsFugacityState],
) -> LogFugacityRatio[State, Any]:
    """Build the fugacity (chemical potential) acceptance criterion for GCMC.

    Constructs a [LogFugacityRatio][kups.mcmc.probability.LogFugacityRatio] that
    computes the grand-canonical acceptance factor arising from particle-number
    changes. Current and proposed motif counts are derived from
    ``state.groups.data.species`` via [motif_counts][kups.mcmc.probability.motif_counts].

    Args:
        state: Lens into the sub-state satisfying
            [IsFugacityState][kups.mcmc.probability.IsFugacityState] (needs
            ``groups``, ``systems.log_activity``, and ``systems.unitcell.volume``).

    Returns:
        [LogFugacityRatio][kups.mcmc.probability.LogFugacityRatio]
        ready to be called with ``(state, patch)``.
    """

    def counts_probe(og_state: State, patch: Move) -> Table[SystemId, Array]:
        systems = state(og_state).systems
        new_state = patch(
            og_state, systems.set_data(jnp.ones(len(systems), dtype=bool))
        )
        return systems.set_data(motif_counts(state(new_state).groups))

    return LogFugacityRatio(
        state.focus(lambda x: x.systems.map_data(lambda s: s.log_activity)),
        state.focus(lambda x: x.systems.map_data(lambda s: s.unitcell.volume)),
        pipe(state, lambda x: x.systems.set_data(motif_counts(x.groups))),
        counts_probe,
    )


def make_muvt_probability_ratio[State, Move: Patch](
    state: Lens[State, IsMuVTState], potential: Potential[State, Any, Any, Move]
) -> tuple[
    CachedPotential[State, EmptyType, EmptyType, Move],
    MuVTLogProbabilityRatio[State, Move],
]:
    """Build the combined acceptance criterion for grand canonical (μVT) Monte Carlo.

    Composes
    [make_fugacity_probability_ratio][kups.mcmc.probability.make_fugacity_probability_ratio]
    and
    [make_boltzmann_probability_ratio][kups.mcmc.probability.make_boltzmann_probability_ratio]
    into a single
    [MuVTLogProbabilityRatio][kups.mcmc.probability.MuVTLogProbabilityRatio]
    whose log-ratio is:

    $$
    \\log p = \\log p_{\\text{fugacity}} + \\log p_{\\text{Boltzmann}}
    $$

    Args:
        state: Lens into the sub-state satisfying
            [IsMuVTState][kups.mcmc.probability.IsMuVTState] (needs ``groups``,
            ``systems.temperature``, ``systems.log_activity``,
            ``systems.unitcell.volume``, and ``previous_energy``).
        potential: Potential to evaluate on the proposed configuration.

    Returns:
        [MuVTLogProbabilityRatio][kups.mcmc.probability.MuVTLogProbabilityRatio]
        ready to be called with ``(state, patch)``.
    """
    potential, boltzmann_ratio = make_boltzmann_probability_ratio(state, potential)
    return potential, MuVTLogProbabilityRatio(
        make_fugacity_probability_ratio(state), boltzmann_ratio
    )
