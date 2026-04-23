# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Widom ghost evaluation primitives and accumulators for flat-histogram Monte Carlo.

This module provides the building blocks for Widom's test-particle method and
the transition-matrix Monte Carlo (TMMC) variant of Witman, Mahynski & Smit
(2018). Ghost evaluations run the full MCMC proposal/patch/energy pipeline and
discard the resulting state change --- the physical configuration is unchanged,
but the log acceptance ratio is accumulated into running statistics.

Contents:

- [widom_test][kups.mcmc.widom.widom_test]: per-system $\ln\alpha$ for a ghost move
- [GhostProbe][kups.mcmc.widom.GhostProbe]: propagator that runs a ghost move
  and accumulates the result via a lens + update callback
- [TransitionStatistics][kups.mcmc.widom.TransitionStatistics]: TMMC C-matrix
  accumulator
- [EnergyMoments][kups.mcmc.widom.EnergyMoments] /
  [EnergyCumulants][kups.mcmc.widom.EnergyCumulants]: Pébay/Welford online
  moments 1--4 for Taylor expansion of $\ln Q_c(\beta)$
- [WidomStatistics][kups.mcmc.widom.WidomStatistics] /
  [WidomResult][kups.mcmc.widom.WidomResult]: accumulator and finalizer for
  plain Widom ($\mu^\mathrm{ex}$, $K_H$, Vlugt $q_\mathrm{st}$)

References:
    Widom, B. (1963). Some Topics in the Theory of Fluids. J. Chem. Phys., 39,
    2808--2812. DOI: 10.1063/1.1734110

    Witman, M., Mahynski, N. A. & Smit, B. (2018). Flat-Histogram Monte Carlo
    as an Efficient Tool To Evaluate Adsorption Processes Involving Rigid and
    Deformable Molecules. J. Chem. Theory Comput., 14, 6149--6158.
    DOI: 10.1021/acs.jctc.8b00534

    Pébay, P. (2008). Formulas for Robust, One-Pass Parallel Computation of
    Covariances and Arbitrary-Order Statistical Moments. Sandia SAND2008-6212.

    Vlugt, T. J. H. et al. (2008). Computing the Heat of Adsorption using
    Molecular Simulations. J. Chem. Theory Comput., 4, 1107--1118.
    DOI: 10.1021/ct700342k
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.lens import Lens
from kups.core.patch import Patch
from kups.core.propagator import (
    ChangesFn,
    LogProbabilityRatioFn,
    PatchFn,
    Propagator,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field, key_chain

type LogAcceptanceRatio = Array
r"""Log Metropolis acceptance ratio $\ln\alpha$ [dimensionless]."""

type Energy = Array
r"""Potential energy [energy]."""

type ParticleCount = Array
r"""Macrostate particle count $N$ [dimensionless, integer]."""

type Temperature = Array
r"""Thermodynamic temperature $T$ [K]."""

type Volume = Array
r"""Simulation-cell volume $V$ [length$^3$]."""


def widom_test[State, Changes, Move: Patch](
    key: Array,
    state: State,
    propose_fn: ChangesFn[State, Changes],
    patch_fn: PatchFn[State, Changes, Move],
    log_probability_ratio_fn: LogProbabilityRatioFn[State, Move],
) -> Table[SystemId, LogAcceptanceRatio]:
    r"""Evaluate per-system $\ln\alpha$ for a ghost move without modifying state.

    Runs the full MCMC proposal $\to$ patch $\to$ log-ratio pipeline and
    intentionally discards the resulting state patch. The physical state is
    untouched --- this is the Widom test-particle method applied as a reusable
    subroutine. The returned value is **raw** $\ln\alpha$, not clamped by
    $\min(1, \cdot)$; callers decide how to consume it:

    - TMMC: clamp with $\min(1, \exp\ln\alpha)$, accumulate into C-matrix
      (Witman 2018, eq 5--7).
    - Excess chemical potential: average $\exp\ln\alpha$, take $-k_BT \ln\langle\cdot\rangle$.
    - Henry coefficient: same average evaluated at $N = 0$.

    Args:
        key: JAX PRNG key.
        state: Current simulation state. Not modified.
        propose_fn: Move proposal (e.g. insertion or deletion).
        patch_fn: Converts proposal to a state patch.
        log_probability_ratio_fn: Evaluates the acceptance log-ratio against
            the proposed patch.

    Returns:
        Per-system log acceptance ratio as ``Table[SystemId, Array]``.
    """
    chain = key_chain(key)
    changes, move_lr = propose_fn(next(chain), state)
    patch = patch_fn(next(chain), state, changes)
    result = log_probability_ratio_fn(state, patch)
    # result.patch is intentionally discarded --- state is NOT modified.
    return move_lr + result.data


@dataclass
class TransitionStatistics:
    r"""TMMC collection-matrix (C-matrix) accumulator for $N \to N \pm 1$ moves.

    Each ghost evaluation contributes $\min(1, \exp\ln\alpha)$ to the
    corresponding row (Witman 2018, eq 5--7). Downstream, transition
    probabilities are recovered as

    $$P(N \to N+1) = \frac{\text{acceptance\_insertion}}
        {n_\text{trials,ins} + n_\text{trials,del}}$$

    All arrays have shape ``(n_systems,)``.

    Attributes:
        acceptance_insertion: $\sum \min(1, \exp\ln\alpha_\text{ins})$.
        acceptance_deletion: $\sum \min(1, \exp\ln\alpha_\text{del})$.
        n_trials_insertion: Number of ghost insertions evaluated.
        n_trials_deletion: Number of ghost deletions evaluated (incremented
            even when $N = 0$; the accepted fraction is zero there).
    """

    acceptance_insertion: Array
    acceptance_deletion: Array
    n_trials_insertion: Array
    n_trials_deletion: Array

    @staticmethod
    def zeros(n_systems: int) -> TransitionStatistics:
        """Create a zero-initialized accumulator for ``n_systems`` macrostates."""
        return TransitionStatistics(
            acceptance_insertion=jnp.zeros(n_systems),
            acceptance_deletion=jnp.zeros(n_systems),
            n_trials_insertion=jnp.zeros(n_systems, dtype=jnp.int32),
            n_trials_deletion=jnp.zeros(n_systems, dtype=jnp.int32),
        )

    def reset(self) -> TransitionStatistics:
        """Zero all fields while preserving shape/dtype."""
        return TransitionStatistics(
            acceptance_insertion=jnp.zeros_like(self.acceptance_insertion),
            acceptance_deletion=jnp.zeros_like(self.acceptance_deletion),
            n_trials_insertion=jnp.zeros_like(self.n_trials_insertion),
            n_trials_deletion=jnp.zeros_like(self.n_trials_deletion),
        )

    def update_insertion(self, ln_alpha: LogAcceptanceRatio) -> TransitionStatistics:
        r"""Accumulate a ghost-insertion $\ln\alpha$. Trial count is incremented unconditionally."""
        acceptance = jnp.minimum(1.0, jnp.exp(ln_alpha))
        return TransitionStatistics(
            acceptance_insertion=self.acceptance_insertion + acceptance,
            acceptance_deletion=self.acceptance_deletion,
            n_trials_insertion=self.n_trials_insertion + 1,
            n_trials_deletion=self.n_trials_deletion,
        )

    def update_deletion(
        self,
        ln_alpha: LogAcceptanceRatio,
        macrostate_n: ParticleCount,
    ) -> TransitionStatistics:
        r"""Accumulate a ghost-deletion $\ln\alpha$; zero contribution when $N = 0$.

        The trial count always increments — the fraction of accepted deletions
        at $N = 0$ is zero, but the denominator still counts the trial, so
        $P(0 \to 1)$ is not inflated.
        """
        acceptance = jnp.minimum(1.0, jnp.exp(ln_alpha))
        acceptance = jnp.where(macrostate_n > 0, acceptance, 0.0)
        return TransitionStatistics(
            acceptance_insertion=self.acceptance_insertion,
            acceptance_deletion=self.acceptance_deletion + acceptance,
            n_trials_insertion=self.n_trials_insertion,
            n_trials_deletion=self.n_trials_deletion + 1,
        )


@dataclass
class EnergyCumulants:
    r"""Finalized central moments of the potential energy distribution.

    These match the $\beta$-derivatives of the configurational partition function
    (Witman 2018, eq 10):

    $$\kappa_k = \partial^k \ln Q_c / \partial(-\beta)^k.$$

    Attributes:
        mean: $\kappa_1 = \langle E \rangle$ [energy].
        variance: $\kappa_2 = \langle (E - \langle E\rangle)^2 \rangle$ [energy$^2$].
        third: $\kappa_3 = -\langle (E - \langle E\rangle)^3 \rangle$ [energy$^3$].
        fourth: $\kappa_4 = \langle (E - \langle E\rangle)^4 \rangle - 3\,\mathrm{Var}^2$
            (excess kurtosis $\times$ variance$^2$) [energy$^4$].
    """

    mean: Energy
    variance: Array
    third: Array
    fourth: Array


@dataclass
class EnergyMoments:
    r"""Pébay one-pass accumulator for central moments 1--4 of per-system energy.

    Maintains the unnormalized central-moment sums

    $$M_k = \sum_{i=1}^{n} (x_i - \bar{x}_n)^k,$$

    updated via the single-sample specialisations of Pébay (2008) eqs 1.2, 1.5,
    1.6. Call :meth:`finalize` to convert to physical cumulants.

    Attributes:
        count: Number of samples accumulated.
        mean: Running sample mean $\bar{x}_n$ [energy].
        m2: Sum of squared deviations [energy$^2$].
        m3: Sum of cubed deviations [energy$^3$].
        m4: Sum of fourth-order deviations [energy$^4$].
    """

    count: Array
    mean: Energy
    m2: Array
    m3: Array
    m4: Array

    @staticmethod
    def zeros(n_systems: int) -> EnergyMoments:
        """Zero-initialize for ``n_systems`` macrostates."""
        return EnergyMoments(
            count=jnp.zeros(n_systems, dtype=jnp.int32),
            mean=jnp.zeros(n_systems),
            m2=jnp.zeros(n_systems),
            m3=jnp.zeros(n_systems),
            m4=jnp.zeros(n_systems),
        )

    def reset(self) -> EnergyMoments:
        """Zero all fields while preserving shape/dtype."""
        return EnergyMoments(
            count=jnp.zeros_like(self.count),
            mean=jnp.zeros_like(self.mean),
            m2=jnp.zeros_like(self.m2),
            m3=jnp.zeros_like(self.m3),
            m4=jnp.zeros_like(self.m4),
        )

    def update(self, energy: Energy) -> EnergyMoments:
        r"""Incorporate one per-system energy sample (Pébay single-sample update).

        Uses the standard Welford-style recurrences for higher moments — cf.
        Pébay (2008) and the parallel-algorithm article on Wikipedia. The
        coefficients $(n-1)(n-2)$ and $(n-1)(n^2-3n+3)$ ensure $M_3 = M_4 = 0$
        at $n = 1$ and $M_3 = 0$ for any symmetric pair at $n = 2$.
        """
        n = self.count + 1
        nf = n.astype(energy.dtype)
        n_prev = nf - 1.0

        delta = energy - self.mean
        delta_n = delta / nf
        delta_n_sq = delta_n * delta_n
        # term1 = δ² (n-1)/n — the single-sample "pair" contribution.
        term1 = delta * delta_n * n_prev

        new_mean = self.mean + delta_n
        # Update m4 and m3 before m2 — they read old m2 / m3 values.
        new_m4 = (
            self.m4
            + term1 * delta_n_sq * (nf * nf - 3.0 * nf + 3.0)
            + 6.0 * delta_n_sq * self.m2
            - 4.0 * delta_n * self.m3
        )
        new_m3 = self.m3 + term1 * delta_n * (nf - 2.0) - 3.0 * delta_n * self.m2
        new_m2 = self.m2 + term1

        return EnergyMoments(
            count=n,
            mean=new_mean,
            m2=new_m2,
            m3=new_m3,
            m4=new_m4,
        )

    def finalize(self) -> EnergyCumulants:
        r"""Normalize $M_k$ by $n$ and map to cumulants for Taylor expansion."""
        nf = self.count.astype(jnp.float64)
        variance = self.m2 / nf
        third_central = self.m3 / nf
        fourth_central = self.m4 / nf
        return EnergyCumulants(
            mean=self.mean,
            variance=variance,
            third=-third_central,
            fourth=fourth_central - 3.0 * variance**2,
        )


@dataclass
class WidomStatistics:
    r"""Online accumulator for plain Widom insertion sums.

    Collects the Boltzmann-weighted running sums required to compute
    $\mu^\mathrm{ex}$, $K_H$, and Vlugt's heat of adsorption.

    Attributes:
        sum_boltzmann: $\sum \exp(-\beta \Delta U)$ [dimensionless].
        sum_energy_boltzmann: $\sum U \exp(-\beta \Delta U)$ [energy].
        sum_energy: $\sum U$ [energy].
        n_samples: Number of evaluations accumulated.
    """

    sum_boltzmann: Array
    sum_energy_boltzmann: Array
    sum_energy: Energy
    n_samples: Array

    @staticmethod
    def zeros(n_systems: int) -> WidomStatistics:
        """Zero-initialize."""
        return WidomStatistics(
            sum_boltzmann=jnp.zeros(n_systems),
            sum_energy_boltzmann=jnp.zeros(n_systems),
            sum_energy=jnp.zeros(n_systems),
            n_samples=jnp.zeros(n_systems, dtype=jnp.int32),
        )

    def reset(self) -> WidomStatistics:
        """Zero all fields while preserving shape/dtype."""
        return WidomStatistics(
            sum_boltzmann=jnp.zeros_like(self.sum_boltzmann),
            sum_energy_boltzmann=jnp.zeros_like(self.sum_energy_boltzmann),
            sum_energy=jnp.zeros_like(self.sum_energy),
            n_samples=jnp.zeros_like(self.n_samples),
        )

    def update(self, ln_alpha: LogAcceptanceRatio, energy: Energy) -> WidomStatistics:
        r"""Accumulate $W = \exp\ln\alpha$ and $U \cdot W$ from one ghost insertion."""
        boltzmann = jnp.exp(ln_alpha)
        return WidomStatistics(
            sum_boltzmann=self.sum_boltzmann + boltzmann,
            sum_energy_boltzmann=self.sum_energy_boltzmann + energy * boltzmann,
            sum_energy=self.sum_energy + energy,
            n_samples=self.n_samples + 1,
        )


@dataclass
class WidomResult:
    r"""Finalized Widom chemical potential / Henry / heat of adsorption.

    Attributes:
        excess_chemical_potential:
            $\mu^\mathrm{ex} = -k_BT \ln\langle W \rangle$ [energy].
        henry_coefficient:
            $K_H = V\langle W\rangle / (k_BT)$ [length$^3$ / energy].
        heat_of_adsorption: Vlugt fluctuation formula
            $q_\mathrm{st} = k_BT - (\langle U W\rangle - \langle U\rangle\langle W\rangle) / \langle W\rangle$
            [energy].
    """

    excess_chemical_potential: Energy
    henry_coefficient: Array
    heat_of_adsorption: Energy


def finalize_widom(
    stats: WidomStatistics,
    temperature: Temperature,
    volume: Volume,
) -> WidomResult:
    r"""Convert accumulated Widom sums into $\mu^\mathrm{ex}$, $K_H$, and $q_\mathrm{st}$.

    Args:
        stats: Accumulated statistics.
        temperature: Per-system $T$ [K].
        volume: Per-system $V$ [length$^3$].

    Returns:
        The finalized :class:`WidomResult`.
    """
    nf = stats.n_samples.astype(jnp.float64)
    mean_w = stats.sum_boltzmann / nf
    mean_u_w = stats.sum_energy_boltzmann / nf
    mean_u = stats.sum_energy / nf
    kT = temperature * BOLTZMANN_CONSTANT
    return WidomResult(
        excess_chemical_potential=-kT * jnp.log(mean_w),
        henry_coefficient=volume * mean_w / kT,
        heat_of_adsorption=kT - (mean_u_w - mean_u * mean_w) / mean_w,
    )


@runtime_checkable
class HasMacrostateN(Protocol):
    """State that exposes a per-system macrostate particle count."""

    @property
    def macrostate_n(self) -> Array: ...


@runtime_checkable
class HasTransitionStatistics(Protocol):
    """State that carries a TMMC C-matrix accumulator."""

    @property
    def transition_statistics(self) -> Table[SystemId, TransitionStatistics]: ...


@runtime_checkable
class HasEnergyMoments(Protocol):
    """State that carries an energy-moments accumulator."""

    @property
    def energy_moments(self) -> Table[SystemId, EnergyMoments]: ...


@runtime_checkable
class HasWidomStatistics(Protocol):
    """State that carries a plain-Widom accumulator."""

    @property
    def widom_statistics(self) -> Table[SystemId, WidomStatistics]: ...


@dataclass
class GhostProbe[State, Changes, Move: Patch, Stat](Propagator[State]):
    r"""Propagator that runs one ghost move and updates a statistic via a lens.

    A single abstract propagator shared by TMMC (insertion probe, deletion probe)
    and plain Widom. The configuration is:

    - ``propose_fn`` / ``patch_fn`` / ``log_probability_ratio_fn``: the standard
      MCMC trio. Exactly the same objects used by
      :class:`~kups.core.propagator.MCMCPropagator`; here the resulting patch is
      discarded.
    - ``stat_lens``: where in ``state`` the accumulator lives.
    - ``update_fn``: how the accumulator integrates the new $\ln\alpha$. Has
      signature ``(state, stat, ln_alpha) -> stat`` so the callback can read
      auxiliary state (e.g. macrostate $N$ for deletion, current energy for
      Widom).

    Compose multiple ``GhostProbe`` instances via
    :class:`~kups.core.propagator.SequentialPropagator` or
    :func:`~kups.core.propagator.compose_propagators` to build higher-level
    schemes.
    """

    propose_fn: ChangesFn[State, Changes] = field(static=True)
    patch_fn: PatchFn[State, Changes, Move] = field(static=True)
    log_probability_ratio_fn: LogProbabilityRatioFn[State, Move] = field(static=True)
    stat_lens: Lens[State, Stat] = field(static=True)
    update_fn: Callable[[State, Stat, Array], Stat] = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        ln_alpha = widom_test(
            key,
            state,
            self.propose_fn,
            self.patch_fn,
            self.log_probability_ratio_fn,
        )
        current = self.stat_lens.get(state)
        updated = self.update_fn(state, current, ln_alpha.data)
        return self.stat_lens.set(state, updated)
