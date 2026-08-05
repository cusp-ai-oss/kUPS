# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Widom test-particle method.

A ghost move runs the full propose/patch/log-ratio pipeline and discards the
resulting state patch; the log acceptance ratio is accumulated into running
statistics.

Contents:

- [widom_test][kups.mcmc.widom.widom_test]: per-system $\ln\alpha$ for a ghost move
- [GhostProbe][kups.mcmc.widom.GhostProbe]: propagator wrapper accumulating
  the ratio via a lens + update callback
- [WidomStatistics][kups.mcmc.widom.WidomStatistics]: running-sum accumulator
  reduced to $\mu^\mathrm{ex}$, $K_H$, $q_\mathrm{st}$ by the post-hoc
  analyzer.
- [TransitionStatistics][kups.mcmc.widom.TransitionStatistics]: TMMC
  collection-matrix (C-matrix) accumulator for flat-histogram runs
  (Witman 2018).
- [EnergyMoments][kups.mcmc.widom.EnergyMoments] /
  [EnergyCumulants][kups.mcmc.widom.EnergyCumulants]: Pébay/Welford online
  central moments 1--4 for Taylor expansion of $\ln Q_c(\beta)$.

References:
    Widom, B. (1963). J. Chem. Phys., 39, 2808.
    Vlugt, T. J. H. et al. (2008). J. Chem. Theory Comput., 4, 1107.
    Witman, M., Mahynski, N. A. & Smit, B. (2018). J. Chem. Theory Comput.,
    14, 6149--6158. DOI: 10.1021/acs.jctc.8b00534
    Pébay, P. (2008). Formulas for Robust, One-Pass Parallel Computation of
    Covariances and Arbitrary-Order Statistical Moments. Sandia SAND2008-6212.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array

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


def widom_test[State, Changes, Move: Patch[Any]](
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
class WidomStatistics:
    r"""Online accumulator for plain Widom insertion sums.

    Attributes:
        sum_boltzmann: $\sum \exp(-\beta \Delta U) = \sum W$ [dimensionless].
        sum_delta_u_boltzmann:
            $\sum \Delta U \cdot \exp(-\beta \Delta U)$ [energy], with
            $\Delta U$ the ghost insertion (host-guest) energy, not the
            cell's total potential energy.
        n_samples: Number of evaluations accumulated.
    """

    sum_boltzmann: Array
    sum_delta_u_boltzmann: Array
    n_samples: Array

    @staticmethod
    def zeros(n_systems: int) -> WidomStatistics:
        """Zero-initialize."""
        return WidomStatistics(
            sum_boltzmann=jnp.zeros(n_systems),
            sum_delta_u_boltzmann=jnp.zeros(n_systems),
            n_samples=jnp.zeros(n_systems, dtype=jnp.int32),
        )

    def reset(self) -> WidomStatistics:
        """Zero all fields."""
        return self.zeros(int(self.n_samples.shape[0]))

    def update(self, ln_alpha: LogAcceptanceRatio, delta_u: Energy) -> WidomStatistics:
        r"""Accumulate one ghost insertion.

        Args:
            ln_alpha: Per-system log Metropolis ratio.
            delta_u: Per-system ghost insertion energy. With a zero-move-log
                insertion proposal and a bare Boltzmann log-ratio,
                $\Delta U = -k_BT \ln\alpha$ exactly.
        """
        boltzmann = jnp.exp(ln_alpha)
        return WidomStatistics(
            sum_boltzmann=self.sum_boltzmann + boltzmann,
            sum_delta_u_boltzmann=self.sum_delta_u_boltzmann + delta_u * boltzmann,
            n_samples=self.n_samples + 1,
        )


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
        """Zero all fields."""
        return self.zeros(int(self.n_trials_insertion.shape[0]))

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
        """Zero all fields."""
        return self.zeros(int(self.count.shape[0]))

    def update(self, energy: Energy) -> EnergyMoments:
        r"""Incorporate one per-system energy sample (Pébay single-sample update).

        Uses the standard Welford-style recurrences for higher moments — cf.
        Pébay (2008). The coefficients $(n-1)(n-2)$ and $(n-1)(n^2-3n+3)$
        ensure $M_3 = M_4 = 0$ at $n = 1$ and $M_3 = 0$ for any symmetric pair
        at $n = 2$.
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
class GhostProbe[State, Changes, Move: Patch[Any], Stat](Propagator[State]):
    r"""Propagator running one ghost move and updating a lens-accessed statistic.

    Attributes:
        propose_fn / patch_fn / log_probability_ratio_fn: standard MCMC trio
            (:class:`~kups.core.propagator.MCMCPropagator` interface); the
            resulting patch is discarded.
        stat_lens: where in ``state`` the accumulator lives.
        update_fn: ``(state, stat, ln_alpha) -> stat``.
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
