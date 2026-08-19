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
  central moments of arbitrary order (default 4) for Taylor expansion of
  $\ln Q_c(\beta)$.

References:
    Widom, B. (1963). J. Chem. Phys., 39, 2808.
    Vlugt, T. J. H. et al. (2008). J. Chem. Theory Comput., 4, 1107.
    Witman, M., Mahynski, N. A. & Smit, B. (2018). J. Chem. Theory Comput.,
    14, 6149--6158. DOI: 10.1021/acs.jctc.8b00534
    Pébay, P. (2008). Formulas for Robust, One-Pass Parallel Computation of
    Covariances and Arbitrary-Order Statistical Moments. Sandia SAND2008-6212.
"""

from __future__ import annotations

from math import comb
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
        """Create a zero-initialized accumulator.

        Args:
            n_systems: Number of systems accumulated in parallel.

        Returns:
            Accumulator with all sums and counts at zero, shape ``(n_systems,)``.
        """
        return WidomStatistics(
            sum_boltzmann=jnp.zeros(n_systems),
            sum_delta_u_boltzmann=jnp.zeros(n_systems),
            n_samples=jnp.zeros(n_systems, dtype=int),
        )

    def reset(self) -> WidomStatistics:
        """Zero all fields.

        Returns:
            Fresh accumulator of the same shape with all fields at zero.
        """
        return self.zeros(int(self.n_samples.shape[0]))

    def update(self, ln_alpha: LogAcceptanceRatio, delta_u: Energy) -> WidomStatistics:
        r"""Accumulate one ghost insertion.

        Args:
            ln_alpha: Per-system log Metropolis ratio.
            delta_u: Per-system ghost insertion energy. With a zero-move-log
                insertion proposal and a bare Boltzmann log-ratio,
                $\Delta U = -k_BT \ln\alpha$ exactly.

        Returns:
            Accumulator with the sample folded into the running sums.
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

    Here $\alpha$ is the Metropolis acceptance ratio of the ghost move (the
    exponential of the log-ratio produced by the proposal pipeline). Each
    ghost evaluation contributes the acceptance probability $\min(1, \alpha)$
    to the corresponding row (Witman 2018, eq 5--7). Downstream, transition
    probabilities are recovered as

    $$P(N \to N+1) = \frac{\text{acceptance\_insertion}}
        {n_\text{trials,ins} + n_\text{trials,del}}$$

    All arrays have shape ``(n_systems,)``.

    Attributes:
        acceptance_insertion: $\sum \min(1, \alpha_\text{ins})$.
        acceptance_deletion: $\sum \min(1, \alpha_\text{del})$.
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
        """Create a zero-initialized accumulator for ``n_systems`` macrostates.

        Args:
            n_systems: Number of macrostates accumulated in parallel.

        Returns:
            Accumulator with all sums and counts at zero, shape ``(n_systems,)``.
        """
        return TransitionStatistics(
            acceptance_insertion=jnp.zeros(n_systems),
            acceptance_deletion=jnp.zeros(n_systems),
            n_trials_insertion=jnp.zeros(n_systems, dtype=int),
            n_trials_deletion=jnp.zeros(n_systems, dtype=int),
        )

    def reset(self) -> TransitionStatistics:
        """Zero all fields.

        Returns:
            Fresh accumulator of the same shape with all fields at zero.
        """
        return self.zeros(int(self.n_trials_insertion.shape[0]))

    def update_insertion(self, ln_alpha: LogAcceptanceRatio) -> TransitionStatistics:
        r"""Accumulate a ghost insertion. Trial count is incremented unconditionally.

        Args:
            ln_alpha: Per-system log Metropolis ratio $\ln\alpha$ of the
                ghost insertion.

        Returns:
            Accumulator with $\min(1, \alpha)$ added to the insertion row.
        """
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
        r"""Accumulate a ghost deletion; zero contribution when $N = 0$.

        The trial count always increments — the fraction of accepted deletions
        at $N = 0$ is zero, but the denominator still counts the trial, so
        $P(0 \to 1)$ is not inflated.

        Args:
            ln_alpha: Per-system log Metropolis ratio $\ln\alpha$ of the
                ghost deletion.
            macrostate_n: Per-system particle count $N$; systems at $N = 0$
                contribute nothing to the acceptance sum.

        Returns:
            Accumulator with $\min(1, \alpha)$ added to the deletion row.
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
    r"""Finalized cumulants of the potential energy distribution.

    Stores the standard cumulants $\kappa_2, \ldots, \kappa_P$ of the energy
    ($\kappa_2 = \mathrm{Var}$, $\kappa_3 = \mu_3$,
    $\kappa_4 = \mu_4 - 3\mu_2^2$, ... with $\mu_k$ the central moments),
    plus $\kappa_1 = \langle E \rangle$ as ``mean``. They determine the
    $\beta$-derivatives of the configurational partition function
    (Witman 2018, eq 10) via

    $$\partial^k \ln Q_c / \partial\beta^k = (-1)^k \kappa_k,$$

    which is what the flat-histogram Taylor extrapolation consumes.

    Attributes:
        mean: $\kappa_1 = \langle E \rangle$, shape ``(n_systems,)`` [energy].
        cumulants: $\kappa_2, \ldots, \kappa_P$ stacked along the trailing
            axis (leading axes stay per-system, so tables/vmap batch over
            them), shape ``(n_systems, max_order - 1)``; entry ``[..., k - 2]``
            holds $\kappa_k$ [energy$^k$].
    """

    mean: Energy
    cumulants: Array

    @property
    def max_order(self) -> int:
        """Highest cumulant order $P$ stored."""
        return self.cumulants.shape[-1] + 1

    @property
    def variance(self) -> Array:
        r"""$\kappa_2 = \mathrm{Var}(E)$, shape ``(n_systems,)``."""
        return self.cumulants[..., 0]


@dataclass
class EnergyMoments:
    r"""Pébay one-pass accumulator for central moments of per-system energy.

    Maintains the unnormalized central-moment sums

    $$M_p = \sum_{i=1}^{n} (x_i - \bar{x}_n)^p, \qquad p = 2, \ldots, P,$$

    for an arbitrary maximum order $P$ (default 4, enough for the third-order
    Taylor expansion of $\ln Q_c(\beta)$), updated via the single-sample
    recurrence of Pébay (2008). Call :meth:`finalize` to convert to cumulants.

    Attributes:
        count: Number of samples accumulated, shape ``(n_systems,)``.
        mean: Running sample mean $\bar{x}_n$, shape ``(n_systems,)`` [energy].
        central_sums: $M_2, \ldots, M_P$ stacked along the trailing axis
            (leading axes stay per-system, so tables/vmap batch over them),
            shape ``(n_systems, max_order - 1)``; entry ``[..., p - 2]`` holds
            $M_p$ [energy$^p$].
    """

    count: Array
    mean: Energy
    central_sums: Array

    @property
    def max_order(self) -> int:
        """Highest moment order $P$ accumulated."""
        return self.central_sums.shape[-1] + 1

    @staticmethod
    def zeros(n_systems: int, max_order: int = 4) -> EnergyMoments:
        """Create a zero-initialized accumulator for ``n_systems`` macrostates.

        Args:
            n_systems: Number of macrostates accumulated in parallel.
            max_order: Highest central-moment order to track (at least 2).

        Returns:
            Accumulator with all sums and counts at zero.
        """
        assert max_order >= 2, "at least mean and variance must be tracked"
        return EnergyMoments(
            count=jnp.zeros(n_systems, dtype=int),
            mean=jnp.zeros(n_systems),
            central_sums=jnp.zeros((n_systems, max_order - 1)),
        )

    def reset(self) -> EnergyMoments:
        """Zero all fields.

        Returns:
            Fresh accumulator of the same shape and order with all fields zero.
        """
        return self.zeros(int(self.count.shape[0]), max_order=self.max_order)

    def update(self, energy: Energy) -> EnergyMoments:
        r"""Incorporate one per-system energy sample (Pébay single-sample update).

        With $\delta = x - \bar{x}_{n-1}$ and $\delta_n = \delta / n$, the
        general recurrence for a single new sample is

        $$M_p \leftarrow M_p
            + \sum_{k=1}^{p-2} \binom{p}{k} (-\delta_n)^k M_{p-k}
            + \delta_n^p (n-1) \left[(n-1)^{p-1} + (-1)^p\right],$$

        the one-value specialisation of Pébay (2008), eq 2.9. For $p \le 4$
        this reduces to the familiar Welford-style updates; each order reads
        only the previous orders' old values.

        Args:
            energy: Per-system energy sample, shape ``(n_systems,)``.

        Returns:
            Accumulator with the sample folded into all tracked moments.
        """
        n = self.count + 1
        nf = n.astype(energy.dtype)
        n_prev = nf - 1.0
        delta = energy - self.mean
        delta_n = delta / nf

        old = {p: self.central_sums[..., p - 2] for p in range(2, self.max_order + 1)}
        new_sums = [
            old[p]
            # Cross terms re-centre the old sums onto the new mean; M_1 = 0
            # and the M_0 = n - 1 term is folded into the tail below.
            + sum(comb(p, k) * (-delta_n) ** k * old[p - k] for k in range(1, p - 1))
            + delta_n**p * n_prev * (n_prev ** (p - 1) + (-1.0) ** p)
            for p in range(2, self.max_order + 1)
        ]
        return EnergyMoments(
            count=n,
            mean=self.mean + delta_n,
            central_sums=jnp.stack(new_sums, axis=-1),
        )

    def finalize(self) -> EnergyCumulants:
        r"""Normalize $M_p$ by $n$ and map central moments to cumulants.

        Uses the standard recursion (valid since $\mu_1 = 0$)

        $$\kappa_p = \mu_p - \sum_{m=2}^{p-2} \binom{p-1}{m-1}
            \kappa_m \mu_{p-m},$$

        which yields $\kappa_2 = \mu_2$, $\kappa_3 = \mu_3$,
        $\kappa_4 = \mu_4 - 3\mu_2^2$, ...

        Returns:
            Cumulants $\kappa_1, \ldots, \kappa_P$ of the accumulated samples.
        """
        nf = self.count.astype(float)
        mu = {
            p: self.central_sums[..., p - 2] / nf for p in range(2, self.max_order + 1)
        }
        kappa: dict[int, Array] = {}
        for p in range(2, self.max_order + 1):
            kappa[p] = mu[p] - sum(
                comb(p - 1, m - 1) * kappa[m] * mu[p - m] for m in range(2, p - 1)
            )
        return EnergyCumulants(
            mean=self.mean,
            cumulants=jnp.stack(
                [kappa[p] for p in range(2, self.max_order + 1)], axis=-1
            ),
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
        r"""Run one ghost move and fold $\ln\alpha$ into the accumulator.

        Args:
            key: JAX PRNG key.
            state: Current simulation state; only the lens-accessed statistic
                changes.

        Returns:
            State with the updated accumulator written back through the lens.
        """
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
