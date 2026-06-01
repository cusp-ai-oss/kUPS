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

References:
    Widom, B. (1963). J. Chem. Phys., 39, 2808.
    Vlugt, T. J. H. et al. (2008). J. Chem. Theory Comput., 4, 1107.
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
