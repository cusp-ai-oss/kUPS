# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Flat-histogram (TMMC) post-processing (Witman, Mahynski & Smit 2018).

Turns one NVT+W simulation at a single temperature into isotherms, isosteric
heats and working capacities:

1. C-matrix $\to$ $P(N \to N \pm 1)$ (eq 7).
2. $P(N \to N \pm 1)$ $\to$ $\ln Q_c(N)$ by a prefix sum (eq 8).
3. Taylor-extrapolate $\ln Q_c$ in $\beta$ from energy cumulants (eq 9).
4. Reweight to the grand-canonical $\Pi(N)$ with a Peng--Robinson fugacity
   (eq 3).

The macrostate $N$ is always the trailing axis; leading axes are per-system.
Only a single adsorbate species is supported.

References:
    Witman, M., Mahynski, N. A. & Smit, B. (2018). J. Chem. Theory Comput.,
    14, 6149--6158. DOI: 10.1021/acs.jctc.8b00534
"""

from __future__ import annotations

from math import factorial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT, KELVIN, PASCAL
from kups.core.data import Table
from kups.core.typing import MotifId, SystemId
from kups.core.utils.jax import dataclass, field
from kups.mcmc.fugacity import peng_robinson_log_fugacity
from kups.mcmc.widom import (
    DeletionProbability,
    EnergyCumulants,
    InsertionProbability,
    TransitionStatistics,
)

type LogPartition = Array
r"""$\ln Q_c(N, V, \beta)$, shape ``(..., n_max+1)``."""

type MacrostateDistribution = Array
r"""Grand-canonical distribution $\Pi(N; \mu, V, \beta)$, shape ``(..., n_max+1)``."""

type Loading = Array
r"""Expected loading $\langle N \rangle$, shape ``(...)`` or ``(..., n_pressures)``."""

type IsostericHeat = Array
r"""Isosteric heat $q_\mathrm{st}$ [energy], shape ``(...)`` or ``(..., n_pressures)``."""


@dataclass
class AdsorbateEOS:
    r"""Peng-Robinson parameters per adsorbate species, in SI units.

    Carried as a ``Table[MotifId, AdsorbateEOS]``.

    Attributes:
        critical_pressure: $P_c$ [Pa].
        critical_temperature: $T_c$ [K].
        acentric_factor: $\omega$ [-].
    """

    critical_pressure: Array
    critical_temperature: Array
    acentric_factor: Array

    @staticmethod
    def single_species(
        critical_pressure: Array | float,
        critical_temperature: Array | float,
        acentric_factor: Array | float,
    ) -> Table[MotifId, AdsorbateEOS]:
        r"""Build a one-row EOS table from SI scalars."""
        return Table.arange(
            AdsorbateEOS(
                critical_pressure=jnp.atleast_1d(jnp.asarray(critical_pressure)),
                critical_temperature=jnp.atleast_1d(jnp.asarray(critical_temperature)),
                acentric_factor=jnp.atleast_1d(jnp.asarray(acentric_factor)),
            ),
            label=MotifId,
        )


def _with_macrostate_axis(per_system: Array) -> Array:
    """Append a trailing axis so a per-system value broadcasts over a ladder."""
    value = jnp.asarray(per_system)
    return value if value.ndim == 0 else value[..., None]


def _single_species(adsorbates: Table[MotifId, AdsorbateEOS]) -> AdsorbateEOS:
    """Unwrap the only row of a single-species EOS table."""
    assert len(adsorbates) == 1, (
        f"flat-histogram post-processing supports a single adsorbate species, "
        f"got {len(adsorbates)}"
    )
    return adsorbates.data


def reconstruct_log_partition(
    insertion_probability: InsertionProbability,
    deletion_probability: DeletionProbability,
    beta: Array,
    log_fugacity: Array,
) -> LogPartition:
    r"""$\ln Q_c(N)$ from transition probabilities (Witman 2018 eq 8).

    $$\ln \frac{Q_c(N+1)}{Q_c(N)}
        = \ln\frac{P(N \to N+1)}{P(N+1 \to N)} - \ln(\beta f)$$

    Args:
        insertion_probability: $P(N \to N+1)$, shape ``(..., n_max+1)``.
        deletion_probability: $P(N \to N-1)$, shape ``(..., n_max+1)``.
        beta: Simulation inverse temperature, per system.
        log_fugacity: Simulation $\ln f$, per system.

    Returns:
        $\ln Q_c$, shape ``(..., n_max+1)``, zero at $N = 0$.
    """
    log_ratio = jnp.log(insertion_probability[..., :-1]) - jnp.log(
        deletion_probability[..., 1:]
    )
    delta_log_qc = log_ratio - _with_macrostate_axis(log_fugacity + jnp.log(beta))
    anchor = jnp.zeros_like(delta_log_qc[..., :1])
    return jnp.concatenate([anchor, jnp.cumsum(delta_log_qc, axis=-1)], axis=-1)


def extrapolate_log_partition(
    log_partition_sim: LogPartition,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    beta_target: Array,
    order: int = 3,
) -> LogPartition:
    r"""Taylor-extrapolate $\ln Q_c$ in $\beta$ (Witman 2018 eq 9).

    $$\ln Q_c(\beta') \approx \ln Q_c(\beta)
        + \sum_{k=1}^{K} \frac{(\beta - \beta')^k}{k!}\,\kappa_k$$

    Args:
        log_partition_sim: $\ln Q_c$ at ``beta_sim``, shape ``(..., n_max+1)``.
        cumulants: Per-macrostate energy cumulants.
        beta_sim: Simulation inverse temperature, per system.
        beta_target: Target inverse temperature.
        order: Taylor order $K \le$ ``cumulants.max_order``.

    Returns:
        $\ln Q_c$ at ``beta_target``, shape ``(..., n_max+1)``.

    Raises:
        ValueError: If ``order`` is out of range.
    """
    if not 1 <= order <= cumulants.max_order:
        raise ValueError(f"order must be in 1..{cumulants.max_order}, got {order}")

    db = _with_macrostate_axis(beta_target - beta_sim)
    result = log_partition_sim - cumulants.mean * db
    for k in range(2, order + 1):
        result = result + cumulants.cumulants[..., k - 2] * (-db) ** k / factorial(k)
    return result


def macrostate_distribution(
    log_partition: LogPartition,
    beta: Array,
    log_fugacity: Array,
) -> MacrostateDistribution:
    r"""Grand-canonical $\Pi(N) \propto (\beta f)^N Q_c(N)$ (Witman 2018 eq 3).

    Args:
        log_partition: $\ln Q_c$, shape ``(..., n_max+1)``.
        beta: Inverse temperature, per system.
        log_fugacity: Reservoir $\ln f$, per system.

    Returns:
        $\Pi(N)$, shape ``(..., n_max+1)``, normalised over $N$.
    """
    n_values = jnp.arange(log_partition.shape[-1])
    log_activity = _with_macrostate_axis(log_fugacity + jnp.log(beta))
    return jax.nn.softmax(log_activity * n_values + log_partition, axis=-1)


def average_loading(distribution: MacrostateDistribution) -> Loading:
    r"""$\langle N \rangle = \sum_N N\,\Pi(N)$ over the trailing axis."""
    n_values = jnp.arange(distribution.shape[-1])
    return jnp.sum(n_values * distribution, axis=-1)


def _log_fugacity_at_pressures(
    pressures_pa: Array,
    temperature_k: Array,
    adsorbates: Table[MotifId, AdsorbateEOS],
) -> Array:
    r"""Peng-Robinson $\ln f$ of the single adsorbate at each pressure [Pa]."""
    eos = _single_species(adsorbates)
    result = peng_robinson_log_fugacity(
        pressures_pa * PASCAL,
        temperature_k * KELVIN,
        eos.critical_pressure * PASCAL,
        eos.critical_temperature * KELVIN,
        eos.acentric_factor,
    )
    return result.log_fugacity[..., 0]


def _isotherm(
    log_partition: LogPartition,
    beta: Array,
    pressures: Array,
    temperature: Array,
    adsorbates: Table[MotifId, AdsorbateEOS],
) -> Loading:
    r"""$\langle N \rangle$ per pressure on a trailing axis; see :meth:`TMMCSummary.isotherm`."""
    log_fugacities = _log_fugacity_at_pressures(pressures, temperature, adsorbates)

    def loading_at(log_fugacity: Array) -> Loading:
        return average_loading(
            macrostate_distribution(log_partition, beta, log_fugacity)
        )

    return jax.vmap(loading_at, out_axes=-1)(log_fugacities)


def _loading_from_temperature_log_pressure(
    temperature: Array,
    log_pressure_pa: Array,
    log_partition_sim: LogPartition,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    order: int,
    adsorbates: Table[MotifId, AdsorbateEOS],
) -> Loading:
    r"""Differentiable $\langle N \rangle(T, \ln P)$ at one state point."""
    beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
    log_qc = extrapolate_log_partition(
        log_partition_sim,
        cumulants,
        beta_sim,
        beta,
        order,
    )
    pressure = jnp.exp(log_pressure_pa)
    log_fugacity = _log_fugacity_at_pressures(
        jnp.atleast_1d(pressure),
        temperature,
        adsorbates,
    )[0]
    return average_loading(macrostate_distribution(log_qc, beta, log_fugacity))


def _isosteric_heat(
    log_partition_sim: LogPartition,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    pressures: Array,
    temperature: Array,
    adsorbates: Table[MotifId, AdsorbateEOS],
    order: int = 3,
) -> IsostericHeat:
    r"""$q_\mathrm{st}$ per pressure; see :meth:`TMMCSummary.isosteric_heat`.

    $$q_\mathrm{st} = k_B T^2
        \frac{(\partial \langle N\rangle / \partial T)_P}
             {(\partial \langle N\rangle / \partial \ln P)_T}$$

    ``jacfwd`` because the inputs are scalars but the loading is batched.
    """

    def loading_fn(t: Array, log_p: Array) -> Loading:
        return _loading_from_temperature_log_pressure(
            t,
            log_p,
            log_partition_sim,
            cumulants,
            beta_sim,
            order,
            adsorbates,
        )

    loading_jacobian = jax.jacfwd(loading_fn, argnums=(0, 1))

    def q_st_at(log_p: Array) -> IsostericHeat:
        dn_dt, dn_dlog_p = loading_jacobian(temperature, log_p)
        return BOLTZMANN_CONSTANT * temperature**2 * dn_dt / dn_dlog_p

    return jax.vmap(q_st_at, out_axes=-1)(jnp.log(pressures))


def _working_capacity(
    log_partition_sim: LogPartition,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    pressure_ads: Array,
    pressure_des: Array,
    temperature: Array,
    adsorbates: Table[MotifId, AdsorbateEOS],
    order: int = 3,
) -> Loading:
    r"""Pressure-swing $n_\mathrm{wc}$; see :meth:`TMMCSummary.working_capacity`."""

    def loading(pressure: Array) -> Loading:
        return _loading_from_temperature_log_pressure(
            temperature,
            jnp.log(pressure),
            log_partition_sim,
            cumulants,
            beta_sim,
            order,
            adsorbates,
        )

    return loading(pressure_ads) - loading(pressure_des)


@dataclass
class TMMCSummary:
    r"""Reconstructed TMMC ladders, one row per system.

    Attributes:
        log_partition_sim: $\ln Q_c$ at ``beta_sim``, rows ``(n_max+1,)``.
        cumulants: Per-macrostate energy cumulants.
        beta_sim: Simulation inverse temperature.
        adsorbates: Reservoir EOS; exactly one species.
        order: Taylor order of the $\beta$-extrapolation.
    """

    log_partition_sim: Table[SystemId, LogPartition]
    cumulants: Table[SystemId, EnergyCumulants]
    beta_sim: Table[SystemId, Array]
    adsorbates: Table[MotifId, AdsorbateEOS]
    order: int = field(static=True, default=3)

    @staticmethod
    def from_transition_statistics(
        statistics: Table[SystemId, TransitionStatistics],
        cumulants: Table[SystemId, EnergyCumulants],
        beta_sim: Table[SystemId, Array],
        log_fugacity_sim: Table[SystemId, Array],
        adsorbates: Table[MotifId, AdsorbateEOS],
        order: int = 3,
    ) -> TMMCSummary:
        r"""Build a summary from accumulated C-matrices (rows indexed by $N$)."""

        def reconstruct(
            statistics: TransitionStatistics, beta: Array, log_fugacity: Array
        ) -> LogPartition:
            probabilities = statistics.probabilities()
            return reconstruct_log_partition(
                probabilities.insertion,
                probabilities.deletion,
                beta,
                log_fugacity,
            )

        return TMMCSummary(
            log_partition_sim=Table.transform(reconstruct)(
                statistics, beta_sim, log_fugacity_sim
            ),
            cumulants=cumulants,
            beta_sim=beta_sim,
            adsorbates=adsorbates,
            order=order,
        )

    def _per_ladder[R](
        self, fn: Callable[[LogPartition, EnergyCumulants, Array], R]
    ) -> Table[SystemId, R]:
        r"""Apply ``fn`` to each system's $\ln Q_c$, cumulants and $\beta_\mathrm{sim}$."""
        return Table.transform(fn)(
            self.log_partition_sim, self.cumulants, self.beta_sim
        )

    def extrapolate(self, beta_target: Array) -> Table[SystemId, LogPartition]:
        r"""$\ln Q_c$ at a target inverse temperature via Taylor expansion."""
        return self._per_ladder(
            lambda log_partition, cumulants, beta_sim: extrapolate_log_partition(
                log_partition, cumulants, beta_sim, beta_target, self.order
            )
        )

    def isotherm(
        self, pressures: Array, temperature: Array
    ) -> Table[SystemId, Loading]:
        r"""$\langle N \rangle$ at each pressure [Pa] and temperature [K]."""
        beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
        return self.extrapolate(beta).map_data(
            lambda log_partition: _isotherm(
                log_partition, beta, pressures, temperature, self.adsorbates
            )
        )

    def isosteric_heat(
        self, pressures: Array, temperature: Array
    ) -> Table[SystemId, IsostericHeat]:
        r"""Isosteric heat [energy] at each pressure [Pa] and temperature [K]."""
        return self._per_ladder(
            lambda log_partition, cumulants, beta_sim: _isosteric_heat(
                log_partition,
                cumulants,
                beta_sim,
                pressures,
                temperature,
                self.adsorbates,
                self.order,
            )
        )

    def working_capacity(
        self,
        pressure_ads: Array,
        pressure_des: Array,
        temperature: Array,
    ) -> Table[SystemId, Loading]:
        r"""Pressure-swing capacity $\langle N\rangle(P_\mathrm{ads}) - \langle N\rangle(P_\mathrm{des})$ (eq 20)."""
        return self._per_ladder(
            lambda log_partition, cumulants, beta_sim: _working_capacity(
                log_partition,
                cumulants,
                beta_sim,
                pressure_ads,
                pressure_des,
                temperature,
                self.adsorbates,
                self.order,
            )
        )
