# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Post-processing for flat-histogram (TMMC) adsorption simulations.

Implements the temperature-extrapolation scheme of Witman, Mahynski & Smit
(2018) [^witman2018] for computing adsorption isotherms, isosteric heats and
working capacities from an NVT+W simulation performed at a single temperature.

Pipeline:

1. C-matrix $\to$ transition probabilities $P(N \to N \pm 1)$ (eq 7).
2. Transition probabilities $\to$ $\ln Q_c(N, V, \beta_\mathrm{sim})$ via
   a prefix sum of the detailed-balance ratio (eq 8). This replaces a
   Python loop with a single [`jnp.cumsum`][jax.numpy.cumsum].
3. Taylor-expand $\ln Q_c(N, V, \beta)$ in $\beta$ using the per-macrostate
   energy cumulants collected during simulation (eq 9--10).
4. Reweight into the grand canonical ensemble (eq 3) using
   [peng_robinson_log_fugacity][kups.mcmc.fugacity.peng_robinson_log_fugacity]
   for the reservoir fugacity.

The isosteric heat of adsorption (eq 19) is computed with [`jax.grad`][jax.grad]
through the full $\langle N \rangle(T, \ln P)$ pipeline, including the
Peng--Robinson equation of state. Autodiff is enabled by the implicit-function
JVP attached to [`cubic_roots`][kups.core.utils.math.cubic_roots] and the
double-where sanitisation in
[`peng_robinson_log_fugacity`][kups.mcmc.fugacity.peng_robinson_log_fugacity].
No ``delta_T`` hyperparameter and no finite-difference truncation error.

All public functions and the [`TMMCSummary`][kups.mcmc.flat_histogram.TMMCSummary]
dataclass operate on plain [`jax.Array`][jax.Array] inputs and are fully
vectorised across pressure grids.

[^witman2018]:
    Witman, M., Mahynski, N. A. & Smit, B. (2018). *J. Chem. Theory Comput.*
    **14**, 6149--6158. DOI: 10.1021/acs.jctc.8b00534
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT, KELVIN, PASCAL
from kups.core.utils.jax import dataclass, field
from kups.mcmc.fugacity import peng_robinson_log_fugacity
from kups.mcmc.widom import EnergyCumulants

type LogPartitionFunction = Array
r"""Natural logarithm of $Q_c(N, V, \beta)$, shape ``(n_max+1,)``."""

type MacrostateDistribution = Array
r"""Grand-canonical distribution $\Pi(N; \mu, V, \beta)$, shape ``(n_max+1,)``."""

type Loading = Array
r"""Expected loading $\langle N \rangle$, scalar or shape ``(n_pressures,)``."""

type IsostericHeat = Array
r"""Isosteric heat $q_\mathrm{st}$ [energy], scalar or shape ``(n_pressures,)``."""

type InsertionProbability = Array
r"""TMMC insertion probability $P(N \to N+1)$, shape ``(n_max+1,)``."""

type DeletionProbability = Array
r"""TMMC deletion probability $P(N \to N-1)$, shape ``(n_max+1,)``."""


def transition_probabilities(
    acceptance_insertion: Array,
    acceptance_deletion: Array,
    n_trials_insertion: Array,
    n_trials_deletion: Array,
) -> tuple[InsertionProbability, DeletionProbability]:
    r"""Normalise accumulated C-matrix counts into per-macrostate transition probabilities.

    Applies Witman 2018 eq 7:

    $$P(N \to N + \Delta) = \frac{C(N, N + \Delta)}{\sum_\Delta C(N, N + \Delta)}.$$

    Args:
        acceptance_insertion: Accumulated $\min(1, \mathrm{acc})$ for
            $N \to N+1$, shape ``(n_max+1,)``.
        acceptance_deletion: Accumulated $\min(1, \mathrm{acc})$ for
            $N \to N-1$, shape ``(n_max+1,)``.
        n_trials_insertion: Ghost-insertion trial count per macrostate.
        n_trials_deletion: Ghost-deletion trial count per macrostate
            (includes trials at $N = 0$, which contribute zero acceptance).

    Returns:
        A pair ``(P(N → N+1), P(N → N-1))``, each shape ``(n_max+1,)``.
    """
    total = n_trials_insertion + n_trials_deletion
    safe_total = jnp.where(total > 0, total, 1)
    return (
        acceptance_insertion / safe_total,
        acceptance_deletion / safe_total,
    )


def reconstruct_log_partition_fn(
    insertion_probability: InsertionProbability,
    deletion_probability: DeletionProbability,
    beta: Array,
    log_fugacity: Array,
) -> LogPartitionFunction:
    r"""Reconstruct $\ln Q_c(N, V, \beta_\mathrm{sim})$ from TMMC transition probabilities.

    Applies Witman 2018 eq 8 as a prefix sum over macrostates:

    $$\ln Q_c(N+1) - \ln Q_c(N)
        = -\ln[q(\beta)\exp(\beta\mu)] + \ln\frac{P(N \to N+1)}{P(N+1 \to N)}.$$

    For an ideal-gas reference, $q(\beta)\exp(\beta\mu) = \beta f$, so the
    correction term is $\ln\beta + \ln f$ at every macrostate. The full
    reconstruction is a single [`jnp.cumsum`][jax.numpy.cumsum] over the
    macrostate axis.

    Args:
        insertion_probability: $P(N \to N+1)$, shape ``(n_max+1,)``.
        deletion_probability: $P(N \to N-1)$, shape ``(n_max+1,)``.
        beta: Simulation inverse temperature $1/(k_B T_\mathrm{sim})$
            [1/energy].
        log_fugacity: Simulation log-fugacity $\ln f$ [dimensionless after
            unit scaling].

    Returns:
        $\ln Q_c(N, V, \beta_\mathrm{sim})$ of shape ``(n_max+1,)``, anchored
        to ``0`` at $N = 0$.
    """
    ratio = jnp.log(insertion_probability[:-1]) - jnp.log(deletion_probability[1:])
    delta_log_qc = ratio - (log_fugacity + jnp.log(beta))
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(delta_log_qc)])


def extrapolate_log_partition_fn(
    log_partition_fn_sim: LogPartitionFunction,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    beta_target: Array,
    order: int = 3,
) -> LogPartitionFunction:
    r"""Taylor-expand $\ln Q_c$ from $\beta_\mathrm{sim}$ to $\beta_\mathrm{target}$.

    Implements Witman 2018 eq 9 to the specified order using the per-macrostate
    energy cumulants collected during simulation. Using the convention that
    [`EnergyCumulants.mean`][kups.mcmc.widom.EnergyCumulants] stores the raw
    running mean $\langle E\rangle$ and ``third``/``fourth`` already carry the
    signs appropriate for $\partial^n \ln Q_c / \partial \beta^n$:

    $$\ln Q_c(\beta')
        \approx \ln Q_c(\beta)
        - \langle E\rangle\,\delta\beta
        + \tfrac{1}{2}\mathrm{Var}\,\delta\beta^2
        + \tfrac{1}{6}\kappa_3\,\delta\beta^3
        + \tfrac{1}{24}\kappa_4\,\delta\beta^4$$

    with $\delta\beta = \beta_\mathrm{target} - \beta_\mathrm{sim}$.

    Args:
        log_partition_fn_sim: $\ln Q_c(N, V, \beta_\mathrm{sim})$, shape
            ``(n_max+1,)``.
        cumulants: Per-macrostate energy cumulants.
        beta_sim: Simulation inverse temperature.
        beta_target: Target inverse temperature.
        order: Taylor order (1--4). Higher orders require more sampling to
            converge.

    Returns:
        Extrapolated $\ln Q_c(N, V, \beta_\mathrm{target})$, shape
        ``(n_max+1,)``.

    Raises:
        ValueError: If ``order`` is not in 1--4.
    """
    if not 1 <= order <= 4:
        raise ValueError(f"order must be in 1..4, got {order}")

    db = beta_target - beta_sim
    result = log_partition_fn_sim - cumulants.mean * db
    if order >= 2:
        result = result + cumulants.variance * db**2 / 2.0
    if order >= 3:
        result = result + cumulants.third * db**3 / 6.0
    if order >= 4:
        result = result + cumulants.fourth * db**4 / 24.0
    return result


def macrostate_distribution(
    log_partition_fn: LogPartitionFunction,
    beta: Array,
    log_fugacity: Array,
) -> MacrostateDistribution:
    r"""Compute $\Pi(N; \mu, V, \beta)$ from $\ln Q_c$ and a reservoir fugacity.

    Witman 2018 eq 3 with $\beta\mu = \ln(\beta f / q(\beta))$; the
    ideal-gas $q$-factor cancels with the one absorbed into ``log_partition_fn``
    by :func:`reconstruct_log_partition_fn`.

    Uses the standard log-sum-exp trick to avoid overflow at large $N$.

    Args:
        log_partition_fn: $\ln Q_c(N, V, \beta)$, shape ``(n_max+1,)``.
        beta: Inverse temperature.
        log_fugacity: $\ln f$ at target conditions.

    Returns:
        Normalised distribution $\Pi(N)$, shape ``(n_max+1,)``.
    """
    n_values = jnp.arange(log_partition_fn.shape[0])
    log_weights = (log_fugacity + jnp.log(beta)) * n_values + log_partition_fn
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    return weights / jnp.sum(weights)


def average_loading(distribution: MacrostateDistribution) -> Loading:
    r"""Compute $\langle N \rangle = \sum_N N\,\Pi(N)$."""
    n_values = jnp.arange(distribution.shape[0])
    return jnp.sum(n_values * distribution)


def _log_fugacity_at_pressures(
    pressures_pa: Array,
    temperature_k: Array,
    critical_pressure_pa: float,
    critical_temperature_k: float,
    acentric_factor: float,
) -> Array:
    r"""Return $\ln f$ at each pressure for a single-component gas. Shape ``(n_p,)``.

    Uses kUPS's internal units (``PASCAL``, ``KELVIN``). Peng-Robinson's
    numpy-style ``vectorize`` broadcasts the leading pressure axis
    automatically.
    """
    result = peng_robinson_log_fugacity(
        pressures_pa * PASCAL,
        temperature_k * KELVIN,
        jnp.atleast_1d(jnp.asarray(critical_pressure_pa)) * PASCAL,
        jnp.atleast_1d(jnp.asarray(critical_temperature_k)) * KELVIN,
        jnp.atleast_1d(jnp.asarray(acentric_factor)),
    )
    return result.log_fugacity[..., 0]


def isotherm(
    log_partition_fn: LogPartitionFunction,
    beta: Array,
    pressures: Array,
    temperature: Array,
    critical_pressure: float,
    critical_temperature: float,
    acentric_factor: float,
) -> Loading:
    r"""Vectorised adsorption isotherm $\langle N \rangle(P)$ at fixed $T$.

    Evaluates Witman 2018 eq 18 for every pressure in ``pressures`` in a single
    JAX computation — no Python loop. Fugacity comes from Peng-Robinson; the
    macrostate distribution is rebuilt at each pressure.

    Args:
        log_partition_fn: $\ln Q_c(N, V, \beta)$ at the target $\beta$,
            shape ``(n_max+1,)``.
        beta: Inverse temperature corresponding to ``temperature``.
        pressures: Pressure grid [Pa], shape ``(n_pressures,)``.
        temperature: Temperature [K].
        critical_pressure: $P_c$ of the adsorbate [Pa].
        critical_temperature: $T_c$ of the adsorbate [K].
        acentric_factor: $\omega$ of the adsorbate [-].

    Returns:
        $\langle N \rangle$ at each pressure, shape ``(n_pressures,)``.
    """
    log_fugacities = _log_fugacity_at_pressures(
        pressures, temperature,
        critical_pressure, critical_temperature, acentric_factor,
    )
    distributions = jax.vmap(
        macrostate_distribution, in_axes=(None, None, 0),
    )(log_partition_fn, beta, log_fugacities)
    return jax.vmap(average_loading)(distributions)


def _loading_from_T_logP(
    temperature: Array,
    log_pressure_pa: Array,
    log_partition_fn_sim: LogPartitionFunction,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    order: int,
    critical_pressure: float,
    critical_temperature: float,
    acentric_factor: float,
) -> Array:
    r"""Pure, differentiable $\langle N \rangle(T, \ln P)$ at one state point.

    Used by :func:`isosteric_heat` as the target of ``jax.grad`` — composes
    the Taylor extrapolation, Peng--Robinson fugacity, macrostate distribution,
    and $\langle N\rangle$ marginalisation.
    """
    beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
    log_qc = extrapolate_log_partition_fn(
        log_partition_fn_sim, cumulants, beta_sim, beta, order,
    )
    pressure = jnp.exp(log_pressure_pa)
    log_fugacity = _log_fugacity_at_pressures(
        jnp.atleast_1d(pressure), temperature,
        critical_pressure, critical_temperature, acentric_factor,
    )[0]
    return average_loading(macrostate_distribution(log_qc, beta, log_fugacity))


def isosteric_heat(
    log_partition_fn_sim: LogPartitionFunction,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    pressures: Array,
    temperature: Array,
    critical_pressure: float,
    critical_temperature: float,
    acentric_factor: float,
    order: int = 3,
) -> IsostericHeat:
    r"""Isosteric heat $q_\mathrm{st}$ via Clausius--Clapeyron with autodiff.

    Uses the triple-product rule

    $$q_\mathrm{st} = k_B T^2
        \frac{(\partial \langle N\rangle / \partial T)_P}
             {(\partial \langle N\rangle / \partial \ln P)_T}.$$

    Both partial derivatives come from [`jax.grad`][jax.grad] on the pure
    loading function $\langle N \rangle(T, \ln P)$, vmapped across the
    pressure grid. No ``delta_T`` hyperparameter, no finite-difference
    truncation error.

    Args:
        log_partition_fn_sim: $\ln Q_c(N, V, \beta_\mathrm{sim})$,
            shape ``(n_max+1,)``.
        cumulants: Per-macrostate energy cumulants.
        beta_sim: Simulation inverse temperature.
        pressures: Pressure grid [Pa], shape ``(n_pressures,)``.
        temperature: Target temperature [K].
        critical_pressure: $P_c$ of the adsorbate [Pa].
        critical_temperature: $T_c$ of the adsorbate [K].
        acentric_factor: $\omega$ of the adsorbate [-].
        order: Taylor order for the $\beta$-extrapolation.

    Returns:
        $q_\mathrm{st}$ [energy] at each pressure, shape ``(n_pressures,)``.
    """

    def loading_fn(t: Array, log_p: Array) -> Array:
        return _loading_from_T_logP(
            t, log_p,
            log_partition_fn_sim, cumulants, beta_sim, order,
            critical_pressure, critical_temperature, acentric_factor,
        )

    dN_dT = jax.grad(loading_fn, argnums=0)
    dN_dlogP = jax.grad(loading_fn, argnums=1)

    def q_st_at(log_p: Array) -> Array:
        return (
            BOLTZMANN_CONSTANT
            * temperature**2
            * dN_dT(temperature, log_p)
            / dN_dlogP(temperature, log_p)
        )

    return jax.vmap(q_st_at)(jnp.log(pressures))


def working_capacity(
    log_partition_fn_sim: LogPartitionFunction,
    cumulants: EnergyCumulants,
    beta_sim: Array,
    temperature_ads: Array,
    pressure_ads: Array,
    temperature_des: Array,
    pressure_des: Array,
    critical_pressure: float,
    critical_temperature: float,
    acentric_factor: float,
    order: int = 3,
) -> Array:
    r"""Working capacity $n_\mathrm{wc} = \langle N\rangle_\mathrm{ads} - \langle N\rangle_\mathrm{des}$.

    Evaluates Witman 2018 eq 20 for a single $(T_\mathrm{ads}, P_\mathrm{ads}) \to (T_\mathrm{des}, P_\mathrm{des})$
    swing process. Both end points use the Taylor-extrapolated $\ln Q_c$.
    """

    def loading(t: Array, p: Array) -> Array:
        beta = 1.0 / (BOLTZMANN_CONSTANT * t)
        log_qc = extrapolate_log_partition_fn(
            log_partition_fn_sim, cumulants, beta_sim, beta, order,
        )
        return isotherm(
            log_qc, beta, jnp.atleast_1d(p), t,
            critical_pressure, critical_temperature, acentric_factor,
        )[0]

    return loading(temperature_ads, pressure_ads) - loading(
        temperature_des, pressure_des
    )


@dataclass
class AdsorbateEOS:
    r"""Peng-Robinson equation-of-state parameters for a single adsorbate species.

    All values are in SI units (Pa, K, dimensionless) at the boundary; internal
    unit conversion is handled downstream by
    [peng_robinson_log_fugacity][kups.mcmc.fugacity.peng_robinson_log_fugacity].
    """

    critical_pressure: float = field(static=True)
    critical_temperature: float = field(static=True)
    acentric_factor: float = field(static=True)


@dataclass
class TMMCSummary:
    r"""Post-processed TMMC simulation at a single temperature.

    Bundles $\ln Q_c(\beta_\mathrm{sim})$, the per-macrostate energy cumulants,
    and the reservoir EOS so that downstream observables are one method call
    away. All methods are pure and JIT/vmap-compatible.

    Attributes:
        log_partition_fn_sim: $\ln Q_c(N, V, \beta_\mathrm{sim})$, shape
            ``(n_max+1,)``.
        cumulants: Per-macrostate energy cumulants.
        beta_sim: Simulation inverse temperature.
        adsorbate: Peng-Robinson parameters for the reservoir species.
        order: Taylor order for the $\beta$-extrapolation (default 3).
    """

    log_partition_fn_sim: LogPartitionFunction
    cumulants: EnergyCumulants
    beta_sim: Array
    adsorbate: AdsorbateEOS
    order: int = field(static=True, default=3)

    @staticmethod
    def from_transition_statistics(
        acceptance_insertion: Array,
        acceptance_deletion: Array,
        n_trials_insertion: Array,
        n_trials_deletion: Array,
        cumulants: EnergyCumulants,
        beta_sim: Array,
        log_fugacity_sim: Array,
        adsorbate: AdsorbateEOS,
        order: int = 3,
    ) -> TMMCSummary:
        r"""Build a summary directly from raw C-matrix counts."""
        p_ins, p_del = transition_probabilities(
            acceptance_insertion,
            acceptance_deletion,
            n_trials_insertion,
            n_trials_deletion,
        )
        log_qc = reconstruct_log_partition_fn(p_ins, p_del, beta_sim, log_fugacity_sim)
        return TMMCSummary(
            log_partition_fn_sim=log_qc,
            cumulants=cumulants,
            beta_sim=beta_sim,
            adsorbate=adsorbate,
            order=order,
        )

    def extrapolate(self, beta_target: Array) -> LogPartitionFunction:
        r"""$\ln Q_c$ at a target inverse temperature via Taylor expansion."""
        return extrapolate_log_partition_fn(
            self.log_partition_fn_sim, self.cumulants,
            self.beta_sim, beta_target, self.order,
        )

    def isotherm(self, pressures: Array, temperature: Array) -> Loading:
        r"""$\langle N \rangle(P)$ at fixed $T$, vectorised across pressures."""
        beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
        log_qc = self.extrapolate(beta)
        return isotherm(
            log_qc, beta, pressures, temperature,
            self.adsorbate.critical_pressure,
            self.adsorbate.critical_temperature,
            self.adsorbate.acentric_factor,
        )

    def isosteric_heat(self, pressures: Array, temperature: Array) -> IsostericHeat:
        r"""Autodiff Clausius--Clapeyron heat across a pressure grid."""
        return isosteric_heat(
            self.log_partition_fn_sim, self.cumulants, self.beta_sim,
            pressures, temperature,
            self.adsorbate.critical_pressure,
            self.adsorbate.critical_temperature,
            self.adsorbate.acentric_factor,
            self.order,
        )

    def working_capacity(
        self,
        temperature_ads: Array,
        pressure_ads: Array,
        temperature_des: Array,
        pressure_des: Array,
    ) -> Array:
        r"""Working capacity between adsorption and desorption conditions."""
        return working_capacity(
            self.log_partition_fn_sim, self.cumulants, self.beta_sim,
            temperature_ads, pressure_ads, temperature_des, pressure_des,
            self.adsorbate.critical_pressure,
            self.adsorbate.critical_temperature,
            self.adsorbate.acentric_factor,
            self.order,
        )
