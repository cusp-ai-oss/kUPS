# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Post-processing for flat-histogram (TMMC) adsorption simulations.

Implements the temperature-extrapolation scheme of Witman, Mahynski & Smit
(2018) [^witman2018] for computing adsorption isotherms, isosteric heats and
working capacities from an NVT+W simulation performed at a single temperature.

Pipeline:

1. C-matrix $\to$ transition probabilities $P(N \to N \pm 1)$ (eq 7), via
   [probabilities][kups.mcmc.widom.TransitionStatistics.probabilities].
2. Transition probabilities $\to$ $\ln Q_c(N, V, \beta_\mathrm{sim})$ via
   a prefix sum of the detailed-balance ratio (eq 8).
3. Taylor-expand $\ln Q_c(N, V, \beta)$ in $\beta$ using the per-macrostate
   energy cumulants collected during simulation (eq 9--10).
4. Reweight into the grand canonical ensemble (eq 3) using
   [peng_robinson_log_fugacity][kups.mcmc.fugacity.peng_robinson_log_fugacity]
   for the reservoir fugacity.

Axis convention: the macrostate $N$ is the **trailing** axis of every array in
the pipeline ([MacrostateId][kups.core.typing.MacrostateId]), so a ladder is
one row of ``n_max + 1`` rungs and any leading axes stay per-system. Every
step above is written against ``axis=-1``, which makes it rank-polymorphic:
a lone ladder of shape ``(n_max+1,)`` and a batch of shape
``(n_systems, n_max+1)`` follow the identical code path and produce bitwise
identical results. Per-system inputs that carry no macrostate axis
($\beta_\mathrm{sim}$, $\ln f$) gain a trailing singleton before broadcasting.

[`TMMCSummary`][kups.mcmc.flat_histogram.TMMCSummary] is the public entry
point: it holds the reconstructed ladders as ``Table[SystemId, ...]`` rows
keyed by system, exposes the observables as methods, and returns
``Table``-typed results so the keys survive the whole post-processing chain.

The isosteric heat of adsorption (eq 19) is computed by autodiff through the
full $\langle N \rangle(T, \ln P)$ pipeline, including the Peng--Robinson
equation of state. Both derivatives are taken with respect to scalars, so a
single forward-mode pass ([`jax.jacfwd`][jax.jacfwd]) yields them for every
system at once. Differentiability is enabled by the implicit-function JVP
attached to [`cubic_roots`][kups.core.utils.math.cubic_roots] and the
double-where sanitisation in
[`peng_robinson_log_fugacity`][kups.mcmc.fugacity.peng_robinson_log_fugacity].

Scope: the macrostate axis is the 1-D loading $N$ of a **single adsorbate
species** --- one ladder per system, exactly the case Witman 2018 treats ---
so every equation-of-state argument is a ``Table[MotifId, AdsorbateEOS]`` of
length one, asserted at the point of use. The multicomponent successor follows
Shen & Errington [^shen2005]: the full $\Pi(N_1, N_2)$ surface is assembled
from two pure-species ladders plus constant-$N_\mathrm{total}$ semigrand
slices (identity-swap dynamics), all reconstructed by the same telescoping
recursion used here --- no least-squares solve on the documented path.

[^witman2018]:
    Witman, M., Mahynski, N. A. & Smit, B. (2018). *J. Chem. Theory Comput.*
    **14**, 6149--6158. DOI: 10.1021/acs.jctc.8b00534

[^shen2005]:
    Shen, V. K. & Errington, J. R. (2005). *J. Chem. Phys.* **122**, 064508.
    DOI: 10.1063/1.1844372
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
r"""Natural logarithm of $Q_c(N, V, \beta)$, shape ``(..., n_max+1)``.

The trailing axis is the macrostate ([MacrostateId][kups.core.typing.MacrostateId]);
leading axes, if any, are per-system.
"""

type MacrostateDistribution = Array
r"""Grand-canonical distribution $\Pi(N; \mu, V, \beta)$, shape ``(..., n_max+1)``."""

type Loading = Array
r"""Expected loading $\langle N \rangle$, shape ``(...)`` or ``(..., n_pressures)``."""

type IsostericHeat = Array
r"""Isosteric heat $q_\mathrm{st}$ [energy], shape ``(...)`` or ``(..., n_pressures)``."""


@dataclass
class AdsorbateEOS:
    r"""Peng-Robinson equation-of-state parameters for adsorbate species.

    Always carried as a ``Table[MotifId, AdsorbateEOS]``, so each field holds
    one entry per species and the whole bundle is the ``(n,)``-shaped
    per-component layout that
    [peng_robinson_log_fugacity][kups.mcmc.fugacity.peng_robinson_log_fugacity]
    expects. The single-ladder pipeline in this module asserts a length of
    one; the multicomponent successor described in the module scope note is
    what lifts that restriction.

    All values are in SI units (Pa, K, dimensionless) at the boundary;
    internal unit conversion happens downstream. The parameters are pytree
    leaves, so they may be traced (e.g. to differentiate observables with
    respect to the EOS parameters).

    Attributes:
        critical_pressure: $P_c$ [Pa], shape ``(n_species,)``.
        critical_temperature: $T_c$ [K], shape ``(n_species,)``.
        acentric_factor: $\omega$ [-], shape ``(n_species,)``.
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
        r"""Build the one-species EOS table the flat-histogram pipeline consumes.

        Args:
            critical_pressure: $P_c$ of the adsorbate [Pa].
            critical_temperature: $T_c$ of the adsorbate [K].
            acentric_factor: $\omega$ of the adsorbate [-].

        Returns:
            A ``Table[MotifId, AdsorbateEOS]`` with a single row.
        """
        return Table.arange(
            AdsorbateEOS(
                critical_pressure=jnp.atleast_1d(jnp.asarray(critical_pressure)),
                critical_temperature=jnp.atleast_1d(jnp.asarray(critical_temperature)),
                acentric_factor=jnp.atleast_1d(jnp.asarray(acentric_factor)),
            ),
            label=MotifId,
        )


def _with_macrostate_axis(per_system: Array) -> Array:
    r"""Append a singleton macrostate axis to a per-system quantity.

    Quantities like $\beta$ and $\ln f$ are one value per system, with no
    macrostate axis of their own. A trailing singleton lets them broadcast
    against a ``(..., n_max+1)`` ladder: a per-system ``(n_systems,)`` becomes
    ``(n_systems, 1)``, which lines up with the ladder's trailing macrostate
    axis. A rank-0 value already broadcasts against every axis, so it is
    passed through untouched and keeps scalar inputs giving scalar results.
    """
    value = jnp.asarray(per_system)
    return value if value.ndim == 0 else value[..., None]


def _single_species(adsorbates: Table[MotifId, AdsorbateEOS]) -> AdsorbateEOS:
    """Unwrap the sole species of an EOS table, or fail loudly.

    Raises:
        AssertionError: If ``adsorbates`` does not hold exactly one species.
    """
    assert len(adsorbates) == 1, (
        f"the flat-histogram ladder samples a single adsorbate species, got "
        f"{len(adsorbates)}. A mixture needs the multi-dimensional macrostate "
        "lattice of the Shen & Errington (2005) spanning-tree construction "
        "described in this module's scope note, not the 1-D ladder here."
    )
    return adsorbates.data


def reconstruct_log_partition(
    insertion_probability: InsertionProbability,
    deletion_probability: DeletionProbability,
    beta: Array,
    log_fugacity: Array,
) -> LogPartition:
    r"""Reconstruct $\ln Q_c(N, V, \beta_\mathrm{sim})$ from TMMC transition probabilities.

    Applies Witman 2018 eq 8 as a prefix sum over macrostates:

    $$\ln Q_c(N+1) - \ln Q_c(N)
        = -\ln[q(\beta)\exp(\beta\mu)] + \ln\frac{P(N \to N+1)}{P(N+1 \to N)}.$$

    For an ideal-gas reference, $q(\beta)\exp(\beta\mu) = \beta f$, so the
    correction term is $\ln\beta + \ln f$ at every macrostate, so the
    reconstruction is a [`jnp.cumsum`][jax.numpy.cumsum] over the trailing
    macrostate axis.

    Args:
        insertion_probability: $P(N \to N+1)$, shape ``(..., n_max+1)``.
        deletion_probability: $P(N \to N-1)$, shape ``(..., n_max+1)``.
        beta: Simulation inverse temperature $1/(k_B T_\mathrm{sim})$
            [1/energy], one value per system.
        log_fugacity: Simulation log-fugacity $\ln f$ [dimensionless after
            unit scaling], one value per system.

    Returns:
        $\ln Q_c(N, V, \beta_\mathrm{sim})$ of shape ``(..., n_max+1)``,
        anchored to ``0`` at $N = 0$.
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
    r"""Taylor-expand $\ln Q_c$ from $\beta_\mathrm{sim}$ to $\beta_\mathrm{target}$.

    Implements Witman 2018 eq 9 to arbitrary order using the per-macrostate
    energy cumulants collected during simulation. Since
    $\partial^k \ln Q_c / \partial\beta^k = (-1)^k \kappa_k$ (see
    [`EnergyCumulants`][kups.mcmc.widom.EnergyCumulants]),

    $$\ln Q_c(\beta')
        \approx \ln Q_c(\beta)
        + \sum_{k=1}^{K} \frac{(-\delta\beta)^k}{k!}\,\kappa_k,
    \qquad \delta\beta = \beta' - \beta.$$

    Args:
        log_partition_sim: $\ln Q_c(N, V, \beta_\mathrm{sim})$, shape
            ``(..., n_max+1)``.
        cumulants: Per-macrostate energy cumulants; ``mean`` shares the
            ladder's shape and ``cumulants`` appends the moment-order axis.
        beta_sim: Simulation inverse temperature, one value per system.
        beta_target: Target inverse temperature.
        order: Taylor order $K$, at most ``cumulants.max_order``. Higher
            orders require more sampling to converge.

    Returns:
        Extrapolated $\ln Q_c(N, V, \beta_\mathrm{target})$, shape
        ``(..., n_max+1)``.

    Raises:
        ValueError: If ``order`` is not in ``1..cumulants.max_order``.
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
    r"""Compute $\Pi(N; \mu, V, \beta)$ from $\ln Q_c$ and a reservoir fugacity.

    Witman 2018 eq 3 with $\beta\mu = \ln(\beta f / q(\beta))$; the
    ideal-gas $q$-factor cancels with the one absorbed into ``log_partition``
    by :func:`reconstruct_log_partition`.

    Args:
        log_partition: $\ln Q_c(N, V, \beta)$, shape ``(..., n_max+1)``.
        beta: Inverse temperature, one value per system.
        log_fugacity: $\ln f$ at target conditions, one value per system.

    Returns:
        Distribution $\Pi(N)$ of shape ``(..., n_max+1)``, normalised over
        the trailing macrostate axis.
    """
    n_values = jnp.arange(log_partition.shape[-1])
    log_activity = _with_macrostate_axis(log_fugacity + jnp.log(beta))
    return jax.nn.softmax(log_activity * n_values + log_partition, axis=-1)


def average_loading(distribution: MacrostateDistribution) -> Loading:
    r"""Compute $\langle N \rangle = \sum_N N\,\Pi(N)$.

    Args:
        distribution: Normalised $\Pi(N)$, shape ``(..., n_max+1)``.

    Returns:
        Expected loading $\langle N \rangle$, one value per system (the
        trailing macrostate axis is summed away).
    """
    n_values = jnp.arange(distribution.shape[-1])
    return jnp.sum(n_values * distribution, axis=-1)


def _log_fugacity_at_pressures(
    pressures_pa: Array,
    temperature_k: Array,
    adsorbates: Table[MotifId, AdsorbateEOS],
) -> Array:
    r"""Return $\ln f$ at each pressure for the sole adsorbate. Shape ``(n_p,)``.

    Uses kUPS's internal units (``PASCAL``, ``KELVIN``). Peng-Robinson's
    numpy-style ``vectorize`` broadcasts the leading pressure axis
    automatically, and the EOS table's single row is already the ``(n,)``
    per-component layout it expects.
    """
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
    r"""Vectorised $\langle N \rangle(P)$ at fixed $T$; see :meth:`TMMCSummary.isotherm`.

    Evaluates Witman 2018 eq 18 for every pressure in ``pressures``, vmapped
    over the pressure grid and stacked onto a trailing pressure axis so the
    leading per-system axes are untouched: ``(..., n_max+1)`` in,
    ``(..., n_pressures)`` out.
    """
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
    r"""Pure, differentiable $\langle N \rangle(T, \ln P)$ at one state point.

    Used by :func:`_isosteric_heat` as the target of forward-mode autodiff —
    composes the Taylor extrapolation, Peng--Robinson fugacity, macrostate
    distribution, and $\langle N\rangle$ marginalisation. Both differentiated
    arguments are scalars; the result carries the ladder's per-system shape.
    """
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
    r"""Clausius--Clapeyron $q_\mathrm{st}$; see :meth:`TMMCSummary.isosteric_heat`.

    Uses the triple-product rule

    $$q_\mathrm{st} = k_B T^2
        \frac{(\partial \langle N\rangle / \partial T)_P}
             {(\partial \langle N\rangle / \partial \ln P)_T}.$$

    Both partial derivatives come from one forward-mode pass over the pure
    loading function $\langle N \rangle(T, \ln P)$ — the differentiation
    targets are two scalars, so [`jax.jacfwd`][jax.jacfwd] returns
    derivatives shaped like the loading itself and the whole batch of systems
    is differentiated at once, where [`jax.grad`][jax.grad] would insist on a
    scalar output. The pressure grid is vmapped onto a trailing axis.
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
    r"""Post-processed TMMC simulation at a single temperature.

    Bundles $\ln Q_c(\beta_\mathrm{sim})$, the per-macrostate energy
    cumulants, and the reservoir EOS so that downstream observables are one
    method call away. Every field is keyed: one row per system, holding that
    system's whole macrostate ladder along the row's trailing axis. The
    observable methods keep those keys, so a batch of hosts is post-processed
    in a single call and stays labelled throughout. All methods are pure and
    JIT/vmap-compatible.

    Attributes:
        log_partition_sim: $\ln Q_c(N, V, \beta_\mathrm{sim})$, rows of shape
            ``(n_max+1,)``.
        cumulants: Per-macrostate energy cumulants, rows of shape
            ``(n_max+1, order-1)`` (plus the ``(n_max+1,)`` mean).
        beta_sim: Simulation inverse temperature, one scalar row per system.
        adsorbates: Peng-Robinson parameters for the reservoir species.
            Exactly one species, asserted at the point of use — see the
            module scope note for the multicomponent successor.
        order: Taylor order for the $\beta$-extrapolation (default 3).
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
        r"""Build a summary directly from accumulated C-matrices.

        Args:
            statistics: Accumulated TMMC C-matrix per system, each row
                indexed by macrostate $N$ along its trailing axis.
            cumulants: Per-macrostate energy cumulants.
            beta_sim: Simulation inverse temperature.
            log_fugacity_sim: Simulation log-fugacity $\ln f$.
            adsorbates: Peng-Robinson parameters for the reservoir species.
            order: Taylor order for the $\beta$-extrapolation.

        Returns:
            Summary ready for isotherm / heat / capacity evaluation.
        """

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
        r"""$\langle N \rangle(P)$ at fixed $T$, vectorised across pressures.

        Args:
            pressures: Pressure grid [Pa], shape ``(n_pressures,)``.
            temperature: Temperature [K].

        Returns:
            $\langle N \rangle$ per system, rows of shape ``(n_pressures,)``.
        """
        beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
        return self.extrapolate(beta).map_data(
            lambda log_partition: _isotherm(
                log_partition, beta, pressures, temperature, self.adsorbates
            )
        )

    def isosteric_heat(
        self, pressures: Array, temperature: Array
    ) -> Table[SystemId, IsostericHeat]:
        r"""Autodiff Clausius--Clapeyron heat across a pressure grid.

        Args:
            pressures: Pressure grid [Pa], shape ``(n_pressures,)``.
            temperature: Temperature [K].

        Returns:
            $q_\mathrm{st}$ [energy] per system, rows of shape
            ``(n_pressures,)``.
        """
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
        r"""Isothermal working capacity
        $n_\mathrm{wc} = \langle N\rangle_\mathrm{ads} - \langle N\rangle_\mathrm{des}$.

        Evaluates Witman 2018 eq 20 for a pressure swing at fixed
        temperature. A temperature swing is the difference of two
        :meth:`isotherm` calls, so it needs no method of its own.

        Args:
            pressure_ads: Adsorption pressure [Pa].
            pressure_des: Desorption pressure [Pa].
            temperature: Swing temperature [K].

        Returns:
            $n_\mathrm{wc}$ [particles], one scalar row per system.
        """
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
