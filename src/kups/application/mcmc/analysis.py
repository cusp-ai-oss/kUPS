# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Post-simulation analysis for MCMC simulations."""

from __future__ import annotations

from dataclasses import dataclass as plain_dataclass
from pathlib import Path
from typing import Any, Protocol

import jax.numpy as jnp
from jax import Array

from kups.application.mcmc.data import StressResult
from kups.application.mcmc.logging import (
    MCMCLoggedData,
    WidomFixedData,
    WidomLoggedData,
)
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Table
from kups.core.storage import HDF5StorageReader
from kups.core.typing import HasTemperature, MotifId, SystemId
from kups.core.utils.block_average import (
    BlockAverageResult,
    block_average,
    block_average_from_blocks,
    compute_block_means,
    optimal_block_average,
)
from kups.core.utils.jax import no_jax_tracing
from kups.mcmc.widom import WidomStatistics


class IsMCMCFixedData(Protocol):
    """Contract for data from the fixed reader group."""

    @property
    def systems(self) -> Table[SystemId, HasTemperature]: ...


class IsMCMCSystemStepData(Protocol):
    """Contract for per-system step data."""

    @property
    def potential_energy(self) -> Array: ...
    @property
    def guest_stress(self) -> StressResult: ...


class IsMCMCStepData(Protocol):
    """Contract for data from the per_step reader group."""

    @property
    def particle_count(self) -> Table[tuple[SystemId, MotifId], Array]: ...
    @property
    def systems(self) -> Table[SystemId, IsMCMCSystemStepData]: ...


@plain_dataclass
class MCMCAnalysisResult:
    """Results from MCMC simulation analysis for a single system.

    Attributes:
        energy: Average total potential energy with SEM (eV).
        loading: Average particle count per species with SEM (dimensionless).
        heat_of_adsorption: Per-species heat of adsorption with SEM (eV).
        total_heat_of_adsorption: Composition-weighted total heat of adsorption (eV).
    """

    energy: BlockAverageResult
    loading: BlockAverageResult
    heat_of_adsorption: BlockAverageResult
    total_heat_of_adsorption: BlockAverageResult
    stress: BlockAverageResult | None = None
    stress_potential: BlockAverageResult | None = None
    stress_tail_correction: BlockAverageResult | None = None
    stress_ideal_gas: BlockAverageResult | None = None
    pressure: BlockAverageResult | None = None
    pressure_potential: BlockAverageResult | None = None
    pressure_tail_correction: BlockAverageResult | None = None
    pressure_ideal_gas: BlockAverageResult | None = None


@no_jax_tracing
def _analyze_single_system(
    energy: Array,
    counts: Array,
    temperature: float,
    n_blocks: int | None,
    stress: Array | None = None,
    stress_potential: Array | None = None,
    stress_tail_correction: Array | None = None,
    stress_ideal_gas: Array | None = None,
) -> MCMCAnalysisResult:
    """Run block-averaging analysis for one system.

    Args:
        energy: Potential energy time series, shape ``(n_steps,)``.
        counts: Particle count time series, shape ``(n_steps, n_species)``.
        temperature: System temperature in K.
        n_blocks: Number of blocks, or ``None`` for automatic selection.
        stress: Total stress tensor time series, shape ``(n_steps, 3, 3)``.
        stress_potential: Configurational stress component.
        stress_tail_correction: Tail correction stress component.
        stress_ideal_gas: Ideal gas stress component.
    """
    if n_blocks is None:
        energy_result = optimal_block_average(energy)
        n_blocks_used = int(energy_result.n_blocks)
    else:
        energy_result = block_average(energy, n_blocks=n_blocks)
        n_blocks_used = n_blocks

    loading = block_average(counts, n_blocks=n_blocks_used, axis=0)

    U_blocks = compute_block_means(energy, n_blocks_used)
    N_blocks = compute_block_means(counts, n_blocks_used)
    UN_blocks = compute_block_means(energy[:, None] * counts, n_blocks_used)
    NN_blocks = compute_block_means(counts[..., None] * counts[:, None], n_blocks_used)

    cov = NN_blocks - N_blocks[..., None] * N_blocks[:, None, :]
    cov_inv = jnp.linalg.inv(cov)
    diff = UN_blocks - U_blocks[:, None] * N_blocks
    hoa_blocks = jnp.einsum("bi,bij->bj", diff, cov_inv)
    hoa_blocks -= temperature * BOLTZMANN_CONSTANT
    heat_of_adsorption = block_average_from_blocks(hoa_blocks)

    weights = N_blocks / N_blocks.sum(axis=1, keepdims=True)
    total_hoa_blocks = (hoa_blocks * weights).sum(axis=1)
    total_heat_of_adsorption = block_average_from_blocks(total_hoa_blocks)

    def _stress_and_pressure(
        s: Array | None,
    ) -> tuple[BlockAverageResult | None, BlockAverageResult | None]:
        if s is None:
            return None, None
        return (
            block_average(s, n_blocks=n_blocks_used),
            block_average(jnp.trace(s, axis1=-2, axis2=-1) / 3, n_blocks=n_blocks_used),
        )

    stress_result, pressure_result = _stress_and_pressure(stress)
    sp_result, pp_result = _stress_and_pressure(stress_potential)
    st_result, pt_result = _stress_and_pressure(stress_tail_correction)
    si_result, pi_result = _stress_and_pressure(stress_ideal_gas)

    return MCMCAnalysisResult(
        energy=energy_result,
        loading=loading,
        heat_of_adsorption=heat_of_adsorption,
        total_heat_of_adsorption=total_heat_of_adsorption,
        stress=stress_result,
        stress_potential=sp_result,
        stress_tail_correction=st_result,
        stress_ideal_gas=si_result,
        pressure=pressure_result,
        pressure_potential=pp_result,
        pressure_tail_correction=pt_result,
        pressure_ideal_gas=pi_result,
    )


@no_jax_tracing
def analyze_mcmc(
    fixed: IsMCMCFixedData,
    per_step: IsMCMCStepData,
    n_blocks: int | None = None,
) -> dict[SystemId, MCMCAnalysisResult]:
    """Analyze MCMC simulation results from pre-loaded data.

    Computes energy, loading (average particle counts), and heat of adsorption
    per species using block averaging for error estimation. Analysis is
    performed independently for each system.

    Args:
        fixed: Fixed (one-shot) logged data containing system metadata.
        per_step: Per-step logged data containing energies and particle counts.
        n_blocks: Number of blocks for error estimation. ``None`` uses
            :func:`~kups.core.utils.block_average.optimal_block_average`.

    Returns:
        Per-system analysis results keyed by ``SystemId``.
    """
    system_keys = fixed.systems.keys
    count_keys = per_step.particle_count.keys
    count_data = per_step.particle_count.data
    temperatures = fixed.systems.data.temperature
    all_energy = per_step.systems.data.potential_energy
    guest_stress = per_step.systems.data.guest_stress
    stress_potential = guest_stress.potential
    stress_tail = guest_stress.tail_correction
    stress_ideal = guest_stress.ideal_gas
    all_stress = stress_potential + stress_tail + stress_ideal

    results: dict[SystemId, MCMCAnalysisResult] = {}
    for i, sys_id in enumerate(system_keys):
        motif_cols = [j for j, label in enumerate(count_keys) if label[0] == sys_id]
        counts = count_data[:, motif_cols].reshape(-1, len(motif_cols))
        energy = all_energy[:, i].reshape(-1)
        s = all_stress[:, i]
        has_stress = bool(jnp.any(s != 0))
        temperature = float(temperatures[i])
        results[sys_id] = _analyze_single_system(
            energy,
            counts,
            temperature,
            n_blocks,
            stress=s if has_stress else None,
            stress_potential=stress_potential[:, i]
            if has_stress and stress_potential is not None
            else None,
            stress_tail_correction=stress_tail[:, i]
            if has_stress and stress_tail is not None
            else None,
            stress_ideal_gas=stress_ideal[:, i]
            if has_stress and stress_ideal is not None
            else None,
        )

    return results


@plain_dataclass
class WidomAnalysisResult:
    """Block-averaged Widom outputs for a single system.

    Attributes:
        excess_chemical_potential: $\\mu^\\mathrm{ex}$ with SEM (eV).
        henry_coefficient: $K_H$ with SEM (Å³/eV).
        heat_of_adsorption: $q_\\mathrm{st}$ with SEM (eV); Vlugt 2008 eq. 16,
            test-particle limit.
    """

    excess_chemical_potential: BlockAverageResult
    henry_coefficient: BlockAverageResult
    heat_of_adsorption: BlockAverageResult


@no_jax_tracing
def _per_cycle_widom_means(
    per_step: Table[SystemId, WidomStatistics],
) -> tuple[Array, Array]:
    """Difference cumulative-cycle snapshots into per-cycle ⟨W⟩ and ⟨ΔU·W⟩.

    Returns arrays of shape ``(n_cycles, n_systems)``.
    """
    sum_w = per_step.data.sum_boltzmann
    sum_du_w = per_step.data.sum_delta_u_boltzmann
    n = per_step.data.n_samples.astype(jnp.float64)

    zero_w = jnp.zeros_like(sum_w[:1])
    zero_du = jnp.zeros_like(sum_du_w[:1])
    zero_n = jnp.zeros_like(n[:1])
    delta_w = jnp.diff(jnp.concatenate([zero_w, sum_w], axis=0), axis=0)
    delta_du_w = jnp.diff(jnp.concatenate([zero_du, sum_du_w], axis=0), axis=0)
    delta_n = jnp.diff(jnp.concatenate([zero_n, n], axis=0), axis=0)
    return delta_w / delta_n, delta_du_w / delta_n


@no_jax_tracing
def _analyze_single_widom_system(
    mean_w_per_cycle: Array,
    mean_du_w_per_cycle: Array,
    temperature: float,
    volume: float,
    n_blocks: int | None,
) -> WidomAnalysisResult:
    """Block-average per-cycle Widom samples into μ_ex / K_H / q_st with SEM.

    μ_ex, K_H, q_st are derived from the block-averaged means of W and ΔU·W
    via the canonical formulas; per-block ratio-then-average estimators are
    biased (Jensen) and would not converge to the canonical quantities.
    """
    if n_blocks is None:
        w_result = optimal_block_average(mean_w_per_cycle)
        n_used = int(w_result.n_blocks)
    else:
        w_result = block_average(mean_w_per_cycle, n_blocks=n_blocks)
        n_used = n_blocks

    block_w = compute_block_means(mean_w_per_cycle, n_used)
    block_du_w = compute_block_means(mean_du_w_per_cycle, n_used)

    mean_w = jnp.mean(block_w)
    mean_du_w = jnp.mean(block_du_w)
    var_w = jnp.var(block_w, ddof=1)
    var_du_w = jnp.var(block_du_w, ddof=1)
    # Block-level covariance for the ratio's delta-method SE.
    cov_w_du_w = (
        jnp.mean((block_w - mean_w) * (block_du_w - mean_du_w)) * n_used / (n_used - 1)
    )

    kT = temperature * BOLTZMANN_CONSTANT

    mu_ex = -kT * jnp.log(mean_w)
    se_mu_ex = kT * w_result.sem / mean_w

    k_h = volume * mean_w / kT
    se_k_h = volume * w_result.sem / kT

    # q_st = kT - ⟨ΔU·W⟩/⟨W⟩; delta method on f(a, b) = -a/b at (mean_du_w, mean_w).
    inv_w = 1.0 / mean_w
    ratio = mean_du_w * inv_w
    q_st = kT - ratio
    var_q_st = (
        inv_w**2 * var_du_w
        + ratio**2 * inv_w**2 * var_w
        - 2 * ratio * inv_w**2 * cov_w_du_w
    ) / n_used
    se_q_st = jnp.sqrt(jnp.maximum(var_q_st, 0.0))

    return WidomAnalysisResult(
        excess_chemical_potential=BlockAverageResult(mu_ex, se_mu_ex, n_used),
        henry_coefficient=BlockAverageResult(k_h, se_k_h, n_used),
        heat_of_adsorption=BlockAverageResult(q_st, se_q_st, n_used),
    )


@no_jax_tracing
def analyze_widom(
    fixed: WidomFixedData,
    per_step: Table[SystemId, WidomStatistics],
    n_blocks: int | None = None,
) -> dict[SystemId, WidomAnalysisResult]:
    """Block-averaged Widom analysis from pre-loaded HDF5 data.

    Args:
        fixed: Fixed-group data carrying per-system temperature and cell.
        per_step: Cumulative ``WidomStatistics`` time series.
        n_blocks: Number of blocks; ``None`` uses ``optimal_block_average``.
    """
    mean_w, mean_du_w = _per_cycle_widom_means(per_step)
    temperatures = fixed.systems.data.temperature
    volumes = fixed.systems.data.cell.volume

    results: dict[SystemId, WidomAnalysisResult] = {}
    for i, sys_id in enumerate(fixed.systems.keys):
        results[sys_id] = _analyze_single_widom_system(
            mean_w[:, i],
            mean_du_w[:, i],
            float(temperatures[i]),
            float(volumes[i]),
            n_blocks,
        )
    return results


@no_jax_tracing
def analyze_widom_file(
    hdf5_path: str | Path,
    n_blocks: int | None = None,
) -> dict[SystemId, WidomAnalysisResult]:
    """Analyze a ``kups_mcmc_widom`` HDF5 output."""
    with HDF5StorageReader[WidomLoggedData[Any]](hdf5_path) as reader:
        fixed = reader.focus_group(lambda s: s.fixed)[...]
        per_step = reader.focus_group(lambda s: s.per_step)[...]
    return analyze_widom(fixed, per_step, n_blocks)


@no_jax_tracing
def analyze_mcmc_file(
    hdf5_path: str | Path,
    n_blocks: int | None = None,
) -> dict[SystemId, MCMCAnalysisResult]:
    """Analyze MCMC simulation results from an HDF5 file.

    Convenience wrapper that reads the HDF5 file and delegates to
    :func:`analyze_mcmc`.

    Args:
        hdf5_path: Path to HDF5 output file from
            :func:`~kups.application.mcmc.simulation.run_mcmc`.
        n_blocks: Number of blocks for error estimation. ``None`` uses
            :func:`~kups.core.utils.block_average.optimal_block_average`.

    Returns:
        Per-system analysis results keyed by ``SystemId``.
    """
    with HDF5StorageReader[MCMCLoggedData[Any]](hdf5_path) as reader:
        fixed = reader.focus_group(lambda s: s.fixed)[...]
        per_step = reader.focus_group(lambda s: s.per_step)[...]

    return analyze_mcmc(fixed, per_step, n_blocks)
