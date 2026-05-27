# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Post-simulation analysis for molecular dynamics."""

from __future__ import annotations

from dataclasses import dataclass as plain_dataclass
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.application.md.logging import MDLoggedData
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Index, Table
from kups.core.storage import HDF5StorageReader
from kups.core.typing import (
    HasMasses,
    HasMomenta,
    HasPositions,
    HasPotentialEnergy,
    HasStressTensor,
    ParticleId,
    SystemId,
)
from kups.core.utils.block_average import (
    BlockAverageResult,
    block_average,
    optimal_block_average,
)
from kups.md.observables import (
    particle_kinetic_energy,
    remove_center_of_mass_momentum,
)


class _IsMDAtoms(HasPositions, Protocol):
    @property
    def system(self) -> Index[SystemId]: ...


class _IsMDStepAtoms(HasMomenta, HasMasses, Protocol): ...


class IsMDInitData(Protocol):
    """Contract for the init reader group."""

    @property
    def atoms(self) -> Table[ParticleId, _IsMDAtoms]: ...


class IsMDStepData(HasPotentialEnergy, HasStressTensor, Protocol):
    """Contract for the step reader group."""

    @property
    def atoms(self) -> Table[ParticleId, _IsMDStepAtoms]: ...

    @property
    def kinetic_energy(self) -> Array: ...

    @property
    def volume(self) -> Array: ...


@plain_dataclass
class MDAnalysisResult:
    """Results from MD simulation analysis for a single system.

    Attributes:
        potential_energy: Average potential energy with SEM (eV).
        kinetic_energy: Average kinetic energy with SEM (eV).
        total_energy: Average total energy with SEM (eV).
        temperature: Average temperature with SEM (K).
        energy_drift: Linear drift rate of total energy (eV/step).
        energy_drift_per_atom: Energy drift normalized by number of atoms.
        pressure: Average pressure with SEM (Pa).
        volume: Average cell volume with SEM (A^3).
        n_atoms: Number of atoms in this system.
        n_steps: Number of simulation steps analyzed.
    """

    potential_energy: BlockAverageResult
    kinetic_energy: BlockAverageResult
    total_energy: BlockAverageResult
    temperature: BlockAverageResult
    energy_drift: float
    energy_drift_per_atom: float
    pressure: BlockAverageResult
    volume: BlockAverageResult
    n_atoms: int
    n_steps: int


def _analyze_single_system(
    potential_energy: Array,
    kinetic_energy: Array,
    stress_tensor: Array,
    volume: Array,
    n_atoms: int,
    n_blocks: int | None,
    internal_kinetic_energy: Array,
) -> MDAnalysisResult:
    """Run block-averaging analysis for one system.

    Args:
        potential_energy: Potential energy time series, shape ``(n_steps,)``.
        kinetic_energy: Kinetic energy time series, shape ``(n_steps,)``.
        stress_tensor: Stress tensor time series, shape ``(n_steps, 3, 3)``.
        volume: Cell volume time series, shape ``(n_steps,)``.
        n_atoms: Number of atoms in this system.
        n_blocks: Number of blocks, or ``None`` for automatic selection.
        internal_kinetic_energy: Center-of-mass-projected kinetic energy time
            series to use when computing internal temperature. The logged
            ``kinetic_energy`` and ``total_energy`` are left unchanged.
    """
    degrees_of_freedom = 3 * n_atoms - 3
    total_energy = potential_energy + kinetic_energy
    n_steps = len(total_energy)

    temperature = (
        2 * internal_kinetic_energy / (BOLTZMANN_CONSTANT * degrees_of_freedom)
    )
    pressure = jnp.trace(stress_tensor, axis1=-2, axis2=-1) / 3

    if n_blocks is None:
        pressure_result = optimal_block_average(pressure)
        n_blocks_used = int(pressure_result.n_blocks)
    else:
        pressure_result = block_average(pressure, n_blocks=n_blocks)
        n_blocks_used = n_blocks

    pe_result = block_average(potential_energy, n_blocks=n_blocks_used)
    ke_result = block_average(kinetic_energy, n_blocks=n_blocks_used)
    te_result = block_average(total_energy, n_blocks=n_blocks_used)
    temp_result = block_average(temperature, n_blocks=n_blocks_used)
    volume_result = block_average(volume, n_blocks=n_blocks_used)

    steps = np.arange(n_steps)
    slope, _ = np.polyfit(steps, np.asarray(total_energy), 1)

    return MDAnalysisResult(
        potential_energy=pe_result,
        kinetic_energy=ke_result,
        total_energy=te_result,
        temperature=temp_result,
        energy_drift=float(slope),
        energy_drift_per_atom=float(slope) / n_atoms,
        pressure=pressure_result,
        volume=volume_result,
        n_atoms=n_atoms,
        n_steps=n_steps,
    )


def _internal_kinetic_energy(
    atoms: Table[ParticleId, _IsMDStepAtoms],
    system: Index[SystemId],
    system_number: int,
) -> Array:
    """Compute per-step internal kinetic energy for one system."""
    mask = system.indices == system_number
    momenta = atoms.data.momenta[:, mask, :]
    masses = atoms.data.masses[:, mask]
    local_system = Index.zeros(momenta.shape[1], label=SystemId)
    projected_momenta = jax.vmap(
        remove_center_of_mass_momentum,
        in_axes=(0, 0, None),
    )(momenta, masses, local_system)

    return jnp.sum(particle_kinetic_energy(projected_momenta, masses), axis=-1)


def analyze_md(
    init_data: IsMDInitData,
    step_data: IsMDStepData,
    n_blocks: int | None = None,
) -> dict[SystemId, MDAnalysisResult]:
    """Analyze MD simulation from pre-loaded data.

    Computes thermodynamic averages and energy conservation metrics
    independently for each system.

    Args:
        init_data: Initial simulation state with atom positions and system index.
        step_data: Per-step thermodynamic data with shape ``(n_steps, n_systems)``.
        n_blocks: Number of blocks for error estimation. If None, uses
            optimal_block_average to auto-select.

    Returns:
        Per-system analysis results keyed by ``SystemId``.
    """
    system_index = init_data.atoms.data.system
    n_atoms_per_system = system_index.counts.data

    results: dict[SystemId, MDAnalysisResult] = {}
    for i, sys_id in enumerate(system_index.keys):
        kinetic_energy = step_data.kinetic_energy[:, i]
        internal_ke = kinetic_energy
        internal_ke = _internal_kinetic_energy(step_data.atoms, system_index, i)
        results[sys_id] = _analyze_single_system(
            potential_energy=step_data.potential_energy[:, i],
            kinetic_energy=kinetic_energy,
            stress_tensor=step_data.stress_tensor[:, i],
            volume=step_data.volume[:, i],
            n_atoms=int(n_atoms_per_system[i]),
            n_blocks=n_blocks,
            internal_kinetic_energy=internal_ke,
        )

    return results


def analyze_md_file(
    hdf5_path: str | Path,
    n_blocks: int | None = None,
) -> dict[SystemId, MDAnalysisResult]:
    """Analyze MD simulation results from HDF5 file.

    Convenience wrapper that reads HDF5 and delegates to ``analyze_md``.

    Args:
        hdf5_path: Path to HDF5 file from MD simulation.
        n_blocks: Number of blocks for error estimation. If None, uses
            optimal_block_average to auto-select.

    Returns:
        Per-system analysis results keyed by ``SystemId``.
    """

    with HDF5StorageReader[MDLoggedData](hdf5_path) as reader:
        init_data = reader.focus_group(lambda state: state.init)[...]
        step_data = reader.focus_group(lambda state: state.step)[...]

    return analyze_md(init_data, step_data, n_blocks=n_blocks)
