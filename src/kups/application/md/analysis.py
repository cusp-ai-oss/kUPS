# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Post-simulation analysis for molecular dynamics."""

from __future__ import annotations

from dataclasses import dataclass as plain_dataclass
from pathlib import Path
from typing import Protocol

import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.application.md.logging import MDLoggedData
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Index, Table
from kups.core.storage import HDF5StorageReader
from kups.core.typing import HasPotentialEnergy, HasStressTensor, ParticleId, SystemId
from kups.core.utils.block_average import (
    BlockAverageResult,
    block_average,
    optimal_block_average,
)


class _IsMDInitAtoms(Protocol):
    @property
    def system(self) -> Index[SystemId]: ...


class IsMDInitData(Protocol):
    """Contract for the init reader group."""

    @property
    def atoms(self) -> Table[ParticleId, _IsMDInitAtoms]: ...


class IsMDStepData(HasPotentialEnergy, HasStressTensor, Protocol):
    """Contract for the step reader group."""

    @property
    def kinetic_energy(self) -> Array: ...

    @property
    def internal_kinetic_energy(self) -> Array: ...

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
    internal_kinetic_energy: Array,
    stress_tensor: Array,
    volume: Array,
    n_atoms: int,
    n_blocks: int | None,
) -> MDAnalysisResult:
    """Block-average one system's time series into thermodynamic averages.

    Args:
        potential_energy: Potential energy, shape ``(n_steps,)``.
        kinetic_energy: Total kinetic energy, shape ``(n_steps,)``.
        internal_kinetic_energy: Center-of-mass-projected kinetic energy used for
            the temperature, shape ``(n_steps,)``.
        stress_tensor: Stress tensor, shape ``(n_steps, 3, 3)``.
        volume: Cell volume, shape ``(n_steps,)``.
        n_atoms: Number of atoms in the system.
        n_blocks: Number of blocks, or ``None`` to auto-select from the pressure.

    Returns:
        Block-averaged thermodynamic results for the system.
    """
    total_energy = potential_energy + kinetic_energy
    dof = 3 * n_atoms - 3
    temperature = 2 * internal_kinetic_energy / (BOLTZMANN_CONSTANT * dof)
    pressure = jnp.trace(stress_tensor, axis1=-2, axis2=-1) / 3

    if n_blocks is None:
        pressure_result = optimal_block_average(pressure)
        n_blocks = int(pressure_result.n_blocks)
    else:
        pressure_result = block_average(pressure, n_blocks=n_blocks)

    slope, _ = np.polyfit(np.arange(len(total_energy)), np.asarray(total_energy), 1)

    return MDAnalysisResult(
        potential_energy=block_average(potential_energy, n_blocks=n_blocks),
        kinetic_energy=block_average(kinetic_energy, n_blocks=n_blocks),
        total_energy=block_average(total_energy, n_blocks=n_blocks),
        temperature=block_average(temperature, n_blocks=n_blocks),
        energy_drift=float(slope),
        energy_drift_per_atom=float(slope) / n_atoms,
        pressure=pressure_result,
        volume=block_average(volume, n_blocks=n_blocks),
        n_atoms=n_atoms,
        n_steps=len(total_energy),
    )


def _analyze_md_systems(
    system: Index[SystemId],
    potential_energy: Array,
    kinetic_energy: Array,
    internal_kinetic_energy: Array,
    stress_tensor: Array,
    volume: Array,
    n_blocks: int | None,
) -> dict[SystemId, MDAnalysisResult]:
    """Block-average each system independently.

    Args:
        system: Per-atom system index supplying keys and atom counts.
        potential_energy: Potential energy, shape ``(n_steps, n_systems)``.
        kinetic_energy: Total kinetic energy, shape ``(n_steps, n_systems)``.
        internal_kinetic_energy: Center-of-mass-projected kinetic energy, shape
            ``(n_steps, n_systems)``.
        stress_tensor: Stress tensor, shape ``(n_steps, n_systems, 3, 3)``.
        volume: Cell volume, shape ``(n_steps, n_systems)``.
        n_blocks: Number of blocks, or ``None`` for automatic selection.

    Returns:
        Per-system analysis results keyed by ``SystemId``.
    """
    n_atoms_per_system = system.counts.data
    return {
        sys_id: _analyze_single_system(
            potential_energy[:, i],
            kinetic_energy[:, i],
            internal_kinetic_energy[:, i],
            stress_tensor[:, i],
            volume[:, i],
            int(n_atoms_per_system[i]),
            n_blocks,
        )
        for i, sys_id in enumerate(system.keys)
    }


def analyze_md(
    init_data: IsMDInitData,
    step_data: IsMDStepData,
    n_blocks: int | None = None,
) -> dict[SystemId, MDAnalysisResult]:
    """Analyze MD simulation from pre-loaded data.

    Computes thermodynamic averages and energy conservation metrics
    independently for each system.

    Args:
        init_data: Initial simulation state providing the per-atom system index.
        step_data: Per-step thermodynamic data with shape ``(n_steps, n_systems)``.
        n_blocks: Number of blocks for error estimation. ``None`` auto-selects via
            ``optimal_block_average``.

    Returns:
        Per-system analysis results keyed by ``SystemId``.
    """
    return _analyze_md_systems(
        init_data.atoms.data.system,
        step_data.potential_energy,
        step_data.kinetic_energy,
        step_data.internal_kinetic_energy,
        step_data.stress_tensor,
        step_data.volume,
        n_blocks,
    )


def analyze_md_file(
    hdf5_path: str | Path,
    n_blocks: int | None = None,
) -> dict[SystemId, MDAnalysisResult]:
    """Analyze MD simulation results from an HDF5 file.

    Reads only the per-step thermodynamic scalars (energies, stress, volume) and
    the initial system index, leaving the per-atom trajectory on disk. Internal
    kinetic energy is logged per step, so no momenta are read here.

    Args:
        hdf5_path: Path to HDF5 file from MD simulation.
        n_blocks: Number of blocks for error estimation. ``None`` auto-selects via
            ``optimal_block_average``.

    Returns:
        Per-system analysis results keyed by ``SystemId``.
    """
    with HDF5StorageReader[MDLoggedData](hdf5_path) as reader:
        system = reader.focus_group(lambda s: s.init).read(
            select=lambda d: d.atoms.data.system
        )
        potential_energy, kinetic_energy, internal_kinetic_energy, stress, volume = (
            reader.focus_group(lambda s: s.step).read(
                select=lambda d: (
                    d.potential_energy,
                    d.kinetic_energy,
                    d.internal_kinetic_energy,
                    d.stress_tensor,
                    d.volume,
                )
            )
        )
    return _analyze_md_systems(
        system,
        potential_energy,
        kinetic_energy,
        internal_kinetic_energy,
        stress,
        volume,
        n_blocks,
    )
