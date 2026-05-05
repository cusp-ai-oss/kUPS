# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Post-simulation analysis for rigid-body MD.

Mirrors :mod:`kups.application.md.analysis` but reads the rigid-MD HDF5
schema (atom positions, per-group COM/quaternion/momenta, total kinetic
energy already aggregated to the system level).
"""

from __future__ import annotations

from dataclasses import dataclass as plain_dataclass
from pathlib import Path
from typing import Protocol

import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.application.md.rigid_logging import RigidMDLoggedData
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data import Index, Table
from kups.core.storage import HDF5StorageReader
from kups.core.typing import (
    HasDegreesOfFreedom,
    HasPositions,
    ParticleId,
    SystemId,
)
from kups.core.utils.block_average import (
    BlockAverageResult,
    block_average,
    optimal_block_average,
)


class _IsRigidMDAtoms(HasPositions, Protocol):
    @property
    def system(self) -> Index[SystemId]: ...


class IsRigidMDInitData(Protocol):
    """Contract for the init reader group."""

    @property
    def atoms(self) -> Table[ParticleId, _IsRigidMDAtoms]: ...
    @property
    def systems(self) -> Table[SystemId, HasDegreesOfFreedom]: ...


class IsRigidMDStepData(Protocol):
    """Contract for the step reader group."""

    @property
    def potential_energy(self) -> Array: ...
    @property
    def kinetic_energy(self) -> Array: ...
    @property
    def stress_tensor(self) -> Array: ...
    @property
    def volume(self) -> Array: ...


@plain_dataclass
class RigidMDAnalysisResult:
    """Block-averaged thermodynamic estimates for one rigid-MD system.

    Attributes:
        potential_energy: ⟨U⟩ ± SEM (eV).
        kinetic_energy: ⟨K⟩ ± SEM (eV).
        total_energy: ⟨E⟩ ± SEM (eV).
        temperature: ⟨T⟩ ± SEM (K).
        pressure: ⟨P⟩ ± SEM (Pa).
        volume: ⟨V⟩ ± SEM (Å³).
        energy_drift: Linear drift slope of total energy (eV/step).
        energy_drift_per_atom: Drift normalised by atom count.
        n_atoms: Number of atoms.
        degrees_of_freedom: DOF used for the kinetic temperature.
        n_steps: Number of production steps analysed.
    """

    potential_energy: BlockAverageResult
    kinetic_energy: BlockAverageResult
    total_energy: BlockAverageResult
    temperature: BlockAverageResult
    pressure: BlockAverageResult
    volume: BlockAverageResult
    energy_drift: float
    energy_drift_per_atom: float
    n_atoms: int
    degrees_of_freedom: float
    n_steps: int


def _analyze_single_system(
    potential_energy: Array,
    kinetic_energy: Array,
    stress_tensor: Array,
    volume: Array,
    n_atoms: int,
    dof: float,
    n_blocks: int | None,
) -> RigidMDAnalysisResult:
    n_steps = int(potential_energy.shape[0])
    total_energy = potential_energy + kinetic_energy
    # Kinetic temperature uses the per-system DOF (already accounting for
    # rotational vs. translational DOF at build time).
    temperature = 2 * kinetic_energy / (BOLTZMANN_CONSTANT * dof)
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
    vol_result = block_average(volume, n_blocks=n_blocks_used)

    steps = np.arange(n_steps)
    slope, _ = np.polyfit(steps, np.asarray(total_energy), 1)

    return RigidMDAnalysisResult(
        potential_energy=pe_result,
        kinetic_energy=ke_result,
        total_energy=te_result,
        temperature=temp_result,
        pressure=pressure_result,
        volume=vol_result,
        energy_drift=float(slope),
        energy_drift_per_atom=float(slope) / max(n_atoms, 1),
        n_atoms=n_atoms,
        degrees_of_freedom=dof,
        n_steps=n_steps,
    )


def analyze_rigid_md(
    init_data: IsRigidMDInitData,
    step_data: IsRigidMDStepData,
    n_blocks: int | None = None,
) -> dict[SystemId, RigidMDAnalysisResult]:
    """Analyse rigid-MD trajectories from pre-loaded data.

    Args:
        init_data: Initial state with atoms (for system membership) and
            systems (for DOF lookup).
        step_data: Per-step trajectory with shape ``(n_steps, n_systems, …)``.
        n_blocks: Block count for SEM. ``None`` selects optimally.

    Returns:
        Per-system results keyed by ``SystemId``.
    """
    system_index = init_data.atoms.data.system
    n_atoms_per_system = system_index.counts.data
    dofs = init_data.systems.data.degrees_of_freedom

    results: dict[SystemId, RigidMDAnalysisResult] = {}
    for i, sys_id in enumerate(system_index.keys):
        results[sys_id] = _analyze_single_system(
            potential_energy=step_data.potential_energy[:, i],
            kinetic_energy=step_data.kinetic_energy[:, i],
            stress_tensor=step_data.stress_tensor[:, i],
            volume=step_data.volume[:, i],
            n_atoms=int(n_atoms_per_system[i]),
            dof=float(dofs[i]),
            n_blocks=n_blocks,
        )

    return results


def analyze_rigid_md_file(
    hdf5_path: str | Path,
    n_blocks: int | None = None,
) -> dict[SystemId, RigidMDAnalysisResult]:
    """Read a rigid-MD HDF5 file and run :func:`analyze_rigid_md`."""
    with HDF5StorageReader[RigidMDLoggedData](hdf5_path) as reader:
        init_data = reader.focus_group(lambda state: state.init)[...]
        step_data = reader.focus_group(lambda state: state.step)[...]

    return analyze_rigid_md(init_data, step_data, n_blocks=n_blocks)
