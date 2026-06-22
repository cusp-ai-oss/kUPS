# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Post-simulation analysis for structure relaxation."""

from __future__ import annotations

from dataclasses import dataclass as plain_dataclass
from pathlib import Path
from typing import Protocol

from jax import Array

from kups.application.relaxation.logging import RelaxLoggedData
from kups.core.data import Table
from kups.core.storage import HDF5StorageReader
from kups.core.typing import HasPotentialEnergy, SystemId


class IsRelaxStepData(HasPotentialEnergy, Protocol):
    """Contract for a single relaxation step from the reader."""

    @property
    def max_force(self) -> Array: ...


@plain_dataclass
class RelaxAnalysisResult:
    """Summary of a completed relaxation for a single system.

    Attributes:
        final_energy: Final potential energy (eV).
        final_max_force: Maximum atomic force at the last step (eV/Ang).
        n_steps: Number of steps actually taken.
    """

    final_energy: float
    final_max_force: float
    n_steps: int


class _IsRelaxInitData(Protocol):
    @property
    def systems(self) -> Table[SystemId, object]: ...


def _analyze_relax(
    system_keys: tuple[SystemId, ...],
    final_energy: Array,
    final_max_force: Array,
    n_steps: int,
) -> dict[SystemId, RelaxAnalysisResult]:
    """Build per-system results from the final-step energy and force arrays."""
    return {
        sys_id: RelaxAnalysisResult(
            final_energy=float(final_energy[i]),
            final_max_force=float(final_max_force[i]),
            n_steps=n_steps,
        )
        for i, sys_id in enumerate(system_keys)
    }


def analyze_relax(
    init_data: _IsRelaxInitData,
    step_data: IsRelaxStepData,
    n_steps: int,
) -> dict[SystemId, RelaxAnalysisResult]:
    """Extract per-system analysis results from a relaxation step.

    Args:
        init_data: Initial state providing system keys.
        step_data: Final step data with per-system potential_energy and
            max_force.
        n_steps: Number of steps actually taken.

    Returns:
        Per-system relaxation results keyed by ``SystemId``.
    """
    return _analyze_relax(
        init_data.systems.keys,
        step_data.potential_energy,
        step_data.max_force,
        n_steps,
    )


def analyze_relax_file(
    hdf5_path: str | Path,
) -> dict[SystemId, RelaxAnalysisResult]:
    """Analyse relaxation results from an HDF5 file.

    Reads only the system keys and the final step's per-system
    ``potential_energy`` and ``max_force``, leaving the per-atom trajectory on
    disk.

    Args:
        hdf5_path: Path to HDF5 output from a relaxation run.

    Returns:
        Per-system relaxation results keyed by ``SystemId``.
    """

    with HDF5StorageReader[RelaxLoggedData](hdf5_path) as reader:
        n_steps = int(reader.file.attrs.get("actual_steps", -1))
        system_keys = (
            reader.focus_group(lambda s: s.init).read(select=lambda d: d.systems).keys
        )
        final_energy, final_max_force = reader.focus_group(lambda s: s.step).read(
            n_steps - 1, select=lambda d: (d.potential_energy, d.max_force)
        )

    return _analyze_relax(system_keys, final_energy, final_max_force, n_steps)
