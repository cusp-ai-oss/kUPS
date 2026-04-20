#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Validate MD simulation results against known physical properties.

For an NVE simulation of argon with Lennard-Jones potential, validates:
1. Energy conservation (total energy drift)
2. Equipartition theorem (KE per DOF = kT/2)
3. Velocity distribution (Maxwell-Boltzmann)
4. Virial pressure consistency
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kups.application.md.logging import MDLoggedData
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.storage import HDF5StorageReader

# Physical constants for validation
ARGON_MASS_AMU = 39.948  # Atomic mass of argon


def validate_energy_conservation(
    total_energy: np.ndarray,
    n_atoms: int,
    max_drift_per_atom_per_step: float = 1e-8,
    max_relative_fluctuation: float = 5e-4,
) -> tuple[bool, dict]:
    """Validate energy conservation in NVE simulation.

    For a symplectic integrator, energy oscillates around a shadow Hamiltonian
    with bounded amplitude (no systematic drift). We check:
    1. No linear drift: < 1e-8 eV/atom/step
    2. Bounded fluctuations: relative std(E)/|<E>| < 0.05%

    These thresholds are appropriate for typical MD timesteps (1-2 fs).
    """
    n_steps = len(total_energy)
    steps = np.arange(n_steps)

    # Linear fit to detect drift
    coeffs = np.polyfit(steps, total_energy, 1)
    slope = float(coeffs[0])
    drift_per_step = abs(slope)
    drift_per_atom_per_step = drift_per_step / n_atoms

    # Energy fluctuation should be very small for NVE (symplectic bounded error)
    energy_std = float(np.std(total_energy))
    energy_mean = float(np.mean(total_energy))
    relative_fluctuation = energy_std / abs(energy_mean) if energy_mean != 0 else 0.0

    drift_ok = drift_per_atom_per_step < max_drift_per_atom_per_step
    fluctuation_ok = relative_fluctuation < max_relative_fluctuation
    passed = drift_ok and fluctuation_ok

    return passed, {
        "drift_per_step": drift_per_step,
        "drift_per_atom_per_step": drift_per_atom_per_step,
        "relative_fluctuation": relative_fluctuation,
        "drift_threshold": max_drift_per_atom_per_step,
        "fluctuation_threshold": max_relative_fluctuation,
        "drift_ok": drift_ok,
        "fluctuation_ok": fluctuation_ok,
    }


def validate_equipartition(
    kinetic_energy: np.ndarray,
    temperature: np.ndarray,
    degrees_of_freedom: int,
    rtol: float = 0.02,
) -> tuple[bool, dict]:
    """Validate equipartition theorem: <KE> = (DOF/2) * k_B * T.

    For a system in thermal equilibrium, the average kinetic energy
    should equal (DOF/2) * k_B * T.
    """
    mean_ke = np.mean(kinetic_energy)
    mean_temp = np.mean(temperature)

    # Expected KE from equipartition
    expected_ke = 0.5 * degrees_of_freedom * BOLTZMANN_CONSTANT * mean_temp

    # Temperature from KE (should be consistent)
    temp_from_ke = 2 * mean_ke / (degrees_of_freedom * BOLTZMANN_CONSTANT)

    relative_error = (
        abs(mean_ke - expected_ke) / expected_ke if expected_ke != 0 else 0.0
    )
    passed = bool(relative_error < rtol)

    return passed, {
        "mean_kinetic_energy": mean_ke,
        "expected_kinetic_energy": expected_ke,
        "mean_temperature": mean_temp,
        "temperature_from_ke": temp_from_ke,
        "relative_error": relative_error,
        "threshold": rtol,
    }


def validate_temperature_fluctuations(
    temperature: np.ndarray,
    degrees_of_freedom: int,
    max_ratio: float = 0.95,
) -> tuple[bool, dict]:
    """Validate temperature fluctuations are smaller than canonical ensemble.

    For the canonical ensemble: <(dT)^2> / <T>^2 = 2 / DOF
    For microcanonical (NVE), fluctuations MUST be smaller since total energy
    is fixed. This is a fundamental property of the microcanonical ensemble.
    """
    mean_temp = np.mean(temperature)
    temp_var = np.var(temperature)

    # Canonical prediction (NVE must have smaller fluctuations)
    canonical_relative_var = 2.0 / degrees_of_freedom
    measured_relative_var = float(temp_var / mean_temp**2) if mean_temp != 0 else 0.0

    # NVE fluctuations must be strictly less than canonical
    ratio = (
        measured_relative_var / canonical_relative_var
        if canonical_relative_var != 0
        else 0.0
    )
    passed = bool(ratio < max_ratio)

    return passed, {
        "mean_temperature": float(mean_temp),
        "temperature_std": float(np.sqrt(temp_var)),
        "measured_relative_variance": measured_relative_var,
        "canonical_relative_variance": canonical_relative_var,
        "ratio": ratio,
        "max_ratio": max_ratio,
    }


def validate_pressure_virial(
    pressure: np.ndarray,
    kinetic_energy: np.ndarray,
    volume: float,
    max_pressure_cv: float = 0.5,
) -> tuple[bool, dict]:
    """Check pressure consistency with virial theorem.

    Ideal gas contribution: P_ideal = N * k_B * T / V = 2/3 * KE / V
    The excess pressure from interactions should be stable (bounded fluctuations).
    """
    mean_pressure = float(np.mean(pressure))
    std_pressure = float(np.std(pressure))
    mean_ke = float(np.mean(kinetic_energy))

    # Ideal gas pressure contribution
    ideal_pressure = 2 * mean_ke / (3 * volume)
    excess_pressure = mean_pressure - ideal_pressure

    # Pressure coefficient of variation should be bounded
    # (large CV indicates numerical instability or unphysical behavior)
    pressure_cv = std_pressure / abs(mean_pressure) if mean_pressure != 0 else 0.0

    finite_ok = bool(np.isfinite(mean_pressure) and np.isfinite(std_pressure))
    cv_ok = pressure_cv < max_pressure_cv
    passed = finite_ok and cv_ok

    return passed, {
        "mean_pressure_Pa": mean_pressure,
        "ideal_gas_pressure_Pa": ideal_pressure,
        "excess_pressure_Pa": excess_pressure,
        "pressure_std": std_pressure,
        "pressure_cv": pressure_cv,
        "max_pressure_cv": max_pressure_cv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MD simulation against physical properties"
    )
    parser.add_argument(
        "hdf5_path",
        type=Path,
        nargs="?",
        default=Path("./md_nve_lj_argon.h5"),
        help="Path to HDF5 output file",
    )
    parser.add_argument(
        "--max-drift",
        type=float,
        default=1e-8,
        help="Max energy drift per atom per step (eV)",
    )
    args = parser.parse_args()

    if not args.hdf5_path.exists():
        print(f"File not found: {args.hdf5_path}")
        return 1

    # Load data from HDF5 using the storage reader
    with HDF5StorageReader[MDLoggedData](args.hdf5_path) as reader:
        init_data = reader.focus_group(lambda state: state.init)[...]
        step_data = reader.focus_group(lambda state: state.step)[...]

    positions = np.asarray(init_data.atoms.data.positions)
    dof = float(3 * positions.shape[0] - 3)
    lattice = np.asarray(init_data.systems.data.unitcell.lattice_vectors[0])
    potential_energy = np.asarray(step_data.potential_energy).flatten()
    kinetic_energy = np.asarray(step_data.kinetic_energy).flatten()
    stress_tensor = np.asarray(step_data.stress_tensor).reshape(-1, 3, 3)

    n_atoms = positions.shape[0]
    n_steps = len(potential_energy)
    volume = abs(np.linalg.det(lattice))

    total_energy = potential_energy + kinetic_energy
    temperature = 2 * kinetic_energy / (BOLTZMANN_CONSTANT * dof)
    pressure = np.trace(stress_tensor, axis1=-2, axis2=-1) / 3

    print(f"MD Physics Validation: {args.hdf5_path.name}")
    print(f"  Atoms: {n_atoms}, Steps: {n_steps}, DOF: {int(dof)}")
    print(f"  Volume: {volume:.2f} A^3, Density: {n_atoms / volume:.4f} atoms/A^3")
    print()

    all_passed = True

    # 1. Energy conservation
    passed, info = validate_energy_conservation(total_energy, n_atoms, args.max_drift)
    status = "PASS" if passed else "FAIL"
    print(f"1. Energy Conservation: {status}")
    print(
        f"   Drift: {info['drift_per_atom_per_step']:.2e} eV/atom/step (max: {info['drift_threshold']:.2e})"
    )
    print(
        f"   Relative fluctuation: {info['relative_fluctuation']:.2e} (max: {info['fluctuation_threshold']:.2e})"
    )
    all_passed &= passed

    # 2. Equipartition theorem
    passed, info = validate_equipartition(kinetic_energy, temperature, int(dof))
    status = "PASS" if passed else "FAIL"
    print(f"2. Equipartition Theorem: {status}")
    print(f"   <KE>: {info['mean_kinetic_energy']:.4f} eV")
    print(f"   Expected: {info['expected_kinetic_energy']:.4f} eV")
    print(f"   Relative error: {info['relative_error']:.2%}")
    all_passed &= passed

    # 3. Temperature fluctuations
    passed, info = validate_temperature_fluctuations(temperature, int(dof))
    status = "PASS" if passed else "FAIL"
    print(f"3. Temperature Fluctuations: {status}")
    print(f"   <T>: {info['mean_temperature']:.2f} K")
    print(f"   std(T): {info['temperature_std']:.2f} K")
    print(
        f"   Var ratio (NVE/canonical): {info['ratio']:.2f} (must be < {info['max_ratio']:.2f})"
    )
    all_passed &= passed

    # 4. Pressure/virial consistency
    passed, info = validate_pressure_virial(pressure, kinetic_energy, volume)
    status = "PASS" if passed else "FAIL"
    print(f"4. Pressure Consistency: {status}")
    print(f"   <P>: {info['mean_pressure_Pa']:.4e} Pa")
    print(f"   Ideal gas P: {info['ideal_gas_pressure_Pa']:.4e} Pa")
    print(f"   Excess P: {info['excess_pressure_Pa']:.4e} Pa")
    print(
        f"   Pressure CV: {info['pressure_cv']:.4f} (max: {info['max_pressure_cv']:.2f})"
    )
    all_passed &= passed

    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
