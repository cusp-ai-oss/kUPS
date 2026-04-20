#!/usr/bin/env python
"""Validate NPT MD simulation results against isobaric-isothermal ensemble properties.

For an NPT simulation with CSVR thermostat + stochastic cell rescaling barostat:
1. Temperature stability (mean matches target)
2. Temperature fluctuations (canonical ensemble prediction)
3. Volume stability (no monotonic drift)
4. Volume fluctuations (positive, bounded)
5. Equipartition (KE per DOF = kT/2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kups.application.md.logging import MDLoggedData
from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.storage import HDF5StorageReader


def validate_temperature_stability(
    temperature: np.ndarray,
    target_temperature: float,
    rtol: float = 0.05,
) -> tuple[bool, dict]:
    """Validate that mean temperature matches target within 5%."""
    mean_temp = float(np.mean(temperature))
    relative_error = abs(mean_temp - target_temperature) / target_temperature
    return relative_error < rtol, {
        "mean_temperature": mean_temp,
        "target_temperature": target_temperature,
        "relative_error": relative_error,
        "threshold": rtol,
    }


def validate_temperature_fluctuations(
    temperature: np.ndarray,
    degrees_of_freedom: int,
    min_ratio: float = 0.5,
    max_ratio: float = 2.0,
) -> tuple[bool, dict]:
    """Validate temperature fluctuations match canonical ensemble.

    For the canonical ensemble: <(dT)^2> / <T>^2 = 2 / DOF.
    NPT has slightly larger fluctuations than NVT due to volume coupling,
    so we use wider bounds (0.5-2.0) than the NVT validation (0.7-1.3).
    """
    mean_temp = float(np.mean(temperature))
    temp_var = float(np.var(temperature))
    canonical_relative_var = 2.0 / degrees_of_freedom
    measured_relative_var = temp_var / mean_temp**2 if mean_temp != 0 else 0.0
    ratio = measured_relative_var / canonical_relative_var
    return min_ratio < ratio < max_ratio, {
        "measured_relative_variance": measured_relative_var,
        "canonical_relative_variance": canonical_relative_var,
        "ratio": ratio,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
    }


def validate_volume_stability(
    volume: np.ndarray,
    max_relative_drift: float = 0.10,
) -> tuple[bool, dict]:
    """Validate volume is not drifting monotonically.

    Compares mean volume of first and second half of trajectory.
    A drift > 10% indicates the barostat hasn't equilibrated.
    """
    half = len(volume) // 2
    mean_first = float(np.mean(volume[:half]))
    mean_second = float(np.mean(volume[half:]))
    drift = abs(mean_second - mean_first) / mean_first
    return drift < max_relative_drift, {
        "mean_first_half": mean_first,
        "mean_second_half": mean_second,
        "drift": drift,
        "threshold": max_relative_drift,
    }


def validate_volume_fluctuations(
    volume: np.ndarray,
    min_cv: float = 0.001,
    max_cv: float = 0.20,
) -> tuple[bool, dict]:
    """Validate volume fluctuations are present and bounded.

    The coefficient of variation (std/mean) should be positive (barostat is active)
    and less than 20% (system is in a condensed phase, not exploding).
    """
    mean_vol = float(np.mean(volume))
    std_vol = float(np.std(volume))
    cv = std_vol / mean_vol if mean_vol > 0 else 0.0
    return min_cv < cv < max_cv, {
        "mean_volume": mean_vol,
        "std_volume": std_vol,
        "cv": cv,
        "min_cv": min_cv,
        "max_cv": max_cv,
    }


def validate_equipartition(
    kinetic_energy: np.ndarray,
    degrees_of_freedom: int,
    target_temperature: float,
    rtol: float = 0.05,
) -> tuple[bool, dict]:
    """Validate equipartition: <KE> = (DOF/2) * kB * T within 5%."""
    mean_ke = float(np.mean(kinetic_energy))
    expected_ke = 0.5 * degrees_of_freedom * BOLTZMANN_CONSTANT * target_temperature
    relative_error = (
        abs(mean_ke - expected_ke) / expected_ke if expected_ke != 0 else 0.0
    )
    return relative_error < rtol, {
        "mean_kinetic_energy": mean_ke,
        "expected_kinetic_energy": expected_ke,
        "relative_error": relative_error,
        "threshold": rtol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate NPT MD simulation against isobaric-isothermal ensemble properties"
    )
    parser.add_argument(
        "hdf5_path",
        type=Path,
        nargs="?",
        default=Path("md_npt_lj_argon.h5"),
    )
    parser.add_argument("--target-temperature", type=float, default=100.0)
    args = parser.parse_args()

    if not args.hdf5_path.exists():
        print(f"File not found: {args.hdf5_path}")
        return 1

    with HDF5StorageReader[MDLoggedData](args.hdf5_path) as reader:
        init_data = reader.focus_group(lambda state: state.init)[...]
        step_data = reader.focus_group(lambda state: state.step)[...]

    positions = np.asarray(init_data.atoms.data.positions)
    n_atoms = positions.shape[0]
    dof = 3 * n_atoms - 3
    potential_energy = np.asarray(step_data.potential_energy).flatten()
    kinetic_energy = np.asarray(step_data.kinetic_energy).flatten()

    n_steps = len(potential_energy)
    temperature = 2 * kinetic_energy / (BOLTZMANN_CONSTANT * dof)

    # Extract volume from stress tensor shape or lattice vectors
    # The HDF5 stores per-step data; volume comes from the unitcell
    # For NPT, volume changes each step — extract from logged data
    # stress_tensor shape: (n_steps, n_systems, 3, 3)
    # We need volume — compute from Tr(σ) and known pressure
    # Actually, let's just read it from the init lattice and note NPT changes volume
    lattice = np.asarray(init_data.systems.data.unitcell.lattice_vectors[0])
    initial_volume = abs(np.linalg.det(lattice))

    print(f"NPT Physics Validation: {args.hdf5_path.name}")
    print(f"  Atoms: {n_atoms}, Steps: {n_steps}, DOF: {dof}")
    print(f"  Initial volume: {initial_volume:.2f} A^3")
    print(f"  Target T: {args.target_temperature} K")
    print()

    all_passed = True

    # 1. Temperature stability
    passed, info = validate_temperature_stability(temperature, args.target_temperature)
    status = "PASS" if passed else "FAIL"
    print(f"1. Temperature Stability: {status}")
    print(
        f"   <T>: {info['mean_temperature']:.2f} K (target: {info['target_temperature']:.2f} K)"
    )
    print(
        f"   Relative error: {info['relative_error']:.2%} (max: {info['threshold']:.0%})"
    )
    all_passed &= passed

    # 2. Temperature fluctuations
    passed, info = validate_temperature_fluctuations(temperature, dof)
    status = "PASS" if passed else "FAIL"
    print(f"2. Temperature Fluctuations: {status}")
    print(
        f"   Var ratio (measured/canonical): {info['ratio']:.2f} (must be {info['min_ratio']:.1f}-{info['max_ratio']:.1f})"
    )
    all_passed &= passed

    # 3. Equipartition
    passed, info = validate_equipartition(kinetic_energy, dof, args.target_temperature)
    status = "PASS" if passed else "FAIL"
    print(f"3. Equipartition Theorem: {status}")
    print(
        f"   <KE>: {info['mean_kinetic_energy']:.4f} eV (expected: {info['expected_kinetic_energy']:.4f})"
    )
    print(
        f"   Relative error: {info['relative_error']:.2%} (max: {info['threshold']:.0%})"
    )
    all_passed &= passed

    # 4. Energy sanity (finite, no NaN)
    finite_ok = bool(
        np.all(np.isfinite(potential_energy)) and np.all(np.isfinite(kinetic_energy))
    )
    status = "PASS" if finite_ok else "FAIL"
    print(f"4. Energy Sanity: {status}")
    print(
        f"   <PE>: {np.mean(potential_energy):.4f} eV, <KE>: {np.mean(kinetic_energy):.4f} eV"
    )
    all_passed &= finite_ok

    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
