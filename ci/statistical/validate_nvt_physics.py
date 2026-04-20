#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Validate NVT MD simulation results against canonical ensemble properties.

For an NVT simulation with Langevin thermostat, validates:
1. Temperature stability (mean matches target)
2. Temperature fluctuations (canonical ensemble prediction)
3. Equipartition theorem (KE per DOF = kT/2)
4. Energy distribution consistency
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
    rtol: float = 0.02,
) -> tuple[bool, dict]:
    """Validate that mean temperature matches target.

    For a well-equilibrated NVT simulation with a properly tuned thermostat,
    the time-averaged temperature should equal the target temperature within
    ~2%. BAOAB Langevin specifically should achieve excellent temperature control.
    """
    mean_temp = float(np.mean(temperature))
    std_temp = float(np.std(temperature))
    sem_temp = std_temp / np.sqrt(len(temperature))

    relative_error = abs(mean_temp - target_temperature) / target_temperature
    passed = relative_error < rtol

    return passed, {
        "mean_temperature": mean_temp,
        "target_temperature": target_temperature,
        "temperature_std": std_temp,
        "temperature_sem": sem_temp,
        "relative_error": relative_error,
        "threshold": rtol,
    }


def validate_temperature_fluctuations(
    temperature: np.ndarray,
    degrees_of_freedom: int,
    min_ratio: float = 0.7,
    max_ratio: float = 1.3,
) -> tuple[bool, dict]:
    """Validate temperature fluctuations match canonical ensemble.

    For the canonical ensemble: <(dT)^2> / <T>^2 = 2 / DOF
    BAOAB Langevin should reproduce canonical fluctuations accurately.
    We allow 0.7-1.3 range to account for finite-size effects and
    correlation time effects in finite trajectories.
    """
    mean_temp = float(np.mean(temperature))
    temp_var = float(np.var(temperature))

    canonical_relative_var = 2.0 / degrees_of_freedom
    measured_relative_var = temp_var / mean_temp**2 if mean_temp != 0 else 0.0

    ratio = measured_relative_var / canonical_relative_var
    passed = min_ratio < ratio < max_ratio

    return passed, {
        "mean_temperature": mean_temp,
        "temperature_std": float(np.sqrt(temp_var)),
        "measured_relative_variance": measured_relative_var,
        "canonical_relative_variance": canonical_relative_var,
        "ratio": ratio,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
    }


def validate_equipartition(
    kinetic_energy: np.ndarray,
    degrees_of_freedom: int,
    target_temperature: float,
    rtol: float = 0.02,
) -> tuple[bool, dict]:
    """Validate equipartition theorem: <KE> = (DOF/2) * k_B * T.

    For NVT with a proper thermostat, the kinetic energy should give
    the correct target temperature within 2%.
    """
    mean_ke = float(np.mean(kinetic_energy))
    expected_ke = 0.5 * degrees_of_freedom * BOLTZMANN_CONSTANT * target_temperature

    relative_error = (
        abs(mean_ke - expected_ke) / expected_ke if expected_ke != 0 else 0.0
    )
    passed = relative_error < rtol

    return passed, {
        "mean_kinetic_energy": mean_ke,
        "expected_kinetic_energy": expected_ke,
        "relative_error": relative_error,
        "threshold": rtol,
    }


def validate_energy_distribution(
    potential_energy: np.ndarray,
    kinetic_energy: np.ndarray,
    degrees_of_freedom: int,
    ke_var_rtol: float = 0.3,
) -> tuple[bool, dict]:
    """Validate energy fluctuations match canonical ensemble.

    In the canonical ensemble:
    - KE follows chi-squared distribution with DOF degrees of freedom
    - Var(KE) / <KE>^2 = 2 / DOF

    We validate that KE variance is within 30% of the expected value,
    which is appropriate for finite simulation lengths.
    """
    mean_pe = float(np.mean(potential_energy))
    std_pe = float(np.std(potential_energy))
    mean_ke = float(np.mean(kinetic_energy))
    std_ke = float(np.std(kinetic_energy))

    # KE relative variance should be 2/DOF for canonical ensemble
    ke_relative_var = (std_ke / mean_ke) ** 2 if mean_ke != 0 else 0.0
    expected_ke_relative_var = 2.0 / degrees_of_freedom

    # Check KE variance is close to canonical prediction
    ke_var_error = (
        abs(ke_relative_var - expected_ke_relative_var) / expected_ke_relative_var
        if expected_ke_relative_var != 0
        else 0.0
    )

    # Check that PE has reasonable fluctuations (not NaN or zero)
    pe_cv = std_pe / abs(mean_pe) if mean_pe != 0 else 0.0

    finite_ok = bool(np.isfinite(mean_pe) and np.isfinite(mean_ke))
    pe_ok = pe_cv > 0
    ke_var_ok = ke_var_error < ke_var_rtol
    passed = finite_ok and pe_ok and ke_var_ok

    return passed, {
        "mean_potential_energy": mean_pe,
        "potential_energy_std": std_pe,
        "potential_energy_cv": pe_cv,
        "mean_kinetic_energy": mean_ke,
        "kinetic_energy_std": std_ke,
        "ke_relative_variance": ke_relative_var,
        "expected_ke_relative_variance": expected_ke_relative_var,
        "ke_variance_error": ke_var_error,
        "ke_var_rtol": ke_var_rtol,
    }


def validate_ergodicity(
    potential_energy: np.ndarray,
    block_size: int = 1000,
    max_relative_block_std: float = 0.05,
) -> tuple[bool, dict]:
    """Check for ergodicity by comparing block averages.

    If the system is ergodic, block averages should converge and have
    similar values across different time windows. We require block means
    to have relative std < 5% of the overall mean.
    """
    n_blocks = len(potential_energy) // block_size
    if n_blocks < 4:
        return True, {"status": "insufficient_data", "n_blocks": n_blocks}

    blocks = potential_energy[: n_blocks * block_size].reshape(n_blocks, block_size)
    block_means = np.mean(blocks, axis=1)

    overall_mean = float(np.mean(potential_energy))
    block_std = float(np.std(block_means))
    max_deviation = float(np.max(np.abs(block_means - overall_mean)))

    # Check that block means don't drift too much
    relative_block_std = block_std / abs(overall_mean) if overall_mean != 0 else 0.0
    passed = relative_block_std < max_relative_block_std

    return passed, {
        "n_blocks": n_blocks,
        "block_size": block_size,
        "overall_mean": overall_mean,
        "block_std": block_std,
        "max_deviation": max_deviation,
        "relative_block_std": relative_block_std,
        "max_relative_block_std": max_relative_block_std,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate NVT MD simulation against canonical ensemble properties"
    )
    parser.add_argument(
        "hdf5_path",
        type=Path,
        nargs="?",
        default=Path("md_nvt_lj_argon.h5"),
        help="Path to HDF5 output file",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=100.0,
        help="Target temperature (K)",
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
    dof = int(3 * positions.shape[0] - 3)
    lattice = np.asarray(init_data.systems.data.unitcell.lattice_vectors[0])
    potential_energy = np.asarray(step_data.potential_energy).flatten()
    kinetic_energy = np.asarray(step_data.kinetic_energy).flatten()

    n_atoms = positions.shape[0]
    n_steps = len(potential_energy)
    volume = abs(np.linalg.det(lattice))

    temperature = 2 * kinetic_energy / (BOLTZMANN_CONSTANT * dof)

    print(f"NVT Physics Validation: {args.hdf5_path.name}")
    print(f"  Atoms: {n_atoms}, Steps: {n_steps}, DOF: {dof}")
    print(f"  Volume: {volume:.2f} A^3, Target T: {args.target_temperature} K")
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
    print(f"   std(T): {info['temperature_std']:.2f} K")
    print(
        f"   Var ratio (measured/canonical): {info['ratio']:.2f} "
        f"(must be {info['min_ratio']:.1f}-{info['max_ratio']:.1f})"
    )
    all_passed &= passed

    # 3. Equipartition theorem
    passed, info = validate_equipartition(kinetic_energy, dof, args.target_temperature)
    status = "PASS" if passed else "FAIL"
    print(f"3. Equipartition Theorem: {status}")
    print(f"   <KE>: {info['mean_kinetic_energy']:.4f} eV")
    print(f"   Expected: {info['expected_kinetic_energy']:.4f} eV")
    print(
        f"   Relative error: {info['relative_error']:.2%} (max: {info['threshold']:.0%})"
    )
    all_passed &= passed

    # 4. Energy distribution
    passed, info = validate_energy_distribution(potential_energy, kinetic_energy, dof)
    status = "PASS" if passed else "FAIL"
    print(f"4. Energy Distribution: {status}")
    print(f"   <PE>: {info['mean_potential_energy']:.4f} eV")
    print(f"   PE coefficient of variation: {info['potential_energy_cv']:.4f}")
    print(
        f"   KE variance error: {info['ke_variance_error']:.2%} "
        f"(max: {info['ke_var_rtol']:.0%})"
    )
    all_passed &= passed

    # 5. Ergodicity check
    passed, info = validate_ergodicity(potential_energy)
    status = "PASS" if passed else "FAIL"
    print(f"5. Ergodicity Check: {status}")
    if "status" in info:
        print(f"   {info['status']}")
    else:
        print(
            f"   Block std / mean: {info['relative_block_std']:.4f} "
            f"(max: {info['max_relative_block_std']:.2f})"
        )
    all_passed &= passed

    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
