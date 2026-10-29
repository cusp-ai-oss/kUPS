#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Compare committed LAMMPS YAML reference statistics to kUPS refs."""

from __future__ import annotations

import argparse
import math

from lammps_reference_utils import (
    CASES,
    ExpectedSpec,
    compared_stats,
    load_kups_expected,
    load_reference_stats,
)


def _compare_to_kups(current: dict[str, ExpectedSpec]) -> bool:
    """Compare the LAMMPS statistics of every case against the kUPS references.

    Args:
        current: LAMMPS expected values keyed by case name.

    Returns:
        Whether every compared observable agrees within its case's tolerance.
    """
    ok = True
    for case, config in CASES.items():
        if not config["compare_to_kups"]:
            continue
        kups_expected = load_kups_expected(config["kups_expected"])
        max_z = config["max_z"]
        abs_tolerances = config.get("abs_tolerances", {})
        print(f"\n{case} vs kUPS {config['kups_expected']} ({max_z:g} sigma)")
        for lammps_obs, kups_obs in config["compare_observables"].items():
            if kups_obs not in kups_expected:
                print(f"  SKIP {lammps_obs}: no kUPS {kups_obs} expected value")
                continue
            lmp_stats = current[case]["observables"][kups_obs]
            kups_stats = kups_expected[kups_obs]
            unit, lmp_mean, lmp_sem = compared_stats(lmp_stats)
            kups_unit, kups_mean, kups_sem = compared_stats(kups_stats)
            if unit != kups_unit:
                print(f"  FAIL {lammps_obs}: LAMMPS is in {unit}, kUPS in {kups_unit}")
                ok = False
                continue
            delta = abs(lmp_mean - kups_mean)
            abs_tol = float(abs_tolerances.get(lammps_obs, 0.0))
            combined = math.sqrt(lmp_sem**2 + kups_sem**2)
            z = delta / combined if combined > 0 else math.inf
            passed = delta <= abs_tol or z <= max_z
            ok &= passed
            mark = "PASS" if passed else "FAIL"
            reason = f"z={z:.2f}"
            if delta <= abs_tol and z > max_z:
                reason = f"abs={delta:.3g}<={abs_tol:.3g}"
            print(
                f"  {lammps_obs:7s}: LAMMPS {lmp_mean:.8g} +/- {lmp_sem:.3g}; "
                f"kUPS {kups_mean:.8g} +/- {kups_sem:.3g} {unit or ''}; "
                f"{reason} {mark}"
            )
    return ok


def main() -> int:
    """Load the committed LAMMPS statistics and compare them to the kUPS ones.

    Returns:
        Zero when every comparison passes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-kups-compare",
        action="store_true",
        help="Only verify that the committed LAMMPS YAML reference files load.",
    )
    args = parser.parse_args()

    references = load_reference_stats()
    print("Loaded committed LAMMPS YAML reference statistics")

    ok = True
    if not args.skip_kups_compare:
        ok &= _compare_to_kups(references)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
