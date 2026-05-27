#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Compare committed LAMMPS YAML reference statistics to kUPS refs."""

from __future__ import annotations

import argparse
import math

from lammps_reference_utils import CASES, load_kups_expected, load_reference_stats


def _compare_to_kups(current: dict) -> bool:
    ok = True
    for case, config in CASES.items():
        if not config.get("compare_to_kups", False):
            continue
        kups_expected = load_kups_expected(config["kups_expected"])
        max_z = float(config["max_z"])
        abs_tolerances = config.get("abs_tolerances", {})
        print(f"\n{case} vs kUPS {config['kups_expected']} ({max_z:g} sigma)")
        compare_observables = config.get("compare_observables", config["observables"])
        for lammps_obs, kups_obs in compare_observables.items():
            if kups_obs not in kups_expected:
                print(f"  SKIP {lammps_obs}: no kUPS {kups_obs} expected value")
                continue
            lmp_stats = current[case]["observables"][kups_obs]
            kups_stats = kups_expected[kups_obs]
            lmp_mean = float(lmp_stats["expected_mean"])
            lmp_sem = float(lmp_stats["expected_sem"])
            kups_mean = float(kups_stats["expected_mean"])
            kups_sem = float(kups_stats["expected_sem"])
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
                f"kUPS {kups_mean:.8g} +/- {kups_sem:.3g}; {reason} {mark}"
            )
    return ok


def main() -> int:
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
