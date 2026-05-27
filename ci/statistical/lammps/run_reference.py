#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Regenerate LAMMPS LJ Argon reference outputs and kUPS-format YAML stats."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from lammps_reference_utils import (
    CASES,
    OUTPUT_DIR,
    ROOT,
    analyze_all_outputs,
    write_reference_stats,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        choices=sorted(CASES),
        help="Case names to run. Defaults to all cases.",
    )
    parser.add_argument("--lmp", default="lmp", help="LAMMPS executable")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only regenerate YAML statistics from existing LAMMPS output tables.",
    )
    args = parser.parse_args()

    selected = args.cases or list(CASES)
    if not args.analyze_only:
        for name in selected:
            config = CASES[name]
            input_path = ROOT / config["input"]
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            log_path = (
                Path("outputs/logs") / Path(config["output"]).with_suffix(".log").name
            )
            (ROOT / log_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"Running {name}: {input_path.name}")
            subprocess.run(
                [
                    args.lmp,
                    "-in",
                    str(input_path.relative_to(ROOT)),
                    "-log",
                    str(log_path),
                ],
                cwd=ROOT,
                check=True,
            )

    stats = analyze_all_outputs()
    write_reference_stats(stats)
    print(f"Wrote LAMMPS YAML references in {ROOT / 'expected'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
