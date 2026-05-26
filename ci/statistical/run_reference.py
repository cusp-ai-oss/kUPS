#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Run all reference simulations in parallel and generate expected values."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REFERENCE_DIR = SCRIPT_DIR / "inputs" / "reference"

# (config_name, sim_type, cli_command)
EXPERIMENTS: list[tuple[str, str, str]] = [
    ("md_nve_lj_argon", "md", "kups_md_lj"),
    ("md_nvt_lj_argon", "md", "kups_md_lj"),
    ("md_npt_lj_argon", "md", "kups_md_lj"),
    ("nvt_50co2_30box", "nvt", "kups_mcmc_rigid"),
    ("gcmc_co2_rubtak", "gcmc", "kups_mcmc_rigid"),
]

SIM_ENV = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def _generate_expected(name: str, sim_type: str) -> bool:
    """Run generate_expected.py for a single experiment. Returns True on success."""
    hdf5 = SCRIPT_DIR / f"{name}_ref.h5"
    if not hdf5.exists():
        print(f"  ✗ Missing HDF5 output: {hdf5.name}")
        return False
    result = subprocess.run(
        ["uv", "run", "python", "generate_expected.py", hdf5.name, sim_type],
        cwd=SCRIPT_DIR,
        env=SIM_ENV,
    )
    if result.returncode != 0:
        print(f"  ✗ Expected value generation failed: {name}")
        return False
    print(f"  ✓ Expected values generated: {name}")
    return True


def _regenerate_only() -> int:
    print(f"Regenerating expected YAMLs for {len(EXPERIMENTS)} experiments")
    failed = [
        name
        for name, sim_type, _ in EXPERIMENTS
        if not _generate_expected(name, sim_type)
    ]
    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All expected values regenerated successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate-only",
        action="store_true",
        help="Skip simulations and only regenerate expected YAMLs from existing HDF5 outputs.",
    )
    args = parser.parse_args()

    if args.regenerate_only:
        return _regenerate_only()

    print(f"Launching {len(EXPERIMENTS)} simulations in parallel")
    procs: list[tuple[str, str, subprocess.Popen]] = []

    # All children join process_group=0 → each becomes its own group leader.
    # We track PIDs and kill individually, but group them for clarity.
    def _kill_all() -> None:
        for _, _, p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for _, _, p in procs:
            p.wait()

    for name, sim_type, cli in EXPERIMENTS:
        config = REFERENCE_DIR / f"{name}.yaml"
        print(f"  Starting: {name} ({cli})")
        proc = subprocess.Popen(
            ["uv", "run", cli, config.name],
            cwd=config.parent,
            env=SIM_ENV,
            start_new_session=True,
        )
        procs.append((name, sim_type, proc))

    failed: list[str] = []
    try:
        for name, sim_type, proc in procs:
            rc = proc.wait()
            if rc != 0:
                print(f"  ✗ Simulation failed: {name}")
                failed.append(name)
                continue
            print(f"  ✓ Simulation done: {name}")

            if not _generate_expected(name, sim_type):
                failed.append(name)
    except BaseException:
        _kill_all()
        raise

    print(f"\n{'=' * 60}")
    if failed:
        _kill_all()
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All reference simulations and expected values generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
