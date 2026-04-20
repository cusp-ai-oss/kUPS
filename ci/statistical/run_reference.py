#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Run all reference simulations in parallel and generate expected values."""

from __future__ import annotations

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


def main() -> int:
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

            hdf5 = SCRIPT_DIR / f"{name}_ref.h5"
            result = subprocess.run(
                ["uv", "run", "python", "generate_expected.py", hdf5.name, sim_type],
                cwd=SCRIPT_DIR,
            )
            if result.returncode != 0:
                print(f"  ✗ Expected value generation failed: {name}")
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
