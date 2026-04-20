#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Run simulations and validate outputs against expected values."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import yaml

from kups.application.mcmc.analysis import MCMCAnalysisResult, analyze_mcmc_file
from kups.application.md.analysis import MDAnalysisResult, analyze_md_file
from kups.core.utils.block_average import BlockAverageResult

SCRIPT_DIR = Path(__file__).parent
INPUTS_DIR = SCRIPT_DIR / "inputs" / "ci"
EXPECTED_DIR = SCRIPT_DIR / "expected"
DEFAULT_TOLERANCE_SIGMAS = 10.0

CLI_COMMANDS: dict[str, str] = {
    "md": "kups_md_lj",
    "nvt": "kups_mcmc_rigid",
    "gcmc": "kups_mcmc_rigid",
}

SIM_ENV = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def _analyze_mcmc_first_system(path: Path) -> MCMCAnalysisResult:
    results = analyze_mcmc_file(path)
    return next(iter(results.values()))


def _analyze_md_first_system(path: Path) -> MDAnalysisResult:
    results = analyze_md_file(path)
    return next(iter(results.values()))


ANALYZERS: dict[str, Callable[[Path], Any]] = {
    "nvt": _analyze_mcmc_first_system,
    "gcmc": _analyze_mcmc_first_system,
    "md": _analyze_md_first_system,
}


def run_simulations(
    specs: list[tuple[str, Path, str]],
) -> dict[str, bool]:
    """Run simulations in parallel. Returns {name: success} mapping."""
    procs: list[tuple[str, subprocess.Popen]] = []

    def _kill_all() -> None:
        for _, p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, p in procs:
            p.wait()

    for name, config_path, sim_type in specs:
        cli = CLI_COMMANDS.get(sim_type)
        if not cli:
            print(f"  ✗ Unknown simulation type: {sim_type}")
            continue
        print(f"  Starting: {name} ({cli})")
        proc = subprocess.Popen(
            ["uv", "run", cli, config_path.name],
            cwd=config_path.parent,
            env=SIM_ENV,
            start_new_session=True,
        )
        procs.append((name, proc))

    results: dict[str, bool] = {}
    try:
        for name, proc in procs:
            rc = proc.wait()
            results[name] = rc == 0
            if rc != 0:
                print(f"  ✗ Simulation failed: {name}")
            else:
                print(f"  ✓ Simulation done: {name}")
    except BaseException:
        _kill_all()
        raise

    return results


def validate_observable(
    name: str,
    result: BlockAverageResult,
    expected_mean: float | list[float],
    expected_sem: float | list[float],
    tolerance_sigmas: float = DEFAULT_TOLERANCE_SIGMAS,
) -> bool:
    """Check if measured value is within tolerance."""
    exp_mean = jnp.asarray(expected_mean).flatten()
    exp_sem = jnp.asarray(expected_sem).flatten()
    measured_mean = jnp.asarray(result.mean).flatten()

    all_pass = True
    for i, (m, e, es) in enumerate(zip(measured_mean, exp_mean, exp_sem)):
        m_f, e_f = float(m), float(e)
        suffix = f"[{i}]" if len(exp_mean) > 1 else ""
        if jnp.isnan(m) and jnp.isnan(e):
            print(f"  - {name}{suffix}: nan (expected nan, skipped)")
            continue
        if jnp.isnan(m) or jnp.isnan(e):
            print(
                f"  ✗ {name}{suffix}: unexpected NaN (measured={m_f}, expected={e_f})"
            )
            all_pass = False
            continue
        diff = abs(m_f - e_f)
        ref_sem = float(es)
        max_diff = tolerance_sigmas * ref_sem if ref_sem > 1e-10 else abs(e_f) * 0.01
        sem_str = f"{ref_sem:.4g}" if ref_sem > 1e-10 else f"{ref_sem:.2e}"
        if diff <= max_diff:
            print(
                f"  ✓ {name}{suffix}: {float(m):.4g} (expected {float(e):.4g}, σ={sem_str})"
            )
        else:
            print(
                f"  ✗ {name}{suffix}: {float(m):.4g} != {float(e):.4g} (diff={diff:.4g} > {tolerance_sigmas}σ={max_diff:.4g})"
            )
            all_pass = False
    return all_pass


def validate(path: Path, sim_type: str, obs: dict) -> bool:
    """Validate simulation results against expected observables."""
    analyze = ANALYZERS.get(sim_type)
    if not analyze:
        print(f"  ⚠ Skipped: unsupported type {sim_type}")
        return True

    result = analyze(path)
    first_field = getattr(result, fields(result)[0].name)
    if isinstance(first_field, BlockAverageResult) and first_field.n_blocks < 2:
        print("  ⚠ Skipped: insufficient samples")
        return True

    all_pass = True
    for key, expected in obs.items():
        value = getattr(result, key)
        if isinstance(value, BlockAverageResult):
            if not validate_observable(key, value, **expected):
                all_pass = False
    return all_pass


def main() -> int:
    """Run simulations and validate outputs."""
    parser = argparse.ArgumentParser(description="Validate simulation outputs")
    parser.add_argument(
        "name",
        nargs="?",
        help="Specific simulation to run (e.g., 'md_nve_lj_argon')",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running simulations, only validate existing outputs",
    )
    args = parser.parse_args()

    if not EXPECTED_DIR.exists():
        print(f"No expected values directory: {EXPECTED_DIR}")
        return 1

    if args.name:
        expected_files = [EXPECTED_DIR / f"{args.name}.yaml"]
        if not expected_files[0].exists():
            print(f"Expected file not found: {expected_files[0]}")
            return 1
    else:
        expected_files = sorted(EXPECTED_DIR.glob("*.yaml"))

    # Load all specs
    specs: list[tuple[str, dict]] = []
    for expected_file in expected_files:
        with open(expected_file) as f:
            spec = yaml.safe_load(f)
        specs.append((expected_file.stem, spec))

    # Run all simulations in parallel
    sim_results: dict[str, bool] = {}
    if not args.no_run:
        to_run: list[tuple[str, Path, str]] = []
        for name, spec in specs:
            input_config = INPUTS_DIR / f"{name}.yaml"
            if input_config.exists():
                to_run.append((name, input_config, spec["simulation_type"]))
            else:
                print(f"  ⚠ No input config found: {input_config}")

        if to_run:
            print("Running simulations:")
            sim_results = run_simulations(to_run)

    # Validate all outputs
    all_pass = True
    for name, spec in specs:
        print(f"\n{name}:")

        if not args.no_run and not sim_results.get(name, True):
            all_pass = False
            continue

        hdf5_path = SCRIPT_DIR / spec["hdf5_output"]
        if not hdf5_path.exists():
            print(f"  ⚠ Skipped: output not found ({hdf5_path})")
            continue

        if not validate(
            hdf5_path, spec["simulation_type"], spec.get("observables", {})
        ):
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
