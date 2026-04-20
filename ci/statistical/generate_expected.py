#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generate expected values YAML from simulation HDF5 output."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import yaml

from kups.application.mcmc.analysis import analyze_mcmc_file
from kups.application.md.analysis import analyze_md_file
from kups.core.utils.block_average import BlockAverageResult


def _analyze_mcmc_first_system(path: Path) -> Any:
    """Analyze MCMC file and return the first system's result."""
    results = analyze_mcmc_file(path)
    return next(iter(results.values()))


def _analyze_md_first_system(path: Path) -> Any:
    """Analyze MD file and return the first system's result."""
    results = analyze_md_file(path)
    return next(iter(results.values()))


def block_result_to_expected(result: BlockAverageResult) -> dict[str, Any]:
    """Convert BlockAverageResult to expected values dict."""
    mean = jnp.atleast_1d(result.mean).tolist()
    sem = jnp.atleast_1d(result.sem).tolist()
    return {
        "expected_mean": mean[0] if len(mean) == 1 else mean,
        "expected_sem": sem[0] if len(sem) == 1 else sem,
    }


def analysis_result_to_expected(
    result: Any, sim_type: str, hdf5_name: str
) -> dict[str, Any]:
    """Convert any analysis result dataclass to expected values dict."""
    observables = {}
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, BlockAverageResult):
            observables[field.name] = block_result_to_expected(value)
    return {
        "simulation_type": sim_type,
        "hdf5_output": hdf5_name,
        "observables": observables,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expected values YAML")
    parser.add_argument("hdf5_path", type=Path, help="Path to HDF5 output file")
    parser.add_argument(
        "sim_type", choices=["nvt", "gcmc", "md"], help="Simulation type"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output YAML path (default: expected/<name>.yaml)",
    )
    args = parser.parse_args()

    analyze_fns = {
        "nvt": _analyze_mcmc_first_system,
        "gcmc": _analyze_mcmc_first_system,
        "md": _analyze_md_first_system,
    }
    analyze_fn = analyze_fns[args.sim_type]
    result = analyze_fn(args.hdf5_path)
    # Strip _ref suffix so hdf5_output points to the CI output filename
    ci_name = args.hdf5_path.name.replace("_ref.h5", ".h5")
    expected = analysis_result_to_expected(result, args.sim_type, ci_name)

    stem = args.hdf5_path.stem.removesuffix("_ref")
    output_path = args.output or Path("expected") / f"{stem}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(expected, f, default_flow_style=False, sort_keys=False)

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
