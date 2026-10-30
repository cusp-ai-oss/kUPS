#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generate expected values YAML from simulation HDF5 output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import jax
from expected_values import AnalysisResult, dump_expected, expected_from_result

from kups.application.mcmc.analysis import MCMCAnalysisResult, analyze_mcmc_file
from kups.application.md.analysis import MDAnalysisResult, analyze_md_file

jax.config.update("jax_enable_x64", True)

N_BLOCKS = 5


def _analyze_mcmc_first_system(path: Path) -> MCMCAnalysisResult:
    """Analyze an MCMC output and return its first system's result.

    Args:
        path: HDF5 output of an MCMC run.

    Returns:
        Block-averaged analysis of the first system.
    """
    results = analyze_mcmc_file(path, n_blocks=N_BLOCKS)
    return next(iter(results.values()))


def _analyze_md_first_system(path: Path) -> MDAnalysisResult:
    """Analyze an MD output and return its first system's result.

    Args:
        path: HDF5 output of an MD run.

    Returns:
        Block-averaged analysis of the first system.
    """
    results = analyze_md_file(path, n_blocks=N_BLOCKS)
    return next(iter(results.values()))


ANALYZERS: dict[str, Callable[[Path], AnalysisResult]] = {
    "nvt": _analyze_mcmc_first_system,
    "gcmc": _analyze_mcmc_first_system,
    "md": _analyze_md_first_system,
}


def main() -> None:
    """Write the expected values of one reference HDF5 output."""
    parser = argparse.ArgumentParser(description="Generate expected values YAML")
    parser.add_argument("hdf5_path", type=Path, help="Path to HDF5 output file")
    parser.add_argument("sim_type", choices=sorted(ANALYZERS), help="Simulation type")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output YAML path (default: expected/<name>.yaml)",
    )
    args = parser.parse_args()

    result = ANALYZERS[args.sim_type](args.hdf5_path)
    # Strip _ref suffix so hdf5_output points to the CI output filename
    ci_name = args.hdf5_path.name.replace("_ref.h5", ".h5")
    expected = expected_from_result(result, args.sim_type, ci_name)

    stem = args.hdf5_path.stem.removesuffix("_ref")
    output_path = args.output or Path("expected") / f"{stem}.yaml"
    dump_expected(expected, output_path)

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
