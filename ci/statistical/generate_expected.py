#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generate expected values YAML from simulation HDF5 output."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml
from expected_values import AnalysisResult, ExpectedSpec, ObservableSpec
from validate import ANALYZERS

from kups.core.utils.block_average import BlockAverageResult

jax.config.update("jax_enable_x64", True)


def block_result_to_expected(result: BlockAverageResult) -> ObservableSpec:
    """Convert BlockAverageResult to expected values dict.

    Args:
        result: Block average of a single observable.

    Returns:
        The observable's expected mean and standard error, as scalars for a
        one-element observable and as lists otherwise.
    """
    mean = jnp.atleast_1d(result.mean).tolist()
    sem = jnp.atleast_1d(result.sem).tolist()
    return {
        "expected_mean": mean[0] if len(mean) == 1 else mean,
        "expected_sem": sem[0] if len(sem) == 1 else sem,
    }


def analysis_result_to_expected(
    result: AnalysisResult, sim_type: str, hdf5_name: str
) -> ExpectedSpec:
    """Convert any analysis result dataclass to expected values dict.

    Args:
        result: Analysis result whose block-averaged fields are recorded.
        sim_type: Simulation kind, `md`, `nvt` or `gcmc`.
        hdf5_name: Name of the output the CI run validates.

    Returns:
        The expected values of every block-averaged field.
    """
    observables: dict[str, ObservableSpec] = {}
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
    expected = analysis_result_to_expected(result, args.sim_type, ci_name)

    stem = args.hdf5_path.stem.removesuffix("_ref")
    output_path = args.output or Path("expected") / f"{stem}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(expected, f, default_flow_style=False, sort_keys=False)

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
