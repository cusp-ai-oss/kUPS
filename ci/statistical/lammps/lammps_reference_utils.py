#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Utilities shared by the LAMMPS LJ Argon statistical references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from kups.core.constants import BAR
from kups.core.utils.block_average import block_average

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
EXPECTED_DIR = ROOT / "expected"
OUTPUT_DIR = ROOT / "outputs" / "reference"
KUPS_EXPECTED_DIR = ROOT.parent / "expected"

PRESSURE_COLUMN = "press_bar"
CONFIG_PRESSURE_COLUMN = "pressure_config"
N_BLOCKS = 5

CASES: dict[str, dict[str, Any]] = {
    "md_nve_lj_argon_lammps": {
        "input": "inputs/reference/in.md_nve_lj_argon",
        "output": "outputs/reference/md_nve_lj_argon_lammps.dat",
        "reference_yaml": "expected/md_nve_lj_argon_lammps.yaml",
        "kups_expected": "md_nve_lj_argon",
        "compare_to_kups": True,
        "max_z": 10.0,
        "abs_tolerances": {"volume": 1.0e-3},
        "observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
        "compare_observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
    },
    "md_nvt_lj_argon_lammps": {
        "input": "inputs/reference/in.md_nvt_lj_argon",
        "output": "outputs/reference/md_nvt_lj_argon_lammps.dat",
        "reference_yaml": "expected/md_nvt_lj_argon_lammps.yaml",
        "kups_expected": "md_nvt_lj_argon",
        "compare_to_kups": True,
        "max_z": 10.0,
        "abs_tolerances": {"volume": 1.0e-3},
        "observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
        "compare_observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
    },
    "md_npt_lj_argon_iso_lammps": {
        "input": "inputs/reference/in.md_npt_lj_argon_iso",
        "output": "outputs/reference/md_npt_lj_argon_iso_lammps.dat",
        "reference_yaml": "expected/md_npt_lj_argon_iso_lammps.yaml",
        "kups_expected": "md_npt_lj_argon",
        "compare_to_kups": True,
        "max_z": 3.0,
        "observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
        "compare_observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
    },
    "md_npt_lj_argon_tri_lammps": {
        "input": "inputs/reference/in.md_npt_lj_argon_tri",
        "output": "outputs/reference/md_npt_lj_argon_tri_lammps.dat",
        "reference_yaml": "expected/md_npt_lj_argon_tri_lammps.yaml",
        "kups_expected": "md_npt_lj_argon_baoab",
        "compare_to_kups": True,
        "max_z": 3.0,
        "observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
        "compare_observables": {
            "pe": "potential_energy",
            "ke": "kinetic_energy",
            "etotal": "total_energy",
            "temp": "temperature",
            "pressure_config": "pressure",
            "volume": "volume",
        },
    },
}


def block_stats(x: np.ndarray, n_blocks: int) -> dict[str, float]:
    """Return mean and SEM using the shared kUPS block-average utility."""
    result = block_average(jnp.asarray(x, dtype=float), n_blocks=n_blocks)
    return {
        "mean": float(result.mean),
        "sem": float(result.sem),
    }


def analyze_output(path: Path) -> dict[str, dict[str, float]]:
    """Analyze a LAMMPS fix-print table using the kUPS MD SEM convention."""
    arr = np.genfromtxt(path, names=True)
    cols = arr.dtype.names or ()
    if PRESSURE_COLUMN not in cols:
        raise ValueError(f"{path} is missing required {PRESSURE_COLUMN!r} column")
    kinetic_pressure = 2.0 * arr["ke"] / (3.0 * arr["volume"])
    config_pressure = arr[PRESSURE_COLUMN] * BAR - kinetic_pressure
    series = {name: arr[name] for name in cols if name not in ("step", PRESSURE_COLUMN)}
    series[CONFIG_PRESSURE_COLUMN] = config_pressure

    return {name: block_stats(values, N_BLOCKS) for name, values in series.items()}


def reference_yaml_path(case_name: str) -> Path:
    """Return the committed kUPS-format YAML path for a LAMMPS case."""
    return ROOT / str(CASES[case_name]["reference_yaml"])


def output_stats_to_expected(
    case_name: str, stats: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Convert LAMMPS column statistics to the kUPS expected-YAML shape."""
    config = CASES[case_name]
    observables = {}
    for lammps_obs, kups_obs in config["observables"].items():
        obs_stats = stats[lammps_obs]
        observables[kups_obs] = {
            "expected_mean": obs_stats["mean"],
            "expected_sem": obs_stats["sem"],
        }
    return {
        "simulation_type": "md",
        "hdf5_output": Path(config["output"]).name,
        "observables": observables,
    }


def analyze_all_outputs() -> dict[str, Any]:
    """Analyze all configured reference outputs into kUPS expected-YAML shape."""
    references = {}
    for case_name, config in CASES.items():
        output = ROOT / config["output"]
        if not output.exists():
            raise FileNotFoundError(output)
        references[case_name] = output_stats_to_expected(
            case_name, analyze_output(output)
        )
    return references


def load_reference_stats() -> dict[str, Any]:
    references = {}
    for case_name in CASES:
        with reference_yaml_path(case_name).open() as f:
            references[case_name] = yaml.safe_load(f)
    return references


def write_reference_stats(stats: dict[str, Any]) -> None:
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    for case_name, expected in stats.items():
        with reference_yaml_path(case_name).open("w") as f:
            yaml.dump(expected, f, sort_keys=False, default_flow_style=False)


def load_kups_expected(name: str) -> dict[str, Any]:
    with (KUPS_EXPECTED_DIR / f"{name}.yaml").open() as f:
        return yaml.safe_load(f)["observables"]
