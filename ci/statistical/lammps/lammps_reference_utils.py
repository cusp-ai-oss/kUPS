#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Utilities shared by the LAMMPS LJ Argon statistical references."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from kups.core.units import catalog as u
from kups.core.utils.block_average import block_average

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from expected_values import (  # noqa: E402  the schema lives next door
    ExpectedSpec,
    ObservableSpec,
    dump_expected,
    load_expected,
    scalar,
)

EXPECTED_DIR = ROOT / "expected"
OUTPUT_DIR = ROOT / "outputs" / "reference"
KUPS_EXPECTED_DIR = ROOT.parent / "expected"

PRESSURE_COLUMN = "press_bar"
CONFIG_PRESSURE_COLUMN = "pressure_config"
N_BLOCKS = 5

OBSERVABLE_UNITS: dict[str, str] = {
    "potential_energy": "eV",
    "kinetic_energy": "eV",
    "total_energy": "eV",
    "temperature": "K",
    "pressure": "eV/angstrom^3",
    "volume": "angstrom^3",
}
"""The unit each kUPS observable's LAMMPS column is converted into."""


class CaseConfig(TypedDict):
    """One LAMMPS reference case.

    Attributes:
        input: LAMMPS input deck, relative to this directory.
        output: Generated fix-print table, relative to this directory.
        reference_yaml: Committed kUPS-format statistics, relative to here.
        kups_expected: Name of the kUPS expected-value file to compare against.
        compare_to_kups: Whether CI compares this case to that file.
        max_z: How many combined SEMs the two means may differ by.
        observables: kUPS observable name per LAMMPS column, for the YAML.
        compare_observables: The subset compared against kUPS.
        abs_tolerances: Absolute tolerance per LAMMPS column, taken as a pass
            even when the z score is larger.
    """

    input: str
    output: str
    reference_yaml: str
    kups_expected: str
    compare_to_kups: bool
    max_z: float
    observables: dict[str, str]
    compare_observables: dict[str, str]
    abs_tolerances: NotRequired[dict[str, float]]


CASES: dict[str, CaseConfig] = {
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


_BAR = float((1.0 * u.bar).to(u.eV / u.angstrom**3))
"""One bar in the eV per cubic Angstrom the kUPS logs are written in."""


def block_stats(x: np.ndarray, n_blocks: int) -> dict[str, float]:
    """Return mean and SEM using the shared kUPS block-average utility.

    Args:
        x: One column of a LAMMPS table.
        n_blocks: Number of blocks to average over.

    Returns:
        The column's ``mean`` and ``sem``.
    """
    result = block_average(jnp.asarray(x, dtype=float), n_blocks=n_blocks)
    return {
        "mean": float(result.mean),
        "sem": float(result.sem),
    }


def analyze_output(path: Path) -> dict[str, dict[str, float]]:
    """Analyze a LAMMPS fix-print table using the kUPS MD SEM convention.

    Args:
        path: Fix-print table with a header row of column names.

    Returns:
        Mean and SEM per column, with the total pressure column replaced by the
        configurational pressure.

    Raises:
        ValueError: If the table has no pressure column.
    """
    arr = np.genfromtxt(path, names=True)
    cols = arr.dtype.names or ()
    if PRESSURE_COLUMN not in cols:
        raise ValueError(f"{path} is missing required {PRESSURE_COLUMN!r} column")
    kinetic_pressure = 2.0 * arr["ke"] / (3.0 * arr["volume"])
    config_pressure = arr[PRESSURE_COLUMN] * _BAR - kinetic_pressure
    series = {name: arr[name] for name in cols if name not in ("step", PRESSURE_COLUMN)}
    series[CONFIG_PRESSURE_COLUMN] = config_pressure

    return {name: block_stats(values, N_BLOCKS) for name, values in series.items()}


def reference_yaml_path(case_name: str) -> Path:
    """Return the committed kUPS-format YAML path for a LAMMPS case.

    Args:
        case_name: Key into `CASES`.

    Returns:
        Path of the case's reference statistics.
    """
    return ROOT / CASES[case_name]["reference_yaml"]


def output_stats_to_expected(
    case_name: str, stats: dict[str, dict[str, float]]
) -> ExpectedSpec:
    """Convert LAMMPS column statistics to the kUPS expected-YAML shape.

    Args:
        case_name: The case the statistics come from.
        stats: Mean and SEM per LAMMPS column.

    Returns:
        The expected values, each naming the unit its column was converted into.
    """
    config = CASES[case_name]
    observables: dict[str, ObservableSpec] = {}
    for lammps_obs, kups_obs in config["observables"].items():
        obs_stats = stats[lammps_obs]
        observables[kups_obs] = {
            "unit": OBSERVABLE_UNITS[kups_obs],
            "expected_mean": obs_stats["mean"],
            "expected_sem": obs_stats["sem"],
        }
    return {
        "simulation_type": "md",
        "hdf5_output": Path(config["output"]).name,
        "observables": observables,
    }


def analyze_all_outputs() -> dict[str, ExpectedSpec]:
    """Analyze all configured reference outputs into kUPS expected-YAML shape.

    Returns:
        The expected values of every case, keyed by case name.

    Raises:
        FileNotFoundError: If a case's generated table is missing.
    """
    references: dict[str, ExpectedSpec] = {}
    for case_name, config in CASES.items():
        output = ROOT / config["output"]
        if not output.exists():
            raise FileNotFoundError(output)
        references[case_name] = output_stats_to_expected(
            case_name, analyze_output(output)
        )
    return references


def load_reference_stats() -> dict[str, ExpectedSpec]:
    """Read the committed LAMMPS statistics of every case.

    Returns:
        The expected values of every case, keyed by case name.
    """
    return {
        case_name: load_expected(reference_yaml_path(case_name)) for case_name in CASES
    }


def write_reference_stats(stats: dict[str, ExpectedSpec]) -> None:
    """Write the LAMMPS statistics of every case.

    Args:
        stats: Expected values keyed by case name.
    """
    for case_name, expected in stats.items():
        dump_expected(expected, reference_yaml_path(case_name))


def compared_stats(spec: ObservableSpec) -> tuple[str | None, float, float]:
    """The unit, mean and SEM of a scalar observable to compare.

    Args:
        spec: One observable's expected values.

    Returns:
        Its unit, or `None` when it is dimensionless, its expected mean and its
        expected standard error.
    """
    return spec.get("unit"), scalar(spec["expected_mean"]), scalar(spec["expected_sem"])


def load_kups_expected(name: str) -> dict[str, ObservableSpec]:
    """Read the observables of a kUPS expected-value file.

    Args:
        name: Stem of the file under the kUPS ``expected`` directory.

    Returns:
        Expected values keyed by analysis field name.
    """
    return load_expected(KUPS_EXPECTED_DIR / f"{name}.yaml")["observables"]
