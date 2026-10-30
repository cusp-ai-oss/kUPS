# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Schema and units of the statistical expected-value YAML files.

Every observable states the unit its numbers are read in, so a committed
reference says what it measures instead of leaving a bare magnitude whose
meaning depends on the base kUPS happens to run in. A dimensionless observable,
such as a loading, names no unit.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import jax.numpy as jnp
import yaml
from jax import Array

from kups.application.mcmc.analysis import MCMCAnalysisResult
from kups.application.md.analysis import MDAnalysisResult
from kups.core import units
from kups.core.units import catalog as u
from kups.core.utils.block_average import BlockAverageResult
from kups.core.utils.units import Measured

type AnalysisResult = MCMCAnalysisResult | MDAnalysisResult
"""An analysis dataclass whose block-averaged fields become expected values."""

type Numbers = float | list[Any]
"""One expected number, or the nested lists of a tensor-valued observable."""

UNIT_NAMES: tuple[str, ...] = ("eV", "K", "angstrom^3", "eV/angstrom^3")
"""The unit an observable is written in, one per dimension the analyses report."""

_UNIT_BY_DIM: dict[tuple[Any, ...], str] = {
    tuple(units.dim_of(u.parse(f"1 {name}"))): name for name in UNIT_NAMES
}


class ObservableSpec(TypedDict):
    """One observable's expected mean and standard error.

    Attributes:
        unit: The unit both numbers are read in, absent for a dimensionless
            observable.
        expected_mean: Expected mean, nested lists for a tensor observable.
        expected_sem: Expected standard error, shaped like `expected_mean`.
    """

    unit: NotRequired[str]
    expected_mean: Numbers
    expected_sem: Numbers


class ExpectedSpec(TypedDict):
    """One committed expected-value file.

    Attributes:
        simulation_type: Simulation kind, `md`, `nvt` or `gcmc`.
        hdf5_output: Name of the output the CI run validates.
        observables: Expected values keyed by analysis field name.
    """

    simulation_type: str
    hdf5_output: str
    observables: dict[str, ObservableSpec]


def unit_of(value: Measured) -> str | None:
    """The unit an expected value of ``value``'s dimension is written in.

    Args:
        value: A block-averaged mean or standard error.

    Returns:
        The unit's name, or `None` when the value carries no dimension.

    Raises:
        ValueError: If no expected-value unit measures what `value` does.
    """
    if not units.is_quantity(value):
        return None
    dim = units.dim_of(value)
    name = _UNIT_BY_DIM.get(tuple(dim))
    if name is None:
        raise ValueError(
            f"no expected-value unit for {units.dim_str(dim)}; "
            f"add one to UNIT_NAMES ({', '.join(UNIT_NAMES)})"
        )
    return name


def magnitudes_in(value: Measured, unit: str | None) -> Array:
    """Read ``value`` in ``unit`` as the plain magnitudes a YAML file holds.

    Args:
        value: A block-averaged mean or standard error.
        unit: The unit to read it in, or `None` for a dimensionless value.

    Returns:
        The magnitudes, as an array of the same shape.

    Raises:
        ValueError: If `value` does not measure what `unit` does, `unit` being
            `None` included.
        units.DimensionError: If `unit` is not a unit.
    """
    if unit is None:
        if units.is_quantity(value):
            raise ValueError(
                f"the value is measured in {unit_of(value)}, "
                "but the expected values name no unit"
            )
        return jnp.asarray(value)
    ratio = value / u.parse(f"1 {unit}")
    if units.is_quantity(ratio):
        raise ValueError(
            f"the value is measured in {unit_of(value) or 'no unit'}, not in {unit}"
        )
    return jnp.asarray(ratio)


def scalar(value: Numbers) -> float:
    """One expected number, rejecting a tensor-valued observable.

    Args:
        value: An expected mean or standard error.

    Returns:
        The number.

    Raises:
        TypeError: If the observable is tensor-valued.
    """
    if isinstance(value, list):
        raise TypeError("a tensor-valued observable has no single expected number")
    return float(value)


def block_result_to_expected(result: BlockAverageResult[Any]) -> ObservableSpec:
    """Convert a block average to the expected values of one observable.

    Args:
        result: Block average of a single observable.

    Returns:
        The observable's expected mean and standard error, in the unit named
        for its dimension.
    """
    unit = unit_of(result.mean)
    mean = jnp.atleast_1d(magnitudes_in(result.mean, unit)).tolist()
    sem = jnp.atleast_1d(magnitudes_in(result.sem, unit)).tolist()
    expected_mean: Numbers = mean[0] if len(mean) == 1 else mean
    expected_sem: Numbers = sem[0] if len(sem) == 1 else sem
    if unit is None:
        return {"expected_mean": expected_mean, "expected_sem": expected_sem}
    return {
        "unit": unit,
        "expected_mean": expected_mean,
        "expected_sem": expected_sem,
    }


def expected_from_result(
    result: AnalysisResult, sim_type: str, hdf5_name: str
) -> ExpectedSpec:
    """Convert an analysis result to a full expected-value file.

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


def load_expected(path: Path) -> ExpectedSpec:
    """Read a committed expected-value file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed expected values.
    """
    with path.open() as f:
        return yaml.safe_load(f)


def dump_expected(spec: ExpectedSpec, path: Path) -> None:
    """Write an expected-value file, creating its directory if needed.

    Args:
        spec: Expected values to write.
        path: Path to write the YAML file to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
