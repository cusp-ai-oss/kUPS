# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Schema of the statistical expected-value YAML files."""

from __future__ import annotations

from typing import TypedDict

from kups.application.mcmc.analysis import MCMCAnalysisResult
from kups.application.md.analysis import MDAnalysisResult

type AnalysisResult = MCMCAnalysisResult | MDAnalysisResult
"""An analysis dataclass whose block-averaged fields become expected values."""

type Numbers = float | list[Numbers]
"""One expected number, or the nested lists of a tensor-valued observable."""


class ObservableSpec(TypedDict):
    """One observable's expected mean and standard error.

    Attributes:
        expected_mean: Expected mean, nested lists for a tensor observable.
        expected_sem: Expected standard error, shaped like `expected_mean`.
    """

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
