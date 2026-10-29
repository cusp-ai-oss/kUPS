# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Units of the statistical CI's expected values."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import pytest

from kups.core import units
from kups.core.units import catalog as u
from kups.core.utils.block_average import BlockAverageResult

sys.path.insert(0, str(Path(__file__).parents[2] / "ci" / "statistical"))

from expected_values import (  # noqa: E402
    block_result_to_expected,
    magnitudes_in,
    unit_of,
)
from validate import validate_observable  # noqa: E402


def test_unit_round_trip():
    pressure = jnp.array([1e-3, 2e-3]) * u.eV / u.angstrom**3
    with units.base(units.SI):
        stored = pressure.rebase()
    for value in (pressure, stored):
        unit = unit_of(value)
        assert unit == "eV/angstrom^3"
        assert magnitudes_in(value, unit).tolist() == pytest.approx([1e-3, 2e-3])
    assert unit_of(jnp.array(3.0)) is None
    assert float(magnitudes_in(jnp.array(3.0), None)) == 3.0


def test_unknown_or_wrong_unit_refused():
    speed = 1.0 * u.angstrom / u.fs
    with pytest.raises(ValueError, match="no expected-value unit"):
        unit_of(speed)
    energy = 1.0 * u.eV
    for wrong in ("K", None):
        with pytest.raises(ValueError, match="not in"):
            magnitudes_in(energy, wrong)


def test_block_result_to_expected():
    scalar = BlockAverageResult(2.0 * u.eV, 0.1 * u.eV, n_blocks=5)
    assert block_result_to_expected(scalar) == {
        "unit": "eV",
        "expected_mean": 2.0,
        "expected_sem": pytest.approx(0.1),
    }
    tensor = BlockAverageResult(
        jnp.array([1.0, 2.0]) * u.K, jnp.array([0.5, 0.5]) * u.K, n_blocks=5
    )
    assert block_result_to_expected(tensor)["expected_mean"] == [1.0, 2.0]
    plain = BlockAverageResult(jnp.array(0.5), jnp.array(0.1), n_blocks=5)
    assert "unit" not in block_result_to_expected(plain)


def test_validate_observable_checks_the_unit():
    result = BlockAverageResult(2.0 * u.eV, 0.1 * u.eV, n_blocks=5)
    spec = block_result_to_expected(result)
    assert validate_observable("energy", result, spec)
    for unit in ("K", "eV/angstrom^3"):
        assert not validate_observable("energy", result, {**spec, "unit": unit})
    missing = {k: v for k, v in spec.items() if k != "unit"}
    assert not validate_observable("energy", result, missing)


def _lammps_validate() -> ModuleType:
    """The LAMMPS comparison script, under its own module name."""
    lammps = Path(__file__).parents[2] / "ci" / "statistical" / "lammps"
    sys.path.insert(0, str(lammps))
    spec = importlib.util.spec_from_file_location(
        "lammps_validate", lammps / "validate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lammps_comparison_checks_units(monkeypatch: pytest.MonkeyPatch):
    validate = _lammps_validate()
    reference_utils = importlib.import_module("lammps_reference_utils")
    compared = [c for c, cfg in validate.CASES.items() if cfg["compare_to_kups"]]
    # every LAMMPS column is tagged with the unit of its kUPS observable
    for case in compared:
        columns = validate.CASES[case]["observables"]
        stats = {col: {"mean": 1.0, "sem": 0.1} for col in columns}
        tagged = reference_utils.output_stats_to_expected(case, stats)["observables"]
        for obs, spec in tagged.items():
            assert spec["unit"] == reference_utils.OBSERVABLE_UNITS[obs]
    obs = "temperature"
    reference = {obs: {"unit": "K", "expected_mean": 1.0, "expected_sem": 0.1}}
    monkeypatch.setattr(validate, "load_kups_expected", lambda name: reference)
    current = {c: {"observables": {obs: dict(reference[obs])}} for c in compared}
    assert validate._compare_to_kups(current)
    for case in compared:
        current[case]["observables"][obs]["unit"] = "eV"
    assert not validate._compare_to_kups(current)
