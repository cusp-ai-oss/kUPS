# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for units.pydantic."""

import ast
import pathlib
import sys
import warnings
from typing import Annotated, Final, Literal, TypeAliasType, get_args, get_origin

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kups.core import units
from kups.core.units import catalog as u
from kups.core.units.pydantic import UnitsModel, _serialize

from ._helpers import arr

# exponents each dimension alias carries at runtime
DIMENSIONS = {
    name: alias.__value__.__metadata__[0].dim
    for name in units.__all__
    if isinstance(alias := getattr(units, name), TypeAliasType)
    and get_origin(alias.__value__) is Annotated
}

CELL = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

type Concentration = units.Quantity  # user alias kups.core.units cannot resolve


class Run(UnitsModel):
    cutoff: units.Length
    e0: units.Energy
    cell: units.Length
    dt: units.Time | None = None
    steps: int = 10
    windows: list[units.Time] = Field(default_factory=list)
    labelled: Annotated[units.Energy, Field(description="a barrier")] | None = None


def test_string_and_mapping_forms():
    run = Run.model_validate(
        {"cutoff": "6 angstrom", "e0": {"eV": 1.5}, "cell": {"angstrom": CELL}}
    )
    assert float(run.cutoff.to(u.angstrom)) == pytest.approx(6.0)
    assert float(run.e0.to(u.eV)) == pytest.approx(1.5)
    assert run.cell.shape == (3, 3)
    assert run.cell.to(u.angstrom).tolist() == CELL


def test_integer_magnitudes_coerced():
    # kups.core.units rejects integer arrays, so the validator casts to float
    run = Run.model_validate(
        {"cutoff": "6 angstrom", "e0": "1 eV", "cell": {"angstrom": [[4, 0, 0]]}}
    )
    assert run.cell.dtype == arr(1.0).dtype
    assert run.cell.to(u.angstrom).tolist() == [[4.0, 0.0, 0.0]]


def test_compound_unit_expression():
    class Grad(UnitsModel):
        forces: units.Force

    grad = Grad.model_validate({"forces": {"eV/angstrom": [[0.1, 0.0, 0.0]]}})
    assert grad.forces.to(u.eV / u.angstrom).tolist() == [
        [pytest.approx(0.1), 0.0, 0.0]
    ]
    with units.base(units.SI):
        assert Grad.model_validate(grad.model_dump()).forces.shape == (1, 3)


def test_dimension_enforced():
    with pytest.raises(ValidationError, match="expected a value in Å"):
        Run.model_validate({"cutoff": "6 eV", "e0": "1 eV", "cell": {"angstrom": CELL}})


@pytest.mark.parametrize(
    "value", [6.0, [1.0, 2.0], {"angstrom": 6, "nm": 6}, {"parsec": 1.0}, None]
)
def test_unreadable_values_rejected(value):
    with pytest.raises(ValidationError):
        Run.model_validate({"cutoff": value, "e0": "1 eV", "cell": {"angstrom": CELL}})


def test_quantities_pass_through():
    run = Run(cutoff=6 * u.angstrom, e0=1.5 * u.eV, cell=arr(CELL) * u.angstrom)
    assert float(run.cutoff.to(u.angstrom)) == pytest.approx(6.0)
    with pytest.raises(ValidationError):
        Run(cutoff=1.5 * u.eV, e0=1.5 * u.eV, cell=arr(CELL) * u.angstrom)


def test_dump_round_trips():
    run = Run.model_validate(
        {"cutoff": "6 angstrom", "e0": "1.5 eV", "cell": {"angstrom": CELL}}
    )
    dumped = run.model_dump()
    assert dumped["cutoff"] == "6.0 Å"
    assert dumped["cell"] == {"Å": CELL}
    back = Run.model_validate(dumped)
    assert float(back.cutoff.to(u.angstrom)) == pytest.approx(6.0)
    assert back.cell.to(u.angstrom).tolist() == CELL


def test_json_round_trips():
    run = Run.model_validate(
        {"cutoff": "6 angstrom", "e0": "1.5 eV", "cell": {"angstrom": CELL}}
    )
    back = Run.model_validate_json(run.model_dump_json())
    assert back.cell.to(u.angstrom).tolist() == CELL
    assert float(back.e0.to(u.eV)) == pytest.approx(1.5)


def test_dump_uses_read_base():
    # the base active at validation time wins, not the one at dump time
    atomistic = Run.model_validate(
        {"cutoff": "6 angstrom", "e0": "1 eV", "cell": {"angstrom": CELL}}
    )
    with units.base(units.SI):
        si = Run.model_validate(
            {"cutoff": "6 angstrom", "e0": "1 eV", "cell": {"m": CELL}}
        )
        assert atomistic.model_dump()["cutoff"] == "6.0 Å"
        assert atomistic.model_dump()["cell"] == {"Å": CELL}
    assert si.model_dump()["cutoff"] == "6e-10 m"


def test_serialized_units_parse_back():
    checked = 0
    for system in (units.SI, units.ATOMISTIC):
        with units.base(system):
            for name, dim in DIMENSIONS.items():
                dumped = _serialize(units.from_dim(1.0, dim))
                read = units.catalog.parse(dumped)
                where = f"{system.name} {name}: {dumped!r}"
                assert units.dim_of(read) == dim, where
                assert float(read.value) == pytest.approx(1.0), where
                checked += 1
            if system is units.SI:
                assert _serialize(units.from_dim(1.0, DIMENSIONS["Entropy"])) == (
                    "1.0 m^2.kg/(s^2.K)"
                )
                assert _serialize(units.from_dim(1.0, DIMENSIONS["Current"])) == "1.0 A"
    assert checked == 2 * len(DIMENSIONS)


def test_nested_annotations():
    run = Run.model_validate(
        {
            "cutoff": "6 angstrom",
            "e0": "1 eV",
            "cell": {"angstrom": CELL},
            "dt": "2 fs",
            "windows": ["1 fs", {"ps": 2}],
            "labelled": "0.5 eV",
        }
    )
    assert run.dt is not None and run.labelled is not None
    assert float(run.dt.to(u.fs)) == pytest.approx(2.0)
    assert [float(w.to(u.fs)) for w in run.windows] == [
        pytest.approx(1.0),
        pytest.approx(2000.0),
    ]
    assert float(run.labelled.to(u.eV)) == pytest.approx(0.5)
    assert (
        Run.model_validate({"cutoff": "1 m", "e0": "1 eV", "cell": {"m": CELL}}).dt
        is None
    )


def test_annotated_metadata_survives():
    class Barrier(UnitsModel):
        height: Annotated[units.Energy, Field(description="a barrier")]

    assert Barrier.model_fields["height"].description == "a barrier"
    assert float(Barrier(height="0.5 eV").height.to(u.eV)) == pytest.approx(0.5)


def test_plain_fields_untouched():
    with pytest.raises(ValidationError):
        Run.model_validate(
            {"cutoff": "1 m", "e0": "1 eV", "cell": {"m": CELL}, "steps": "many"}
        )


def test_literal_members_kept():
    class Choice(UnitsModel):
        mode: Literal["min", "max"]
        cutoff: units.Length | Literal["auto"] = "auto"

    assert get_args(Choice.model_fields["mode"].annotation) == ("min", "max")
    assert Choice.model_validate({"mode": "min"}).mode == "min"
    assert Choice.model_validate({"mode": "max", "cutoff": "1 m"}).cutoff is not None


def test_final_annotation():
    class Frozen(UnitsModel):
        cutoff: Final[units.Length]

    assert float(Frozen(cutoff="6 angstrom").cutoff.to(u.angstrom)) == pytest.approx(
        6.0
    )


def test_user_alias_left_alone():
    # a user alias sharing a dimension name keeps its own meaning
    source = (
        "type Energy = Later\n"
        "from kups.core.units.pydantic import UnitsModel\n"
        "class Mine(UnitsModel):\n"
        "    e: Energy\n"
        "class Later:\n"
        "    pass\n"
    )
    module = type(sys)("units_pydantic_alias_cfg")
    sys.modules["units_pydantic_alias_cfg"] = module
    exec(compile(source, "<alias>", "exec"), module.__dict__)
    assert module.Mine.model_fields["e"].annotation is module.Energy


@pytest.mark.parametrize(
    "magnitudes", [{"a": 1}, {}, [{"a": 1}], "six", [[1.0], [2.0, 3.0]]]
)
def test_bad_magnitudes_rejected(magnitudes):
    with pytest.raises(ValidationError):
        Run.model_validate(
            {"cutoff": "1 m", "e0": "1 eV", "cell": {"angstrom": magnitudes}}
        )


def test_no_raw_exception_escapes():
    for value in ({"m": 10**400}, {"m": [10**400]}, "1 m^1/0"):
        with pytest.raises(ValidationError):
            Run.model_validate(
                {"cutoff": value, "e0": "1 eV", "cell": {"angstrom": CELL}}
            )
    with pytest.raises(units.DimensionError, match="denominator"):
        u.parse("1 m^1/0")


def test_large_value_elided():
    with pytest.raises(ValidationError) as caught:
        Run.model_validate(
            {"cutoff": [1.0] * 100_000, "e0": "1 eV", "cell": {"angstrom": CELL}}
        )
    assert len(str(caught.value)) < 600


def test_future_annotations_resolved():
    source = (
        "from __future__ import annotations\n"
        "from kups.core import units\n"
        "from kups.core.units.pydantic import UnitsModel\n"
        "class Cfg(UnitsModel):\n"
        "    cutoff: units.Length\n"
        "    cells: list[units.Length] = []\n"
    )
    namespace: dict[str, object] = {"__name__": "units_pydantic_future_cfg"}
    sys.modules["units_pydantic_future_cfg"] = type(sys)("units_pydantic_future_cfg")
    sys.modules["units_pydantic_future_cfg"].__dict__.update(namespace)
    exec(
        compile(source, "<cfg>", "exec"),
        sys.modules["units_pydantic_future_cfg"].__dict__,
    )
    cfg = sys.modules["units_pydantic_future_cfg"].Cfg
    parsed = cfg.model_validate(
        {"cutoff": "6 angstrom", "cells": [{"angstrom": [1.0]}]}
    )
    assert float(parsed.cutoff.to(u.angstrom)) == pytest.approx(6.0)
    assert parsed.cells[0].to(u.angstrom).tolist() == [pytest.approx(1.0)]


def test_yaml_document():
    yaml = pytest.importorskip("yaml")
    text = (
        "cutoff: 6 angstrom\n"
        "e0: {eV: 1.5}\n"
        "cell:\n"
        "  angstrom:\n"
        "    - [4, 0, 0]\n"
        "    - [0, 4, 0]\n"
        "    - [0, 0, 4]\n"
        "windows: [1 fs, 2 fs]\n"
    )
    run = Run.model_validate(yaml.safe_load(text))
    assert run.cell.to(u.angstrom).tolist() == CELL
    assert len(run.windows) == 2


def test_json_schema():
    schema = Run.model_json_schema()
    assert schema["properties"]["cutoff"] == {"$ref": "#/$defs/Length"}
    assert schema["$defs"]["Length"]["anyOf"][0]["type"] == "string"
    assert "a quantity in Å" in schema["$defs"]["Length"]["description"]
    assert "a quantity in eV" in schema["$defs"]["Energy"]["description"]
    with units.base(units.SI):
        schema = Run.model_json_schema()
    assert "a quantity in m" in schema["$defs"]["Length"]["description"]


def test_json_schema_quantity_default():
    class Held(UnitsModel):
        temperature: units.Temperature = 300.0 * u.K
        spacing: units.Length = Field(
            default_factory=lambda: arr([[4.0, 0.0]]) * u.angstrom
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # pydantic warns when it drops a default
        properties = Held.model_json_schema()["properties"]
    assert properties["temperature"]["default"] == "300.0 K"
    assert properties["spacing"].get("default") is None  # a factory has no default


def test_multi_unit_mapping_rejected():
    merged = {"angstrom": [[4.0, 0.0]], "nm": [[0.4, 0.0]]}
    with pytest.raises(ValidationError, match="names 2 units"):
        Run.model_validate({"cutoff": "1 m", "e0": "1 eV", "cell": merged})


def test_core_does_not_import_pydantic():
    # only units.pydantic imports pydantic, so importing kups.core.units stays light
    package = pathlib.Path(units.__file__).parent
    for module in package.glob("*.py"):
        if module.name == "pydantic.py":
            continue
        for node in ast.walk(ast.parse(module.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(n.split(".")[0] == "pydantic" for n in names), (
                f"{module.name} imports pydantic; only units.pydantic may"
            )


def test_dimensions_match_aliases():
    stubs = pathlib.Path(units.__file__).with_name("_core.py").read_text()
    declared: dict[str, tuple[int, ...]] = {}
    for node in ast.walk(ast.parse(stubs)):
        if (
            not isinstance(node, ast.TypeAlias)
            or not isinstance(node.value, ast.Subscript)
            or node.name.id not in units.__all__
        ):
            continue
        exponents = node.value.slice
        if (
            isinstance(exponents, ast.Tuple)
            and ast.unparse(node.value.value) == "Quantity"
        ):
            declared[node.name.id] = tuple(ast.literal_eval(e) for e in exponents.elts)
    assert declared == DIMENSIONS and len(DIMENSIONS) == 27


def test_model_eq_raises_dumps_compare():
    values = {"cutoff": "6 angstrom", "e0": "1 eV", "cell": {"angstrom": CELL}}
    one, two = Run.model_validate(values), Run.model_validate(values)
    with pytest.raises(units.DimensionError, match="not defined on quantities"):
        _ = one == two
    assert one.model_dump() == two.model_dump()


def test_bare_quantity_refused():
    with pytest.raises(units.DimensionError, match="names no dimension"):

        class Mix(UnitsModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            amount: Concentration


def test_plain_base_model_validates():
    class Plain(BaseModel):
        cutoff: units.Length

    assert float(Plain(cutoff="6 angstrom").cutoff.to(u.angstrom)) == pytest.approx(6.0)
    with pytest.raises(ValidationError, match="expected a value in"):
        Plain(cutoff="1 eV")
