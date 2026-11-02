# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pydantic support: the annotation fixes the dimension, the data supplies the unit.

The core imports it only when pydantic builds a schema, so pydantic loads on use.

    >>> from pydantic import BaseModel
    >>> from kups.core import units
    >>> class Run(BaseModel):
    ...     cutoff: units.Length
    ...     cell: units.Length
    >>> run = Run.model_validate(
    ...     {"cutoff": "6 angstrom", "cell": {"angstrom": [[4, 0], [0, 4]]}}
    ... )
    >>> run.cutoff
    6.0 Å
    >>> run.cell.shape
    (2, 2)

Each dimension alias carries a schema hook that enforces its dimension; a bare
``units.Quantity`` field takes a quantity of any dimension. The unit always comes
from the data; any unit of the right dimension is accepted, for scalars and arrays
alike. Dumps, hints and quantity defaults in the JSON schema use the alias's
display unit, whatever base was active:

    >>> Run.model_validate({"cutoff": "0.6 nm", "cell": "4 angstrom"}).model_dump()
    {'cutoff': '6.0 Å', 'cell': '4.0 Å'}
"""

import reprlib
from collections.abc import Mapping
from typing import Final, TypeAliasType

import jax
import jax.numpy as jnp
import numpy as np
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from kups.core import units
from kups.core.units import catalog as u
from kups.core.units._core import AnyQuantity, Dimension
from kups.core.units._unit import AnyUnit, Dim

# the display unit of each dimension, as its alias declares it
_DISPLAY: Final[dict[Dim, AnyUnit]] = {
    hook.unit.dim: hook.unit
    for name in units.__all__
    if isinstance(alias := getattr(units, name), TypeAliasType)
    for hook in getattr(alias.__value__, "__metadata__", ())
    if isinstance(hook, Dimension)
}


def _forms(unit: AnyUnit | None) -> str:
    """Describe the accepted input forms, with examples in ``unit`` if one is fixed."""
    shown = unit or u.angstrom
    return f"'1.5 {shown}' or {{'{shown}': [1.5, 2.5]}}; " + (
        "any parseable unit of the same dimension is accepted"
        if unit
        else "any parseable unit is accepted"
    )


def _wanted(unit: AnyUnit | None) -> str:
    """Name what a field takes: a value in ``unit``, or any quantity."""
    return f"a value in {unit}" if unit else "a quantity"


# truncates rejected values in error messages, like pydantic
_REPR: Final[reprlib.Repr] = reprlib.Repr(maxlist=8, maxstring=80, maxother=200)


def _parse(text: str) -> AnyQuantity | jax.Array:
    """Parse ``text`` with ``u.parse``, raising ValueError so pydantic reports it."""
    try:
        return u.parse(text)
    except units.DimensionError as err:
        raise ValueError(str(err)) from err


def _read(value: object, unit: AnyUnit | None) -> AnyQuantity | jax.Array:
    """Parse a field value into a quantity, without checking its dimension."""
    if units.is_quantity(value):
        return value
    if isinstance(value, str):
        return _parse(value)
    if isinstance(value, Mapping) and len(value) == 1:
        ((key, magnitudes),) = value.items()
        if isinstance(key, str):
            try:  # force float rather than rely on promotion
                numbers = jnp.asarray(magnitudes, dtype=float)
            except Exception as err:  # ragged shapes etc.
                raise ValueError(
                    f"units.pydantic: {_REPR.repr(magnitudes)} is not magnitudes "
                    f"{key!r} can measure: {err}"
                ) from err
            return numbers * _parse(f"1 {key}")
    if isinstance(value, Mapping) and len(value) > 1:
        raise ValueError(
            f"units.pydantic: this names {len(value)} units, {tuple(value)}. "
            "Supply one unit-to-magnitudes entry."
        )
    raise ValueError(
        f"units.pydantic: cannot read {_REPR.repr(value)} as {_wanted(unit)}. "
        f"Give it as {_forms(unit)}."
    )


def _validate(value: object, unit: AnyUnit | None) -> AnyQuantity | jax.Array:
    """Validate a field value against its annotated dimension, if it has one."""
    quantity = _read(value, unit)
    if unit is None and not units.is_quantity(quantity):
        raise ValueError(
            f"units.pydantic: expected a quantity, got {_REPR.repr(value)}, which "
            "has no unit"
        )
    try:
        if unit is not None:
            units.require_dim(quantity, unit)
    except units.DimensionError as err:
        raise ValueError(str(err)) from err
    return quantity


def _serialize(quantity: AnyQuantity) -> str | dict[str, object]:
    """Serialize to ``"6.0 Å"`` for scalars or ``{"Å": [[...]]}`` for arrays.

    The magnitudes are in the display unit of the quantity's dimension, rounded
    to the digits their dtype resolves so conversion noise does not show. The
    dump is for display: values with more digits than that lose them, so
    validating it back is exact only for magnitudes that short.
    """
    dim = units.dim_of(quantity)
    unit = _DISPLAY.get(dim) or units.active_base().unit(dim)
    magnitudes = quantity.to(unit)
    digits = jnp.finfo(magnitudes.dtype).precision
    values = np.vectorize(lambda x: float(f"{x:.{digits}g}"), otypes=[float])(
        magnitudes
    ).tolist()
    if magnitudes.ndim == 0:
        return f"{values} {unit}"
    return {str(unit): values}


def _serialize_field(value: object, unit: AnyUnit | None) -> str | dict[str, object]:
    """Serialize a field's value, refusing one that is not ``_wanted(unit)``.

    A model without ``validate_assignment`` can hold any value after assignment.
    """
    if not units.is_quantity(value) or (unit and units.dim_of(value) != unit.dim):
        raise ValueError(
            f"units.pydantic: {_REPR.repr(value)} is not {_wanted(unit)}, so it "
            "cannot be dumped; assign a quantity or turn on validate_assignment."
        )
    return _serialize(value)


def unit_core_schema(unit: AnyUnit | None) -> core_schema.CoreSchema:
    """Build a plain validator schema, which also stops dataclass introspection.

    ``unit`` fixes the field's dimension; None takes a quantity of any dimension.
    """
    return core_schema.no_info_plain_validator_function(
        lambda value: _validate(value, unit),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda value: _serialize_field(value, unit), when_used="always"
        ),
    )


def unit_json_schema(unit: AnyUnit | None) -> JsonSchemaValue:
    """Build the JSON schema for the two accepted input forms."""
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "minProperties": 1, "maxProperties": 1},
        ],
        "description": f"{f'a quantity in {unit}' if unit else 'a quantity'}: "
        f"{_forms(unit)}",
    }
