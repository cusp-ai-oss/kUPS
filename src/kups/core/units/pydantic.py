# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Pydantic support: the annotation fixes the dimension, the data supplies the unit.

The core imports it only when pydantic builds a schema, so pydantic loads on use.

    >>> from kups.core import units
    >>> from kups.core.units.pydantic import UnitsModel
    >>> class Run(UnitsModel):
    ...     cutoff: units.Length
    ...     cell: units.Length
    >>> run = Run.model_validate(
    ...     {"cutoff": "6 angstrom", "cell": {"angstrom": [[4, 0], [0, 4]]}}
    ... )
    >>> run.cutoff
    6.0 Å
    >>> run.cell.shape
    (2, 2)

Each dimension alias carries a schema hook that enforces its dimension, so plain
``BaseModel`` fields work too; ``UnitsModel`` adds quantity defaults to the JSON
schema. The unit always comes from the data; any unit of the right dimension is
accepted, for scalars and arrays alike.
"""

import reprlib
from collections.abc import Mapping
from typing import Any, Final, Literal, override

import jax.numpy as jnp
from pydantic import BaseModel
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
)
from pydantic_core import core_schema

from kups.core import units
from kups.core.units import catalog as u
from kups.core.units._unit import Dim

__all__ = ["UnitsModel"]


def _forms(dim: Dim) -> str:
    """Describe the accepted input forms, with examples in ``dim``'s unit."""
    symbol = units.active_base().symbol(dim) or "1"
    return (
        f"'1.5 {symbol}' or {{'{symbol}': [1.5, 2.5]}}; "
        "any parseable unit of the same dimension is accepted"
    )


# truncates rejected values in error messages, like pydantic
_REPR: Final[reprlib.Repr] = reprlib.Repr(maxlist=8, maxstring=80, maxother=200)


def _parse(text: str) -> Any:
    """Parse ``text`` with ``u.parse``, raising ValueError so pydantic reports it."""
    try:
        return u.parse(text)
    except units.DimensionError as err:
        raise ValueError(str(err)) from err


def _read(value: Any, dim: Dim) -> Any:
    """Parse a field value into a quantity, without checking its dimension."""
    if units.is_quantity(value):
        return value
    if isinstance(value, str):
        return _parse(value)
    if isinstance(value, Mapping) and len(value) == 1:
        ((unit, magnitudes),) = value.items()
        if isinstance(unit, str):
            try:  # force float rather than rely on promotion
                numbers = jnp.asarray(magnitudes, dtype=float)
            except Exception as err:  # ragged shapes etc.
                raise ValueError(
                    f"units.pydantic: {_REPR.repr(magnitudes)} is not magnitudes "
                    f"{unit!r} can measure: {err}"
                ) from err
            return numbers * _parse(f"1 {unit}")
    if isinstance(value, Mapping) and len(value) > 1:
        raise ValueError(
            f"units.pydantic: this names {len(value)} units, {tuple(value)}. "
            "Supply one unit-to-magnitudes entry."
        )
    raise ValueError(
        f"units.pydantic: cannot read {_REPR.repr(value)} as a value in "
        f"{units.active_base().symbol(dim)}. Give it as {_forms(dim)}."
    )


def _validate(value: Any, dim: Dim) -> Any:
    """Validate a field value against its annotated dimension."""
    quantity = _read(value, dim)
    got = units.dim_of(quantity)
    if got == dim:
        return quantity
    base = units.active_base()
    raise ValueError(
        f"units.pydantic: expected a value in {base.symbol(dim)}, got one in "
        f"{base.symbol(got) or 'no unit at all'}\n"
        f"  expected {units.dim_str(dim)}\n  received {units.dim_str(got)}"
    )


def _serialize(quantity: Any) -> Any:
    """Serialize to ``"6.0 Å"`` for scalars or ``{"Å": [[...]]}`` for arrays."""
    symbol = units.base_of(quantity).symbol(units.dim_of(quantity))
    magnitudes = quantity.value
    if magnitudes.ndim == 0:
        return f"{float(magnitudes)} {symbol}"
    return {symbol: magnitudes.tolist()}


def _core_schema(dim: Dim) -> Any:
    """Build a plain validator schema, which also stops dataclass introspection."""
    return core_schema.no_info_plain_validator_function(
        lambda value: _validate(value, dim),
        serialization=core_schema.plain_serializer_function_ser_schema(
            _serialize, when_used="always"
        ),
    )


def _json_schema(dim: Dim) -> dict[str, Any]:
    """Build the JSON schema for the two accepted input forms."""
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "minProperties": 1, "maxProperties": 1},
        ],
        "description": f"a quantity in {units.active_base().symbol(dim)}: {_forms(dim)}",
    }


class _JsonSchema(GenerateJsonSchema):
    """JSON schema generator that encodes quantity field defaults.

    pydantic's default encoder cannot dump the JAX array inside a ``Quantity``
    and would drop the default from the schema.
    """

    @override
    def encode_default(self, dft: Any) -> Any:
        """Encode a quantity default as its serialized form, else defer to pydantic."""
        if units.is_quantity(dft):
            return _serialize(dft)
        return super().encode_default(dft)


class UnitsModel(BaseModel):
    """A pydantic model whose JSON schema includes quantity field defaults.

    A field annotated with a dimension such as ``units.Length`` accepts a string
    like ``"6 angstrom"``, a one-entry mapping from a unit to magnitudes of any
    shape, or a quantity of that dimension; anything else raises a
    ``ValidationError``. That holds on any ``BaseModel``; this subclass also lets
    ``model_json_schema`` encode a quantity default, which pydantic would refuse.
    """

    @override
    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = _JsonSchema,
        mode: JsonSchemaMode = "validation",
        *,
        union_format: Literal["any_of", "primitive_type_array"] = "any_of",
    ) -> dict[str, Any]:
        """Generate the model's JSON schema, encoding quantity defaults.

        Takes pydantic's arguments; only the default ``schema_generator`` differs.

        Returns:
          The JSON schema as a dict.
        """
        return super().model_json_schema(
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
            union_format=union_format,
        )
