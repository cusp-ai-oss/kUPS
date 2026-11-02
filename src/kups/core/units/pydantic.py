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
accepted, for scalars and arrays alike. Dumps and hints use the alias's display
unit, whatever base was active:

    >>> Run.model_validate({"cutoff": "0.6 nm", "cell": "4 angstrom"}).model_dump()
    {'cutoff': '6.0 Å', 'cell': '4.0 Å'}
"""

import reprlib
from collections.abc import Mapping
from typing import Any, Final, Literal, TypeAliasType, override

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
    JsonSchemaValue,
)
from pydantic_core import core_schema

from kups.core import units
from kups.core.units import catalog as u
from kups.core.units._core import AnyQuantity, _Dimension
from kups.core.units._unit import AnyUnit, Dim

__all__ = ["UnitsModel"]

# the display unit of each dimension, as its alias declares it
_DISPLAY: Final[dict[Dim, AnyUnit]] = {
    hook.unit.dim: hook.unit
    for name in units.__all__
    if isinstance(alias := getattr(units, name), TypeAliasType)
    for hook in getattr(alias.__value__, "__metadata__", ())
    if isinstance(hook, _Dimension)
}


def _forms(unit: AnyUnit) -> str:
    """Describe the accepted input forms, with examples in ``unit``."""
    return (
        f"'1.5 {unit}' or {{'{unit}': [1.5, 2.5]}}; "
        "any parseable unit of the same dimension is accepted"
    )


# truncates rejected values in error messages, like pydantic
_REPR: Final[reprlib.Repr] = reprlib.Repr(maxlist=8, maxstring=80, maxother=200)


def _parse(text: str) -> AnyQuantity | jax.Array:
    """Parse ``text`` with ``u.parse``, raising ValueError so pydantic reports it."""
    try:
        return u.parse(text)
    except units.DimensionError as err:
        raise ValueError(str(err)) from err


def _read(value: object, unit: AnyUnit) -> AnyQuantity | jax.Array:
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
        f"units.pydantic: cannot read {_REPR.repr(value)} as a value in "
        f"{unit}. Give it as {_forms(unit)}."
    )


def _validate(value: object, unit: AnyUnit) -> AnyQuantity | jax.Array:
    """Validate a field value against its annotated dimension."""
    quantity = _read(value, unit)
    got = units.dim_of(quantity)
    if got == unit.dim:
        return quantity
    raise ValueError(
        f"units.pydantic: expected a value in {unit}, got one in "
        f"{units.active_base().symbol(got) or 'no unit at all'}\n"
        f"  expected {units.dim_str(unit.dim)}\n  received {units.dim_str(got)}"
    )


def _serialize(quantity: AnyQuantity) -> str | dict[str, object]:
    """Serialize to ``"6.0 Å"`` for scalars or ``{"Å": [[...]]}`` for arrays.

    The magnitudes are in the display unit of the quantity's dimension, rounded
    to the digits their dtype resolves so conversion noise does not show. The
    dump is for display: values with more digits than that lose them, so
    validating it back is exact only for magnitudes that short.
    """
    unit = _DISPLAY[units.dim_of(quantity)]
    magnitudes = quantity.to(unit)
    digits = jnp.finfo(magnitudes.dtype).precision
    values = np.vectorize(lambda x: float(f"{x:.{digits}g}"), otypes=[float])(
        magnitudes
    ).tolist()
    if magnitudes.ndim == 0:
        return f"{values} {unit}"
    return {str(unit): values}


def _core_schema(unit: AnyUnit) -> core_schema.CoreSchema:
    """Build a plain validator schema, which also stops dataclass introspection."""
    return core_schema.no_info_plain_validator_function(
        lambda value: _validate(value, unit),
        serialization=core_schema.plain_serializer_function_ser_schema(
            _serialize, when_used="always"
        ),
    )


def _json_schema(unit: AnyUnit) -> JsonSchemaValue:
    """Build the JSON schema for the two accepted input forms."""
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "minProperties": 1, "maxProperties": 1},
        ],
        "description": f"a quantity in {unit}: {_forms(unit)}",
    }


class _JsonSchema(GenerateJsonSchema):
    """JSON schema generator that encodes quantity field defaults.

    pydantic's default encoder cannot dump the JAX array inside a ``Quantity``
    and would drop the default from the schema.
    """

    @override
    def encode_default(self, dft: object) -> object:
        """Encode a quantity default as its serialized form, else defer to pydantic."""
        if units.is_quantity(dft):
            return _serialize(dft)
        return super().encode_default(dft)


class UnitsModel(BaseModel):
    """A pydantic model whose JSON schema includes quantity field defaults, which plain ``BaseModel`` drops."""

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
