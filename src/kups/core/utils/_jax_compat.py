# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Adapters for JAX API changes within the supported ``jax`` range.

kUPS is tested against both the lowest and the highest ``jax`` it allows (see
the CI matrix), so the few internals that changed shape in between are read and
written through this module only. Each helper names the release that changed
the API; once the ``jax`` floor in ``pyproject.toml`` reaches it, inline the new
form and delete the helper.
"""

import inspect
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from jax import ShapeDtypeStruct
from jax.extend.core import JaxprEqn, Primitive
from jax.interpreters import partial_eval as pe

type DCERule = Callable[[list[bool], JaxprEqn], tuple[list[bool], JaxprEqn | None]]


def get_bind_params(eqn: JaxprEqn) -> tuple[list[Any], dict[str, Any]]:
    """Return ``(subfuns, bind_params)`` for re-binding ``eqn``'s primitive.

    Remove once ``jax>=0.9.2``: from then on ``Primitive.get_bind_params``
    returns only the params and ``subfuns`` is always empty.
    """
    result: Any = eqn.primitive.get_bind_params(eqn.params)
    if isinstance(result, tuple):
        return result[0], result[1]
    return [], result


_MANUAL_AXES_KWARG = (
    "manual_axis_type"
    if "manual_axis_type" in inspect.signature(ShapeDtypeStruct.__init__).parameters
    else "vma"
)


def manual_axes_kwarg(aval: Any) -> dict[str, Any]:
    """Keyword forwarding ``aval``'s manual mesh axes to a new aval or struct.

    Applies to ``ShapedArray`` and ``ShapeDtypeStruct``. Remove once
    ``jax>=0.10``, which renamed ``vma`` to ``manual_axis_type``.
    """
    return {_MANUAL_AXES_KWARG: getattr(aval, _MANUAL_AXES_KWARG, None)}


def split_scan_operands[T](
    params: dict[str, Any], operands: Sequence[T]
) -> tuple[list[T], list[T], list[T]]:
    """Split a ``scan`` equation's operands into ``(consts, carry, xs)``.

    Remove once ``jax>=0.11``, which replaced ``num_consts``/``num_carry`` with
    the ``ft_in`` flat tree.
    """
    if "ft_in" in params:
        consts, carry, xs = params["ft_in"].update(operands).unpack()
        return list(consts), list(carry), list(xs)
    n_consts, n_carry = params["num_consts"], params["num_carry"]
    return (
        list(operands[:n_consts]),
        list(operands[n_consts : n_consts + n_carry]),
        list(operands[n_consts + n_carry :]),
    )


def split_scan_results[T](
    params: dict[str, Any], results: Sequence[T]
) -> tuple[list[T], list[T]]:
    """Split a ``scan`` body's results into ``(carry, ys)``.

    Remove once ``jax>=0.11``, which replaced ``num_carry`` with the ``ft_out``
    flat tree.
    """
    if "ft_out" in params:
        carry, ys = params["ft_out"].update(results).unpack()
        return list(carry), list(ys)
    n_carry = params["num_carry"]
    return list(results[:n_carry]), list(results[n_carry:])


def shard_map_manual_axes(params: dict[str, Any]) -> frozenset[Hashable]:
    """Mesh axes a ``shard_map`` equation makes manual (its ``axis_names``).

    Remove once ``jax>=0.11``, which renamed the ``manual_axes`` parameter to
    ``newly_manual_axes``.
    """
    if "newly_manual_axes" in params:
        return params["newly_manual_axes"]
    return params["manual_axes"]


def register_dce_rule(primitive: Primitive, rule: DCERule) -> None:
    """Register a dead-code-elimination rule taking ``(used_outputs, eqn)``.

    JAX 0.12 passes ``(used_outputs, live_ins, eqn)``. Once ``jax>=0.12``,
    register rules with that signature directly.
    """

    def adapted(
        used_outputs: list[bool], *args: Any
    ) -> tuple[list[bool], JaxprEqn | None]:
        return rule(used_outputs, args[-1])

    pe.dce_rules[primitive] = adapted
