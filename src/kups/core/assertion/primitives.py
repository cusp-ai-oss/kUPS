# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""JAX primitives that record runtime assertions in traced code."""

from __future__ import annotations

import traceback
from typing import Any, Final

import jax
import jax.numpy as jnp
from jax import Array
from jax.core import ShapedArray
from jax.extend.core import JaxprEqn, Primitive
from jax.interpreters import ad, batching, mlir

from kups.core.assertion.runtime_assertion import _NO_ARGS, NO_ARGS, Fix, LoopMerge
from kups.core.interpreter._compat import (
    manual_axes_kwarg,
    register_dce_rule,
)

_TRACEBACK_MARKER: Final[str] = "\nAssertion created at:\n"


def _make_noop_primitive(name: str) -> Primitive:
    primitive = Primitive(name)
    primitive.multiple_results = True
    primitive.def_impl(lambda *args, **kwargs: args)
    primitive.def_abstract_eval(lambda *args, **kwargs: args)
    mlir.register_lowering(
        primitive,
        lambda ctx, *args, **kwargs: mlir.lower_fun(
            lambda *args: args, multiple_results=True
        )(ctx, *args),
    )

    def noop_p_jvp(
        primals: tuple[Any, ...], tangents: tuple[Any, ...], **kwargs: Any
    ) -> tuple[Any, tuple[Any, ...]]:
        primal_out = primitive.bind(*primals, **kwargs)
        tangent_out = tangents
        return primal_out, tangent_out

    ad.primitive_jvps[primitive] = noop_p_jvp

    def noop_p_transpose(
        cotangent: Any, primals: tuple[Any, ...], **kwargs: Any
    ) -> list[Any]:
        return [cotangent]

    ad.primitive_transposes[primitive] = noop_p_transpose

    def noop_p_batcher(
        batched_args: tuple[Any, ...], batch_dims: tuple[Any, ...], **kwargs: Any
    ) -> tuple[Any, tuple[Any, ...]]:
        return primitive.bind(*batched_args, **kwargs), batch_dims

    batching.primitive_batchers[primitive] = noop_p_batcher

    def noop_p_dce_rule(
        used_outputs: list[bool], eqn: JaxprEqn
    ) -> tuple[list[bool], JaxprEqn]:
        return [True] * len(used_outputs), eqn

    register_dce_rule(primitive, noop_p_dce_rule)
    return primitive


def _scalar_bool(x: ShapedArray) -> ShapedArray:
    return ShapedArray((), jnp.bool, sharding=x.sharding, **manual_axes_kwarg(x))


def _make_constant_primitive(name: str) -> Primitive:
    primitive = Primitive(name)
    primitive.multiple_results = False
    primitive.def_impl(lambda x: jnp.ones_like(x, shape=(), dtype=jnp.bool))
    primitive.def_abstract_eval(_scalar_bool)
    mlir.register_lowering(
        primitive,
        lambda ctx, *args, **kwargs: [mlir.ir_constant(True, aval=ctx.avals_out[0])],
    )

    def constant_p_jvp(
        primals: tuple[Any, ...], tangents: tuple[Any, ...]
    ) -> tuple[Array, Array]:
        primal_out = primitive.bind(*primals)
        return primal_out, jnp.zeros_like(primal_out)

    ad.primitive_jvps[primitive] = constant_p_jvp

    def constant_p_transpose(cotangent: Any, primals: tuple[Any, ...]) -> list[Any]:
        return [cotangent]

    ad.primitive_transposes[primitive] = constant_p_transpose

    def constant_p_batcher(
        batched_args: tuple[Any, ...], batch_dims: tuple[Any, ...], **kwargs: Any
    ) -> tuple[Any, int]:
        # The scalar result is broadcast onto the mapped axis of ``like``.
        (like,), (bdim,) = batched_args, batch_dims
        out = primitive.bind(like, **kwargs)
        return jnp.broadcast_to(out, (like.shape[bdim],)), 0

    batching.primitive_batchers[primitive] = constant_p_batcher

    def constant_p_dce_rule(
        used_outputs: list[bool], eqn: JaxprEqn
    ) -> tuple[list[bool], JaxprEqn]:
        # The primitive is opaque, so all inputs are used.
        return [True] * len(eqn.invars), eqn

    register_dce_rule(primitive, constant_p_dce_rule)
    return primitive


assertion_p = _make_noop_primitive("assertion")
check_assertion_p = _make_constant_primitive("check_assertion")


def _capture_traceback() -> str:
    """Capture the caller's stack trace for debugging deferred callbacks.

    Returns:
        Formatted stack trace string of the caller's site.
    """
    return "".join(traceback.format_stack()[:-2])


def runtime_assert[State, FixArgs](
    predicate: Array,
    message: str = "",
    fmt_args: dict[str, Array] | None = None,
    exception_type: type[Exception] = AssertionError,
    static_info: dict[str, Any] | None = None,
    fix_fn: Fix[State, FixArgs] | None = None,
    fix_args: FixArgs | _NO_ARGS = NO_ARGS,
    loop_merge: LoopMerge | None = None,
) -> None:
    """
    Create a runtime assertion that integrates with JAX transformations.

    This function creates assertions that can be traced through JAX transformations
    including JIT compilation, automatic differentiation, and vectorization. The
    assertion acts as an identity function during execution but records assertion
    metadata for later inspection.

    Args:
        predicate: A boolean array indicating whether the assertion passes
        message: Error message with optional format placeholders (e.g., "Value {val} is invalid")
        fmt_args: Dictionary mapping format placeholder names to values
        exception_type: Type of exception to raise if assertion fails during checking
        static_info: Additional metadata for debugging (not traced by JAX)
        fix_fn: Optional function to repair state when assertion fails
        fix_args: Arguments for the fix function (can be complex PyTree structures)
        loop_merge: How ``scan``/``while_loop`` fold ``fmt_args`` and ``fix_args``
            across iterations. ``None`` keeps the loop's default:
            [LAST_FAILURE][kups.core.assertion.LAST_FAILURE] for ``scan`` and
            [ELEMENTWISE_MAX][kups.core.assertion.ELEMENTWISE_MAX] for ``while_loop``.

    Type Parameters:
        State: Type of state that can be modified by the fix function
        FixArgs: Type of arguments passed to the fix function (supports PyTree structures)

    Example:
        Basic assertion:
        ```python
        x = jnp.array(5.0)
        runtime_assert(
            predicate=x > 0,
            message="Value must be positive, got {val}",
            fmt_args={"val": x}
        )
        ```

        Assertion with fixing:
        ```python
        runtime_assert(
            predicate=x > threshold,
            message="Value {val} below threshold {thresh}",
            fmt_args={"val": x, "thresh": threshold},
            fix_fn=lambda state, args: jnp.maximum(state, args["min_val"]),
            fix_args={"min_val": jnp.array(1.0)}
        )
        ```

        Complex PyTree fix_args:
        ```python
        complex_args = {
            "thresholds": {"min": jnp.array(0.1), "max": jnp.array(10.0)},
            "multipliers": (jnp.array(2.0), jnp.array(3.0))
        }
        runtime_assert(
            predicate=x > 0,
            message="Invalid value",
            fix_args=complex_args
        )
        ```

    Note:
        The fix_args parameter supports arbitrarily nested PyTree structures including
        dictionaries, tuples, and arrays. These are automatically flattened and
        unflattened during JAX transformations.
    """
    if fmt_args is None:
        fmt_args = {}
    if static_info is None:
        static_info = {}

    tb = _capture_traceback().replace("{", "{{").replace("}", "}}")
    # Convert static_info to hashable format (tuple of key-value pairs)
    static_info_hashable = tuple(sorted(static_info.items()))

    # Prepare inputs: predicate, fmt_args values, fix_args flattened (if present)
    inputs = [predicate, *fmt_args.values()]
    fix_args_tree = None
    if not isinstance(fix_args, _NO_ARGS):
        # Flatten fix_args PyTree and add to inputs
        fix_args_flat, fix_args_tree = jax.tree.flatten(fix_args)
        inputs.extend(fix_args_flat)

    assertion_p.bind(
        *inputs,
        fmt_arg_names=tuple(fmt_args.keys()),
        message=message + f"{_TRACEBACK_MARKER}{tb}",
        exception_type=exception_type,
        static_info_hashable=static_info_hashable,
        fix_fn=fix_fn,
        fix_args_tree=fix_args_tree,
        loop_merge=loop_merge,
    )


def check_assertions(like: Array | None = None) -> Array:
    """A primitive that returns a scalar bool. When not wrapped in with_runtime_assertions,
    it always returns True. When wrapped, it will return the conjunction of all
    runtime assertions in the current context.

    Args:
        like: An array whose device placement and sharding will be used for the output.
            If None, the output will be placed on the default device.
    Returns:
        A scalar boolean array indicating whether all assertions pass.
    """
    if like is None:
        like = jnp.array(True, dtype=jnp.bool)
    return check_assertion_p.bind(like)
