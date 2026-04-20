# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""
JAX-compatible assertion tracing system with optional automatic fixing.

This module provides a comprehensive assertion system that works seamlessly with
JAX transformations, including JIT compilation, automatic differentiation, and
vectorization. Assertions can include optional fix functions for automatic
error recovery.

Key Components:

- **[RuntimeAssertion][kups.core.assertion.RuntimeAssertion]**: Core assertion dataclass with optional fixing capabilities
- **[runtime_assert][kups.core.assertion.runtime_assert]**: Function to create assertions that work with JAX transformations
- **[with_runtime_assertions][kups.core.assertion.with_runtime_assertions]**: Decorator to enable assertion tracing in functions
"""

from __future__ import annotations

import dataclasses
import traceback
import typing
from collections.abc import Callable
from functools import partial
from typing import Any, Final, Self, no_type_check

import jax
import jax.interpreters.partial_eval as pe
import jax.numpy as jnp
from jax import Array
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive, jaxpr_as_fun
from jax.interpreters import ad, batching, mlir
from slub.handlers import (
    ScanSemantics,
    default_jit_handler,
    default_primitive_handler,
    default_scan_handler,
    default_shard_map_handler,
    default_while_handler,
)
from slub.interpreter import (
    Dispatcher,
    HandlerResult,
    Interpreter,
    InterpreterContext,
    InterpreterPolicy,
    TracerValue,
    contains_subjaxprs,
    reinterpret,
)
from slub.util import get_bind_params

from kups.core.lens import bind
from kups.core.utils.jax import dataclass, field

try:
    from jax import typeof as get_aval
except ImportError:
    from jax.core import get_aval  # pyright: ignore[reportAttributeAccessIssue]

# Type alias for fix functions that take a state and fix arguments, returning a new state
type Fix[State, FixArgs] = Callable[[State, FixArgs], State]


@dataclass
class _NO_ARGS: ...  # We cannot use `None` as a default value because it may be an actual argument.


NO_ARGS: Final[_NO_ARGS] = _NO_ARGS()
"""Sentinel instance to distinguish between no arguments and None as an argument."""

_TRACEBACK_MARKER: Final[str] = "\nAssertion created at:\n"


def _strip_traceback(message: str) -> str:
    """Strip the traceback suffix appended by runtime_assert."""
    idx = message.find(_TRACEBACK_MARKER)
    return message[:idx] if idx != -1 else message


@dataclass
class RuntimeAssertion[State, FixArgs]:
    """
    A runtime assertion that validates computations with optional automatic fixing.

    This class encapsulates a predicate that should evaluate to True, along with
    metadata for error reporting and optional repair mechanisms. Assertions are
    designed to work seamlessly with JAX transformations while providing rich
    debugging information.

    Type Parameters:
        State: The type of state that can be modified by the fix function
        FixArgs: The types of arguments passed to the fix function (can be a PyTree)

    Attributes:
        predicate: A scalar boolean array indicating whether the assertion passes
        message: Human-readable error message with optional format placeholders
        fmt_args: Dictionary of values to substitute into the message format string
        exception_type: Type of exception to raise on assertion failure
        static_info: Additional metadata for debugging (not traced by JAX)
        fix_fn: Optional function to repair the state when assertion fails
        fix_args: Arguments to pass to the fix function (can be complex PyTree structures)

    Example:
        ```python
        assertion = RuntimeAssertion(
            predicate=jnp.array(x > 0),
            message="Value must be positive, got {val}",
            fmt_args={"val": x},
            fix_fn=lambda state, threshold: jnp.maximum(state, threshold),
            fix_args=0.1
        )
        ```

    Note:
        The fix_args can be complex PyTree structures including nested dictionaries,
        tuples, and arrays. The assertion system properly handles flattening and
        unflattening these structures during JAX transformations.
    """

    predicate: Array
    message: str = field(static=True)
    fmt_args: dict[str, Array] = field(default_factory=dict)
    exception_type: type[Exception] = field(static=True, default=AssertionError)
    static_info: dict[str, Any] = field(static=True, default_factory=dict)
    fix_fn: Fix[State, FixArgs] | None = field(static=True, default=None)
    fix_args: FixArgs | _NO_ARGS = field(default=NO_ARGS)

    def valid(self) -> bool:
        """Check if the assertion is valid (i.e., passes)."""
        return bool(self.predicate)

    def failed(self) -> bool:
        """Check if the assertion has failed."""
        return not self.valid()

    def __str__(self) -> str:
        """Return the formatted assertion message."""
        return self.message.format(**self.fmt_args)

    def check(self):
        """
        Check the assertion and raise an exception if it fails.

        Raises:
            Exception: The configured exception type if the assertion fails
        """
        if not bool(jnp.all(self.predicate)):
            raise self.exception_type(self.message.format(**self.fmt_args))

    @property
    def exception(self) -> Exception:
        """
        Create the exception instance that would be raised on assertion failure.

        Returns:
            An exception instance with the formatted error message
        """
        return self.exception_type(
            self.message.format(**(self.fmt_args | self.static_info))
        )

    def fix(self, state: State) -> State:
        """
        Attempt to fix the assertion failure by modifying the provided state.

        Args:
            state: The current state that needs to be repaired

        Returns:
            The modified state after applying the fix function

        Raises:
            NotImplementedError: If no fix function is available
            AssertionError: If fix arguments are missing when a fix function exists
        """
        if self.fix_fn is None:
            raise self.exception
        assert not isinstance(self.fix_args, _NO_ARGS), (
            "Fix arguments were not provided."
        )
        return self.fix_fn(state, self.fix_args)


def _make_noop_primitive(name: str):
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

    def noop_p_jvp(primals, tangents, **kwargs):
        primal_out = primitive.bind(*primals, **kwargs)
        tangent_out = tangents
        return primal_out, tangent_out

    ad.primitive_jvps[primitive] = noop_p_jvp

    def noop_p_transpose(cotangent, primals, **kwargs):
        return [cotangent]

    ad.primitive_transposes[primitive] = noop_p_transpose

    def noop_p_batcher(batched_args, batch_dims, **kwargs):
        return primitive.bind(*batched_args, **kwargs), batch_dims

    batching.primitive_batchers[primitive] = noop_p_batcher

    def noop_p_dce_rule(used_outputs, eqn: JaxprEqn, **kwargs):
        return [True] * len(used_outputs), eqn

    pe.dce_rules[primitive] = noop_p_dce_rule
    return primitive


@no_type_check
def _scalar_bool(x: ShapedArray) -> ShapedArray:
    # JAX 0.10 renamed the ShapeDtypeStruct kwarg ``vma`` → ``manual_axis_type``.
    mat = getattr(x, "manual_axis_type", getattr(x, "vma", None))
    try:
        return ShapedArray((), jnp.bool, sharding=x.sharding, manual_axis_type=mat)
    except TypeError:
        return ShapedArray((), jnp.bool, sharding=x.sharding, vma=mat)


def _make_constant_primitive(name: str):
    primitive = Primitive(name)
    primitive.multiple_results = False
    primitive.def_impl(lambda x: jnp.ones_like(x, shape=(), dtype=jnp.bool))
    primitive.def_abstract_eval(_scalar_bool)
    mlir.register_lowering(
        primitive,
        lambda ctx, *args, **kwargs: [mlir.ir_constant(True, aval=ctx.avals_out[0])],
    )

    def constant_p_jvp(primals, tangents):
        primal_out = primitive.bind(*primals)
        return primal_out, jnp.zeros_like(primal_out)

    ad.primitive_jvps[primitive] = constant_p_jvp

    def constant_p_transpose(cotangent, primals):
        return [cotangent]

    ad.primitive_transposes[primitive] = constant_p_transpose

    def constant_p_batcher(batched_args, batch_dims, **kwargs):
        return primitive.bind(*batched_args, **kwargs), batch_dims

    batching.primitive_batchers[primitive] = constant_p_batcher

    def noop_p_dce_rule(used_outputs, eqn: JaxprEqn, **kwargs):
        return [], eqn

    pe.dce_rules[primitive] = noop_p_dce_rule
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
):
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


@dataclass
class AssertionContext:
    assertions: tuple[RuntimeAssertion, ...] = ()

    def add_assertion(self: Self, assertion: RuntimeAssertion) -> Self:
        return (
            bind(self)
            .focus(lambda ctx: ctx.assertions)
            .apply(lambda assertions: assertions + (assertion,))
        )

    def check_assertions(self) -> Array:
        if len(self.assertions) > 0:
            return jnp.all(
                jnp.concatenate([a.predicate.ravel() for a in self.assertions])
            )
        return jnp.array(True, dtype=jnp.bool)

    def push(self: Self) -> Self:
        return self

    def pop(self: Self) -> Self:
        return self


def _strip_ctx_tracebacks(ctx: AssertionContext, *, note: str = "") -> AssertionContext:
    """Strip traceback suffixes from all assertion messages in a context."""
    if not ctx.assertions:
        return ctx

    def _replace_msg(a: RuntimeAssertion) -> RuntimeAssertion:
        stripped = _strip_traceback(a.message)
        if note and stripped != a.message:
            stripped += note
        return dataclasses.replace(a, message=stripped)

    return AssertionContext(assertions=tuple(map(_replace_msg, ctx.assertions)))


def _normalize_ctx_for_comparison(out_info: tuple) -> tuple:
    """Strip traceback suffixes from assertion messages for structural comparison."""
    outvals, ctx = out_info
    if not isinstance(ctx, AssertionContext):
        return out_info
    return (outvals, _strip_ctx_tracebacks(ctx))


def _assert_same_tree[PyTree](old: PyTree, new: PyTree):
    """Compare pytree structure and leaf shapes/dtypes."""
    old_leaves, old_tree_def = jax.tree.flatten(old)
    new_leaves, new_tree_def = jax.tree.flatten(new)
    if old_tree_def != new_tree_def:
        raise ValueError(
            f"Function modified the tree structure: {new_tree_def} != {old_tree_def}"
        )
    leaf_mismatches = []
    for x, y in zip(old_leaves, new_leaves, strict=True):
        xaval = get_aval(x)
        yaval = get_aval(y)
        if any(
            getattr(xaval, attr) != getattr(yaval, attr) for attr in ["shape", "dtype"]
        ):
            leaf_mismatches.append(f"{xaval} != {yaval}")
    if leaf_mismatches:
        raise ValueError(f"Function modified the tree values: {leaf_mismatches}")


def check_assertion_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
):
    return HandlerResult(ctx, [ctx.check_assertions()])


def assertion_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
):
    # meta-data for assertion
    _, bind_params = get_bind_params(eqn)
    message = bind_params["message"]
    fmt_arg_names = bind_params["fmt_arg_names"]
    exception_type = bind_params["exception_type"]
    static_info_hashable = bind_params["static_info_hashable"]
    fix_fn = bind_params["fix_fn"]
    fix_args_tree = bind_params["fix_args_tree"]

    # Convert static_info back from hashable format
    static_info = dict(static_info_hashable)

    pred = invals[0]
    num_fmt_args = len(fmt_arg_names)
    fmt_arg_values = invals[1 : 1 + num_fmt_args]

    # Extract and reconstruct fix_args if present
    if fix_args_tree is not None:
        fix_args_flat = invals[1 + num_fmt_args :]
        fix_args = jax.tree.unflatten(fix_args_tree, fix_args_flat)
    else:
        fix_args = NO_ARGS

    if len(fmt_arg_names) != len(fmt_arg_values):
        raise ValueError(
            f"Expected {len(fmt_arg_names)} format arguments, but got {len(fmt_arg_values)}."
        )
    fmt_args = {
        name: value for name, value in zip(fmt_arg_names, fmt_arg_values, strict=True)
    }

    ctx = ctx.add_assertion(
        RuntimeAssertion(
            predicate=pred,
            message=message,
            fmt_args=fmt_args,
            exception_type=exception_type,
            static_info=static_info,
            fix_fn=fix_fn,
            fix_args=fix_args,
        )
    )

    return default_primitive_handler(interpreter, ctx, eqn, invals)


def cond_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    """Custom cond handler that normalizes traceback strings before branch comparison.

    runtime_assert appends source-location tracebacks to assertion messages. Since
    messages are static pytree fields, assertions at different source lines produce
    different treedefs. This handler strips traceback suffixes so that branches with
    the same base assertion compare as equal.
    """
    _, bind_params = get_bind_params(eqn)
    branches = bind_params["branches"]

    context_aware_branch_fns = [
        reinterpret(jaxpr_as_fun(jaxpr), interpreter) for jaxpr in branches
    ]

    # Check that all branches produce the same context structure by tracing them with a dummy context
    branch_ctx_trees = [
        jax.jit(branch_fn).trace(ctx.push(), *invals[1:]).out_info
        for branch_fn in context_aware_branch_fns
    ]
    assert len(branches) > 0, "cond must have at least one branch"
    # Normalize by stripping tracebacks before structural comparison
    normalized = [_normalize_ctx_for_comparison(t) for t in branch_ctx_trees]
    try:
        for tree in normalized[1:]:
            _assert_same_tree(normalized[0], tree)
    except ValueError as e:
        raise ValueError("Cond branches return inconsistent contexts.") from e

    # Wrap branches to strip tracebacks from output contexts so that
    # jax.lax.cond sees matching pytree structures across branches.
    def _wrap_branch(fn: Callable) -> Callable:
        def wrapped(ctx: AssertionContext, *args: Any) -> Any:
            outvals, ctx_out = fn(ctx, *args)
            return outvals, _strip_ctx_tracebacks(
                ctx_out, note="\n(traceback unavailable: assertion inside cond branch)"
            )

        return wrapped

    normalized_branch_fns = [_wrap_branch(fn) for fn in context_aware_branch_fns]

    outvals, ctx_out = jax.lax.cond(
        invals[0], *reversed(normalized_branch_fns), ctx, *invals[1:]
    )
    return HandlerResult(ctx_out, outvals)


def scan_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
):
    def init_with_true(
        old: AssertionContext, new: AssertionContext
    ) -> AssertionContext:
        """Initialize all assertions to be true."""

        def initialize(a: RuntimeAssertion) -> RuntimeAssertion:
            a = bind(a).focus(lambda a: a.predicate).apply(jnp.ones_like)
            a = (
                bind(a)
                .focus(lambda a: (a.fix_args, a.fmt_args))
                .apply(partial(jax.tree.map, jnp.empty_like))
            )
            return a

        return AssertionContext(
            old.assertions
            + tuple(map(initialize, new.assertions[len(old.assertions) :]))
        )

    def update_on_fail(
        old: AssertionContext, new: AssertionContext
    ) -> AssertionContext:
        """Always keep the first assertion that fails."""
        return AssertionContext(
            tuple(
                typing.cast(
                    RuntimeAssertion,
                    jax.tree.map(partial(jnp.where, new.predicate), old, new),
                )
                for old, new in zip(old.assertions, new.assertions)
            )
        )

    return default_scan_handler(
        interpreter,
        ctx,
        eqn,
        invals,
        threading=ScanSemantics.CARRY,
        initializer=init_with_true,
        updater=update_on_fail,
    )


def while_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
):
    def init_with_true(
        old: AssertionContext, new: AssertionContext
    ) -> AssertionContext:
        """Initialize all assertions to be true."""

        def _sentinel_like(x: jax.Array) -> jax.Array:
            """Fill with -inf for floats, min value for integers."""
            if jnp.issubdtype(x.dtype, jnp.integer):
                return jnp.full_like(x, fill_value=jnp.iinfo(x.dtype).min)
            return jnp.full_like(x, fill_value=-jnp.inf)

        def initialize(a: RuntimeAssertion) -> RuntimeAssertion:
            a = bind(a).focus(lambda a: a.predicate).apply(jnp.ones_like)
            a = (
                bind(a)
                .focus(lambda a: (a.fix_args, a.fmt_args))
                .apply(partial(jax.tree.map, _sentinel_like))
            )
            return a

        return AssertionContext(
            old.assertions
            + tuple(map(initialize, new.assertions[len(old.assertions) :]))
        )

    def update_on_fail(
        old: AssertionContext, new: AssertionContext
    ) -> AssertionContext:
        """Always keep the first assertion that fails."""
        return AssertionContext(
            tuple(
                RuntimeAssertion(
                    predicate=jnp.logical_and(old.predicate, new.predicate),
                    message=old.message,
                    fmt_args=jax.tree.map(jnp.maximum, old.fmt_args, new.fmt_args),
                    exception_type=old.exception_type,
                    static_info=old.static_info,
                    fix_fn=old.fix_fn,
                    fix_args=jax.tree.map(jnp.maximum, old.fix_args, new.fix_args),
                )
                for old, new in zip(old.assertions, new.assertions)
            )
        )

    return default_while_handler(
        interpreter,
        ctx,
        eqn,
        invals,
        initializer=init_with_true,
        updater=update_on_fail,
    )


def shard_map_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    context_sharding: jax.sharding.PartitionSpec | None = None,
):
    def declare_ctx_in_specs(ctx: AssertionContext) -> AssertionContext:
        return jax.tree.map(lambda _: context_sharding, ctx)

    def declare_ctx_out_specs(ctx: AssertionContext) -> AssertionContext:
        return jax.tree.map(lambda _: context_sharding, ctx)

    return default_shard_map_handler(
        interpreter,
        ctx,
        eqn,
        invals,
        declare_ctx_in_specs=declare_ctx_in_specs
        if context_sharding is not None
        else None,
        declare_ctx_out_specs=declare_ctx_out_specs
        if context_sharding is not None
        else None,
    )


def _contains_assertion_primitive[Context: InterpreterContext](
    ctx: Context,  # type: ignore
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> bool:
    for leaf in jax.tree.leaves(
        eqn.params, is_leaf=lambda x: isinstance(x, (Jaxpr, ClosedJaxpr))
    ):
        if isinstance(leaf, ClosedJaxpr):
            jaxpr = leaf.jaxpr
        elif isinstance(leaf, Jaxpr):
            jaxpr = leaf
        else:
            continue
        assert isinstance(jaxpr, Jaxpr)
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "assertion":
                return True
            # Recursively check for nested jaxprs
            if contains_subjaxprs(ctx, eqn, invals):
                if _contains_assertion_primitive(ctx, eqn, invals):
                    return True
    return False


def with_runtime_assertions[**P, R](
    fn: Callable[P, R],
    policy: InterpreterPolicy = InterpreterPolicy.RAISE,
    context_sharding: jax.sharding.PartitionSpec | None = None,
) -> Callable[P, tuple[R, tuple[RuntimeAssertion, ...]]]:
    """
    Decorator that enables runtime assertion tracing for JAX functions.

    This decorator wraps a function to intercept and collect all runtime assertions
    created with `runtime_assert` during execution. The wrapped function returns
    both the original result and a tuple of all assertions that were evaluated,
    allowing for post-execution analysis, debugging, and optional error recovery.

    Args:
        fn: The function to wrap with assertion tracing capabilities
        policy: Controls interpreter behavior on unhandled operations:
            - RAISE: Raise exception on unknown operations (default, safest)
            - WARN: Issue warning and continue with original function
            - SKIP: Silently continue with original function
        context_sharding: Optional sharding specification for distributed contexts.
            When provided, assertion contexts are sharded according to this spec
            for multi-device computations.

    Returns:
        A wrapped function that returns a tuple of (original_result, assertions_tuple).
        The assertions tuple contains all RuntimeAssertion instances encountered
        during execution, preserving order and enabling post-hoc analysis.

    Type Parameters:
        P: Parameter specification of the wrapped function (ParamSpec)
        R: Return type of the wrapped function

    Example:
        Basic usage:
        ```python
        @with_runtime_assertions
        def validate_computation(x):
            runtime_assert(x > 0, "x must be positive")
            return x ** 2

        result, assertions = validate_computation(jnp.array(5.0))
        # result = 25.0, assertions contains one RuntimeAssertion
        ```

        With custom policy:
        ```python
        traced_fn = with_runtime_assertions(
            my_function,
            policy=InterpreterPolicy.WARN
        )
        result, assertions = traced_fn(inputs)
        ```

        Distributed computation:
        ```python
        sharded_fn = with_runtime_assertions(
            distributed_computation,
            context_sharding=jax.sharding.PartitionSpec('data', None)
        )
        ```

        Error analysis and recovery:
        ```python
        result, assertions = traced_fn(initial_state)

        # Check for failures
        failed_assertions = [a for a in assertions if a.failed()]
        if failed_assertions:
            # Attempt automatic fixing
            fixed_state = initial_state
            for assertion in failed_assertions:
                if assertion.fix_fn is not None:
                    fixed_state = assertion.fix(fixed_state)

            # Re-run with fixed state
            result, _ = traced_fn(fixed_state)
        ```

    Note:
        The decorator integrates seamlessly with JAX transformations including
        jit, vmap, grad, and scan. Assertions are properly threaded through
        control flow operations and maintain correct semantics under
        transformations.
    """
    dispatcher = Dispatcher(
        handlers={
            "assertion": assertion_handler,
            "check_assertion": check_assertion_handler,
            "jit": default_jit_handler,
            "scan": scan_handler,
            "while": while_handler,
            "cond": cond_handler,
            "shard_map": partial(shard_map_handler, context_sharding=context_sharding),
        }
    ).register_custom_matching_rule(_contains_assertion_primitive)

    interpreter = Interpreter(dispatcher, policy=policy, label="assertion_interpreter")

    reinterpreted = reinterpret(fn, interpreter=interpreter)

    def wrapped(
        *args: P.args, **kwargs: P.kwargs
    ) -> tuple[R, tuple[RuntimeAssertion, ...]]:
        ctx = AssertionContext()
        outvals, ctx = reinterpreted(ctx, *args, **kwargs)
        return outvals, ctx.assertions

    return wrapped
