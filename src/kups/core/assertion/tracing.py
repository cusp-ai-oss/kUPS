# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Collect runtime assertions by reinterpreting traced functions."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import Array
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, jaxpr_as_fun

from kups.core.assertion.runtime_assertion import (
    ELEMENTWISE_MAX,
    LAST_FAILURE,
    NO_ARGS,
    LoopMerge,
    RuntimeAssertion,
)
from kups.core.interpreter._compat import (
    get_bind_params,
)
from kups.core.interpreter.handlers import (
    default_checkpoint_handler,
    default_jit_handler,
    default_primitive_handler,
    default_scan_handler,
    default_shard_map_handler,
    default_while_handler,
)
from kups.core.interpreter.interpreter import (
    Dispatcher,
    HandlerResult,
    Interpreter,
    InterpreterPolicy,
    TracerValue,
    contains_subjaxprs,
    reinterpret,
)
from kups.core.lens import bind
from kups.core.utils.jax import dataclass


@dataclass
class AssertionContext:
    assertions: tuple[RuntimeAssertion[Any, Any], ...] = ()

    def add_assertion(self: Self, assertion: RuntimeAssertion[Any, Any]) -> Self:
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


def check_assertion_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    return HandlerResult(ctx, [ctx.check_assertions()])


def assertion_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    # meta-data for assertion
    _, bind_params = get_bind_params(eqn)
    message = bind_params["message"]
    fmt_arg_names = bind_params["fmt_arg_names"]
    exception_type = bind_params["exception_type"]
    static_info_hashable = bind_params["static_info_hashable"]
    fix_fn = bind_params["fix_fn"]
    fix_args_tree = bind_params["fix_args_tree"]
    loop_merge = bind_params["loop_merge"]

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
            loop_merge=loop_merge,
        )
    )

    return default_primitive_handler(interpreter, ctx, eqn, invals)


def cond_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    """Custom cond handler that unions assertions across branches.

    Branches may declare different assertions. The output context holds the
    incoming assertions followed by every branch's assertions concatenated in
    branch order. Each branch fills its own slots with real predicates and the
    slots owned by other branches with passing placeholders (True predicate,
    zeroed fmt_args/fix_args), so jax.lax.cond observes matching pytree
    structures across branches. An unchosen branch thus contributes only passing
    assertions.
    """
    _, bind_params = get_bind_params(eqn)
    branches = bind_params["branches"]
    assert len(branches) > 0, "cond must have at least one branch"

    branch_fns = [
        jax.jit(reinterpret(jaxpr_as_fun(jaxpr), interpreter)) for jaxpr in branches
    ]

    # Dry-run trace each branch to learn the assertions it appends; the abstract
    # leaves (shape/dtype) are used to build placeholders for the other branches.
    n_in = len(ctx.assertions)
    suffix_templates = [
        fn.trace(ctx, *invals[1:]).out_info[1].assertions[n_in:] for fn in branch_fns
    ]

    def passing(a: RuntimeAssertion[Any, Any]) -> RuntimeAssertion[Any, Any]:
        """Passing copy of an assertion: True predicate, zeroed fmt/fix args."""
        a = bind(a).focus(lambda a: a.predicate).apply(jnp.ones_like)
        return (
            bind(a)
            .focus(lambda a: (a.fmt_args, a.fix_args))
            .apply(partial(jax.tree.map, jnp.zeros_like))
        )

    def wrap(
        index: int, fn: Callable[..., tuple[Any, AssertionContext]]
    ) -> Callable[..., tuple[Any, AssertionContext]]:
        def wrapped(ctx: AssertionContext, *args: Any) -> tuple[Any, AssertionContext]:
            with jax.disable_jit(False):
                outvals, ctx_out = fn(ctx, *args)
            merged = list(ctx_out.assertions[:n_in])
            for i, templates in enumerate(suffix_templates):
                merged.extend(
                    ctx_out.assertions[n_in:] if i == index else map(passing, templates)
                )
            return outvals, AssertionContext(tuple(merged))

        return wrapped

    wrapped_fns = [wrap(i, fn) for i, fn in enumerate(branch_fns)]

    # jax.lax.switch selects branch ``invals[0]`` (the cond/switch index) directly,
    # matching the primitive's branch order and supporting any number of branches.
    outvals, ctx_out = jax.lax.switch(invals[0], wrapped_fns, ctx, *invals[1:])
    return HandlerResult(ctx_out, outvals)


def _loop_initializer(
    default: LoopMerge,
) -> Callable[[AssertionContext, AssertionContext], AssertionContext]:
    """Seed the assertions a loop body appends: passing, with initial args."""

    def initialize(a: RuntimeAssertion[Any, Any]) -> RuntimeAssertion[Any, Any]:
        merge = a.loop_merge or default
        a = bind(a).focus(lambda a: a.predicate).apply(jnp.ones_like)
        return (
            bind(a)
            .focus(lambda a: (a.fix_args, a.fmt_args))
            .apply(partial(jax.tree.map, merge.init))
        )

    def initializer(old: AssertionContext, new: AssertionContext) -> AssertionContext:
        appended = new.assertions[len(old.assertions) :]
        return AssertionContext(old.assertions + tuple(map(initialize, appended)))

    return initializer


def _loop_updater(
    default: LoopMerge,
) -> Callable[[AssertionContext, AssertionContext], AssertionContext]:
    """Fold one iteration's assertions into those accumulated so far."""

    def merge(
        old: RuntimeAssertion[Any, Any], new: RuntimeAssertion[Any, Any]
    ) -> RuntimeAssertion[Any, Any]:
        combine = (old.loop_merge or default).combine
        fmt_args, fix_args = jax.tree.map(
            lambda o, n: combine(old.predicate, o, new.predicate, n),
            (old.fmt_args, old.fix_args),
            (new.fmt_args, new.fix_args),
        )
        return dataclasses.replace(
            old,
            predicate=jnp.where(new.predicate, old.predicate, new.predicate),
            fmt_args=fmt_args,
            fix_args=fix_args,
        )

    def updater(old: AssertionContext, new: AssertionContext) -> AssertionContext:
        return AssertionContext(tuple(map(merge, old.assertions, new.assertions)))

    return updater


def scan_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    return default_scan_handler(
        interpreter,
        ctx,
        eqn,
        invals,
        initializer=_loop_initializer(LAST_FAILURE),
        updater=_loop_updater(LAST_FAILURE),
    )


def while_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    return default_while_handler(
        interpreter,
        ctx,
        eqn,
        invals,
        initializer=_loop_initializer(ELEMENTWISE_MAX),
        updater=_loop_updater(ELEMENTWISE_MAX),
    )


def shard_map_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    context_sharding: jax.sharding.PartitionSpec | None = None,
) -> HandlerResult[AssertionContext]:
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


def _contains_assertion_primitive[Context](
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
) -> Callable[P, tuple[R, tuple[RuntimeAssertion[Any, Any], ...]]]:
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
            "remat2": default_checkpoint_handler,
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
    ) -> tuple[R, tuple[RuntimeAssertion[Any, Any], ...]]:
        ctx = AssertionContext()
        outvals, ctx = reinterpreted(ctx, *args, **kwargs)
        return outvals, ctx.assertions

    return wrapped
