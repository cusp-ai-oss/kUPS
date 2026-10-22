# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Collect runtime assertions by reinterpreting traced functions."""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable
from functools import partial
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, jaxpr_as_fun

from kups.core.assertion.primitives import assertion_p, check_assertion_p
from kups.core.assertion.runtime_assertion import (
    NO_ARGS,
    RuntimeAssertion,
)
from kups.core.effects import (
    Handler,
    HandlerResult,
    Interpreter,
    InterpreterPolicy,
    TracerValue,
    default_checkpoint_handler,
    default_jit_handler,
    default_primitive_handler,
    default_shard_map_handler,
    reinterpret,
)
from kups.core.lens import bind
from kups.core.utils._jax_compat import (
    get_bind_params,
    split_scan_operands,
    split_scan_results,
)
from kups.core.utils.jax import PyTreeDef, dataclass


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


class _AssertionPacker:
    """Pack assertion leaves into one flat array per dtype.

    Control flow then carries a few arrays instead of one per assertion leaf.
    """

    def __init__(self, templates: tuple[RuntimeAssertion[Any, Any], ...]) -> None:
        self.treedefs: list[PyTreeDef[RuntimeAssertion[Any, Any]]] = []
        self.specs: list[list[tuple[int, int, tuple[int, ...]]]] = []
        self.dtypes: list[jnp.dtype] = []
        sizes: list[int] = []
        for a in templates:
            leaves, treedef = jax.tree.flatten(a)
            self.treedefs.append(treedef)
            spec: list[tuple[int, int, tuple[int, ...]]] = []
            for leaf in leaves:
                dtype = jnp.dtype(leaf.dtype)
                if dtype not in self.dtypes:
                    self.dtypes.append(dtype)
                    sizes.append(0)
                g = self.dtypes.index(dtype)
                shape = tuple(leaf.shape)
                spec.append((g, sizes[g], shape))
                sizes[g] += int(np.prod(shape, dtype=int))
            self.specs.append(spec)

    def pack(self, assertions: tuple[RuntimeAssertion[Any, Any], ...]) -> list[Array]:
        """Flatten assertions into one array per dtype.

        Args:
            assertions: Assertions matching the templates' structure.

        Returns:
            One flat array per dtype.
        """
        parts: list[list[Array]] = [[] for _ in self.dtypes]
        for a, spec in zip(assertions, self.specs, strict=True):
            for leaf, (g, _, _) in zip(jax.tree.leaves(a), spec, strict=True):
                parts[g].append(jnp.ravel(leaf).astype(self.dtypes[g]))
        return [jnp.concatenate(p) for p in parts]

    def unpack(self, packed: list[Array]) -> tuple[RuntimeAssertion[Any, Any], ...]:
        """Inverse of ``pack``.

        Args:
            packed: One flat array per dtype, as returned by ``pack``.

        Returns:
            Assertions with the templates' structure.
        """
        out: list[RuntimeAssertion[Any, Any]] = []
        for treedef, spec in zip(self.treedefs, self.specs):
            leaves = [
                packed[g][o : o + int(np.prod(shape, dtype=int))].reshape(shape)
                for g, o, shape in spec
            ]
            out.append(jax.tree.unflatten(treedef, leaves))
        return tuple(out)


_primitive_names_cache: weakref.WeakKeyDictionary[Jaxpr, frozenset[str]] = (
    weakref.WeakKeyDictionary()
)


def _sub_jaxprs(params: Any) -> list[Jaxpr]:
    """The jaxprs among the leaves of ``params``, unwrapping closed jaxprs."""
    leaves = jax.tree.leaves(
        params, is_leaf=lambda x: isinstance(x, (Jaxpr, ClosedJaxpr))
    )
    return [
        leaf.jaxpr if isinstance(leaf, ClosedJaxpr) else leaf
        for leaf in leaves
        if isinstance(leaf, (Jaxpr, ClosedJaxpr))
    ]


def _primitive_names(jaxpr: Jaxpr) -> frozenset[str]:
    """Names of the primitives bound in ``jaxpr`` or any nested jaxpr."""
    if (names := _primitive_names_cache.get(jaxpr)) is None:
        names = frozenset(eqn.primitive.name for eqn in jaxpr.eqns).union(
            *(
                _primitive_names(sub)
                for eqn in jaxpr.eqns
                for sub in _sub_jaxprs(eqn.params)
            )
        )
        _primitive_names_cache[jaxpr] = names
    return names


def _contains_primitive(jaxprs: object, *names: str) -> bool:
    """Whether any (nested) jaxpr in ``jaxprs`` binds one of ``names``."""
    return any(not _primitive_names(j).isdisjoint(names) for j in _sub_jaxprs(jaxprs))


def _passing(a: RuntimeAssertion[Any, Any]) -> RuntimeAssertion[Any, Any]:
    """Passing copy of an assertion: True predicate, zeroed fmt/fix args."""
    a = bind(a).focus(lambda a: a.predicate).apply(jnp.ones_like)
    return (
        bind(a)
        .focus(lambda a: (a.fmt_args, a.fix_args))
        .apply(partial(jax.tree.map, jnp.zeros_like))
    )


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
    # Incoming assertions bypass the conditional unless a branch reads them.
    needs_ctx = _contains_primitive(branches, check_assertion_p.name)
    base = ctx if needs_ctx else AssertionContext()
    n_in = len(base.assertions)

    # Dry-run trace each branch to learn the assertions it appends; the abstract
    # leaves (shape/dtype) are used to build placeholders for the other branches.
    suffix_templates = [
        fn.trace(base, *invals[1:]).out_info[1].assertions[n_in:] for fn in branch_fns
    ]
    packer = _AssertionPacker(sum(suffix_templates, ()))

    def wrap(
        index: int, fn: Callable[..., tuple[Any, AssertionContext]]
    ) -> Callable[..., tuple[Any, list[Array]]]:
        def wrapped(ctx_in: AssertionContext, *args: Any) -> tuple[Any, list[Array]]:
            with jax.disable_jit(False):
                outvals, ctx_out = fn(ctx_in, *args)
            merged: list[RuntimeAssertion[Any, Any]] = []
            for i, templates in enumerate(suffix_templates):
                merged.extend(
                    ctx_out.assertions[n_in:]
                    if i == index
                    else map(_passing, templates)
                )
            return outvals, packer.pack(tuple(merged))

        return wrapped

    wrapped_fns = [wrap(i, fn) for i, fn in enumerate(branch_fns)]

    # jax.lax.switch selects branch ``invals[0]`` (the cond/switch index) directly,
    # matching the primitive's branch order and supporting any number of branches.
    # Branches only append, so the incoming assertions need not be returned.
    outvals, packed = jax.lax.switch(invals[0], wrapped_fns, base, *invals[1:])
    return HandlerResult(
        AssertionContext(ctx.assertions + packer.unpack(packed)), outvals
    )


def _initialize_loop_assertion(
    a: RuntimeAssertion[Any, Any],
) -> RuntimeAssertion[Any, Any]:
    a = bind(a).focus(lambda a: a.predicate).apply(jnp.ones_like)
    return (
        bind(a)
        .focus(lambda a: (a.fix_args, a.fmt_args))
        .apply(partial(jax.tree.map, a.loop_merge.init))
    )


def _merge_loop_assertion(
    old: RuntimeAssertion[Any, Any], new: RuntimeAssertion[Any, Any]
) -> RuntimeAssertion[Any, Any]:
    combine = old.loop_merge.combine
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


def scan_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    """Carry only the body's assertions through the scan.

    Each iteration folds the new assertions into the accumulated ones with their
    ``loop_merge``. Assertions created before the scan stay outside it. The body
    is traced once: the dry run that discovers its assertions is reused by the
    scan.
    """
    _, bind_params = get_bind_params(eqn)
    consts, carry, xs = split_scan_operands(bind_params, invals)
    n_in = len(ctx.assertions)
    body = jax.jit(partial(interpreter, bind_params["jaxpr"]))
    _, ctx_out_tree = body.trace(ctx, *consts, *carry, *(x[0] for x in xs)).out_info
    init = tuple(map(_initialize_loop_assertion, ctx_out_tree.assertions[n_in:]))

    def step(
        state: tuple[tuple[RuntimeAssertion[Any, Any], ...], list[Any]],
        x: list[Any],
    ) -> tuple[tuple[tuple[RuntimeAssertion[Any, Any], ...], list[Any]], list[Any]]:
        kept, args = state
        out, new_ctx = body(ctx, *consts, *args, *x)
        kept = tuple(
            _merge_loop_assertion(old, new)
            for old, new in zip(kept, new_ctx.assertions[n_in:], strict=True)
        )
        carry_out, ys = split_scan_results(bind_params, out)
        return (kept, carry_out), ys

    (kept, carry), ys = jax.lax.scan(
        step,
        (init, carry),
        xs,
        length=bind_params["length"],
        unroll=bind_params["unroll"],
        reverse=bind_params["reverse"],
    )
    return HandlerResult(AssertionContext(ctx.assertions + kept), [*carry, *ys])


def while_handler(
    interpreter: Interpreter[AssertionContext],
    ctx: AssertionContext,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[AssertionContext]:
    """Thread only the loop's own assertions, packed, through the carry.

    Each iteration folds the new assertions into the accumulated ones with their
    ``loop_merge``. Assertions created before the loop stay outside it.
    """
    _, bind_params = get_bind_params(eqn)
    cond_jaxpr: ClosedJaxpr = bind_params["cond_jaxpr"]
    body_jaxpr: ClosedJaxpr = bind_params["body_jaxpr"]
    num_cond_consts = bind_params["cond_nconsts"]
    n_consts = num_cond_consts + bind_params["body_nconsts"]
    cond_consts = invals[:num_cond_consts]
    body_consts = invals[num_cond_consts:n_consts]
    nonconst_invals = invals[n_consts:]
    n_in = len(ctx.assertions)
    body = jax.jit(partial(interpreter, body_jaxpr))
    _, ctx_out_tree = body.trace(ctx, *body_consts, *nonconst_invals).out_info
    templates = ctx_out_tree.assertions[n_in:]
    packer = _AssertionPacker(templates)

    def full_ctx(packed: list[Array]) -> AssertionContext:
        return AssertionContext(ctx.assertions + packer.unpack(packed))

    def _cond(carry: tuple[list[Array], list[Any]]) -> Array:
        packed, args = carry
        out, new_ctx = interpreter(cond_jaxpr, full_ctx(packed), *cond_consts, *args)
        if len(new_ctx.assertions) != n_in + len(templates):
            raise ValueError("While cond modified the context.")
        return jax.tree.leaves(out)[0]

    def _body(
        carry: tuple[list[Array], list[Any]],
    ) -> tuple[list[Array], list[Any]]:
        packed, args = carry
        out, new_ctx = body(ctx, *body_consts, *args)
        merged = map(
            _merge_loop_assertion, packer.unpack(packed), new_ctx.assertions[n_in:]
        )
        return packer.pack(tuple(merged)), list(out)

    init = packer.pack(tuple(map(_initialize_loop_assertion, templates)))
    packed, outvals = jax.lax.while_loop(_cond, _body, (init, list(nonconst_invals)))
    return HandlerResult(full_ctx(packed), outvals)


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


def _only_with_assertions(
    handler: Handler[AssertionContext],
) -> Handler[AssertionContext]:
    """Bind ``eqn`` unchanged unless its sub-jaxprs create or read assertions."""

    def wrapped(
        interpreter: Interpreter[AssertionContext],
        ctx: AssertionContext,
        eqn: JaxprEqn,
        invals: list[TracerValue],
    ) -> HandlerResult[AssertionContext]:
        if _contains_primitive(eqn.params, assertion_p.name, check_assertion_p.name):
            return handler(interpreter, ctx, eqn, invals)
        return default_primitive_handler(interpreter, ctx, eqn, invals)

    return wrapped


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
        policy: What to do with assertions the interpreter cannot reach, e.g.
            inside ``jax.custom_jvp``: RAISE (default), WARN or IGNORE (see
            [InterpreterPolicy][kups.core.effects.interpreter.InterpreterPolicy]).
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
    interpreter = Interpreter(
        {
            assertion_p: assertion_handler,
            check_assertion_p: check_assertion_handler,
            # Re-tracing jit and cond bodies yields slightly faster CPU programs.
            "jit": default_jit_handler,
            "cond": cond_handler,
            "remat2": default_checkpoint_handler,
            "scan": _only_with_assertions(scan_handler),
            "while": _only_with_assertions(while_handler),
            "shard_map": partial(shard_map_handler, context_sharding=context_sharding),
        },
        effects=(assertion_p, check_assertion_p),
        policy=policy,
        label="assertion_interpreter",
    )

    reinterpreted = reinterpret(fn, interpreter=interpreter)

    def wrapped(
        *args: P.args, **kwargs: P.kwargs
    ) -> tuple[R, tuple[RuntimeAssertion[Any, Any], ...]]:
        ctx = AssertionContext()
        outvals, ctx = reinterpreted(ctx, *args, **kwargs)
        return outvals, ctx.assertions

    return wrapped
