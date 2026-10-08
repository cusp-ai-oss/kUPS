# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Hashable
from enum import Enum
from functools import partial
from typing import Any, Literal, overload

import jax
from jax import ShapeDtypeStruct
from jax._src.named_sharding import UNSPECIFIED as UnspecifiedSharding
from jax.core import AbstractValue
from jax.extend.core import ClosedJaxpr, JaxprEqn, jaxpr_as_fun
from jax.sharding import PartitionSpec

from kups.core.interpreter.interpreter import (
    HandlerResult,
    Interpreter,
    InterpreterContext,
    TracerValue,
    reinterpret,
)
from kups.core.interpreter.util import get_aval, get_bind_params, split_sequence


def default_primitive_handler[Context: InterpreterContext](
    _: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[Context]:
    subfuns, bind_params = get_bind_params(eqn)
    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
    if not eqn.primitive.multiple_results:
        outvals = [outvals]
    return HandlerResult(ctx, outvals)


def default_jit_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[Context]:
    _, bind_params = get_bind_params(eqn)

    jaxpr: ClosedJaxpr = bind_params["jaxpr"]
    in_shardings = bind_params["in_shardings"]
    out_shardings = bind_params["out_shardings"]
    donated_invars = bind_params["donated_invars"]
    keep_unused = bind_params["keep_unused"]
    inline = bind_params["inline"]
    compiler_options_kvs = bind_params["compiler_options_kvs"]

    fn = jax.jit(
        partial(interpreter, jaxpr),
        in_shardings=(
            UnspecifiedSharding,
            *in_shardings,
        ),
        out_shardings=(list(out_shardings), UnspecifiedSharding),
        donate_argnums=[i for i, is_donated in enumerate(donated_invars) if is_donated],
        keep_unused=keep_unused,
        inline=inline,
        compiler_options=dict(compiler_options_kvs),
    )
    outvals, ctx_out = fn(ctx, *invals)
    return HandlerResult(ctx_out, outvals)


class Uninitialized(ShapeDtypeStruct):
    def __init__(self, aval: AbstractValue, ignore_sharding: bool = False):
        assert hasattr(aval, "shape") and hasattr(aval, "dtype"), (
            f"{aval} does not have a shape or dtype"
        )
        shape = getattr(aval, "shape")
        dtype = getattr(aval, "dtype")
        if ignore_sharding:
            sharding = None
            mat = None
        else:
            sharding = getattr(aval, "sharding", None)
            mat = getattr(aval, "manual_axis_type", getattr(aval, "vma", None))
        weak_type = getattr(aval, "weak_type", False)
        is_ref = getattr(aval, "is_ref", False)
        # JAX 0.10 renamed the ShapeDtypeStruct kwarg ``vma`` → ``manual_axis_type``.
        try:
            super().__init__(
                shape,
                dtype,
                sharding=sharding,
                weak_type=weak_type,
                manual_axis_type=mat,  # type: ignore[call-arg]
                is_ref=is_ref,
            )
        except TypeError:
            super().__init__(
                shape,
                dtype,
                sharding=sharding,
                weak_type=weak_type,
                vma=mat,  # pyrefly: ignore[unexpected-keyword]
                is_ref=is_ref,
            )


def _sentinel_preserving_parent[Context: InterpreterContext](
    ctx_out_tree: Context, ctx: Context
) -> Context:
    """Build sentinel from trace output, preserving real parent values.

    The dry-run trace wraps ALL leaves (including the parent chain) as abstract
    values. Naive jax.tree.map(Uninitialized, ...) would cause the initializer
    to zero pre-existing parent data. This function keeps real parent values
    intact and only marks child-level leaves as Uninitialized.

    Relies on register_dataclass flattening data_fields in declaration order:
    parent leaves come first, then child value leaves.
    """
    trace_leaves, trace_treedef = jax.tree.flatten(ctx_out_tree)
    parent_leaves = jax.tree.leaves(ctx)
    n_parent = len(parent_leaves)

    sentinel_leaves = list(parent_leaves) + [
        Uninitialized(leaf) for leaf in trace_leaves[n_parent:]
    ]

    return jax.tree.unflatten(trace_treedef, sentinel_leaves)


class ScanSemantics(Enum):
    CARRY = "carry"
    RESULT = "result"


def _split_scan_inputs(
    bind_params: dict[str, Any], invals: list[TracerValue]
) -> tuple[list[TracerValue], list[TracerValue], list[TracerValue]]:
    """Partition scan inputs across JAX's old and new parameter schemas."""
    if "ft_in" in bind_params:
        consts, carry, xs = bind_params["ft_in"].update(invals).unpack()
        return list(consts), list(carry), list(xs)

    consts, carry, xs = split_sequence(
        invals, (bind_params["num_consts"], bind_params["num_carry"])
    )
    return consts, carry, xs


def _split_scan_outputs(
    bind_params: dict[str, Any], outvals: list[TracerValue]
) -> tuple[list[TracerValue], list[TracerValue]]:
    """Partition scan outputs across JAX's old and new parameter schemas."""
    if "ft_out" in bind_params:
        carry, results = bind_params["ft_out"].update(outvals).unpack()
        return list(carry), list(results)

    carry, results = split_sequence(outvals, (bind_params["num_carry"],))
    return carry, results


def _assert_same_tree[PyTree](old: PyTree, new: PyTree):
    old_leaves, old_tree_def = jax.tree.flatten(old)
    new_leaves, new_tree_def = jax.tree.flatten(new)
    is_same_def = old_tree_def == new_tree_def
    if not is_same_def:
        raise ValueError(
            f"Function modified the tree structure: {new_tree_def} != {old_tree_def}"
        )
    leaf_mismatches: list[str] = []
    for x, y in zip(old_leaves, new_leaves, strict=True):
        xaval = get_aval(x)
        yaval = get_aval(y)
        match_attrs = ["shape", "dtype"]
        if any(getattr(xaval, attr) != getattr(yaval, attr) for attr in match_attrs):
            leaf_mismatches.append(f"{xaval} != {yaval}")
    if not len(leaf_mismatches) == 0:
        raise ValueError(f"Function modified the tree values: {leaf_mismatches}")


def _scan_handler_result[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[Context]:
    _, bind_params = get_bind_params(eqn)
    jaxpr: ClosedJaxpr = bind_params["jaxpr"]
    consts, carry, xs = _split_scan_inputs(bind_params, invals)

    # Trace to discover child context structure for reconstruction after scan
    _, ctx_out_tree = (
        jax.jit(partial(interpreter, jaxpr))
        .trace(ctx.push(), *consts, *carry, *(x[0] for x in xs))
        .out_info
    )
    ctx_out_treedef = jax.tree_util.tree_structure(ctx_out_tree)
    n_parent = len(jax.tree.leaves(ctx))

    def _new_body_fn(carry: list[TracerValue], xs: list[TracerValue]):
        out_flat, ctx_out = interpreter(jaxpr, ctx.push(), *consts, *carry, *xs)
        carry_out_flat, results_flat = _split_scan_outputs(bind_params, out_flat)
        # Only return child-level leaves as scan results — parent is constant
        child_leaves = jax.tree.leaves(ctx_out)[n_parent:]
        return carry_out_flat, (child_leaves, results_flat)

    carry_out, (stacked_child_leaves, results) = jax.lax.scan(
        _new_body_fn,
        carry,
        xs,
        length=bind_params.get("length"),
        unroll=bind_params.get("unroll", 1),
        reverse=bind_params.get("reverse", False),
    )

    # Reconstruct: real parent (unstacked) + stacked child values
    all_leaves = list(jax.tree.leaves(ctx)) + list(stacked_child_leaves)
    ctx_out = jax.tree.unflatten(ctx_out_treedef, all_leaves)
    ctx_out = ctx_out.pop()

    return HandlerResult(ctx_out, jax.tree.leaves(carry_out + results))


def _scan_handler_carry[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    initializer: Callable[[Context, Context], Context],
    updater: Callable[[Context, Context], Context],
) -> HandlerResult[Context]:
    _, bind_params = get_bind_params(eqn)
    jaxpr: ClosedJaxpr = bind_params["jaxpr"]
    consts, carry, xs = _split_scan_inputs(bind_params, invals)

    _, ctx_out_tree = (
        jax.jit(partial(interpreter, jaxpr))
        .trace(ctx.push(), *consts, *carry, *(x[0] for x in xs))
        .out_info
    )
    sentinel_ctx = _sentinel_preserving_parent(ctx_out_tree, ctx)
    initialized_ctx = initializer(ctx, sentinel_ctx)
    if any(isinstance(x, Uninitialized) for x in jax.tree.leaves(initialized_ctx)):
        raise ValueError(
            f"All context variables must be initialized within the initializer. "
            f" Found uninitialized variables: {jax.tree.leaves(initialized_ctx)}"
        )

    def _body_fn(carry: tuple[Context, list[TracerValue]], xs: list[TracerValue]):
        old_ctx, carry_in_flat = carry
        out_flat, new_ctx = interpreter(jaxpr, ctx.push(), *consts, *carry_in_flat, *xs)
        new_ctx = updater(old_ctx, new_ctx)
        try:
            _assert_same_tree(old_ctx, new_ctx)
        except ValueError as e:
            raise ValueError("Scan body modified the context.") from e
        carry_out_flat, results_flat = _split_scan_outputs(bind_params, out_flat)
        return (new_ctx, carry_out_flat), results_flat

    (ctx_out, carry), results = jax.lax.scan(
        _body_fn,
        (initialized_ctx, carry),
        xs,
        length=bind_params.get("length"),
        unroll=bind_params.get("unroll", 1),
        reverse=bind_params.get("reverse", False),
    )
    return HandlerResult(ctx_out.pop(), jax.tree.leaves(carry + results))


@overload
def default_scan_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    threading: Literal[ScanSemantics.CARRY],
    initializer: Callable[[Context, Context], Context],
    updater: Callable[[Context, Context], Context],
) -> HandlerResult[Context]: ...


@overload
def default_scan_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    threading: Literal[ScanSemantics.RESULT],
) -> HandlerResult[Context]: ...


def default_scan_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    threading: ScanSemantics = ScanSemantics.CARRY,
    initializer: Callable[[Context, Context], Context] | None = None,
    updater: Callable[[Context, Context], Context] | None = None,
) -> HandlerResult[Context]:
    if threading == ScanSemantics.CARRY:
        assert initializer is not None, (
            "Initializer must be provided for carry threading."
        )
        assert updater is not None, "Updater must be provided for carry threading."
        return _scan_handler_carry(
            interpreter, ctx, eqn, invals, initializer=initializer, updater=updater
        )
    elif threading == ScanSemantics.RESULT:
        return _scan_handler_result(interpreter, ctx, eqn, invals)
    else:
        raise ValueError(f"Unknown scan threading: {threading}")


def default_while_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    initializer: Callable[[Context, Context], Context],
    updater: Callable[[Context, Context], Context],
) -> HandlerResult[Context]:
    _, bind_params = get_bind_params(eqn)
    cond_jaxpr: ClosedJaxpr = bind_params["cond_jaxpr"]
    body_jaxpr: ClosedJaxpr = bind_params["body_jaxpr"]
    num_cond_consts = bind_params["cond_nconsts"]
    num_body_consts = bind_params["body_nconsts"]

    cond_consts, body_consts, nonconst_invals = split_sequence(
        invals, (num_cond_consts, num_body_consts)
    )

    _, ctx_out_tree = (
        jax.jit(partial(interpreter, body_jaxpr))
        .trace(ctx.push(), *body_consts, *nonconst_invals)
        .out_info
    )
    sentinel_ctx = _sentinel_preserving_parent(ctx_out_tree, ctx)
    initialized_ctx = initializer(ctx, sentinel_ctx)
    if any(isinstance(x, Uninitialized) for x in jax.tree.leaves(initialized_ctx)):
        raise ValueError(
            f"All context variables must be initialized within the initializer. "
            f" Found uninitialized variables: {jax.tree.leaves(initialized_ctx)}"
        )

    def _cond(packed: tuple[Context, list[TracerValue]]):
        old_ctx, args = packed
        args_flat, _ = jax.tree.flatten(args)
        out, new_ctx = interpreter(cond_jaxpr, old_ctx, *cond_consts, *args_flat)
        try:
            _assert_same_tree(old_ctx, new_ctx)
        except ValueError as e:
            raise ValueError("While cond modified the context.") from e
        return jax.tree.leaves(out)[0]

    def _body(packed: tuple[Context, list[TracerValue]]):
        old_ctx, args = packed
        args_flat, _ = jax.tree.flatten(args)
        out, new_ctx = interpreter(body_jaxpr, ctx.push(), *body_consts, *args_flat)
        new_ctx = updater(old_ctx, new_ctx)
        try:
            _assert_same_tree(old_ctx, new_ctx)
        except ValueError as e:
            raise ValueError("While body modified the context.") from e
        return new_ctx, out

    ctx_out, outvals = jax.lax.while_loop(
        _cond, _body, (initialized_ctx, nonconst_invals)
    )
    return HandlerResult(ctx_out.pop(), outvals)


def default_cond_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[Context]:
    _, bind_params = get_bind_params(eqn)
    branches = bind_params["branches"]

    context_aware_branch_fns = [
        reinterpret(jaxpr_as_fun(jaxpr), interpreter) for jaxpr in branches
    ]
    branch_ctx_trees = [
        jax.jit(branch_fn).trace(ctx.push(), *invals[1:]).out_info
        for branch_fn in context_aware_branch_fns
    ]
    assert len(branches) > 0
    try:
        for tree in branch_ctx_trees[1:]:
            _assert_same_tree(branch_ctx_trees[0], tree)
    except ValueError as e:
        raise ValueError("Cond branches return inconsistent contexts.") from e

    outvals, ctx_out = jax.lax.cond(
        invals[0], *reversed(context_aware_branch_fns), ctx, *invals[1:]
    )
    return HandlerResult(ctx_out, outvals)


def default_checkpoint_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[Context]:
    jaxpr = ClosedJaxpr(eqn.params["jaxpr"], ())
    outvals, ctx_out = interpreter(jaxpr, ctx, *invals)
    return HandlerResult(ctx_out, outvals)


def default_shard_map_handler[Context: InterpreterContext](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    declare_ctx_in_specs: Callable[[Context], Context | PartitionSpec] | None = None,
    declare_ctx_out_specs: Callable[[Context], Context | PartitionSpec] | None = None,
    reduce_ctx: Callable[[Context], Context] | None = None,
) -> HandlerResult[Context]:
    """Reinterpret a ``shard_map`` equation through ``interpreter``.

    Args:
        interpreter: The interpreter walking the surrounding jaxpr.
        ctx: Input context (lives outside the manual axis scope, typically
            replicated across all shards).
        eqn: The ``shard_map`` equation being handled.
        invals: User-data inputs to the equation.
        declare_ctx_in_specs: Optional. Map the input context tree to the
            ``in_specs`` shard_map should use for it. Defaults to ``P()``
            (replicated) for every leaf.
        declare_ctx_out_specs: Optional. Map the post-body context tree to
            the ``out_specs`` shard_map should use for it. Defaults to ``P()``
            (replicated, broadcast across leaves). Note: deriving the tree
            shape requires re-tracing the body outside the manual axis scope,
            which fails for bodies that bind manual axes (collectives, pvary,
            axis_index). Use ``reduce_ctx`` instead in that case.
        reduce_ctx: Optional. Transform applied to the per-shard post-body
            context inside the ``shard_map`` body before returning. The
            natural place to invoke collectives (``all_gather``, ``psum``, …)
            to combine per-shard data into globally-replicated values, since
            it executes inside the manual axis scope. When the result is
            fully replicated, the default ``out_specs=P()`` is satisfied
            without ``declare_ctx_out_specs``.
    """
    jaxpr = ClosedJaxpr(eqn.params["jaxpr"], ())

    def fn(ctx: Context, *invals: TracerValue) -> tuple[list[Any], Context]:
        outvals, ctx_post = interpreter(jaxpr, ctx, *invals)
        if reduce_ctx is not None:
            ctx_post = reduce_ctx(ctx_post)
        return outvals, ctx_post

    if declare_ctx_in_specs is not None:
        ctx_in_specs = declare_ctx_in_specs(jax.tree.map(Uninitialized, ctx))
    else:
        ctx_in_specs = jax.P()

    if declare_ctx_out_specs is not None:
        abstract_invals = [Uninitialized(x) for x in jaxpr.in_avals]
        ctx_out_tree = (
            jax.jit(fn)
            .trace(jax.tree.map(Uninitialized, ctx), *abstract_invals)
            .out_info[1]
        )
        ctx_out_specs = declare_ctx_out_specs(ctx_out_tree)
    else:
        ctx_out_specs = jax.P()

    # JAX 0.11 renamed the shard_map primitive parameter while retaining
    # ``axis_names`` in the public API.
    axis_names: frozenset[Hashable] = (
        eqn.params["newly_manual_axes"]
        if "newly_manual_axes" in eqn.params
        else eqn.params.get("manual_axes", frozenset())
    )
    sharded_fn = jax.shard_map(
        out_specs=(list(eqn.params["out_specs"]), ctx_out_specs),
        in_specs=(ctx_in_specs, *eqn.params["in_specs"]),
        mesh=eqn.params["mesh"],
        axis_names=axis_names,
        check_vma=eqn.params["check_vma"],
    )(fn)

    outvals, ctx_out = sharded_fn(ctx, *invals)
    return HandlerResult(ctx_out, outvals)
