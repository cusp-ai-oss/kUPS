# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
from jax import ShapeDtypeStruct
from jax._src.named_sharding import UNSPECIFIED as UnspecifiedSharding
from jax.core import AbstractValue
from jax.extend.core import ClosedJaxpr, JaxprEqn, jaxpr_as_fun
from jax.sharding import PartitionSpec

from kups.core.interpreter._compat import (
    get_bind_params,
    manual_axes_kwarg,
    shard_map_manual_axes,
    split_scan_operands,
    split_scan_results,
)
from kups.core.interpreter.interpreter import (
    HandlerResult,
    Interpreter,
    TracerValue,
    reinterpret,
)
from kups.core.interpreter.util import split_sequence


def default_primitive_handler[Context](
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


def default_jit_handler[Context](
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
    def __init__(self, aval: AbstractValue):
        assert hasattr(aval, "shape") and hasattr(aval, "dtype"), (
            f"{aval} does not have a shape or dtype"
        )
        shape = getattr(aval, "shape")
        dtype = getattr(aval, "dtype")
        super().__init__(
            shape,
            dtype,
            sharding=getattr(aval, "sharding", None),
            weak_type=getattr(aval, "weak_type", False),
            is_ref=getattr(aval, "is_ref", False),
            **manual_axes_kwarg(aval),
        )


def _sentinel_for_new_leaves[Context](ctx_out_tree: Context, ctx: Context) -> Context:
    """Abstract body output context whose new leaves are marked uninitialized.

    A loop body may only extend the context, so the incoming context's leaves
    are a prefix of the body's output leaves. Those keep their real values; the
    leaves the body appends become ``Uninitialized`` for the initializer to fill.
    """
    trace_leaves, trace_treedef = jax.tree.flatten(ctx_out_tree)
    leaves = jax.tree.leaves(ctx)
    sentinel_leaves = list(leaves) + [
        Uninitialized(leaf) for leaf in trace_leaves[len(leaves) :]
    ]
    return jax.tree.unflatten(trace_treedef, sentinel_leaves)


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
        xaval = jax.typeof(x)
        yaval = jax.typeof(y)
        match_attrs = ["shape", "dtype"]
        if any(getattr(xaval, attr) != getattr(yaval, attr) for attr in match_attrs):
            leaf_mismatches.append(f"{xaval} != {yaval}")
    if not len(leaf_mismatches) == 0:
        raise ValueError(f"Function modified the tree values: {leaf_mismatches}")


def default_scan_handler[Context](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
    *,
    initializer: Callable[[Context, Context], Context],
    updater: Callable[[Context, Context], Context],
) -> HandlerResult[Context]:
    _, bind_params = get_bind_params(eqn)
    jaxpr: ClosedJaxpr = bind_params["jaxpr"]
    consts, carry, xs = split_scan_operands(bind_params, invals)

    _, ctx_out_tree = (
        jax.jit(partial(interpreter, jaxpr))
        .trace(ctx, *consts, *carry, *(x[0] for x in xs))
        .out_info
    )
    sentinel_ctx = _sentinel_for_new_leaves(ctx_out_tree, ctx)
    initialized_ctx = initializer(ctx, sentinel_ctx)
    if any(isinstance(x, Uninitialized) for x in jax.tree.leaves(initialized_ctx)):
        raise ValueError(
            f"All context variables must be initialized within the initializer. "
            f" Found uninitialized variables: {jax.tree.leaves(initialized_ctx)}"
        )

    def _body_fn(carry: tuple[Context, list[TracerValue]], xs: list[TracerValue]):
        old_ctx, carry_in_flat = carry
        out_flat, new_ctx = interpreter(jaxpr, ctx, *consts, *carry_in_flat, *xs)
        new_ctx = updater(old_ctx, new_ctx)
        try:
            _assert_same_tree(old_ctx, new_ctx)
        except ValueError as e:
            raise ValueError("Scan body modified the context.") from e
        carry_out_flat, results_flat = split_scan_results(bind_params, out_flat)
        return (new_ctx, carry_out_flat), results_flat

    (ctx_out, carry), results = jax.lax.scan(
        _body_fn,
        (initialized_ctx, carry),
        xs,
        length=bind_params.get("length"),
        unroll=bind_params.get("unroll", 1),
        reverse=bind_params.get("reverse", False),
    )
    return HandlerResult(ctx_out, jax.tree.leaves(carry + results))


def default_while_handler[Context](
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
        .trace(ctx, *body_consts, *nonconst_invals)
        .out_info
    )
    sentinel_ctx = _sentinel_for_new_leaves(ctx_out_tree, ctx)
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
        out, new_ctx = interpreter(body_jaxpr, ctx, *body_consts, *args_flat)
        new_ctx = updater(old_ctx, new_ctx)
        try:
            _assert_same_tree(old_ctx, new_ctx)
        except ValueError as e:
            raise ValueError("While body modified the context.") from e
        return new_ctx, out

    ctx_out, outvals = jax.lax.while_loop(
        _cond, _body, (initialized_ctx, nonconst_invals)
    )
    return HandlerResult(ctx_out, outvals)


def default_cond_handler[Context](
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
        jax.jit(branch_fn).trace(ctx, *invals[1:]).out_info
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


def default_checkpoint_handler[Context](
    interpreter: Interpreter[Context],
    ctx: Context,
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> HandlerResult[Context]:
    jaxpr = ClosedJaxpr(eqn.params["jaxpr"], ())
    outvals, ctx_out = interpreter(jaxpr, ctx, *invals)
    return HandlerResult(ctx_out, outvals)


def default_shard_map_handler[Context](
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

    sharded_fn = jax.shard_map(
        out_specs=(list(eqn.params["out_specs"]), ctx_out_specs),
        in_specs=(ctx_in_specs, *eqn.params["in_specs"]),
        mesh=eqn.params["mesh"],
        axis_names=shard_map_manual_axes(eqn.params),
        check_vma=eqn.params["check_vma"],
    )(fn)

    outvals, ctx_out = sharded_fn(ctx, *invals)
    return HandlerResult(ctx_out, outvals)
