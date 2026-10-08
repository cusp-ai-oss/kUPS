# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any, Concatenate, NamedTuple, Self

import jax
import jax.core
from jax.extend import linear_util as lu
from jax.extend import source_info_util
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive
from jax.interpreters import partial_eval as pe

from kups.core.interpreter._compat import get_bind_params
from kups.core.interpreter.environment import InterpreterEnvironment

type PyTreeDef = Any
type TracerValue = Any


class InterpreterPolicy(Enum):
    RAISE = "raise"

    WARN = "warn"

    IGNORE = "ignore"


class HandlerResult[Context](NamedTuple):
    ctx: Context
    outvals: list[TracerValue]


type Handler[Context] = Callable[
    [Interpreter[Context], Context, JaxprEqn, list[TracerValue]], HandlerResult[Context]
]

type MatchingRule[Context] = Callable[[Context, JaxprEqn, list[TracerValue]], bool]


def contains_subjaxprs[Context](
    ctx: Context,  # type: ignore
    eqn: JaxprEqn,
    invals: list[TracerValue],
) -> bool:
    leaves = jax.tree.leaves(
        eqn.params, is_leaf=lambda x: isinstance(x, (Jaxpr, ClosedJaxpr))
    )
    return any(isinstance(leaf, (Jaxpr, ClosedJaxpr)) for leaf in leaves)


class Dispatcher[Context]:
    _custom_matching_rules: list[MatchingRule[Context]]
    _handler_registry: dict[str, Handler[Context]]

    def __init__(self, handlers: dict[Primitive | str, Handler[Context]] | None = None):
        self._custom_matching_rules = []
        self._handler_registry = {}
        if handlers:
            for prim, handler in handlers.items():
                self.register_handler(prim, handler)

    def has_custom_matching_rule(self) -> bool:
        return len(self._custom_matching_rules) > 0

    def matches_custom_rule(
        self, ctx: Context, eqn: JaxprEqn, invals: list[TracerValue]
    ) -> bool:
        return any(rule(ctx, eqn, invals) for rule in self._custom_matching_rules)

    def register_custom_matching_rule(self, rule: MatchingRule[Context]) -> Self:
        self._custom_matching_rules.append(rule)
        return self

    def register_handler(
        self, prim: Primitive | str, handler: Handler[Context]
    ) -> Self:
        if isinstance(prim, Primitive):
            prim = prim.name
        self._handler_registry[prim] = handler
        return self

    def is_registered(self, prim: Primitive | str) -> bool:
        if isinstance(prim, Primitive):
            prim = prim.name
        return prim in self._handler_registry

    def unregister_handler(self, prim: Primitive | str) -> Self:
        if isinstance(prim, Primitive):
            prim = prim.name
        if prim not in self._handler_registry:
            raise KeyError(f"Primitive {prim} not registered")
        del self._handler_registry[prim]
        return self

    def dispatch(
        self,
        interpreter: "Interpreter[Context]",
        ctx: Context,
        eqn: JaxprEqn,
        invals: list[Any],
    ) -> HandlerResult[Context]:
        try:
            handler = self._handler_registry[eqn.primitive.name]
        except KeyError:
            raise NotImplementedError(f"Primitive {eqn.primitive} not registered")
        return handler(interpreter, ctx, eqn, invals)


class Interpreter[Context]:
    policy: InterpreterPolicy
    dispatcher: Dispatcher[Context]

    def __init__(
        self,
        dispatcher: Dispatcher[Context],
        policy: InterpreterPolicy = InterpreterPolicy.RAISE,
        label: str = "",
    ):
        self.dispatcher = dispatcher
        self.policy = policy
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def _handle_default_primitive(
        self,
        ctx: Context,
        eqn: JaxprEqn,
        invals: list[TracerValue],
    ) -> HandlerResult[Context]:
        subfuns, bind_params = get_bind_params(eqn)
        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return HandlerResult(ctx, outvals)

    def _requires_dispatch(
        self, ctx: Context, eqn: JaxprEqn, invals: list[Any]
    ) -> bool:
        return any(
            (
                self.dispatcher.matches_custom_rule(ctx, eqn, invals),
                self.dispatcher.is_registered(eqn.primitive),
            )
        )

    def __call__(
        self, jaxpr: ClosedJaxpr, ctx: Context, *args: Any
    ) -> tuple[list[TracerValue], Context]:
        env = InterpreterEnvironment(jaxpr.jaxpr, jaxpr.consts, args)
        for eqn in jaxpr.jaxpr.eqns:
            try:
                invals = env.read_many(eqn.invars)
            except Exception as e:
                raise ValueError(
                    f"Error reading from env: primitive={eqn.primitive}, invars={eqn.invars}"
                ) from e

            name_stack = (
                source_info_util.current_name_stack() + eqn.source_info.name_stack
            )
            with source_info_util.user_context(
                eqn.source_info.traceback, name_stack=name_stack
            ):
                if self._requires_dispatch(ctx, eqn, invals):
                    try:
                        ctx, outvals = self.dispatcher.dispatch(self, ctx, eqn, invals)
                    except NotImplementedError:
                        if self.policy == InterpreterPolicy.RAISE:
                            raise
                        elif self.policy == InterpreterPolicy.WARN:
                            warnings.warn(
                                f"Handling of {eqn.primitive} is not implemented. Potential sub expressions will thus be ignored."
                            )
                        ctx, outvals = self._handle_default_primitive(ctx, eqn, invals)
                else:
                    ctx, outvals = self._handle_default_primitive(ctx, eqn, invals)

            try:
                env.write_many(eqn.outvars, outvals)
            except Exception as e:
                raise ValueError(
                    f"Error writing to env: primitive={eqn.primitive}, outvars={eqn.outvars} outvals={outvals}"
                ) from e

        try:
            out = env.read_many(jaxpr.jaxpr.outvars)
        except Exception as e:
            raise ValueError(f"Error reading from env: {jaxpr.jaxpr.outvars}") from e

        return out, ctx


def reinterpret[Context, **P, R](
    fn: Callable[P, R], interpreter: Interpreter[Context]
) -> Callable[Concatenate[Context, P], tuple[R, Context]]:
    def inner(ctx: Context, *args: P.args, **kwargs: P.kwargs) -> tuple[R, Context]:
        def fn_closed():
            return fn(*args, **kwargs)

        in_args: tuple[()] = ()
        in_kwargs: dict[str, Any] = {}
        args_flat, in_tree = jax.tree.flatten((in_args, in_kwargs))
        fn_wrapped = lu.wrap_init(
            fn_closed,
            debug_info=jax.api_util.debug_info(
                f"{interpreter.label}_trace", fn_closed, in_args, in_kwargs
            ),
        )
        fn_wrapped, out_tree_thunk = jax.api_util.flatten_fun(fn_wrapped, in_tree)
        jaxpr, _, const = pe.trace_to_jaxpr_dynamic(
            fn_wrapped, [jax.typeof(x) for x in args_flat]
        )
        out, ctx = interpreter(ClosedJaxpr(jaxpr, const), ctx, *args_flat)
        out_tree = out_tree_thunk()
        return jax.tree.unflatten(out_tree, out), ctx

    return inner
