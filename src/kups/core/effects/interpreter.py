# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Iterator, Mapping
from enum import Enum
from typing import Any, Concatenate, NamedTuple

import jax
import jax.core
from jax.extend import linear_util as lu
from jax.extend import source_info_util
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Primitive
from jax.interpreters import partial_eval as pe

from kups.core.effects.environment import InterpreterEnvironment
from kups.core.utils._jax_compat import get_bind_params

type TracerValue = Any


class InterpreterPolicy(Enum):
    """What to do with an effect the interpreter cannot reach.

    That is an effect primitive without a handler, or an unhandled primitive
    whose sub-jaxprs contain an effect primitive.
    """

    RAISE = "raise"
    """Raise ``NotImplementedError``."""

    WARN = "warn"
    """Warn, then bind the primitive unchanged. Its effects are lost."""

    IGNORE = "ignore"
    """Bind the primitive unchanged. Its effects are lost."""


class HandlerResult[Context](NamedTuple):
    ctx: Context
    outvals: list[TracerValue]


type Handler[Context] = Callable[
    [Interpreter[Context], Context, JaxprEqn, list[TracerValue]], HandlerResult[Context]
]


def _name(primitive: Primitive | str) -> str:
    return primitive.name if isinstance(primitive, Primitive) else primitive


def _sub_jaxprs(eqn: JaxprEqn) -> Iterator[Jaxpr]:
    leaves = jax.tree.leaves(
        eqn.params, is_leaf=lambda x: isinstance(x, (Jaxpr, ClosedJaxpr))
    )
    for leaf in leaves:
        if isinstance(leaf, ClosedJaxpr):
            yield leaf.jaxpr
        elif isinstance(leaf, Jaxpr):
            yield leaf


def bind_primitive(eqn: JaxprEqn, invals: list[TracerValue]) -> list[TracerValue]:
    """Bind ``eqn``'s primitive to ``invals`` unchanged."""
    subfuns, bind_params = get_bind_params(eqn)
    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
    return outvals if eqn.primitive.multiple_results else [outvals]


class Interpreter[Context]:
    """Evaluate a jaxpr while threading a context through effect handlers.

    Args:
        handlers: Handler per primitive. Every equation of a listed primitive
            goes to its handler; all other primitives are bound unchanged.
        effects: Primitives that perform effects. Applies ``policy`` when one is
            not handled, directly or inside an unhandled primitive's sub-jaxpr.
        policy: What to do with such an unreachable effect.
        label: Name used in tracing debug info.
    """

    def __init__(
        self,
        handlers: Mapping[Primitive | str, Handler[Context]],
        *,
        effects: Collection[Primitive | str] = (),
        policy: InterpreterPolicy = InterpreterPolicy.RAISE,
        label: str = "",
    ):
        self.handlers: Mapping[str, Handler[Context]] = {
            _name(p): h for p, h in handlers.items()
        }
        self.effects: frozenset[str] = frozenset(map(_name, effects))
        self.policy = policy
        self.label = label

    def _has_effect(self, eqn: JaxprEqn) -> bool:
        """Whether ``eqn`` is an effect or has one in a sub-jaxpr."""
        return eqn.primitive.name in self.effects or any(
            self._has_effect(inner)
            for jaxpr in _sub_jaxprs(eqn)
            for inner in jaxpr.eqns
        )

    def _handler(self, eqn: JaxprEqn) -> Handler[Context] | None:
        handler = self.handlers.get(eqn.primitive.name)
        if handler is not None or not self._has_effect(eqn):
            return handler
        message = f"No handler for {eqn.primitive}, which performs or contains effects."
        if self.policy == InterpreterPolicy.RAISE:
            raise NotImplementedError(message)
        if self.policy == InterpreterPolicy.WARN:
            warnings.warn(f"{message} Its effects are ignored.", stacklevel=2)
        return None

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
                handler = self._handler(eqn)
                if handler is None:
                    outvals = bind_primitive(eqn, invals)
                else:
                    ctx, outvals = handler(self, ctx, eqn, invals)

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
