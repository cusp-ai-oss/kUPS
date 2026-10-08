# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from kups.core.interpreter.handlers import (
    ScanSemantics,
    Uninitialized,
    default_checkpoint_handler,
    default_cond_handler,
    default_jit_handler,
    default_primitive_handler,
    default_scan_handler,
    default_while_handler,
)
from kups.core.interpreter.interpreter import (
    Dispatcher,
    Handler,
    HandlerResult,
    Interpreter,
    InterpreterContext,
    InterpreterPolicy,
    JaxprEqn,
    TracerValue,
    contains_subjaxprs,
)


@partial(
    jax.tree_util.register_dataclass,
    meta_fields=("metadata", "level"),
    data_fields=("parent", "value"),
)
@dataclass(frozen=True)
class MockContext(InterpreterContext):
    """Test implementation of InterpreterContext.

    Note: Named MockContext to avoid pytest collection warnings.
    """

    metadata: tuple[str, ...]
    parent: MockContext | None
    level: int
    value: tuple[TracerValue, ...]

    def add_meta(self, key: str) -> MockContext:
        return MockContext(self.metadata + (key,), self.parent, self.level, self.value)

    def add_value(self, value: TracerValue) -> MockContext:
        return MockContext(
            self.metadata, self.parent, self.level, self.value + (value,)
        )

    def push(self) -> MockContext:
        return MockContext((), self, self.level + 1, ())

    def pop(self) -> MockContext:
        parent = self.parent
        if parent is None:
            raise ValueError("Cannot pop from root context")
        for key in self.metadata:
            parent = parent.add_meta(key)
        for value in self.value:
            parent = parent.add_value(value)
        return parent

    @property
    def total_metadata_count(self) -> int:
        return len(self.metadata)

    @property
    def total_value_count(self) -> int:
        return len(self.value)


class HandlerFactory:
    @staticmethod
    def create_primitive_handler(
        meta_key: str,
        value_fn: Callable[[list[TracerValue]], TracerValue] | None = None,
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn(invals: list[TracerValue]) -> TracerValue:
                return invals[0]

            value_fn = default_value_fn

        def handler(
            _: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            return default_primitive_handler(_, ctx, eqn, invals)

        return handler

    @staticmethod
    def create_primitive_handler_with_extra_value(
        meta_key: str,
        value_fn: Callable[[list[TracerValue]], TracerValue] | None = None,
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn(invals: list[TracerValue]) -> TracerValue:
                return invals[0]

            value_fn = default_value_fn

        def handler(
            _: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn(invals))
            return default_primitive_handler(_, ctx, eqn, invals)

        return handler

    @staticmethod
    def create_jit_handler(
        meta_key: str = "jit", value_fn: Callable[[], TracerValue] | None = None
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.ones(1, dtype=jnp.int32)

            value_fn = default_value_fn

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            return default_jit_handler(interpreter, ctx, eqn, invals)

        return handler

    @staticmethod
    def create_jit_handler_with_extra_value(
        meta_key: str = "jit", value_fn: Callable[[], TracerValue] | None = None
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.ones(1, dtype=jnp.int32)

            value_fn = default_value_fn

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn())
            return default_jit_handler(interpreter, ctx, eqn, invals)

        return handler

    @staticmethod
    def create_scan_handler(
        meta_key: str = "scan", value_fn: Callable[[], TracerValue] | None = None
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.zeros(1, dtype=jnp.float32)

            value_fn = default_value_fn

        def initializer(_old_ctx, sentinel_ctx):
            # Zero-initialize Uninitialized leaves discovered via dry run trace
            leaves, tree = jax.tree.flatten(sentinel_ctx)
            leaves = [
                jnp.zeros_like(x) if isinstance(x, Uninitialized) else x for x in leaves
            ]
            return jax.tree.unflatten(tree, leaves)

        def updater(old_ctx, new_ctx):  # default updater prefers new context
            return new_ctx

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn())
            return default_scan_handler(
                interpreter,
                ctx,
                eqn,
                invals,
                threading=ScanSemantics.CARRY,
                initializer=initializer,
                updater=updater,
            )

        return handler

    @staticmethod
    def create_scan_handler_with_mode(
        meta_key: str = "scan",
        *,
        constant_context_threading: bool,
        value_fn: Callable[[], TracerValue] | None = None,
    ) -> Handler[MockContext]:
        """Create a scan handler with control over constant_context_threading.

        When constant_context_threading is True, context is threaded via consts; otherwise via carry.
        """
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.zeros(1, dtype=jnp.float32)

            value_fn = default_value_fn

        # Convert the old parameter to new ScanSemantics enum
        threading = (
            ScanSemantics.RESULT if constant_context_threading else ScanSemantics.CARRY
        )

        def initializer(_old_ctx, sentinel_ctx):
            leaves, tree = jax.tree.flatten(sentinel_ctx)
            leaves = [
                jnp.zeros_like(x) if isinstance(x, Uninitialized) else x for x in leaves
            ]
            return jax.tree.unflatten(tree, leaves)

        def updater(old_ctx, new_ctx):
            return new_ctx

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn())
            if threading == ScanSemantics.CARRY:
                return default_scan_handler(
                    interpreter,
                    ctx,
                    eqn,
                    invals,
                    threading=threading,
                    initializer=initializer,
                    updater=updater,
                )
            else:  # RESULT threading
                return default_scan_handler(
                    interpreter,
                    ctx,
                    eqn,
                    invals,
                    threading=threading,
                )

        return handler

    @staticmethod
    def create_scan_handler_with_updater(
        meta_key: str = "scan",
        *,
        value_fn: Callable[[], TracerValue] | None = None,
        updater: Callable[[MockContext, MockContext], MockContext] | None = None,
    ) -> Handler[MockContext]:
        """Create a scan handler with carry threading and an updater function."""
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.zeros(1, dtype=jnp.float32)

            value_fn = default_value_fn

        def initializer(_old_ctx, sentinel_ctx):
            leaves, tree = jax.tree.flatten(sentinel_ctx)
            leaves = [
                jnp.zeros_like(x) if isinstance(x, Uninitialized) else x for x in leaves
            ]
            return jax.tree.unflatten(tree, leaves)

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn())
            # If no updater supplied, fall back to identity merge
            eff_updater: Callable[[MockContext, MockContext], MockContext] = (
                updater or (lambda old_ctx, new_ctx: new_ctx)
            )
            return default_scan_handler(
                interpreter,
                ctx,
                eqn,
                invals,
                threading=ScanSemantics.CARRY,
                initializer=initializer,
                updater=eff_updater,
            )

        return handler

    @staticmethod
    def create_while_handler(
        meta_key: str = "while", value_fn: Callable[[], TracerValue] | None = None
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.ones(1, dtype=jnp.float32)

            value_fn = default_value_fn

        def initializer(_old_ctx, sentinel_ctx):
            leaves, tree = jax.tree.flatten(sentinel_ctx)
            leaves = [
                jnp.zeros_like(x) if isinstance(x, Uninitialized) else x for x in leaves
            ]
            return jax.tree.unflatten(tree, leaves)

        def updater(old_ctx, new_ctx):
            return new_ctx

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn())
            return default_while_handler(
                interpreter, ctx, eqn, invals, initializer=initializer, updater=updater
            )

        return handler

    @staticmethod
    def create_minimal_scan_handler(meta_key: str = "scan") -> Handler[MockContext]:
        """Create a scan handler that only adds metadata, no values."""

        def initializer(_old_ctx, sentinel_ctx):
            leaves, tree = jax.tree.flatten(sentinel_ctx)
            leaves = [
                jnp.zeros_like(x) if isinstance(x, Uninitialized) else x for x in leaves
            ]
            return jax.tree.unflatten(tree, leaves)

        def updater(old_ctx, new_ctx):
            return new_ctx

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            return default_scan_handler(
                interpreter,
                ctx,
                eqn,
                invals,
                threading=ScanSemantics.CARRY,
                initializer=initializer,
                updater=updater,
            )

        return handler

    @staticmethod
    def create_checkpoint_handler(
        meta_key: str = "checkpoint",
        value_fn: Callable[[], TracerValue] | None = None,
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.ones(1, dtype=jnp.int32)

            value_fn = default_value_fn

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn())
            return default_checkpoint_handler(interpreter, ctx, eqn, invals)

        return handler

    @staticmethod
    def create_cond_handler(
        meta_key: str = "cond", value_fn: Callable[[], TracerValue] | None = None
    ) -> Handler[MockContext]:
        if value_fn is None:

            def default_value_fn() -> TracerValue:
                return jnp.ones(1, dtype=jnp.float32)

            value_fn = default_value_fn

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            ctx = ctx.add_meta(meta_key)
            ctx = ctx.add_value(value_fn())
            return default_cond_handler(interpreter, ctx, eqn, invals)

        return handler


def create_test_dispatcher(
    handlers: dict[Any, Handler[MockContext]],
) -> Dispatcher[MockContext]:
    dispatcher = Dispatcher(handlers)
    dispatcher.register_custom_matching_rule(contains_subjaxprs)
    return dispatcher


def create_test_interpreter(
    handlers: dict[Any, Handler[MockContext]],
    policy: InterpreterPolicy = InterpreterPolicy.WARN,
    include_in_match: frozenset[str] = frozenset(),
) -> Interpreter[MockContext]:
    dispatcher = create_test_dispatcher(handlers).register_custom_matching_rule(
        lambda ctx, eqn, invals: eqn.primitive.name in include_in_match
    )
    return Interpreter(dispatcher=dispatcher, policy=policy)
