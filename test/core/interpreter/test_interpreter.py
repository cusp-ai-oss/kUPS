# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Pruned interpreter tests: retain only high-signal behaviors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from kups.core.interpreter.interpreter import (
    Dispatcher,
    Interpreter,
    InterpreterPolicy,
    contains_subjaxprs,
)

from ._builders import HandlerFactory, MockContext, create_test_interpreter


class TestDispatcher:
    def test_basic_registration_cycle(self, handler_factory: HandlerFactory):
        dispatcher = Dispatcher({})
        sin_handler = handler_factory.create_primitive_handler("sin")
        assert not dispatcher.is_registered(jax.lax.sin_p)
        dispatcher.register_handler(jax.lax.sin_p, sin_handler)
        assert dispatcher.is_registered(jax.lax.sin_p)
        dispatcher.unregister_handler(jax.lax.sin_p)
        assert not dispatcher.is_registered(jax.lax.sin_p)
        dispatcher.register_custom_matching_rule(contains_subjaxprs)
        assert dispatcher.has_custom_matching_rule()
        with pytest.raises(KeyError):
            dispatcher.unregister_handler(jax.lax.sin_p)


class TestInterpreter:
    def test_policy_and_label(self):
        dispatcher = Dispatcher({})
        assert Interpreter(dispatcher=dispatcher).policy == InterpreterPolicy.RAISE
        assert (
            Interpreter(dispatcher=dispatcher, policy=InterpreterPolicy.WARN).policy
            == InterpreterPolicy.WARN
        )
        assert (
            Interpreter(dispatcher=dispatcher, policy=InterpreterPolicy.IGNORE).policy
            == InterpreterPolicy.IGNORE
        )
        assert Interpreter(dispatcher=dispatcher, label="x").label == "x"


class TestContextThreading:
    def test_jit_threads_metadata(self, handler_factory: HandlerFactory):
        jit_handler = handler_factory.create_jit_handler_with_extra_value("jit")
        sin_handler = handler_factory.create_primitive_handler("sin")
        interpreter = create_test_interpreter(
            {"jit": jit_handler, jax.lax.sin_p: sin_handler}
        )
        from kups.core.interpreter.interpreter import reinterpret

        def f(x):
            @jax.jit
            def inner(y):
                return jnp.sin(y)

            return inner(x)

        f_rt = reinterpret(f, interpreter)
        ctx = MockContext((), None, 0, ())
        _, ctx2 = f_rt(ctx, jnp.array(1.0))
        assert {"jit", "sin"}.issubset(set(ctx2.metadata))

    def test_scan_threads_carry_and_values(self, handler_factory: HandlerFactory):
        scan_handler = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_handler})
        from kups.core.interpreter.interpreter import reinterpret

        def f(x):
            def body(carry, xi):
                return carry + 1, xi * 2

            return jax.lax.scan(body, 0, jnp.arange(5))

        f_rt = reinterpret(f, interpreter)
        ctx = MockContext((), None, 0, ())
        (carry, out), ctx2 = f_rt(ctx, jnp.array(1.0))
        assert carry == 5 and jnp.allclose(out, jnp.array([0, 2, 4, 6, 8]))
        assert "scan" in ctx2.metadata
