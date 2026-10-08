# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Pruned integration tests: retain representative end-to-end flows."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from kups.core.interpreter.interpreter import InterpreterPolicy, reinterpret

from ._builders import HandlerFactory, MockContext, create_test_interpreter


class TestIntegration:
    def test_nested_jit_scan(self, handler_factory: HandlerFactory):
        sin_handler = handler_factory.create_primitive_handler("sin")
        jit_handler = handler_factory.create_jit_handler_with_extra_value("jit")
        scan_handler = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter(
            {jax.lax.sin_p: sin_handler, "jit": jit_handler, "scan": scan_handler}
        )

        def fn(x):
            @jax.jit
            def inner(y):
                return jax.lax.scan(
                    lambda c, xi: (c + 1, jnp.sin(xi)), 0, jnp.arange(4)
                )

            return inner(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, outs), ctx1 = wrapped(ctx0, jnp.array(1.0))
        assert carry == 4 and outs.shape[0] == 4
        assert {"jit", "scan", "sin"}.issubset(set(ctx1.metadata))

    def test_jit_while_and_warn_fallback(self, handler_factory: HandlerFactory):
        jit_handler = handler_factory.create_jit_handler_with_extra_value("jit")
        while_handler = handler_factory.create_while_handler("while")
        # Omit scan handler to exercise WARN fallback
        interpreter = create_test_interpreter(
            {"jit": jit_handler, "while": while_handler}, policy=InterpreterPolicy.WARN
        )

        def fn(x):
            @jax.jit
            def inner(y):
                def cond(state):
                    a, _ = state
                    return a < 3

                def body(state):
                    a, b = state
                    return (a + 1, b + 1)

                res = jax.lax.while_loop(cond, body, (0, 0))
                # Unhandled scan triggers warning
                _ = jax.lax.scan(lambda c, xi: (c + xi, c), 0, jnp.arange(3))
                return res

            return inner(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        with pytest.warns(UserWarning):
            (a, b), ctx1 = wrapped(ctx0, jnp.array(2.0))
        assert a == 3 and b == 3
        assert (
            "while" in ctx1.metadata
            and "jit" in ctx1.metadata
            and "scan" not in ctx1.metadata
        )

    def test_jit_with_static_argnums(self, handler_factory: HandlerFactory):
        add_handler = handler_factory.create_primitive_handler("add")
        mul_handler = handler_factory.create_primitive_handler("mul")
        jit_handler = handler_factory.create_jit_handler_with_extra_value("jit")
        interpreter = create_test_interpreter(
            {jax.lax.add_p: add_handler, jax.lax.mul_p: mul_handler, "jit": jit_handler}
        )

        def fn(x, scale):
            @partial(jax.jit, static_argnums=(1,))
            def inner(y, static_val):
                # static_val is treated as a static argument
                return y * jnp.arange(static_val).sum() + jnp.array(2.0)

            return inner(x, scale)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        result, ctx1 = wrapped(ctx0, jnp.array(3.0), 5)  # scale=5 as static arg

        expected = 3.0 * (1 + 2 + 3 + 4) + 2.0  # 17.0
        assert jnp.allclose(result, expected)
        assert {"jit", "mul", "add"}.issubset(set(ctx1.metadata))
