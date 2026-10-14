# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Pruned interpreter tests: retain only high-signal behaviors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from kups.core.effects.interpreter import (
    Interpreter,
    InterpreterPolicy,
    reinterpret,
)

from ._builders import HandlerFactory, MockContext, create_test_interpreter


def _sin_inside_scan(x: jax.Array) -> jax.Array:
    return jax.lax.scan(lambda c, xi: (c + jnp.sin(xi), None), x, jnp.arange(2.0))[0]


class TestInterpreter:
    def test_defaults(self):
        interpreter = Interpreter({"sin": lambda *_: None}, label="x")
        assert interpreter.policy == InterpreterPolicy.RAISE
        assert interpreter.label == "x"
        assert set(interpreter.handlers) == {"sin"}

    def test_keys_accept_primitives_and_names(self):
        interpreter = Interpreter(
            {jax.lax.sin_p: lambda *_: None}, effects=(jax.lax.cos_p, "exp")
        )
        assert set(interpreter.handlers) == {"sin"}
        assert interpreter.effects == {"cos", "exp"}

    def test_unhandled_primitive_without_effects_is_bound(self):
        interpreter = create_test_interpreter({}, policy=InterpreterPolicy.RAISE)
        out, _ = reinterpret(_sin_inside_scan, interpreter)(
            MockContext((), ()), jnp.array(0.0)
        )
        assert jnp.allclose(out, jnp.sin(0.0) + jnp.sin(1.0))

    def test_effect_inside_unhandled_primitive_raises(self):
        interpreter = create_test_interpreter(
            {}, policy=InterpreterPolicy.RAISE, effects=frozenset({"sin"})
        )
        with pytest.raises(NotImplementedError, match="scan"):
            reinterpret(_sin_inside_scan, interpreter)(
                MockContext((), ()), jnp.array(0.0)
            )

    def test_handler_exceptions_are_not_swallowed(self):
        def failing(*_):
            raise NotImplementedError("from handler")

        interpreter = create_test_interpreter(
            {"sin": failing}, policy=InterpreterPolicy.IGNORE
        )
        with pytest.raises(NotImplementedError, match="from handler"):
            reinterpret(jnp.sin, interpreter)(MockContext((), ()), jnp.array(0.0))


class TestContextThreading:
    def test_jit_threads_metadata(self, handler_factory: HandlerFactory):
        jit_handler = handler_factory.create_jit_handler_with_extra_value("jit")
        sin_handler = handler_factory.create_primitive_handler("sin")
        interpreter = create_test_interpreter(
            {"jit": jit_handler, jax.lax.sin_p: sin_handler}
        )

        def f(x):
            @jax.jit
            def inner(y):
                return jnp.sin(y)

            return inner(x)

        f_rt = reinterpret(f, interpreter)
        ctx = MockContext((), ())
        _, ctx2 = f_rt(ctx, jnp.array(1.0))
        assert {"jit", "sin"}.issubset(set(ctx2.metadata))

    def test_scan_threads_carry_and_values(self, handler_factory: HandlerFactory):
        scan_handler = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_handler})

        def f(x):
            def body(carry, xi):
                return carry + 1, xi * 2

            return jax.lax.scan(body, 0, jnp.arange(5))

        f_rt = reinterpret(f, interpreter)
        ctx = MockContext((), ())
        (carry, out), ctx2 = f_rt(ctx, jnp.array(1.0))
        assert carry == 5 and jnp.allclose(out, jnp.array([0, 2, 4, 6, 8]))
        assert "scan" in ctx2.metadata
