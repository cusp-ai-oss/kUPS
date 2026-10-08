# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Minimal utility & structural tests (license retained)."""

from __future__ import annotations

import jax.numpy as jnp

from kups.core.interpreter._compat import get_bind_params
from kups.core.interpreter.util import split_sequence


class TestUtilities:
    def test_split_sequence_basic_and_multi(self):
        seq = tuple(range(6))
        first, rest = split_sequence(seq, (2,))
        assert first == (0, 1) and rest == (2, 3, 4, 5)
        a, b, c = split_sequence(seq, (2, 2))
        assert a == (0, 1) and b == (2, 3) and c == (4, 5)

    def test_split_sequence_arrays(self):
        arrs = [jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([5, 6])]
        (p1,), (p2, p3) = split_sequence(arrs, (1,))
        assert jnp.allclose(p1, jnp.array([1, 2]))
        assert jnp.allclose(p2, jnp.array([3, 4])) and jnp.allclose(
            p3, jnp.array([5, 6])
        )


class TestGetBindParams:
    def test_returns_subfuns_and_params(self):
        """get_bind_params returns (subfuns, bind_params) for any JAX version."""
        import jax

        def inc(x: jax.Array) -> jax.Array:
            return x + 1

        jaxpr = jax.make_jaxpr(inc)(jnp.array(1.0))
        eqn = jaxpr.jaxpr.eqns[0]
        subfuns, params = get_bind_params(eqn)
        assert isinstance(subfuns, list)
        assert isinstance(params, dict)

    def test_roundtrip_via_bind(self):
        """subfuns + bind_params can be forwarded to Primitive.bind correctly."""
        import jax

        def inc(x: jax.Array) -> jax.Array:
            return x + 1

        jaxpr = jax.make_jaxpr(inc)(jnp.array(1.0))
        eqn = jaxpr.jaxpr.eqns[0]
        subfuns, params = get_bind_params(eqn)
        result = eqn.primitive.bind(*subfuns, jnp.array(2.0), jnp.array(1.0), **params)
        assert jnp.allclose(result, jnp.array(3.0))

    def test_higher_order_primitive(self):
        """get_bind_params handles higher-order primitives (e.g. jit/pjit)."""
        import jax

        @jax.jit
        def f(x):
            return x * 2

        jaxpr = jax.make_jaxpr(f)(jnp.array(1.0))
        for eqn in jaxpr.jaxpr.eqns:
            subfuns, params = get_bind_params(eqn)
            assert isinstance(subfuns, list)
            assert isinstance(params, dict)


class TestHandlerStructures:
    def test_handler_result_annotations(self):
        from kups.core.interpreter.interpreter import HandlerResult

        assert set(getattr(HandlerResult, "__annotations__", {})) == {"ctx", "outvals"}

    def test_tracer_value_typing(self):
        from kups.core.interpreter.interpreter import TracerValue

        x: TracerValue = jnp.array([1.0])
        assert jnp.allclose(x, jnp.array([1.0]))
        y: TracerValue = 7
        assert y == 7
