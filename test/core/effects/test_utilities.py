# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Minimal utility & structural tests (license retained)."""

from __future__ import annotations

import jax.numpy as jnp

from kups.core.utils._jax_compat import get_bind_params


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
        from kups.core.effects.interpreter import HandlerResult

        assert set(getattr(HandlerResult, "__annotations__", {})) == {"ctx", "outvals"}

    def test_tracer_value_typing(self):
        from kups.core.effects.interpreter import TracerValue

        x: TracerValue = jnp.array([1.0])
        assert jnp.allclose(x, jnp.array([1.0]))
        y: TracerValue = 7
        assert y == 7
