# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``default_shard_map_handler``, including the ``reduce_ctx``
injection point that lets handlers aggregate per-shard context entries
into a globally-replicated value at the ``shard_map`` boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from kups.core.interpreter.handlers import default_shard_map_handler
from kups.core.interpreter.interpreter import (
    HandlerResult,
    Interpreter,
    JaxprEqn,
    TracerValue,
    reinterpret,
)

from ._builders import MockContext, create_test_interpreter


def _mesh() -> Mesh:
    devs = jax.devices()
    if len(devs) < 4:
        pytest.skip(f"shard_map tests need >=4 devices, got {len(devs)}")
    return Mesh(devs[:4], axis_names=("data",))


def _make_capture_handler():
    """Primitive handler that captures invals[0] into ctx.value."""

    def capture(
        _: Interpreter[MockContext],
        ctx: MockContext,
        eqn: JaxprEqn,
        invals: list[TracerValue],
    ) -> HandlerResult[MockContext]:
        ctx = ctx.add_meta("captured")
        ctx = ctx.add_value(invals[0])
        # Propagate the primitive's actual computation.
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return HandlerResult(ctx, outvals)

    return capture


class TestShardMapHandler:
    def test_default_path_replicated_body(self):
        """Default behavior unchanged: body that produces no per-shard ctx data
        flows through with default ``out_specs=P()``."""
        mesh = _mesh()
        sin_h = _make_capture_handler()

        def shard_map_h(interp, ctx, eqn, invals):
            return default_shard_map_handler(interp, ctx, eqn, invals)

        interpreter = create_test_interpreter(
            {jax.lax.sin_p: sin_h, "shard_map": shard_map_h}
        )

        def fn(x):
            # Body uses a per-shard input, but ctx only captures... the input
            # itself, which has VMA={data}. Without reduce_ctx, this fails.
            return jax.shard_map(
                jnp.sin, mesh=mesh, in_specs=P("data"), out_specs=P("data")
            )(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), ())

        # No reduce_ctx → the captured per-shard value violates out_specs=P().
        with pytest.raises(Exception):
            wrapped(ctx0, jnp.ones(4))

    def test_reduce_ctx_aggregates_per_shard(self):
        """``reduce_ctx`` runs inside the shard_map body, where collectives
        work, and combines per-shard ctx entries into a globally-replicated
        value satisfying the default ``out_specs=P()``."""
        mesh = _mesh()
        sin_h = _make_capture_handler()

        def reduce_ctx(ctx: MockContext) -> MockContext:
            # psum across the manual axis: output is replicated (VMA={}).
            summed = tuple(jax.lax.psum(v, axis_name="data") for v in ctx.value)
            return MockContext(ctx.metadata, summed)

        def shard_map_h(interp, ctx, eqn, invals):
            return default_shard_map_handler(
                interp, ctx, eqn, invals, reduce_ctx=reduce_ctx
            )

        interpreter = create_test_interpreter(
            {jax.lax.sin_p: sin_h, "shard_map": shard_map_h}
        )

        def fn(x):
            return jax.shard_map(
                jnp.sin, mesh=mesh, in_specs=P("data"), out_specs=P("data")
            )(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), ())
        x = jnp.arange(4, dtype=jnp.float32)
        out, ctx1 = wrapped(ctx0, x)

        # Body output passes through unchanged (still per-shard).
        assert jnp.allclose(out, jnp.sin(x))
        # ctx carries the cross-shard sum (replicated): 0 + 1 + 2 + 3 = 6.
        assert len(ctx1.value) == 1
        assert ctx1.value[0].shape == (1,)
        assert float(ctx1.value[0].squeeze()) == 6.0

    def test_reduce_ctx_with_axis_dependent_body(self):
        """Body uses ``axis_index`` (manual-axis-bound op) — works because
        the per-shard data only flows through ``reduce_ctx`` (inside the
        shard_map body) and the trace path isn't taken."""
        mesh = _mesh()

        def capture_axis_index(
            _: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            outvals = eqn.primitive.bind(**eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            ctx = ctx.add_value(outvals[0])
            return HandlerResult(ctx, outvals)

        def reduce_ctx(ctx: MockContext) -> MockContext:
            # pmax across manual axis: replicated max of per-shard axis_index.
            reduced = tuple(jax.lax.pmax(v, axis_name="data") for v in ctx.value)
            return MockContext(ctx.metadata, reduced)

        def shard_map_h(interp, ctx, eqn, invals):
            return default_shard_map_handler(
                interp, ctx, eqn, invals, reduce_ctx=reduce_ctx
            )

        interpreter = create_test_interpreter(
            {"axis_index": capture_axis_index, "shard_map": shard_map_h}
        )

        def fn(x):
            def body(x):
                idx = jax.lax.axis_index("data")
                return x + idx.astype(x.dtype)

            return jax.shard_map(
                body, mesh=mesh, in_specs=P("data"), out_specs=P("data")
            )(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), ())
        out, ctx1 = wrapped(ctx0, jnp.zeros(4, dtype=jnp.float32))

        # Each shard's axis index added to its element.
        assert jnp.array_equal(out, jnp.arange(4, dtype=jnp.float32))
        # Cross-shard max of axis_index = 3.
        assert len(ctx1.value) == 1
        assert int(ctx1.value[0]) == 3

    def test_reduce_ctx_default_none_passes_through(self):
        """When ``reduce_ctx`` is ``None``, ctx is returned unchanged from the
        body — backward compatible with pre-PR behavior."""
        mesh = _mesh()

        def shard_map_h(interp, ctx, eqn, invals):
            return default_shard_map_handler(interp, ctx, eqn, invals)

        interpreter = create_test_interpreter({"shard_map": shard_map_h})

        def fn(x):
            return jax.shard_map(
                lambda v: v + 1.0,
                mesh=mesh,
                in_specs=P("data"),
                out_specs=P("data"),
            )(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext(("seed",), ())
        out, ctx1 = wrapped(ctx0, jnp.zeros(4, dtype=jnp.float32))

        assert jnp.array_equal(out, jnp.ones(4, dtype=jnp.float32))
        # Metadata preserved; no values added.
        assert ctx1.metadata == ("seed",)
        assert ctx1.value == ()
