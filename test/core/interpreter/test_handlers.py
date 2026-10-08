# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Minimal focused handler tests.

Copyright 2025 CuspAI
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import checkpoint

from kups.core.interpreter.handlers import (
    ScanSemantics,
    default_scan_handler,
    default_while_handler,
)
from kups.core.interpreter.interpreter import (
    HandlerResult,
    Interpreter,
    InterpreterPolicy,
    JaxprEqn,
    TracerValue,
    reinterpret,
)

from ._builders import HandlerFactory, MockContext, create_test_interpreter


class TestPrimitiveAndJit:
    def test_jit_with_primitive(self, handler_factory: HandlerFactory):
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        jit_h = handler_factory.create_jit_handler_with_extra_value("jit")
        interpreter = create_test_interpreter({jax.lax.sin_p: sin_h, "jit": jit_h})

        def fn(x: jax.Array):
            @jax.jit
            def inner(v):
                return jnp.sin(v) + 2.0

            return inner(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(1.0))
        assert jnp.allclose(out, jnp.sin(1.0) + 2.0)
        assert {"jit", "sin"}.issubset(ctx1.metadata)
        assert ctx1.total_value_count == 2


class TestScan:
    def test_scan_carry_threading(self, handler_factory: HandlerFactory):
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_h})

        def fn(_: jax.Array):
            def body(carry, x):
                return carry + 1, x * 2

            return jax.lax.scan(body, 0, jnp.arange(3))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, ys), ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert carry == 3
        assert jnp.allclose(ys, jnp.array([0, 2, 4]))
        assert "scan" in ctx1.metadata and ctx1.total_value_count == 1

    def test_scan_result_threading(self, handler_factory: HandlerFactory):
        scan_h = handler_factory.create_scan_handler_with_mode(
            "scan", constant_context_threading=True
        )
        interpreter = create_test_interpreter({"scan": scan_h})

        def fn(_: jax.Array):
            def body(carry, x):
                return carry + 1, x * 2

            return jax.lax.scan(body, 0, jnp.arange(4))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, ys), ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert carry == 4
        assert jnp.allclose(ys, jnp.array([0, 2, 4, 6]))
        assert "scan" in ctx1.metadata and ctx1.total_value_count == 1

    def test_scan_with_nested_primitive(self, handler_factory: HandlerFactory):
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        scan_h = handler_factory.create_scan_handler("scan")
        inner = create_test_interpreter({"scan": scan_h, jax.lax.sin_p: sin_h})

        def fn(_: jax.Array):
            def body(carry, x):
                v = jnp.sin(x)  # adds context via primitive handler
                return carry + v, x * 2

            return jax.lax.scan(body, 0.0, jnp.arange(3, dtype=float))

        wrapped = reinterpret(fn, inner)
        ctx0 = MockContext((), None, 0, ())
        (_, ys), ctx1 = wrapped(ctx0, jnp.array(0.0))
        # Both handlers should have contributed once
        assert {"scan", "sin"}.issubset(ctx1.metadata)
        assert jnp.allclose(ys, jnp.array([0.0, 2.0, 4.0]))

    def test_scan_carry_only_no_xs(self, handler_factory: HandlerFactory):
        """Carry-only scan (no xs, explicit length) must not crash. Fixes #42."""
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_h})

        def fn(_: jax.Array):
            def body(carry, _xs):
                return carry + 1, None

            return jax.lax.scan(body, 0, length=5)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, _), ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert carry == 5
        assert "scan" in ctx1.metadata

    def test_scan_carry_only_result_threading(self, handler_factory: HandlerFactory):
        """Carry-only scan with RESULT threading must forward length. Fixes #42."""
        scan_h = handler_factory.create_scan_handler_with_mode(
            "scan", constant_context_threading=True
        )
        interpreter = create_test_interpreter({"scan": scan_h})

        def fn(_: jax.Array):
            def body(carry, _xs):
                return carry + 1, None

            return jax.lax.scan(body, 0, length=5)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, _), ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert carry == 5
        assert "scan" in ctx1.metadata

    def test_scan_result_threading_no_parent_stacking(
        self, handler_factory: HandlerFactory
    ):
        """Result threading must NOT stack parent values — only child values.

        The parent is constant across scan iterations. Stacking it produces
        N redundant copies and causes shape mismatches when nested inside
        while_loop. After pop, parent values should retain their original
        (unstacked) shape, while child values are stacked [N, ...].
        """
        sin_h = handler_factory.create_primitive_handler_with_extra_value(
            "sin", value_fn=lambda invals: invals[0]
        )
        scan_h = handler_factory.create_scan_handler_with_mode(
            "scan", constant_context_threading=True
        )
        interpreter = create_test_interpreter({"scan": scan_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            pre = jnp.sin(x)  # pre-call: scalar value

            def body(carry, elem):
                return carry + jnp.sin(elem), elem

            return jax.lax.scan(body, pre, jnp.arange(4.0))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        x_in = jnp.array(1.0)
        (_, _), ctx1 = wrapped(ctx0, x_in)

        # Pre-call sin value (from parent) should be scalar, not stacked [4]
        has_scalar_precall = any(
            jnp.allclose(v, x_in) and v.shape == () for v in ctx1.value
        )
        assert has_scalar_precall, (
            f"Pre-call value should be scalar {x_in}, got shapes: "
            f"{[v.shape for v in ctx1.value]}"
        )

        # Scan handler's own value should also be scalar (added before push)
        # Child sin values from body should be stacked [4, ...]
        has_stacked_child = any(v.shape[0:1] == (4,) for v in ctx1.value)
        assert has_stacked_child, (
            f"Child values should be stacked [4, ...], got shapes: "
            f"{[v.shape for v in ctx1.value]}"
        )

    def test_scan_result_threading_reverse_unroll(
        self, handler_factory: HandlerFactory
    ):
        """RESULT threading must also forward reverse/unroll args."""
        scan_h = handler_factory.create_scan_handler_with_mode(
            "scan", constant_context_threading=True
        )
        interpreter = create_test_interpreter({"scan": scan_h})

        def fn(_: jax.Array):
            def body(carry, x):
                return carry + x, x * 2

            return jax.lax.scan(body, 0, jnp.arange(4), reverse=True, unroll=2)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, ys), _ = wrapped(ctx0, jnp.array(0.0))
        assert carry == 6
        assert jnp.allclose(ys, jnp.array([0, 2, 4, 6]))

    def test_scan_args_reverse_unroll(self, handler_factory: HandlerFactory):
        """Test that scan args (reverse, unroll) are passed through correctly."""
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_h})

        def fn(_: jax.Array):
            def body(carry, x):
                return carry + x, x * 2

            return jax.lax.scan(body, 0, jnp.arange(4), reverse=True, unroll=2)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, ys), ctx1 = wrapped(ctx0, jnp.array(0.0))
        # With reverse=True, scan processes [3,2,1,0] but outputs in original order
        assert carry == 6  # 0+1+2+3
        assert jnp.allclose(ys, jnp.array([0, 2, 4, 6]))
        assert "scan" in ctx1.metadata

    @pytest.mark.parametrize("threading", list(ScanSemantics))
    def test_scan_partitions_consts_carry_and_results(
        self, handler_factory: HandlerFactory, threading: ScanSemantics
    ):
        """Scan schema partitions closed constants, carry, xs, and results."""
        if threading == ScanSemantics.CARRY:
            scan_h = handler_factory.create_scan_handler("scan")
        else:
            scan_h = handler_factory.create_scan_handler_with_mode(
                "scan", constant_context_threading=True
            )
        interpreter = create_test_interpreter({"scan": scan_h})
        offsets = jnp.array([10, 20, 30], dtype=jnp.int32)
        biases = jnp.array([5, 100], dtype=jnp.int32)

        def fn(values: jax.Array):
            def body(carry, inputs):
                total, count = carry
                value, offset = inputs
                adjusted = value + offset + biases[0]
                return (total + adjusted, count + 1), (adjusted, total)

            return jax.lax.scan(
                body,
                (jnp.int32(0), jnp.int32(0)),
                (values, offsets),
            )

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        ((total, count), (adjusted, prior_totals)), ctx1 = wrapped(
            ctx0, jnp.array([1, 2, 3], dtype=jnp.int32)
        )

        assert total == 81
        assert count == 3
        assert jnp.array_equal(adjusted, jnp.array([16, 27, 38]))
        assert jnp.array_equal(prior_totals, jnp.array([0, 16, 43]))
        assert "scan" in ctx1.metadata


class TestWhile:
    def test_basic_while(self, handler_factory: HandlerFactory):
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h})

        def fn(_: jax.Array):
            def cond(a):
                return a < 5

            def body(a):
                return a + 1

            return jax.lax.while_loop(cond, body, 0)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert out == 5
        assert "while" in ctx1.metadata and ctx1.total_value_count == 1


class TestWhileStress:
    """Adversarial stress tests for while_loop handler (#40)."""

    def test_while_multiple_carry_values(self, handler_factory: HandlerFactory):
        """While loop with multiple carry values must preserve all correctly."""
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h})

        def fn(_: jax.Array):
            def cond(args):
                i, _, _ = args
                return i < 4

            def body(args):
                i, total, product = args
                return i + 1, total + i, product * (i + 1)

            return jax.lax.while_loop(cond, body, (0, 0, 1))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (i, total, product), ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert i == 4
        assert total == 6  # 0+1+2+3
        assert product == 24  # 1*1*2*3*4
        assert "while" in ctx1.metadata

    def test_while_with_handled_primitive_in_body(
        self, handler_factory: HandlerFactory
    ):
        """While body containing a handled primitive accumulates context per iteration."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h, jax.lax.sin_p: sin_h})

        def fn(_: jax.Array):
            def cond(a):
                return a < 3.0

            def body(a):
                return a + jnp.sin(jnp.array(0.0)) + 1.0

            return jax.lax.while_loop(cond, body, 0.0)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert out == 3.0
        assert {"while", "sin"}.issubset(ctx1.metadata)

    def test_while_zero_iterations(self, handler_factory: HandlerFactory):
        """While loop that exits immediately (cond false from start)."""
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h})

        def fn(_: jax.Array):
            def cond(a):
                return a < 0  # always false

            def body(a):
                return a + 1

            return jax.lax.while_loop(cond, body, 5)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(0.0))
        assert out == 5
        assert "while" in ctx1.metadata

    def test_while_with_array_carry(self, handler_factory: HandlerFactory):
        """While loop carrying an array, not just a scalar."""
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h})

        def fn(_: jax.Array):
            def cond(args):
                i, _ = args
                return i < 3

            def body(args):
                i, arr = args
                return i + 1, arr + 1.0

            return jax.lax.while_loop(cond, body, (0, jnp.zeros(4)))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (i, arr), _ = wrapped(ctx0, jnp.array(0.0))
        assert i == 3
        assert jnp.allclose(arr, jnp.full(4, 3.0))


class TestNestingAdversarial:
    """Deep adversarial tests for context accumulation in nested control flow (#40).

    Every test checks EXACT metadata counts and value counts to detect
    any context duplication, leakage, or loss across nesting boundaries.
    """

    def test_while_in_scan_exact_context_counts(self, handler_factory: HandlerFactory):
        """While inside scan: verify exact metadata/value counts after all iterations.

        Expected: scan handler adds 1 meta + 1 value (before push).
        While handler adds 1 meta + 1 value (before push).
        Sin handler adds 1 meta + 1 value per body equation.
        With default updater (replace), only last iteration of each loop survives.
        After pops merge up: scan_meta + while_meta + sin_meta = 3 entries.
        """
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        wh_h = handler_factory.create_while_handler("while")
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter(
            {"scan": scan_h, "while": wh_h, jax.lax.sin_p: sin_h}
        )

        def fn(_: jax.Array):
            def scan_body(carry, x):
                def w_cond(a):
                    return a < 3.0

                def w_body(a):
                    return a + jnp.sin(jnp.array(0.0)) + 1.0

                result = jax.lax.while_loop(w_cond, w_body, 0.0)
                return carry + result, x

            return jax.lax.scan(scan_body, 0.0, jnp.arange(4.0))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, _ys), ctx1 = wrapped(ctx0, jnp.array(0.0))

        # Each scan iteration: while runs 3 times, result=3.0. carry accumulates.
        assert carry == 12.0  # 4 * 3.0
        # Metadata: "scan" (scan handler) + "while" (while handler) + "sin" (primitive)
        assert ctx1.metadata.count("scan") == 1
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("sin") == 1
        assert len(ctx1.metadata) == 3
        # Values: one from each handler
        assert ctx1.total_value_count == 3

    def test_scan_in_while_exact_context_counts(self, handler_factory: HandlerFactory):
        """Scan inside while: verify no context duplication across while iterations.

        The while body runs N iterations. Each contains a scan.
        With default updater (replace), only the LAST while iteration's context survives.
        That last iteration's scan adds its own meta/value, runs its body.
        """
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        wh_h = handler_factory.create_while_handler("while")
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter(
            {"scan": scan_h, "while": wh_h, jax.lax.sin_p: sin_h}
        )

        def fn(_: jax.Array):
            def w_cond(state):
                i, _ = state
                return i < 3

            def w_body(state):
                i, total = state

                def scan_body(carry, x):
                    return carry + jnp.sin(x), x

                final_carry, _ = jax.lax.scan(scan_body, 0.0, jnp.ones(2))
                return i + 1, total + final_carry

            return jax.lax.while_loop(w_cond, w_body, (0, 0.0))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (i, total), ctx1 = wrapped(ctx0, jnp.array(0.0))

        assert i == 3
        # Each iteration: scan(sin(1.0), sin(1.0)) → carry ≈ 2*sin(1)
        expected_total = 3 * 2 * jnp.sin(1.0)
        assert jnp.allclose(total, expected_total)

        # Context: "while" + "scan" + "sin" — one each, no duplication
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("scan") == 1
        assert ctx1.metadata.count("sin") == 1
        assert len(ctx1.metadata) == 3
        assert ctx1.total_value_count == 3

    def test_while_in_while_no_context_leakage(self, handler_factory: HandlerFactory):
        """Nested while-in-while: inner while must not leak extra context entries.

        Outer: 2 iterations. Inner: 3 iterations each.
        Both use the "while" meta key — so we expect exactly 2 "while" entries
        (one from each handler), NOT 2*3=6 or other accumulation.
        """
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h})

        def fn(_: jax.Array):
            def outer_cond(state):
                i, _ = state
                return i < 2

            def outer_body(state):
                i, total = state

                def inner_cond(j):
                    return j < 3

                def inner_body(j):
                    return j + 1

                inner_result = jax.lax.while_loop(inner_cond, inner_body, 0)
                return i + 1, total + inner_result

            return jax.lax.while_loop(outer_cond, outer_body, (0, 0))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (i, total), ctx1 = wrapped(ctx0, jnp.array(0.0))

        assert i == 2
        assert total == 6  # 2 * 3
        # Exactly 2 "while" entries: one from outer handler, one from inner handler
        assert ctx1.metadata.count("while") == 2
        assert len(ctx1.metadata) == 2
        assert ctx1.total_value_count == 2

    def test_while_with_accumulating_updater(self, handler_factory: HandlerFactory):
        """Custom updater that SUMS values across iterations.

        Body runs N iterations, each adding a value V via handled sin.
        Updater sums old + new values. After N iterations, value should be N*V.
        """
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        scan_h = handler_factory.create_scan_handler_with_updater(
            "scan",
            updater=lambda old_ctx, new_ctx: MockContext(
                new_ctx.metadata,
                new_ctx.parent,
                new_ctx.level,
                tuple(o + n for o, n in zip(old_ctx.value, new_ctx.value)),
            ),
        )
        interpreter = create_test_interpreter({"scan": scan_h, jax.lax.sin_p: sin_h})

        def fn(_: jax.Array):
            def body(carry, x):
                return carry + jnp.sin(x), x

            return jax.lax.scan(body, 0.0, jnp.ones(5))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, _), ctx1 = wrapped(ctx0, jnp.array(0.0))

        assert jnp.allclose(carry, 5 * jnp.sin(1.0))
        # Metadata: "scan" + "sin" (replace semantics for metadata in updater)
        assert ctx1.metadata.count("scan") == 1
        assert ctx1.metadata.count("sin") == 1
        # Values: scan handler adds 1 value, sin handler adds 1 value.
        # The sin value was accumulated 5 times via the summing updater.
        # After pop, scan_value + accumulated_sin_value = 2 values total.
        assert ctx1.total_value_count == 2
        # The accumulated sin value should be ~5x the per-iteration value
        sin_values = [v for v in ctx1.value if jnp.allclose(v, 5 * jnp.ones(1))]
        assert len(sin_values) > 0, f"Expected accumulated sin value, got {ctx1.value}"

    def test_while_body_multiple_handled_primitives(
        self, handler_factory: HandlerFactory
    ):
        """While body with TWO different handled primitives: both must contribute."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        cos_h = handler_factory.create_primitive_handler_with_extra_value("cos")
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter(
            {"while": wh_h, jax.lax.sin_p: sin_h, jax.lax.cos_p: cos_h}
        )

        def fn(_: jax.Array):
            def cond(a):
                return a < 3.0

            def body(a):
                return a + jnp.sin(jnp.array(0.0)) + jnp.cos(jnp.array(0.0))

            return jax.lax.while_loop(cond, body, 0.0)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(0.0))

        # sin(0)+cos(0) = 0+1 = 1 per iteration, 3 iterations → 3.0
        assert out == 3.0
        # while + sin + cos: one entry each
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("sin") == 1
        assert ctx1.metadata.count("cos") == 1
        assert len(ctx1.metadata) == 3
        assert ctx1.total_value_count == 3

    def test_checkpoint_in_while_context_threading(
        self, handler_factory: HandlerFactory
    ):
        """Checkpoint inside while body: context must thread through both."""
        ckpt_h = handler_factory.create_checkpoint_handler("checkpoint")
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter(
            {"while": wh_h, "remat2": ckpt_h, jax.lax.sin_p: sin_h}
        )

        def fn(_: jax.Array):
            def cond(a):
                return a < 2.0

            def body(a):
                @checkpoint
                def inner(v):
                    return v + jnp.sin(jnp.array(0.0)) + 1.0

                return inner(a)

            return jax.lax.while_loop(cond, body, 0.0)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(0.0))

        assert out == 2.0
        # while + checkpoint + sin: one entry each
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("checkpoint") == 1
        assert ctx1.metadata.count("sin") == 1
        assert len(ctx1.metadata) == 3
        assert ctx1.total_value_count == 3

    def test_result_threaded_scan_in_while(self, handler_factory: HandlerFactory):
        """RESULT-threaded scan inside while: stacked context values are carried correctly.

        Result threading stacks ctx values along scan axis [N, ...].
        The while handler's updater must carry these stacked values across iterations.
        """
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        scan_h = handler_factory.create_scan_handler_with_mode(
            "scan", constant_context_threading=True
        )
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter(
            {"while": wh_h, "scan": scan_h, jax.lax.sin_p: sin_h}
        )

        def fn(_: jax.Array):
            def w_cond(state):
                i, _ = state
                return i < 2

            def w_body(state):
                i, total = state

                def scan_body(carry, x):
                    return carry + jnp.sin(x), x

                final_carry, _ = jax.lax.scan(scan_body, 0.0, jnp.ones(3))
                return i + 1, total + final_carry

            return jax.lax.while_loop(w_cond, w_body, (0, 0.0))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (i, total), ctx1 = wrapped(ctx0, jnp.array(0.0))

        assert i == 2
        expected = 2 * 3 * jnp.sin(1.0)
        assert jnp.allclose(total, expected)
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("scan") == 1
        assert ctx1.metadata.count("sin") == 1
        assert len(ctx1.metadata) == 3

    def test_triple_nesting_scan_while_cond(self, handler_factory: HandlerFactory):
        """3-level nesting: scan → while → cond with CONSISTENT branches.

        Both cond branches use sin (same handler → same context shape).
        This is the deepest nesting: scan body → while body → cond branches.
        """
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        wh_h = handler_factory.create_while_handler("while")
        scan_h = handler_factory.create_scan_handler("scan")
        cond_h = handler_factory.create_cond_handler("cond")
        interpreter = create_test_interpreter(
            {
                "scan": scan_h,
                "while": wh_h,
                "cond": cond_h,
                jax.lax.sin_p: sin_h,
            }
        )

        def fn(_: jax.Array):
            def scan_body(carry, x):
                def w_cond(a):
                    return a < 2.0

                def w_body(a):
                    # cond inside while inside scan — both branches use sin
                    def branch(v: jax.Array) -> jax.Array:
                        return v + jnp.sin(jnp.array(0.0)) + 1.0

                    return jax.lax.cond(a < 1.0, branch, branch, a)

                result = jax.lax.while_loop(w_cond, w_body, 0.0)
                return carry + result, x

            return jax.lax.scan(scan_body, 0.0, jnp.arange(3.0))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, _), ctx1 = wrapped(ctx0, jnp.array(0.0))

        # while body: each iteration adds sin(0)+1 = 1.0, runs 2 iterations → 2.0
        assert jnp.allclose(carry, 6.0)  # 3 * 2.0
        # 4 handlers contributed: scan, while, cond, sin — one entry each
        assert ctx1.metadata.count("scan") == 1
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("cond") == 1
        assert ctx1.metadata.count("sin") == 1
        assert len(ctx1.metadata) == 4
        assert ctx1.total_value_count == 4

    def test_jit_in_while_context_threading(self, handler_factory: HandlerFactory):
        """JIT inside while body: context threads through jit boundary."""
        jit_h = handler_factory.create_jit_handler_with_extra_value("jit")
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter(
            {"while": wh_h, "jit": jit_h, jax.lax.sin_p: sin_h}
        )

        def fn(_: jax.Array):
            def cond(a):
                return a < 2.0

            def body(a):
                @jax.jit
                def inner(v):
                    return v + jnp.sin(jnp.array(0.0)) + 1.0

                return inner(a)

            return jax.lax.while_loop(cond, body, 0.0)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(0.0))

        assert out == 2.0
        # while + jit + sin: one entry each
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("jit") == 1
        assert ctx1.metadata.count("sin") == 1
        assert len(ctx1.metadata) == 3
        assert ctx1.total_value_count == 3


class TestInitializerCorruption:
    """Reproduce #40: dry-run trace + initializer corrupt pre-existing context.

    Both `_scan_handler_carry` and `default_while_handler` do a dry-run trace
    to discover the output context shape, then `jax.tree.map(Uninitialized, ...)`
    over the entire tree (including the parent chain). The standard initializer
    zeros ALL Uninitialized leaves — destroying pre-existing values in the parent.

    Bug manifests as:
    - Extra metadata entries (from the dry-run trace's handler calls)
    - Zeroed values for pre-call context entries
    - Worst with 0 iterations: initialized_ctx IS the final context

    """

    # ── while_loop ──────────────────────────────────────────────────────

    def test_while_precall_exact_counts(self, handler_factory: HandlerFactory):
        """Handled call before while_loop: expect exactly 2x primitive entries."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        cos_h = handler_factory.create_primitive_handler_with_extra_value("cos")
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter(
            {"while": wh_h, jax.lax.sin_p: sin_h, jax.lax.cos_p: cos_h}
        )

        def fn(x: jax.Array):
            # "propagator" — multiple handled primitives per call
            a = jnp.sin(x)
            b = jnp.cos(x)
            pre_result = a + b

            def cond(state):
                i, _ = state
                return i < 2

            def body(state):
                i, val = state
                # same "propagator" inside while body
                a = jnp.sin(val)
                b = jnp.cos(val)
                return i + 1, a + b

            return jax.lax.while_loop(cond, body, (jnp.int32(1), pre_result))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (i, result), ctx1 = wrapped(ctx0, jnp.array(1.0))

        assert i == 2
        # Pre-call: sin(1) + cos(1). While body: sin(sin(1)+cos(1)) + cos(sin(1)+cos(1))
        pre = jnp.sin(1.0) + jnp.cos(1.0)
        expected = jnp.sin(pre) + jnp.cos(pre)
        assert jnp.allclose(result, expected)

        # Context: "while" handler adds 1 entry.
        # Pre-call adds: sin + cos = 2 entries.
        # While body adds: sin + cos = 2 entries (from last iteration, replaces).
        # Total metadata: "while"(1) + "sin"(2) + "cos"(2) = 5
        # Total values: 1 (while) + 2 (pre-call sin+cos) + 2 (while body sin+cos) = 5
        assert ctx1.metadata.count("while") == 1
        assert ctx1.metadata.count("sin") == 2  # one from pre-call, one from while body
        assert ctx1.metadata.count("cos") == 2  # one from pre-call, one from while body
        assert len(ctx1.metadata) == 5
        assert ctx1.total_value_count == 5

    def test_while_precall_values_not_corrupted(self, handler_factory: HandlerFactory):
        """Pre-call context VALUES must survive the while handler's initializer."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value(
            "sin", value_fn=lambda invals: invals[0]
        )
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            pre = jnp.sin(x)  # pre-call: adds value = x (the input to sin)

            def cond(state):
                i, _ = state
                return i < 2

            def body(state):
                i, val = state
                return i + 1, jnp.sin(val)

            return jax.lax.while_loop(cond, body, (jnp.int32(1), pre))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        x_in = jnp.array(1.0)
        (i, result), ctx1 = wrapped(ctx0, x_in)

        assert i == 2
        # Pre-call: sin(1.0). While body (1 iteration): sin(sin(1.0))
        expected = jnp.sin(jnp.sin(1.0))
        assert jnp.allclose(result, expected)

        assert ctx1.metadata.count("sin") == 2
        assert ctx1.metadata.count("while") == 1
        has_precall_value = any(jnp.allclose(v, x_in) for v in ctx1.value)
        assert has_precall_value, (
            f"Pre-call value corrupted! Expected {x_in}, "
            f"got {[float(v) for v in ctx1.value if hasattr(v, '__float__')]}"
        )

    def test_while_zero_iterations_preserves_precall_values(
        self, handler_factory: HandlerFactory
    ):
        """0-iteration while: pre-call VALUES must be preserved.

        Known limitation: 0-iteration while_loop produces phantom child metadata
        because JAX requires fixed-structure carry. The child metadata from the
        dry-run trace is baked into the carry structure. With 0 iterations, the
        initialized carry IS the result, and pop() merges phantom metadata.

        What MUST work: parent values are NOT zeroed.
        """
        sin_h = handler_factory.create_primitive_handler_with_extra_value(
            "sin", value_fn=lambda invals: invals[0]
        )
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter({"while": wh_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            pre = jnp.sin(x)

            def cond(state):
                i, _ = state
                return i < 0  # NEVER true → 0 iterations

            def body(state):
                i, val = state
                return i + 1, jnp.sin(val)

            return jax.lax.while_loop(cond, body, (jnp.int32(1), pre))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        x_in = jnp.array(1.0)
        (i, result), ctx1 = wrapped(ctx0, x_in)

        assert i == 1
        assert jnp.allclose(result, jnp.sin(1.0))

        # CRITICAL: pre-call value must be preserved, not zeroed
        has_precall_value = any(jnp.allclose(v, x_in) for v in ctx1.value)
        assert has_precall_value, (
            f"Pre-call value corrupted by initializer! Expected {x_in}, "
            f"got {ctx1.value}"
        )

    def test_while_precall_multiple_values_all_preserved(
        self, handler_factory: HandlerFactory
    ):
        """Multiple pre-call values: ALL must survive, not just the first."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value(
            "sin", value_fn=lambda invals: invals[0]
        )
        cos_h = handler_factory.create_primitive_handler_with_extra_value(
            "cos", value_fn=lambda invals: invals[0]
        )
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter(
            {"while": wh_h, jax.lax.sin_p: sin_h, jax.lax.cos_p: cos_h}
        )

        def fn(x: jax.Array):
            a = jnp.sin(x)  # value = x stored in context
            b = jnp.cos(a)  # value = sin(x) stored in context
            pre = a + b

            def cond(state):
                i, _ = state
                return i < 2

            def body(state):
                i, val = state
                return i + 1, jnp.sin(val) + jnp.cos(val)

            return jax.lax.while_loop(cond, body, (jnp.int32(1), pre))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        x_in = jnp.array(2.0)
        (i, _), ctx1 = wrapped(ctx0, x_in)

        assert i == 2
        # Both pre-call values must be present: x_in=2.0 and sin(2.0)
        has_sin_input = any(jnp.allclose(v, x_in) for v in ctx1.value)
        has_cos_input = any(jnp.allclose(v, jnp.sin(x_in)) for v in ctx1.value)
        assert has_sin_input, f"Pre-call sin input lost: {ctx1.value}"
        assert has_cos_input, f"Pre-call cos input lost: {ctx1.value}"

    def test_while_in_scan_precall_preserved(self, handler_factory: HandlerFactory):
        """Nested while-in-scan with pre-call: parent chain preserved at depth."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value(
            "sin", value_fn=lambda invals: invals[0]
        )
        scan_h = handler_factory.create_scan_handler("scan")
        wh_h = handler_factory.create_while_handler("while")
        interpreter = create_test_interpreter(
            {"scan": scan_h, "while": wh_h, jax.lax.sin_p: sin_h}
        )

        def fn(x: jax.Array):
            pre = jnp.sin(x)  # pre-call value = x

            def scan_body(carry, elem):
                def w_cond(state):
                    i, _ = state
                    return i < 2

                def w_body(state):
                    i, val = state
                    return i + 1, jnp.sin(val)

                _, result = jax.lax.while_loop(w_cond, w_body, (jnp.int32(1), carry))
                return result, elem

            return jax.lax.scan(scan_body, pre, jnp.ones(2))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        x_in = jnp.array(1.0)
        (_, _), ctx1 = wrapped(ctx0, x_in)

        # Pre-call value x_in must survive through scan + while nesting
        has_precall = any(jnp.allclose(v, x_in) for v in ctx1.value)
        assert has_precall, (
            f"Pre-call value lost through nested scan+while: {ctx1.value}"
        )

    # ── scan (carry threading) ──────────────────────────────────────────

    def test_scan_carry_precall_exact_counts(self, handler_factory: HandlerFactory):
        """Handled call before scan (carry threading): expect 2x primitive entries."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            pre = jnp.sin(x)  # pre-call: 1 sin entry

            def body(carry, elem):
                return carry + jnp.sin(elem), elem  # body: 1 sin entry

            return jax.lax.scan(body, pre, jnp.ones(3))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        (carry, _), ctx1 = wrapped(ctx0, jnp.array(1.0))

        assert jnp.allclose(carry, jnp.sin(1.0) + 3 * jnp.sin(1.0))
        # "scan"(1) + "sin"(2: pre + body) = 3
        assert ctx1.metadata.count("scan") == 1
        assert ctx1.metadata.count("sin") == 2
        assert len(ctx1.metadata) == 3
        assert ctx1.total_value_count == 3

    def test_scan_carry_precall_values_not_corrupted(
        self, handler_factory: HandlerFactory
    ):
        """Pre-call context VALUES must survive scan carry handler's initializer."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value(
            "sin", value_fn=lambda invals: invals[0]
        )
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            pre = jnp.sin(x)  # adds value = x

            def body(carry, elem):
                return carry + jnp.sin(elem), elem

            return jax.lax.scan(body, pre, jnp.ones(3))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        x_in = jnp.array(1.0)
        (_, _), ctx1 = wrapped(ctx0, x_in)

        assert ctx1.metadata.count("sin") == 2
        assert ctx1.metadata.count("scan") == 1
        has_precall_value = any(jnp.allclose(v, x_in) for v in ctx1.value)
        assert has_precall_value, (
            f"Pre-call value corrupted by scan initializer! Expected {x_in}, "
            f"got {[float(v) for v in ctx1.value if hasattr(v, '__float__')]}"
        )

    def test_scan_carry_length1_preserves_precall(
        self, handler_factory: HandlerFactory
    ):
        """Scan with length=1: minimal iteration, pre-call must survive."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value(
            "sin", value_fn=lambda invals: invals[0]
        )
        scan_h = handler_factory.create_scan_handler("scan")
        interpreter = create_test_interpreter({"scan": scan_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            pre = jnp.sin(x)

            def body(carry, elem):
                return carry + jnp.sin(elem), elem

            return jax.lax.scan(body, pre, jnp.ones(1))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        x_in = jnp.array(2.0)
        (carry, _), ctx1 = wrapped(ctx0, x_in)

        assert jnp.allclose(carry, jnp.sin(2.0) + jnp.sin(1.0))
        assert ctx1.metadata.count("sin") == 2
        assert ctx1.metadata.count("scan") == 1
        assert len(ctx1.metadata) == 3
        has_precall_value = any(jnp.allclose(v, x_in) for v in ctx1.value)
        assert has_precall_value, (
            f"Pre-call value corrupted! Expected {x_in}, "
            f"got {[float(v) for v in ctx1.value if hasattr(v, '__float__')]}"
        )


class TestCond:
    def test_cond_basic(self, handler_factory: HandlerFactory):
        cond_h = handler_factory.create_cond_handler("cond")
        interpreter = create_test_interpreter({"cond": cond_h})

        def fn(x: jax.Array):
            def true_fn(v: jax.Array) -> jax.Array:
                return v + 1

            def false_fn(v: jax.Array) -> jax.Array:
                return v - 1

            return jax.lax.cond(x > 0, true_fn, false_fn, x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out_pos, ctx_pos = wrapped(ctx0, jnp.array(2.0))
        out_neg, ctx_neg = wrapped(ctx0, jnp.array(-2.0))
        assert out_pos == 3.0 and out_neg == -3.0
        assert "cond" in ctx_pos.metadata and "cond" in ctx_neg.metadata
        assert ctx_pos.total_value_count == 1 and ctx_neg.total_value_count == 1

    def test_cond_context_mismatch_error(self, handler_factory: HandlerFactory):
        cond_h = handler_factory.create_cond_handler("cond")
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        interpreter = create_test_interpreter({"cond": cond_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            def t(v):
                return jnp.sin(v) + 1  # adds context value via sin handler

            def f(v):
                return v - 1

            return jax.lax.cond(x > 0, t, f, x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        with pytest.raises(ValueError):
            wrapped(ctx0, jnp.array(1.0))


class TestCheckpoint:
    def test_checkpoint_basic(self, handler_factory: HandlerFactory):
        """Checkpoint handler threads context through the rematerialized body."""
        ckpt_h = handler_factory.create_checkpoint_handler("checkpoint")
        interpreter = create_test_interpreter({"remat2": ckpt_h})

        def fn(x: jax.Array):
            @checkpoint
            def body(v):
                return v * 2 + 1

            return body(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(3.0))
        assert out == 7.0
        assert "checkpoint" in ctx1.metadata
        assert ctx1.total_value_count == 1

    def test_checkpoint_with_handled_primitive(self, handler_factory: HandlerFactory):
        """Handled primitives inside checkpoint accumulate context."""
        sin_h = handler_factory.create_primitive_handler_with_extra_value("sin")
        ckpt_h = handler_factory.create_checkpoint_handler("checkpoint")
        interpreter = create_test_interpreter({"remat2": ckpt_h, jax.lax.sin_p: sin_h})

        def fn(x: jax.Array):
            @checkpoint
            def body(v):
                return jnp.sin(v) + 1

            return body(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(1.0))
        assert jnp.allclose(out, jnp.sin(1.0) + 1.0)
        assert {"checkpoint", "sin"}.issubset(ctx1.metadata)
        assert ctx1.total_value_count == 2  # one from checkpoint, one from sin

    def test_checkpoint_nested(self, handler_factory: HandlerFactory):
        """Nested checkpoints each contribute to context."""
        ckpt_h = handler_factory.create_checkpoint_handler("checkpoint")
        interpreter = create_test_interpreter({"remat2": ckpt_h})

        def fn(x: jax.Array):
            @checkpoint
            def outer(v):
                @checkpoint
                def inner(w):
                    return w * 3

                return inner(v) + 1

            return outer(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, ctx1 = wrapped(ctx0, jnp.array(2.0))
        assert out == 7.0  # 2*3 + 1
        assert ctx1.metadata.count("checkpoint") == 2
        assert ctx1.total_value_count == 2

    def test_checkpoint_preserves_computation(self, handler_factory: HandlerFactory):
        """Checkpoint must not alter the computed values."""
        ckpt_h = handler_factory.create_checkpoint_handler("checkpoint")
        interpreter = create_test_interpreter({"remat2": ckpt_h})

        def fn(x: jax.Array):
            @checkpoint
            def body(v):
                return jnp.exp(v) + jnp.log(v)

            return body(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        out, _ = wrapped(ctx0, jnp.array(2.0))
        expected = jnp.exp(2.0) + jnp.log(2.0)
        assert jnp.allclose(out, expected)


class TestPolicies:
    def test_policy_raise_unimplemented(self):
        interpreter = create_test_interpreter(
            {}, policy=InterpreterPolicy.RAISE, include_in_match=frozenset({"sin"})
        )

        def fn(x: jax.Array):
            return jnp.sin(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        with pytest.raises(NotImplementedError):
            wrapped(ctx0, jnp.array(1.0))

    def test_policy_warn_unimplemented(self):
        interpreter = create_test_interpreter(
            {}, policy=InterpreterPolicy.WARN, include_in_match=frozenset({"sin"})
        )

        def fn(x: jax.Array):
            return jnp.sin(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        with pytest.warns(UserWarning):
            wrapped(ctx0, jnp.array(1.0))

    def test_policy_ignore_unimplemented(self):
        interpreter = create_test_interpreter(
            {}, policy=InterpreterPolicy.IGNORE, include_in_match=frozenset({"sin"})
        )

        def fn(x: jax.Array):
            return jnp.sin(x)

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        # Should not raise or warn
        wrapped(ctx0, jnp.array(1.0))


class TestErrorPaths:
    def test_scan_missing_handler_policy_warn(self):
        # No scan handler; policy WARN should emit warning when scan encountered
        interpreter = create_test_interpreter(
            {}, policy=InterpreterPolicy.WARN, include_in_match=frozenset({"scan"})
        )

        def fn(_: jax.Array):
            def body(c, x):
                return c + 1, x

            return jax.lax.scan(body, 0, jnp.arange(2))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        with pytest.warns(UserWarning):
            wrapped(ctx0, jnp.array(0.0))


class TestScanInitializerError:
    def test_uninitialized_leaves_raises(self):
        """Scan with carry threading raises if initializer leaves Uninitialized leaves."""

        def bad_initializer(_old_ctx, sentinel_ctx):
            return sentinel_ctx  # leaves Uninitialized as-is

        def updater(_old_ctx, new_ctx):
            return new_ctx

        # Use a primitive handler that adds a value so the context has array leaves
        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            return default_scan_handler(
                interpreter,
                ctx,
                eqn,
                invals,
                threading=ScanSemantics.CARRY,
                initializer=bad_initializer,
                updater=updater,
            )

        sin_h = HandlerFactory.create_primitive_handler_with_extra_value("sin")
        interpreter = create_test_interpreter({"scan": handler, jax.lax.sin_p: sin_h})

        def fn(_: jax.Array):
            def body(carry, x):
                return carry + jnp.sin(x), x

            return jax.lax.scan(body, 0.0, jnp.arange(3.0))

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        with pytest.raises(ValueError, match="initialized"):
            wrapped(ctx0, jnp.array(0.0))


class TestWhileInitializerError:
    def test_uninitialized_leaves_raises(self):
        """While loop raises if initializer leaves Uninitialized leaves."""

        def bad_initializer(_old_ctx, sentinel_ctx):
            return sentinel_ctx

        def updater(_old_ctx, new_ctx):
            return new_ctx

        sin_h = HandlerFactory.create_primitive_handler_with_extra_value("sin")

        def handler(
            interpreter: Interpreter[MockContext],
            ctx: MockContext,
            eqn: JaxprEqn,
            invals: list[TracerValue],
        ) -> HandlerResult[MockContext]:
            return default_while_handler(
                interpreter,
                ctx,
                eqn,
                invals,
                initializer=bad_initializer,
                updater=updater,
            )

        interpreter = create_test_interpreter({"while": handler, jax.lax.sin_p: sin_h})

        def fn(_: jax.Array):
            return jax.lax.while_loop(
                lambda a: a < 5.0,
                lambda a: a + jnp.sin(a * 0.0 + 1.0),
                jnp.float32(0.0),
            )

        wrapped = reinterpret(fn, interpreter)
        ctx0 = MockContext((), None, 0, ())
        with pytest.raises(ValueError, match="initialized"):
            wrapped(ctx0, jnp.array(0.0))


class TestContextBasics:
    def test_context_push_pop(self):
        ctx = MockContext(("root",), None, 0, (jnp.array(1.0),))
        child = ctx.push().add_meta("inner").add_value(jnp.array(2.0))
        merged = child.pop()
        assert merged.total_metadata_count == 2
        assert merged.total_value_count == 2
