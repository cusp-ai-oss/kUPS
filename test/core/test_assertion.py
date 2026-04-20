# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for assertion functionality."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from kups.core.assertion import (
    NO_ARGS,
    AssertionContext,
    InterpreterPolicy,
    RuntimeAssertion,
    check_assertions,
    runtime_assert,
    with_runtime_assertions,
)
from kups.core.capacity import CapacityError, LensCapacityFix
from kups.core.lens import lens
from kups.core.utils.jax import dataclass


class SimpleState(NamedTuple):
    """Simple state for testing purposes."""

    capacity: int
    data: Array


@dataclass
class DataclassState:
    """Dataclass state for testing purposes."""

    capacity: int
    data: Array


class TestCapacityFix:
    """Tests for CapacityFix class."""

    def test_capacity_fix(self):
        """Test CapacityFix: basic set, keep higher, and array-to-int conversion."""

        def get_dc_capacity(state: DataclassState) -> int:
            return state.capacity

        def get_capacity(state: SimpleState) -> int:
            return state.capacity

        # Basic dataclass fix
        dc_fix = LensCapacityFix(lens=lens(get_dc_capacity))
        state_dc = DataclassState(capacity=5, data=jnp.array([1, 2]))
        new_dc = dc_fix(state_dc, jnp.array([20]))
        assert new_dc.capacity == 20

        # Keeps higher capacity
        fix = LensCapacityFix(lens=lens(get_capacity))
        state_high = SimpleState(capacity=25, data=jnp.array([1, 2, 3]))
        new_high = fix(state_high, jnp.array([10]))
        assert new_high.capacity == 25

        # Array-to-int conversion (float truncated)
        state_conv = SimpleState(capacity=10, data=jnp.array([1, 2]))
        new_conv = fix(state_conv, jnp.array([15.7]))
        assert new_conv.capacity == 15


class TestRuntimeAssertion:
    """Tests for RuntimeAssertion class."""

    def test_runtime_assertion_properties(self):
        """Test RuntimeAssertion creation, formatting, custom exception, and static info."""
        # Basic creation and defaults
        predicate = jnp.array(True)
        assertion = RuntimeAssertion(
            predicate=predicate, message="Test assertion failed"
        )
        assert jnp.array_equal(assertion.predicate, predicate)
        assert assertion.message == "Test assertion failed"
        assert assertion.fmt_args == {}
        assert assertion.exception_type is AssertionError
        assert assertion.static_info == {}
        assert assertion.fix_fn is None
        assert assertion.fix_args is NO_ARGS

        # With message formatting
        fmt_assertion = RuntimeAssertion(
            predicate=jnp.array(False),
            message="Value {value} exceeds limit {limit}",
            fmt_args={"value": jnp.array(15), "limit": jnp.array(10)},
        )
        exception = fmt_assertion.exception
        assert isinstance(exception, AssertionError)
        assert "Value 15 exceeds limit 10" in str(exception)

        # Custom exception type
        cap_assertion = RuntimeAssertion(
            predicate=jnp.array(False),
            message="Capacity exceeded",
            exception_type=CapacityError,
        )
        cap_exception = cap_assertion.exception
        assert isinstance(cap_exception, CapacityError)
        assert str(cap_exception) == "Capacity exceeded"

        # Static info
        info_assertion = RuntimeAssertion(
            predicate=jnp.array(True),
            message="Test message",
            static_info={"source": "test", "level": "debug"},
        )
        assert info_assertion.static_info == {"source": "test", "level": "debug"}

    def test_runtime_assertion_fix_behavior(self):
        """Test fix: without fix_fn, with fix_fn, NO_ARGS, capacity fix, complex args."""
        state = SimpleState(capacity=10, data=jnp.array([1, 2, 3]))

        # Fix without fix_fn raises AssertionError
        assertion_no_fix = RuntimeAssertion(
            predicate=jnp.array(False), message="Test assertion"
        )
        with pytest.raises(AssertionError):
            assertion_no_fix.fix(state)

        # Fix with fix_fn
        def test_fix_fn(state: SimpleState, target_capacity: Array) -> SimpleState:
            return state._replace(capacity=int(target_capacity))

        assertion_with_fix = RuntimeAssertion(
            predicate=jnp.array(False),
            message="Capacity insufficient",
            fix_fn=test_fix_fn,
            fix_args=jnp.array(20),
        )
        fixed = assertion_with_fix.fix(state)
        assert fixed.capacity == 20
        assert jnp.array_equal(fixed.data, state.data)

        # Fix with NO_ARGS raises AssertionError
        assertion_no_args = RuntimeAssertion(
            predicate=jnp.array(False),
            message="Test assertion",
            fix_fn=test_fix_fn,
            fix_args=NO_ARGS,
        )
        with pytest.raises(AssertionError, match="Fix arguments were not provided"):
            assertion_no_args.fix(state)

        # Integrated with CapacityFix
        def get_capacity(state: SimpleState) -> int:
            return state.capacity

        capacity_fix = LensCapacityFix(lens=lens(get_capacity))
        assertion_cap = RuntimeAssertion(
            predicate=jnp.array(False),
            message="Capacity {current} is less than required {required}",
            fmt_args={"current": jnp.array(10), "required": jnp.array(20)},
            fix_fn=capacity_fix,
            fix_args=jnp.array([20]),
        )
        fixed_cap = assertion_cap.fix(state)
        assert fixed_cap.capacity == 20
        assert "Capacity 10 is less than required 20" in str(assertion_cap.exception)

        # Complex fix args
        def complex_fix_fn(
            state: SimpleState, args: tuple[Array, Array]
        ) -> SimpleState:
            new_capacity, multiplier = args
            return state._replace(capacity=int(new_capacity * multiplier))

        assertion_complex = RuntimeAssertion(
            predicate=jnp.array(False),
            message="Complex fix test",
            fix_fn=complex_fix_fn,
            fix_args=(jnp.array(15), jnp.array(2)),
        )
        fixed_complex = assertion_complex.fix(state)
        assert fixed_complex.capacity == 30  # 15 * 2


class TestAssertionTracing:
    """Tests for assertion tracing through JAX transformations."""

    def test_scan_tracing(self):
        """Test assertion tracing through jax.lax.scan."""

        @jax.jit
        @with_runtime_assertions
        def scanned(x):
            def f(carry, x):
                runtime_assert(
                    predicate=(carry < 10) & (x < 10),
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": x, "b": carry},
                )
                return carry**2, x * 2

            return jax.lax.scan(f, x, jnp.arange(10), length=10)

        out, assertions = scanned(jnp.array(2))
        assert len(out) == 2  # carry, xs
        assert len(assertions) == 1
        assert isinstance(assertions[0], RuntimeAssertion)
        assert assertions[0].message.startswith("Assertion failed: {a} < 10")
        assert "a" in assertions[0].fmt_args
        assert "b" in assertions[0].fmt_args

    def test_jvp_tracing(self):
        """Test assertion tracing through forward-mode differentiation."""

        @jax.jit
        @with_runtime_assertions
        def jvp_fn(x):
            def f(x):
                runtime_assert(
                    predicate=x < 10,
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": x},
                )
                return 2 * x

            return jax.jvp(f, (x,), (jnp.ones_like(x),))

        out, assertions = jvp_fn(jnp.array(20.0))
        assert len(out) == 2  # primal, tangent
        assert len(assertions) == 1
        assert isinstance(assertions[0], RuntimeAssertion)
        assert not assertions[0].predicate  # Should fail since 20 >= 10

    def test_vjp_tracing(self):
        """Test assertion tracing through reverse-mode differentiation."""

        @jax.jit
        @with_runtime_assertions
        def vjp_fn(x):
            def f(x):
                runtime_assert(
                    predicate=x < 10,
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": x},
                )
                return 2 * x

            return jax.vjp(f, x)[1](jnp.ones_like(x))

        out, assertions = vjp_fn(jnp.array(20.0))
        assert len(out) == 1  # gradient
        assert len(assertions) == 1
        assert isinstance(assertions[0], RuntimeAssertion)
        assert not assertions[0].predicate  # Should fail since 20 >= 10

    def test_nested_jit_tracing(self):
        """Test assertion tracing through nested jit compilation."""

        @jax.jit
        @with_runtime_assertions
        def nested(x):
            @jax.jit
            def f(x):
                runtime_assert(
                    predicate=x < 10,
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": x},
                )
                return 2 * x

            return f(x)

        out, assertions = nested(jnp.array(20))
        assert isinstance(out, jax.Array)
        assert len(assertions) == 1
        assert isinstance(assertions[0], RuntimeAssertion)
        assert not assertions[0].predicate  # Should fail since 20 >= 10

    def test_while_loop_tracing(self):
        """Test assertion tracing through jax.lax.while_loop."""

        @jax.jit
        @with_runtime_assertions
        def while_loop(x):
            def cond(arg):
                x, i = arg
                return (i < 10) & (x < 10)

            def body(arg):
                x, i = arg
                runtime_assert(
                    predicate=(i < 10) & (x < 10),
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": x, "b": i},
                )
                return x * 2, i + 1

            return jax.lax.while_loop(cond, body, (x, 0))[0]

        out, assertions = while_loop(jnp.array(1))
        assert isinstance(out, jax.Array)
        assert len(assertions) == 1
        assert isinstance(assertions[0], RuntimeAssertion)

    def test_cond_tracing_fails_for_divergent_branches(self):
        """Test assertion tracing through jax.lax.cond."""

        @jax.jit
        @with_runtime_assertions
        def cond_fn(x):
            def f(x, y):
                runtime_assert(
                    predicate=x > 10,
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": x, "c": jnp.array(3.0)},
                )
                return 2 * x

            def g(x, y):
                runtime_assert(
                    predicate=x > 10,
                    message="Assertion failed: {b} > 10",
                    fmt_args={"b": y + 5, "c": x},
                )
                return 3 * x

            return jax.lax.cond(x == 6, f, g, x, x * 2)

        with pytest.raises(
            ValueError, match="Cond branches return inconsistent contexts."
        ):
            out, assertions = cond_fn(jnp.array(5.0))

    def test_cond_tracing(self):
        """Test assertion tracing through jax.lax.cond."""

        @with_runtime_assertions
        @jax.jit
        def cond_fn(x):
            def falsy_fun(x, y):
                runtime_assert(
                    predicate=x > 10,
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": x, "c": jnp.array(3.0)},
                )
                return 2 * x

            def truthy_fun(x, y):
                runtime_assert(
                    predicate=y > 10,
                    message="Assertion failed: {a} < 10",
                    fmt_args={"a": y, "c": x},
                )
                return 3 * x

            return jax.lax.cond(x > 5, truthy_fun, falsy_fun, x, x * 2)

        out, assertions = cond_fn(jnp.array(5.0))
        assert isinstance(out, jax.Array)
        assert len(assertions) == 1
        assert not assertions[0].predicate  # should fail since 5 < 10

        out, assertions = cond_fn(jnp.array(6.0))
        assert assertions[0].predicate  # should pass since 12 > 10

    def test_traceback_present_in_message(self):
        """Test that assertion messages include the creation-site traceback."""

        @jax.jit
        @with_runtime_assertions
        def fn(x):
            runtime_assert(predicate=x > 0, message="x must be positive")
            return x

        _, assertions = fn(jnp.array(-1.0))
        assert "\nAssertion created at:\n" in assertions[0].message
        assert "test_traceback_present_in_message" in assertions[0].message

    def test_traceback_unavailable_in_cond(self):
        """Test that assertions inside cond branches note the missing traceback."""

        @with_runtime_assertions
        @jax.jit
        def cond_fn(x):
            def true_branch(x):
                runtime_assert(predicate=x > 10, message="too small", fmt_args={"x": x})
                return x * 2

            def false_branch(x):
                runtime_assert(predicate=x < 0, message="too small", fmt_args={"x": x})
                return x * 3

            return jax.lax.cond(x > 5, true_branch, false_branch, x)

        _, assertions = cond_fn(jnp.array(3.0))
        assert len(assertions) == 1
        assert "\nAssertion created at:\n" not in assertions[0].message
        assert "traceback unavailable" in assertions[0].message

    def test_complex_nested_transformations(self):
        """Test assertion tracing through complex nested transformations."""

        @jax.jit
        @with_runtime_assertions
        def complex_fn(x, y):
            def scan_body(carry, elem):
                runtime_assert(
                    predicate=carry > 0,
                    message="Carry must be positive: {carry}",
                    fmt_args={"carry": carry},
                )
                return carry + elem, carry * elem

            def cond_fn(val):
                return val < 100

            def while_body(val):
                runtime_assert(
                    predicate=val < 50,
                    message="Value too large: {val}",
                    fmt_args={"val": val},
                )
                return val * 1.1

            # Scan through some values
            final_carry, scan_outputs = jax.lax.scan(scan_body, x, y)

            # Use while loop
            result = jax.lax.while_loop(cond_fn, while_body, final_carry)

            return result, scan_outputs

        x = jnp.array(2.0)
        y = jnp.arange(3.0)

        out, assertions = complex_fn(x, y)
        assert len(out) == 2
        assert len(assertions) >= 1  # Should have at least one assertion

    def test_vmap_compatibility(self):
        """Test that assertion tracing works with vectorized functions."""

        @jax.jit
        @with_runtime_assertions
        def vmapped_fn(x):
            @jax.vmap
            def inner(x_elem):
                runtime_assert(
                    predicate=x_elem > 0,
                    message="Element must be positive: {elem}",
                    fmt_args={"elem": x_elem},
                )
                return x_elem**2

            return inner(x)

        x = jnp.array([1.0, 2.0, 3.0])
        out, assertions = vmapped_fn(x)
        assert out.shape == x.shape
        assert len(assertions) >= 1

    def test_complex_fix_args_tracing(self):
        """Test complex fix_args PyTree structures across scan, jit, cond, nested."""

        # --- scan ---
        @with_runtime_assertions
        def scan_with_complex_fix_args(x):
            def scan_fn(carry, x_elem):
                complex_fix_args = ((jnp.array(0.1), jnp.array(2.0)), jnp.array(0.5))
                runtime_assert(
                    predicate=x_elem > 0,
                    message="Element must be positive",
                    fix_args=complex_fix_args,
                )
                return carry + x_elem, x_elem**2

            return jax.lax.scan(scan_fn, 0.0, x)

        x = jnp.array([1.0, 2.0, 3.0])
        out, assertions = scan_with_complex_fix_args(x)
        assert len(assertions) >= 1
        for assertion in assertions:
            if hasattr(assertion, "fix_args") and assertion.fix_args is not NO_ARGS:
                fix_args = assertion.fix_args
                assert isinstance(fix_args, tuple)
                assert len(fix_args) == 2
                assert isinstance(fix_args[0], tuple)
                assert len(fix_args[0]) == 2

        # --- jit ---
        @jax.jit
        @with_runtime_assertions
        def jit_with_complex_fix_args(x):
            dict_fix_args = {
                "thresholds": {"min": jnp.array(0.1), "max": jnp.array(10.0)},
                "multipliers": (jnp.array(2.0), jnp.array(3.0)),
                "offset": jnp.array(0.5),
            }
            runtime_assert(
                predicate=x > 0,
                message="Input must be positive",
                fix_args=dict_fix_args,
            )
            return x * 2

        out, assertions = jit_with_complex_fix_args(jnp.array(5.0))
        assert out == 10.0
        assert len(assertions) >= 1
        assertion = assertions[0]
        if assertion.fix_args is not NO_ARGS:
            fix_args = assertion.fix_args
            assert isinstance(fix_args, dict)
            assert "thresholds" in fix_args
            assert "multipliers" in fix_args
            assert "offset" in fix_args
            assert isinstance(fix_args["thresholds"], dict)
            assert "min" in fix_args["thresholds"]
            assert "max" in fix_args["thresholds"]

        # --- cond ---
        @with_runtime_assertions
        def cond_with_complex_fix_args(x, flag):
            def true_branch(x):
                mixed_fix_args = (
                    {"scale": jnp.array(2.0), "bias": jnp.array(1.0)},
                    (jnp.array(0.1), jnp.array(0.9), jnp.array(0.8)),
                )
                runtime_assert(
                    predicate=x > 5, message="Branch", fix_args=mixed_fix_args
                )
                return x * 2

            def false_branch(x):
                array_fix_args = (
                    {"scale": jnp.array(3.0), "bias": jnp.array(2.0)},
                    (jnp.array(1.0), jnp.array(2.0), jnp.array(3.0)),
                )
                runtime_assert(
                    predicate=x < 5, message="Branch", fix_args=array_fix_args
                )
                return x / 2

            return jax.lax.cond(flag, true_branch, false_branch, x)

        out1, assertions1 = cond_with_complex_fix_args(jnp.array(6.0), True)
        assert out1 == 12.0
        assert len(assertions1) >= 1

        out2, assertions2 = cond_with_complex_fix_args(jnp.array(3.0), False)
        assert out2 == 1.5
        assert len(assertions2) >= 1

        # --- deeply nested ---
        @with_runtime_assertions
        def deeply_nested_with_fix_args(x):
            @jax.jit
            def inner_jit(y):
                def scan_body(carry, elem):
                    nested_fix_args = {
                        "level1": {
                            "level2": {
                                "arrays": (jnp.array(1.0), jnp.array(2.0)),
                                "scalars": {"a": jnp.array(0.5), "b": jnp.array(1.5)},
                            },
                            "tuples": (
                                (jnp.array(0.1), jnp.array(0.2)),
                                (jnp.array(0.3), jnp.array(0.4)),
                            ),
                        },
                        "top_level": jnp.array(42.0),
                    }
                    runtime_assert(
                        predicate=elem > carry,
                        message="Element must be greater than carry",
                        fix_args=nested_fix_args,
                    )
                    return carry + elem, elem

                carry, ys = jax.lax.scan(scan_body, 0.0, y)
                return carry, ys

            return inner_jit(x)

        out, assertions = deeply_nested_with_fix_args(jnp.array([1.0, 2.0, 3.0]))
        assert len(assertions) >= 1
        for assertion in assertions:
            if assertion.fix_args is not NO_ARGS:
                fix_args = assertion.fix_args
                assert isinstance(fix_args, dict)
                assert "level1" in fix_args
                assert "top_level" in fix_args
                assert isinstance(fix_args["level1"], dict)
                assert "level2" in fix_args["level1"]
                assert "tuples" in fix_args["level1"]
                assert isinstance(fix_args["level1"]["level2"], dict)
                assert "arrays" in fix_args["level1"]["level2"]
                assert "scalars" in fix_args["level1"]["level2"]


class TestAssertionValidation:
    """Tests for assertion validation and error checking."""

    def test_check_assertions_empty(self):
        """Test check_assertions returns True when no assertions are present."""
        ctx = AssertionContext()
        result = ctx.check_assertions()
        assert result == jnp.array(True, dtype=jnp.bool)

    def test_check_assertions_without_like_detects_failure(self):
        """Test that check_assertions() without like arg returns False on failing assertion."""

        @jax.jit
        @with_runtime_assertions
        def test_fn(x):
            runtime_assert(predicate=x > 0, message="x must be positive")
            return check_assertions()

        # Passing: check_assertions should return True
        result, _ = test_fn(jnp.array(1.0))
        assert result

        # Failing: check_assertions should return False (not always True)
        result, _ = test_fn(jnp.array(-1.0))
        assert not result

    def test_assertion_check_and_str(self):
        """Test check method, custom exception type check, and string representation."""
        # Passing assertion check
        passing = RuntimeAssertion(
            predicate=jnp.array(True), fmt_args={}, message="This should pass"
        )
        passing.check()  # Should not raise

        # Failing assertion check
        failing = RuntimeAssertion(
            predicate=jnp.array(False),
            fmt_args={"value": jnp.array(5)},
            message="Value is {value}",
            exception_type=ValueError,
        )
        with pytest.raises(ValueError, match="Value is 5"):
            failing.check()

        # Custom exception type
        rt_assertion = RuntimeAssertion(
            predicate=jnp.array(False),
            fmt_args={},
            message="Custom error",
            exception_type=RuntimeError,
        )
        with pytest.raises(RuntimeError, match="Custom error"):
            rt_assertion.check()

        # String representation
        str_assertion = RuntimeAssertion(
            predicate=jnp.array(False),
            fmt_args={"x": jnp.array(10), "y": jnp.array(20)},
            message="x={x}, y={y}",
        )
        assert str(str_assertion) == "x=10, y=20"

    def test_runtime_assert_basic(self):
        """Test basic runtime_assert functionality."""

        @jax.jit
        @with_runtime_assertions
        def test_fn(x):
            runtime_assert(
                predicate=x > 0,
                message="x must be positive",
                fmt_args={"x": x},
            )
            return x * 2

        # Test passing case
        out, assertions = test_fn(jnp.array(5.0))
        assert out == 10.0
        assert len(assertions) == 1
        assert assertions[0].predicate

        # Test failing case
        out, assertions = test_fn(jnp.array(-5.0))
        assert out == -10.0
        assert len(assertions) == 1
        assert not assertions[0].predicate

    def test_multiple_assertions_in_function(self):
        """Test multiple assertions in a single function."""

        @jax.jit
        @with_runtime_assertions
        def test_fn(x, y):
            runtime_assert(
                predicate=x > 0,
                message="x must be positive: {x}",
                fmt_args={"x": x},
            )
            runtime_assert(
                predicate=y < 10,
                message="y must be less than 10: {y}",
                fmt_args={"y": y},
            )
            return x + y

        out, assertions = test_fn(jnp.array(5.0), jnp.array(15.0))
        assert out == 20.0
        assert len(assertions) == 2
        assert assertions[0].predicate  # x > 0 should pass
        assert not assertions[1].predicate  # y < 10 should fail

    def test_assertion_without_fmt_args(self):
        """Test assertions without format arguments."""

        @jax.jit
        @with_runtime_assertions
        def test_fn(x):
            runtime_assert(
                predicate=x > 0,
                message="x must be positive",
            )
            return x

        out, assertions = test_fn(jnp.array(-1.0))
        assert len(assertions) == 1
        assert assertions[0].fmt_args == {}
        assert assertions[0].message.startswith("x must be positive")

    def test_interpreter_policy_ignore(self):
        """Test InterpreterPolicy.IGNORE behavior."""

        @jax.jit
        @partial(with_runtime_assertions, policy=InterpreterPolicy.IGNORE)
        def test_fn(x):
            runtime_assert(
                predicate=x > 0,
                message="x must be positive",
                fmt_args={"x": x},
            )
            # Use an operation that might not be implemented
            return x.at[0].add(1)

        # Should not raise even if scatter is not implemented
        out, assertions = test_fn(jnp.array([5.0]))
        assert isinstance(out, jax.Array)

    def test_interpreter_policy_raise(self):
        """Test InterpreterPolicy.RAISE collects failed assertions that raise on check."""

        @jax.jit
        @partial(with_runtime_assertions, policy=InterpreterPolicy.RAISE)
        def test_fn(x):
            runtime_assert(
                predicate=x > 0,
                message="x must be positive",
                fmt_args={"x": x},
            )
            return x * 2

        out, assertions = test_fn(jnp.array(-5.0))
        assert out == -10.0
        assert len(assertions) == 1
        assert not assertions[0].predicate
        with pytest.raises(AssertionError, match="x must be positive"):
            assertions[0].check()

    def test_empty_assertions(self):
        """Test functions that don't contain any assertions."""

        @jax.jit
        @with_runtime_assertions
        def test_fn(x):
            return x * 2

        out, assertions = test_fn(jnp.array(5.0))
        assert out == 10.0
        assert len(assertions) == 0

    def test_assertion_with_array_predicates(self):
        """Test assertions with array predicates."""

        @jax.jit
        @with_runtime_assertions
        def test_fn(x):
            runtime_assert(
                predicate=jnp.all(x > 0),
                message="All elements must be positive",
                fmt_args={"x": x},
            )
            return jnp.sum(x)

        # Test with all positive
        out, assertions = test_fn(jnp.array([1.0, 2.0, 3.0]))
        assert len(assertions) == 1
        assert assertions[0].predicate

        # Test with some negative
        out, assertions = test_fn(jnp.array([1.0, -2.0, 3.0]))
        assert len(assertions) == 1
        assert not assertions[0].predicate


class TestAssertionErrorHandling:
    """Tests for error handling in assertion tracing."""

    def test_invalid_while_cond_with_assertions(self):
        """Test that assertions in while loop conditions raise errors."""

        with pytest.raises(ValueError, match="modified the context"):

            @jax.jit
            @with_runtime_assertions
            def invalid_while(x):
                def cond(arg):
                    x, i = arg
                    runtime_assert(
                        predicate=i < 10,
                        message="Invalid assertion in cond",
                    )
                    return i < 5

                def body(arg):
                    x, i = arg
                    return x + 1, i + 1

                return jax.lax.while_loop(cond, body, (x, 0))

            invalid_while(jnp.array(1.0))

    def test_cond_branch_mismatch_errors(self):
        """Test cond branches with mismatched assertions/fmt_args/shapes/dtypes/predicates."""
        # Each factory creates a JIT function with inconsistent cond branches

        # 1. Different number of assertions
        def make_mismatch_assertion_count():
            @jax.jit
            @with_runtime_assertions
            def test_fn(x):
                def branch_a(x):
                    runtime_assert(
                        predicate=x > 0,
                        message="Branch A assertion",
                        fmt_args={"x": x},
                    )
                    return x * 2

                def branch_b(x):
                    runtime_assert(
                        predicate=x < 10,
                        message="Branch B assertion 1",
                        fmt_args={"x": x},
                    )
                    runtime_assert(
                        predicate=x != 5,
                        message="Branch B assertion 2",
                        fmt_args={"x": x},
                    )
                    return x * 3

                return jax.lax.cond(x > 5, branch_a, branch_b, x)

            return test_fn

        # 2. Different fmt_args count
        def make_mismatch_fmt_args_count():
            @jax.jit
            @with_runtime_assertions
            def test_fn(x):
                def branch_a(x):
                    runtime_assert(
                        predicate=x > 0,
                        message="Branch A: {x}",
                        fmt_args={"x": x},
                    )
                    return x * 2

                def branch_b(x):
                    runtime_assert(
                        predicate=x < 10,
                        message="Branch B: {x} and {y}",
                        fmt_args={"x": x, "y": x + 1},
                    )
                    return x * 3

                return jax.lax.cond(x > 5, branch_a, branch_b, x)

            return test_fn

        # 3. Different fmt_args shapes
        def make_mismatch_fmt_args_shapes():
            @jax.jit
            @with_runtime_assertions
            def test_fn(x):
                def branch_a(x):
                    runtime_assert(
                        predicate=x > 0,
                        message="Branch A: {arr}",
                        fmt_args={"arr": jnp.array([x])},
                    )
                    return x * 2

                def branch_b(x):
                    runtime_assert(
                        predicate=x < 10,
                        message="Branch B: {arr}",
                        fmt_args={"arr": jnp.array([[x, x]])},
                    )
                    return x * 3

                return jax.lax.cond(x > 5, branch_a, branch_b, x)

            return test_fn

        # 4. Different fmt_args dtypes
        def make_mismatch_fmt_args_dtypes():
            @jax.jit
            @with_runtime_assertions
            def test_fn(x):
                def branch_a(x):
                    runtime_assert(
                        predicate=x > 0,
                        message="Branch A: {val}",
                        fmt_args={"val": x.astype(float)},
                    )
                    return x * 2

                def branch_b(x):
                    runtime_assert(
                        predicate=x < 10,
                        message="Branch B: {val}",
                        fmt_args={"val": x.astype(int)},
                    )
                    return x * 3

                return jax.lax.cond(x > 5, branch_a, branch_b, x)

            return test_fn

        # 5. Different predicate shapes
        def make_mismatch_predicate_shapes():
            @jax.jit
            @with_runtime_assertions
            def test_fn(x):
                def branch_a(x):
                    runtime_assert(
                        predicate=jnp.array(x > 0),
                        message="Branch assertion",
                        fmt_args={},
                    )
                    return x * 2

                def branch_b(x):
                    runtime_assert(
                        predicate=jnp.array([x < 10, x != 5]),
                        message="Branch assertion",
                        fmt_args={},
                    )
                    return x * 3

                return jax.lax.cond(x > 5, branch_a, branch_b, x)

            return test_fn

        # 6. Different predicate dtypes
        def make_mismatch_predicate_dtypes():
            @jax.jit
            @with_runtime_assertions
            def test_fn(x):
                def branch_a(x):
                    runtime_assert(
                        predicate=(x > 0).astype(jnp.bool_),
                        message="Branch assertion",
                        fmt_args={},
                    )
                    return x * 2

                def branch_b(x):
                    runtime_assert(
                        predicate=(x < 10).astype(int),
                        message="Branch assertion",
                        fmt_args={},
                    )
                    return x * 3

                return jax.lax.cond(x > 5, branch_a, branch_b, x)

            return test_fn

        factories = [
            make_mismatch_assertion_count,
            make_mismatch_fmt_args_count,
            make_mismatch_fmt_args_shapes,
            make_mismatch_fmt_args_dtypes,
            make_mismatch_predicate_shapes,
            make_mismatch_predicate_dtypes,
        ]

        for factory in factories:
            fn = factory()
            with pytest.raises(ValueError, match="return inconsistent contexts"):
                fn(jnp.array(3.0))


class TestAssertionIntegrationWithJAXEcosystem:
    """Tests for integration with the broader JAX ecosystem."""

    def test_assertion_with_pytrees(self):
        """Test assertions with complex PyTree structures."""

        @dataclass
        class State:
            position: Array
            velocity: Array
            energy: Array

        @jax.jit
        @with_runtime_assertions
        def update_state(state: State):
            runtime_assert(
                predicate=state.energy > 0,
                message="Energy must be positive: {energy}",
                fmt_args={"energy": state.energy},
            )
            runtime_assert(
                predicate=jnp.all(jnp.isfinite(state.position)),
                message="Position must be finite",
                fmt_args={"pos": state.position},
            )
            return State(
                position=state.position + state.velocity,
                velocity=state.velocity * 0.99,
                energy=state.energy * 0.999,
            )

        initial_state = State(
            position=jnp.array([1.0, 2.0]),
            velocity=jnp.array([0.1, -0.1]),
            energy=jnp.array(10.0),
        )

        new_state, assertions = update_state(initial_state)
        assert isinstance(new_state, State)
        assert len(assertions) == 2
        assert all(assertion.predicate for assertion in assertions)

    def test_assertion_with_scan_over_complex_state(self):
        """Test assertions in scan operations over complex state."""

        @dataclass
        class State:
            x: Array
            count: Array

        @jax.jit
        @with_runtime_assertions
        def scan_with_assertions(initial_state: State, steps: Array):
            def step_fn(state: State, step_size: Array):
                runtime_assert(
                    predicate=state.x >= 0,
                    message="State x must be non-negative: {x}",
                    fmt_args={"x": state.x},
                )
                runtime_assert(
                    predicate=step_size > 0,
                    message="Step size must be positive: {step}",
                    fmt_args={"step": step_size},
                )
                new_state = State(x=state.x + step_size, count=state.count + 1)
                return new_state, state.x

            return jax.lax.scan(step_fn, initial_state, steps)

        initial = State(x=jnp.array(0.0), count=jnp.array(0))
        steps = jnp.array([1.0, 2.0, 3.0])

        (final_state, trajectory), assertions = scan_with_assertions(initial, steps)
        assert isinstance(final_state, State)
        # Note: trajectory might be empty array due to scan carry behavior
        assert isinstance(trajectory, jax.Array)
        assert len(assertions) >= 1

    def test_higher_order_transformations(self):
        """Test assertions through higher-order JAX transformations."""

        @jax.jit
        @with_runtime_assertions
        def fn_with_assertions(params, x):
            runtime_assert(
                predicate=jnp.all(params > 0),
                message="Parameters must be positive",
                fmt_args={"params": params},
            )
            return jnp.sum(params * x**2)

        # Test with grad
        @jax.jit
        def grad_fn(params, x):
            def wrapped_fn(p):
                result, assertions = fn_with_assertions(p, x)
                return result

            return jax.grad(wrapped_fn)(params)

        params = jnp.array([1.0, 2.0, 3.0])
        x = jnp.array([0.5, 1.0, 1.5])

        gradients = grad_fn(params, x)
        assert gradients.shape == params.shape
