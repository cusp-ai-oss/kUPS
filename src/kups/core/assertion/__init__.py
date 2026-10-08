# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""
JAX-compatible assertion tracing system with optional automatic fixing.

This package provides a comprehensive assertion system that works seamlessly with
JAX transformations, including JIT compilation, automatic differentiation, and
vectorization. Assertions can include optional fix functions for automatic
error recovery.

Key Components:

- **[RuntimeAssertion][kups.core.assertion.RuntimeAssertion]**: Core assertion dataclass with optional fixing capabilities
- **[runtime_assert][kups.core.assertion.runtime_assert]**: Function to create assertions that work with JAX transformations
- **[with_runtime_assertions][kups.core.assertion.with_runtime_assertions]**: Decorator to enable assertion tracing in functions
- **[LoopMerge][kups.core.assertion.LoopMerge]**: How loops fold an assertion's arguments across iterations
"""

from kups.core.assertion.primitives import check_assertions, runtime_assert
from kups.core.assertion.runtime_assertion import (
    ELEMENTWISE_MAX,
    FIRST_FAILURE,
    LAST_FAILURE,
    NO_ARGS,
    Fix,
    LoopMerge,
    RuntimeAssertion,
)
from kups.core.assertion.tracing import AssertionContext, with_runtime_assertions
from kups.core.interpreter.interpreter import InterpreterPolicy

__all__ = [
    "ELEMENTWISE_MAX",
    "FIRST_FAILURE",
    "LAST_FAILURE",
    "NO_ARGS",
    "AssertionContext",
    "Fix",
    "InterpreterPolicy",
    "LoopMerge",
    "RuntimeAssertion",
    "check_assertions",
    "runtime_assert",
    "with_runtime_assertions",
]
