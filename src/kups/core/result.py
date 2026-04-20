# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Result types for simulation computation management.

This module provides a system for managing simulation results and runtime
assertions. The key component is:

- **[Result][kups.core.result.Result]**: Encapsulates computation results with assertions

Results integrate with JAX transformations while maintaining runtime validation
through [RuntimeAssertion][kups.core.assertion.RuntimeAssertion].
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array
from slub.interpreter import InterpreterPolicy

from kups.core.assertion import RuntimeAssertion, with_runtime_assertions
from kups.core.utils.jax import (
    dataclass,
    field,
    no_jax_tracing,
    skip_post_init_if_disabled,
)

type Fix[State, *FixArgs] = Callable[[State, *FixArgs], State]


@dataclass
class Result[State, Return]:
    """
    A result of a computation, containing the actual result and a sequence of runtime
    assertions that will be checked after the computation is done (outside of
    JIT-compiled functions).
    """

    value: Return
    assertions: tuple[RuntimeAssertion[State, Any], ...] = ()
    _all_pass: Array = field(default=None, kw_only=True)  # type: ignore

    def __iter__(self):
        yield self.value
        yield from self.assertions

    @skip_post_init_if_disabled
    def __post_init__(self):
        # Let's precompute _all_pass to compute its value during initialization.
        # This way we avoid computing it multiple times when checking assertions later.
        if self._all_pass is None:
            object.__setattr__(
                self,
                "_all_pass",
                jnp.all(jnp.array([a.predicate for a in self.assertions])),
            )

    @no_jax_tracing
    def raise_assertion(self):
        """Raises an assertion error if any of the runtime assertions failed.

        Note:
        This method should be called outside of JIT-compiled functions.
        """
        for assertion in self.failed_assertions:
            raise assertion.exception

    @property
    def all_assertions_pass(self) -> Array:
        """
        Returns True if all runtime assertions pass, i.e., their predicates
        evaluate to True. By compressing all predicates into a single boolean array,
        we only need to transfer one boolean value from device to host to check if
        all assertions passed.
        """
        return self._all_pass

    @property
    @no_jax_tracing
    def failed_assertions(self) -> tuple[RuntimeAssertion[State, Any], ...]:
        """
        Returns a tuple of runtime assertions that failed, i.e., those
        whose predicate evaluates to False.

        Note:
        This method should be called outside of JIT-compiled functions.
        """
        try:
            if self.all_assertions_pass.item():
                return ()
            return tuple(
                e for e in self.assertions if not bool(jnp.min(e.predicate).item())
            )
        except jax.errors.ConcretizationTypeError as e:
            logging.error(
                "Attempting to evaluate runtime assertions with non-concrete values."
                " You most likely attempted this within a jit-compiled function."
                " Check assertions outside of jit-compiled functions."
            )
            raise e

    @no_jax_tracing
    def fix_or_raise(self, state: State) -> State:
        """Apply fixes for all failed assertions, raising for any without a fix function.

        If all assertions pass, the state is returned unchanged. Otherwise each
        failed assertion's fix function is applied in sequence. Assertions that
        have no fix function registered will raise their configured exception.

        Args:
            state: Current simulation state to repair.

        Returns:
            State with all fixable assertion failures corrected.

        Raises:
            Exception: The configured exception of any failed assertion that has no fix function.
        """
        assertions = self.failed_assertions
        if not assertions:
            return state
        logging.info("Fixing failed assertions.")
        for assertion in assertions:
            logging.info("\t" + str(assertion.exception))
            state = assertion.fix(state)
        return state


def as_result_function[**P, R](
    fn: Callable[P, R],
    policy: InterpreterPolicy = InterpreterPolicy.RAISE,
    context_sharding: jax.sharding.PartitionSpec | None = None,
) -> Callable[P, Result[Any, R]]:
    """Wrap a function to return a Result with runtime assertion tracking.

    This decorator transforms a regular function into one that returns a Result
    object containing both the return value and any runtime assertions encountered
    during execution. Assertions are extracted through JAX tracing.

    Args:
        fn: Function to wrap
        policy: Interpreter policy for handling assertions during tracing
        context_sharding: Optional sharding specification for distributed execution

    Returns:
        Wrapped function that returns Result[Any, R] instead of R

    Example:
        ```python
        @as_result_function
        def compute(x):
            runtime_assert(x > 0, "x must be positive")
            return x ** 2

        result = compute(5.0)
        result.raise_assertion()  # Check all assertions
        value = result.value  # Get the actual result
        ```
    """
    fn_with_assertions = with_runtime_assertions(fn, policy, context_sharding)

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[Any, R]:
        return Result(*fn_with_assertions(*args, **kwargs))

    return wrapper
