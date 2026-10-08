# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Runtime assertion record and the rules loops use to fold it."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Final, override

import jax.numpy as jnp
from jax import Array

from kups.core.utils.jax import dataclass, field

# Type alias for fix functions that take a state and fix arguments, returning a new state
type Fix[State, FixArgs] = Callable[[State, FixArgs], State]


@dataclass
class _NO_ARGS: ...  # We cannot use `None` as a default value because it may be an actual argument.


NO_ARGS: Final[_NO_ARGS] = _NO_ARGS()
"""Sentinel instance to distinguish between no arguments and None as an argument."""


@dataclasses.dataclass(frozen=True)
class LoopMerge:
    """How a loop folds an assertion's ``fmt_args``/``fix_args`` across iterations.

    The predicate of an assertion inside ``scan`` or ``while_loop`` always
    accumulates as a conjunction over iterations. ``LoopMerge`` decides, leaf by
    leaf, which values the reported ``fmt_args`` and ``fix_args`` carry.

    Attributes:
        init: Value of an argument leaf before the first iteration, given a leaf
            from the loop body.
        combine: ``(old_predicate, old, new_predicate, new) -> merged`` for one
            leaf, where ``old`` is accumulated over the previous iterations and
            ``new`` comes from the current one.
    """

    init: Callable[[Array], Array]
    combine: Callable[[Array, Array, Array, Array], Array]


def _keep_last_failure(
    old_predicate: Array, old: Array, new_predicate: Array, new: Array
) -> Array:
    return jnp.where(new_predicate, old, new)


def _keep_first_failure(
    old_predicate: Array, old: Array, new_predicate: Array, new: Array
) -> Array:
    return jnp.where(old_predicate, new, old)


def _keep_largest(
    old_predicate: Array, old: Array, new_predicate: Array, new: Array
) -> Array:
    return jnp.maximum(old, new)


def _lowest_like(x: Array) -> Array:
    """Fill with the dtype's minimum for integers and ``-inf`` otherwise."""
    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.full_like(x, fill_value=jnp.iinfo(x.dtype).min)
    return jnp.full_like(x, fill_value=-jnp.inf)


LAST_FAILURE: Final[LoopMerge] = LoopMerge(jnp.empty_like, _keep_last_failure)
"""Report the arguments of the last failing iteration (``scan`` default)."""

FIRST_FAILURE: Final[LoopMerge] = LoopMerge(jnp.empty_like, _keep_first_failure)
"""Report the arguments of the first failing iteration."""

ELEMENTWISE_MAX: Final[LoopMerge] = LoopMerge(_lowest_like, _keep_largest)
"""Report the elementwise maximum over all iterations (``while_loop`` default).

Suits fixes that must cover every iteration, such as capacity requirements.
"""


@dataclass
class RuntimeAssertion[State, FixArgs]:
    """
    A runtime assertion that validates computations with optional automatic fixing.

    This class encapsulates a predicate that should evaluate to True, along with
    metadata for error reporting and optional repair mechanisms. Assertions are
    designed to work seamlessly with JAX transformations while providing rich
    debugging information.

    Type Parameters:
        State: The type of state that can be modified by the fix function
        FixArgs: The types of arguments passed to the fix function (can be a PyTree)

    Attributes:
        predicate: A scalar boolean array indicating whether the assertion passes
        message: Human-readable error message with optional format placeholders
        fmt_args: Dictionary of values to substitute into the message format string
        exception_type: Type of exception to raise on assertion failure
        static_info: Additional metadata for debugging (not traced by JAX)
        fix_fn: Optional function to repair the state when assertion fails
        fix_args: Arguments to pass to the fix function (can be complex PyTree structures)
        loop_merge: How loops fold ``fmt_args``/``fix_args`` across iterations;
            ``None`` uses the loop's default (see [LoopMerge][kups.core.assertion.LoopMerge])

    Example:
        ```python
        assertion = RuntimeAssertion(
            predicate=jnp.array(x > 0),
            message="Value must be positive, got {val}",
            fmt_args={"val": x},
            fix_fn=lambda state, threshold: jnp.maximum(state, threshold),
            fix_args=0.1
        )
        ```

    Note:
        The fix_args can be complex PyTree structures including nested dictionaries,
        tuples, and arrays. The assertion system properly handles flattening and
        unflattening these structures during JAX transformations.
    """

    predicate: Array
    message: str = field(static=True)
    fmt_args: dict[str, Array] = field(default_factory=dict)
    exception_type: type[Exception] = field(static=True, default=AssertionError)
    static_info: dict[str, Any] = field(static=True, default_factory=dict)
    fix_fn: Fix[State, FixArgs] | None = field(static=True, default=None)
    fix_args: FixArgs | _NO_ARGS = field(default=NO_ARGS)
    loop_merge: LoopMerge | None = field(static=True, default=None)

    def valid(self) -> bool:
        """Check if the assertion is valid (i.e., passes)."""
        return bool(self.predicate)

    def failed(self) -> bool:
        """Check if the assertion has failed."""
        return not self.valid()

    @override
    def __str__(self) -> str:
        """Return the formatted assertion message."""
        return self.message.format(**self.fmt_args)

    def check(self) -> None:
        """
        Check the assertion and raise an exception if it fails.

        Raises:
            Exception: The configured exception type if the assertion fails
        """
        if not bool(jnp.all(self.predicate)):
            raise self.exception_type(self.message.format(**self.fmt_args))

    @property
    def exception(self) -> Exception:
        """
        Create the exception instance that would be raised on assertion failure.

        Returns:
            An exception instance with the formatted error message
        """
        return self.exception_type(
            self.message.format(**(self.fmt_args | self.static_info))
        )

    def fix(self, state: State) -> State:
        """
        Attempt to fix the assertion failure by modifying the provided state.

        Args:
            state: The current state that needs to be repaired

        Returns:
            The modified state after applying the fix function

        Raises:
            NotImplementedError: If no fix function is available
            AssertionError: If fix arguments are missing when a fix function exists
        """
        if self.fix_fn is None:
            raise self.exception
        assert not isinstance(self.fix_args, _NO_ARGS), (
            "Fix arguments were not provided."
        )
        return self.fix_fn(state, self.fix_args)
