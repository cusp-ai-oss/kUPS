# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Dynamic capacity management with automatic resizing for fixed-size arrays.

This module provides utilities for managing array capacities in JAX computations
where the required size may change dynamically. The system automatically detects
capacity violations and can resize arrays using a growth strategy (typically
powers of 2) to amortize allocation costs.

Capacity sizes can be scalar integers or arbitrary pytrees of integers (e.g.,
tuples, dicts) for tracking multiple independent capacities that resize
individually.

Key features:

- Scalar or pytree capacity sizes for independent tracking
- Automatic capacity detection and assertion generation
- Configurable growth strategies (default: exponential with base 2)
- Integration with [runtime assertion system][kups.core.assertion] for automatic fixing
- Lens-based state modification for type-safe resizing
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Protocol

import jax.core
import jax.numpy as jnp
from jax import Array

from kups.core.assertion import Fix, runtime_assert
from kups.core.lens import Lens, bind
from kups.core.utils.jax import dataclass, field, tree_map
from kups.core.utils.math import next_higher_power


class CapacityError(ValueError):
    """Exception raised when array capacity is exceeded.

    This error is raised when attempting to store more elements than the current
    capacity allows. When used with the runtime assertion system, this error can
    trigger automatic resizing.
    """


class Capacity[Value](Protocol):
    """Protocol defining the interface for capacity management.

    Implementations track array capacity and generate runtime assertions
    that can automatically resize arrays when capacity is exceeded.

    Type Parameters:
        Value: The capacity size type - either int or a pytree of ints
    """

    @property
    def size(self) -> Value:
        """Current capacity as scalar int or pytree of ints."""
        ...

    def generate_assertion(self, required_capacity: Array) -> Capacity[Value]:
        """Generate a runtime assertion that checks and fixes capacity violations.

        Args:
            required_capacity: The minimum capacity needed

        Returns:
            Updated Capacity with potentially increased size
        """
        ...

    def multiply(self, factor: int) -> Capacity[Value]:
        """Create a scaled view of this capacity.

        Args:
            factor: Multiplier for the capacity size

        Returns:
            A new Capacity with effective size = self.size * factor
        """
        ...


@dataclass
class FixedCapacity[Value]:
    """Static capacity that asserts without automatic resizing.

    Unlike ``LensCapacity``, a ``FixedCapacity`` cannot grow automatically
    because it has no lens into the state. Its ``generate_assertion`` emits
    a runtime assertion that raises ``CapacityError`` when the required
    capacity exceeds the current size, but does not attach a fix function.

    Type Parameters:
        Value: The capacity size type -- either int or a pytree of ints.

    Attributes:
        size: Current capacity as int or pytree of ints.
        error_msg: Optional message appended to the assertion error.
    """

    size: Value = field(static=True)
    error_msg: str = field(static=True, default="")

    def generate_assertion(self, required_capacity: Array) -> FixedCapacity[Value]:
        # As new capacity we set the closest power of `base` that is >= required_capacity.
        # If the base is smaller than or equal to 1, we just set the new capacity to required_capacity.
        required_capacity = jnp.ceil(required_capacity).astype(int)
        sizes, tree_def = jax.tree.flatten(self.size)
        size = jnp.asarray(sizes)

        # Handle the scalar case
        if (
            required_capacity.ndim == 0 or required_capacity.shape[-1] != size.shape[-1]
        ) and size.shape[-1] == 1:
            required_capacity = required_capacity[..., None]

        max_fn = functools.partial(
            jnp.max, axis=list(range(0, required_capacity.ndim - 1))
        )
        runtime_assert(
            (required_capacity <= size).all(),
            f"Insufficient capacity: {{affected}} > {self.size}.{f'\n{self.error_msg}' if self.error_msg else ''}",
            fmt_args=dict(affected=max_fn(required_capacity)),
            exception_type=CapacityError,
        )
        return self

    def multiply(self, factor: int) -> FixedCapacity[Value]:
        return FixedCapacity(tree_map(lambda x: x * factor, self.size), self.error_msg)


@dataclass
class LensCapacity[State, Value]:
    """Lens-based implementation of the Capacity protocol.

    Manages dynamic capacity with automatic resizing for fixed-size arrays.
    Uses a lens to update the capacity value within the state, enabling
    integration with the runtime assertion system for automatic resizing.

    Type Parameters:
        State: The type of simulation state containing the capacity value.
        Value: The capacity size type -- either int or a pytree of ints.

    Attributes:
        size: Current capacity as int or pytree of ints (e.g., tuple, dict).
        size_lens: Lens focusing on the capacity *value* within the state.
        base: Growth factor for exponential resizing (default: 2.0).

    Example:
        Scalar capacity:
        ```python
        capacity = LensCapacity(size=100, size_lens=lens(lambda s: s.avg_edges))
        capacity = capacity.generate_assertion(required_capacity=150)
        # Resizes to next power of 2: 256
        ```

        Pytree capacity for independent tracking:
        ```python
        capacity = LensCapacity(
            size={"atoms": 100, "bonds": 50},
            size_lens=lens(lambda s: s.capacity),
        )
        # Requirements array matches flattened pytree (sorted keys: atoms, bonds)
        capacity = capacity.generate_assertion(jnp.array([150, 40]))
        # Only atoms resizes: {"atoms": 256, "bonds": 50}
        ```
    """

    size: Value = field(static=True)
    size_lens: Lens[State, Value] = field(static=True)
    base: float = field(static=True, default=2.0)

    def generate_assertion(
        self, required_capacity: Array
    ) -> LensCapacity[State, Value]:
        """Generate a runtime assertion that checks and fixes capacity violations.

        This method creates an assertion that validates whether the current capacity
        is sufficient for the required size. If not, it generates a fix that resizes
        to the next appropriate capacity based on the growth strategy.

        For pytree sizes, each element is checked and resized independently.
        The required_capacity array's last dimension should match the flattened
        pytree length. Batch dimensions are reduced via max.

        Growth strategy:
        - If `base > 1`: Grows to the smallest power of `base` ≥ required_capacity
        - If `base ≤ 1`: Grows exactly to required_capacity (linear growth)

        Args:
            required_capacity: The minimum capacity needed. For pytree sizes,
                shape should be `(..., n)` where n matches flattened pytree length.

        Returns:
            Updated Capacity object with potentially increased size (if not traced)

        Note:
            During JAX tracing, returns self unchanged. The actual resize happens
            when the assertion fix is applied to the state.
        """
        # As new capacity we set the closest power of `base` that is >= required_capacity.
        # If the base is smaller than or equal to 1, we just set the new capacity to required_capacity.
        required_capacity = jnp.ceil(required_capacity).astype(int)
        sizes, tree_def = jax.tree.flatten(self.size)
        size = jnp.asarray(sizes)

        # Handle the scalar case
        if (
            required_capacity.ndim == 0 or required_capacity.shape[-1] != size.shape[-1]
        ) and size.shape[-1] == 1:
            required_capacity = required_capacity[..., None]

        new_capacity = next_higher_power(required_capacity, self.base)
        new_capacity = jnp.where(required_capacity <= size, size, new_capacity)

        max_fn = functools.partial(
            jnp.max, axis=list(range(0, required_capacity.ndim - 1))
        )
        runtime_assert(
            (required_capacity <= size).all(),
            f"Insufficient capacity: {{affected}} > {self.size}.",
            fmt_args=dict(affected=max_fn(required_capacity)),
            exception_type=CapacityError,
            fix_fn=LensCapacityFix(self.size_lens),
            fix_args=new_capacity,
        )
        if isinstance(new_capacity, jax.core.Tracer):
            return self
        new_size = jax.tree.unflatten(
            tree_def, max_fn(new_capacity).astype(int).tolist()
        )
        return bind(self).focus(lambda c: c.size).set(new_size)

    def multiply(self, factor: int) -> Capacity[Value]:
        return MultipliedCapacity(self, factor)


@dataclass
class MultipliedCapacity[Value]:
    """A scaled view of another Capacity.

    Wraps a base capacity and multiplies its effective size by a constant factor.
    When checking capacity, the required amount is divided by the factor before
    delegating to the base capacity, enabling capacity sharing across related arrays.

    For pytree sizes, each element is multiplied by the factor independently.

    Type Parameters:
        Value: The capacity size type - either int or a pytree of ints

    Attributes:
        base_capacity: The underlying capacity to scale
        factor: Multiplier applied to each element of the base capacity's size

    Example:
        ```python
        # Scalar: base size=100, factor=3 -> effective size=300
        position_capacity = base_capacity.multiply(3)

        # Pytree: base size=(10, 20), factor=2 -> effective size=(20, 40)
        scaled = LensCapacity(size=(10, 20), ...).multiply(2)
        ```
    """

    base_capacity: Capacity[Value]
    factor: int = field(static=True)

    def multiply(self, factor: int) -> Capacity[Value]:
        """Create a further scaled view of this capacity."""
        return MultipliedCapacity(self, factor)

    @property
    def size(self) -> Value:
        """Effective capacity: each element of base_capacity.size * factor."""
        return tree_map(lambda x: x * self.factor, self.base_capacity.size)

    def generate_assertion(self, required_capacity: Array) -> Capacity[Value]:
        """Generate assertion by delegating to base with scaled requirement."""
        return MultipliedCapacity(
            self.base_capacity.generate_assertion(required_capacity / self.factor),
            self.factor,
        )


@dataclass
class LensCapacityFix[State, Value]:
    """Fix function for automatically resizing capacity in the state.

    This callable fix function is used by the runtime assertion system to
    automatically resize arrays when capacity is exceeded. It updates the
    capacity field in the state using the provided lens.

    For pytree sizes, each element is updated independently, taking the max
    of the current value and the target to ensure capacity never decreases.

    Type Parameters:
        State: The type of simulation state to modify
        Value: The capacity size type - either int or a pytree of ints

    Attributes:
        lens: Lens focusing on the capacity value within the state

    Example:
        Used internally by Capacity.generate_assertion(), but can be used directly:
        ```python
        fix = CapacityFix(lens=lens(lambda s: s.capacity))
        new_state = fix(state, jnp.array(256))  # Resize to capacity 256

        # For pytree capacity:
        new_state = fix(state, jnp.array([150, 80]))  # Updates each element
        ```
    """

    lens: Lens[State, Value] = field(static=True)

    def __call__(self, state: State, target_capacity: Array) -> State:
        """Apply the capacity fix by updating the state.

        For batched targets, reduces via max over batch dimensions. Each pytree
        element is set to max(current, target) to ensure capacity never decreases.

        Args:
            state: Current simulation state
            target_capacity: New capacity. Shape `(..., n)` where n matches
                flattened pytree length. Batch dims reduced via max.

        Returns:
            State with updated capacity
        """
        sizes, treedef = jax.tree.flatten(self.lens.get(state))
        new_sizes = (
            target_capacity.max(list(range(target_capacity.ndim - 1)))
            .astype(int)
            .tolist()
        )
        new_sizes = jax.tree.unflatten(treedef, list(map(max, sizes, new_sizes)))
        return self.lens.set(state, new_sizes)


if TYPE_CHECKING:

    def _[S, V](
        a: LensCapacityFix[S, V],
        c: LensCapacity[S, V],
        c2: MultipliedCapacity[V],
    ):
        _: Fix[S, Array] = a
        __: Capacity[V] = c
        ___: Capacity[V] = c2
