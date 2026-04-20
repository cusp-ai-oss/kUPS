# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Patch system for simulation state modifications.

This module provides a comprehensive system for managing composable state
modifications in simulations. The key components are:

- **[Patch][kups.core.patch.Patch]**: Protocol for modifications to simulation state
- **[Probe][kups.core.patch.Probe]**: Protocol for extracting information from state and patches
- **[IdPatch][kups.core.patch.IdPatch]**: Identity patch that returns state unchanged
- **[ExplicitPatch][kups.core.patch.ExplicitPatch]**: Patch with explicitly stored proposed state
- **[IndexLensPatch][kups.core.patch.IndexLensPatch]**: Patch that modifies state at specific indices via a lens
- **[ComposedPatch][kups.core.patch.ComposedPatch]**: Composition of multiple patches applied sequentially

The patch system allows for composable, type-safe state modifications that
integrate with JAX transformations while maintaining runtime validation.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Self

import jax
from jax import Array

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import Lens, View
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field, tree_map
from kups.core.utils.ops import where_broadcast_last

type Accept = Table[SystemId, Array]


class Patch[State](Protocol):
    """A patch represents a modification to simulation state.

    Patches are composable transformations that accept a state and an acceptance
    array, returning a modified state. The acceptance array controls which
    modifications are applied (useful for Monte Carlo acceptance/rejection).

    When called, takes (state, accept) and returns the modified state with
    updates applied according to the acceptance mask.
    """

    def __call__(self, state: State, accept: Accept, /) -> State: ...


class Probe[State, P: Patch, R](Protocol):
    """Protocol for functions that extract information from state and patches.

    Probes are used to query simulation state and patch information, typically
    for observables, energy calculations, or other diagnostics. They provide
    a typed interface for extracting data during simulation runs.

    When called, takes (state, patch) and returns extracted information of type R.
    """

    def __call__(self, state: State, patch: P, /) -> R: ...


@dataclass
class IdPatch[State](Patch[State]):
    """A patch that does nothing, i.e., returns the state unchanged."""

    def __call__(self, state: State, accept: Accept) -> State:
        return state


@dataclass
class ExplicitPatch[State, T](Patch[State]):
    """A patch that applies a custom function with payload data.

    This patch type provides maximum flexibility by accepting an arbitrary
    function that defines how the state should be modified based on the payload.

    Attributes:
        payload: Data to pass to the apply function
        apply_fn: Function that applies the patch given state, payload, and acceptance
    """

    payload: T
    apply_fn: Callable[[State, T, Accept], State] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        return self.apply_fn(state, self.payload, accept)


@dataclass
class IndexLensPatch[State, T](Patch[State]):
    """A patch that uses a lens to update indexed elements in the state.

    This patch combines [lens][kups.core.lens.Lens]-based state access with indexed updates and
    acceptance masking. It's particularly useful for updating specific
    elements in arrays or nested structures based on particle indices.

    Attributes:
        data: New data values to apply
        mask_idx: A prefix pytree to match against Index leaves in the state; determines which indices to update.
        lens: Lens that focuses on the part of state to modify
    """

    data: T
    mask_idx: Any
    lens: Lens[State, T] = field(static=True)

    def __call__(self, state: State, accept: Accept) -> State:
        def inner(idx, new_val, old_val):
            assert isinstance(idx, Index), "mask_idx must be an Index"
            return tree_map(
                lambda a, b: where_broadcast_last(accept[idx], a, b), new_val, old_val
            )

        result = jax.tree.map(
            inner,
            self.mask_idx,
            self.data,
            self.lens.get(state),
            is_leaf=lambda x: isinstance(x, Index),
        )
        return self.lens.set(state, result)


@dataclass
class ComposedPatch[State](Patch[State]):
    """A patch that composes multiple patches together by applying them in sequence."""

    patches: tuple[Patch[State], ...]

    def __call__(self, state: State, accept: Accept) -> State:
        assert len(self.patches) > 0, "No patches provided"
        result = self.patches[0](state, accept)
        for patch in self.patches[1:]:
            result = patch(result, accept)
        return result


class Addable(Protocol):
    """Protocol for types that support addition."""

    def __add__(self, other: Self) -> Self: ...


@dataclass
class WithPatch[T_Data, T_Patch: Patch]:
    """Generic wrapper pairing data with a patch.

    This class provides a unified pattern for operations that return both
    a result (data) and a side-effect (patch).

    Type Parameters:
        T_Data: The data type being wrapped (e.g., Energy, PotentialOut, Array)
        T_Patch: The patch type (must satisfy the Patch protocol)

    Attributes:
        data: The primary data result
        patch: The patch to apply

    Example:
        ```python
        result: WithPatch[PotentialOut[Grads, Hess], Patch[State]] = potential(state)
        energies = result.data.total_energies
        new_state = result.patch(state, accept)
        ```
    """

    data: T_Data
    patch: T_Patch

    def map_data[T](self, view: View[T_Data, T]) -> WithPatch[T, T_Patch]:
        """Transform data while preserving patch.

        Args:
            view: Function to transform the data

        Returns:
            New WithPatch with transformed data and same patch
        """
        return WithPatch(view(self.data), self.patch)

    def map_patch[P: Patch](self, view: View[T_Patch, P]) -> WithPatch[T_Data, P]:
        """Transform patch while preserving data.

        Args:
            view: Function to transform the patch

        Returns:
            New WithPatch with same data and transformed patch
        """
        return WithPatch(self.data, view(self.patch))

    def compose_patch(self, other: Patch) -> WithPatch[T_Data, ComposedPatch]:
        """Compose another patch after this one.

        Args:
            other: Patch to compose after this one

        Returns:
            New WithPatch with composed patch
        """
        return WithPatch(self.data, ComposedPatch((self.patch, other)))

    def __add__[D: Addable, P1: Patch, P2: Patch](
        self: WithPatch[D, P1], other: WithPatch[D, P2]
    ) -> WithPatch[D, ComposedPatch]:
        """Add two WithPatch instances by adding data and composing patches.

        Requires T_Data to support __add__. Returns WithPatch with summed data
        and patches composed in sequence.

        Args:
            other: Another WithPatch with compatible data type

        Returns:
            New WithPatch with summed data and composed patches
        """
        return WithPatch(
            self.data + other.data, ComposedPatch((self.patch, other.patch))
        )
