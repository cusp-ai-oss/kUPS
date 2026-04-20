# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Lens library for functional data manipulation in JAX.

This module provides a lens-based approach to accessing and modifying nested data structures
in a functional manner. Lenses allow you to focus on specific parts of a data structure and
perform operations without mutating the original structure.

The main abstractions are:

- **[View][kups.core.lens.View]**: A function that extracts a value from a data structure
- **[Update][kups.core.lens.Update]**: A function that sets a value in a data structure
- **[Lens][kups.core.lens.Lens]**: A bidirectional interface for getting and setting values
- **[BoundLens][kups.core.lens.BoundLens]**: A lens that has been bound to a specific data structure

Lenses satisfy both the View and Update protocols through their `__call__` method:

- `lens(state)` acts as a View, returning the focused value
- `lens(state, value)` acts as an Update, returning the modified structure

This allows lenses to be used directly wherever a View or Update function is expected.
"""

from __future__ import annotations

import abc
import copy
import enum
import functools as ft
import inspect
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import is_dataclass, replace
from functools import partial
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    overload,
    override,
    runtime_checkable,
)

import jax
from jax import Array
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

from kups.core.utils.field_handler import FieldMetaAccess, register_field_handler
from kups.core.utils.functools import flip, identity, pipe
from kups.core.utils.jax import (
    HasScatterArgs,
    PyTreeDef,
    ScatterArgs,
    dataclass,
    field,
    no_post_init,
    tree_map,
    tree_scatter_set,
)


class View[S, R](Protocol):
    """Protocol for a view function that extracts a value from a data structure.

    A view is a read-only operation that focuses on a specific part of a data structure.

    Type Parameters:
        S: The type of the source data structure
        R: The type of the result value
    """

    def __call__(self, state: S, /) -> R: ...


@dataclass
class _View[S, R]:
    """Concrete implementation of the View protocol.

    Wraps a callable function to provide a view into a data structure.
    """

    where: Callable[[S], R] = field(static=True)

    def __call__[S2](self: _View[S2, R], state: S2) -> R:
        return self.where(state)

    def __repr__(self) -> str:
        return f"View(where={self.where})"


def view[S, R](where: Callable[[S], R], /, cls: type[S] | None = None) -> View[S, R]:
    """Create a view from a callable function.

    Args:
        where: A function that extracts a value from the data structure

    Returns:
        A View instance that wraps the provided function
    """
    return _View(where)


class Update[S, R](Protocol):
    """Protocol for an update function that sets a value in a data structure.

    Type Parameters:
        S: The type of the source data structure
        R: The type of the value being set
    """

    def __call__[S2](self: Update[S2, R], state: S2, /, value: R) -> S2: ...


type Modifier[R] = Callable[[R], R]
"""Type alias for a function that transforms a value."""


class _NOT_SET_TYPE(enum.Enum):
    OBJ = enum.auto()


_NOT_SET = _NOT_SET_TYPE.OBJ


S = TypeVar("S", covariant=True)
R = TypeVar("R")
R2 = TypeVar("R2")


@runtime_checkable
class Lens(Protocol[S, R]):
    """Protocol for a lens that provides bidirectional access to data structures.

    A lens combines a getter and setter, allowing functional access and modification
    of nested data structures. Lenses are composable and can be focused on specific
    parts of a data structure.

    Lenses satisfy both the View and Update protocols through their `__call__` method,
    allowing them to be used directly wherever a view or update function is expected.

    Generic in ``S`` (source data structure) and ``R`` (focused value).

    Examples:
        >>> state = MyState(value=10)
        >>> my_lens = lens(lambda s: s.value)
        >>> my_lens(state)  # View: returns 10
        >>> my_lens(state, 20)  # Update: returns MyState(value=20)
    """

    @overload
    def __call__[S2](self: Lens[S2, R], state: S2, /, value: R) -> S2: ...

    @overload
    def __call__[S2](self: Lens[S2, R], state: S2, /) -> R: ...

    def __call__[S2](
        self: Lens[S2, R], state: S2, /, value: R | _NOT_SET_TYPE = _NOT_SET
    ) -> S2 | R:
        """Get or set the focused value, satisfying View and Update protocols.

        When called with one argument, acts as a View and returns the focused value.
        When called with two arguments, acts as an Update and returns the modified state.

        Args:
            state: The data structure to operate on
            value: If provided, the new value to set

        Returns:
            The focused value (one arg) or modified state (two args)
        """
        ...

    def focus[B](self, where: Callable[[R], B]) -> Lens[S, B]:
        """Focus this lens on a deeper part of the data structure.

        Args:
            where: A function that extracts a value from the current focus

        Returns:
            A new lens focused on the result of the where function
        """
        ...

    def get[S2](self: Lens[S2, R], state: S2, /) -> R:
        """Extract the focused value from the data structure.

        Args:
            state: The data structure to extract from

        Returns:
            The focused value
        """
        ...

    def set[S2](self: Lens[S2, R], state: S2, /, value: R) -> S2:
        """Set the focused value in the data structure.

        Args:
            state: The data structure to modify
            value: The new value to set

        Returns:
            A new data structure with the value set
        """
        ...

    def apply[S2](self: Lens[S2, R], state: S2, /, modifier: Modifier[R]) -> S2:
        """Apply a modifier function to the focused value.

        Args:
            state: The data structure to modify
            modifier: A function that transforms the focused value

        Returns:
            A new data structure with the modified value
        """
        ...

    def bind[S2](self: Lens[S2, R], state: S2, /) -> BoundLens[S2, R]:
        """Bind this lens to a specific data structure.

        Args:
            state: The data structure to bind to

        Returns:
            A bound lens that operates on the given state
        """
        ...

    def at(self, idxs: Any, *, args: ScatterArgs | None = None) -> Lens[S, R]:
        """Create a lens that slices the focused pytree at the given indices.

        Functionally equivalent to composing jax.tree.map and jax.array.at[idxs].get/set.

        Args:
            idxs: The indices to slice the pytree at.
            args: Optional ScatterArgs controlling scatter behavior (mode,
                wrap_negative_indices, fill_value, indices_are_sorted, unique_indices).

        Returns:
            A new lens that slices the focused pytree at the given indices.
        """
        ...

    def merge[S2, R2](self: Lens[S2, R], other: Lens[S2, R2]) -> Lens[S2, tuple[R, R2]]:
        """Merge this lens with another lens to access multiple values.

        Args:
            other: Another lens to merge with

        Returns:
            A new lens that accesses both focused values as a tuple
        """
        ...

    def nest[U](self, other: Lens[R, U]) -> Lens[S, U]:
        """Nest another lens or view within this lens.

        This provides an alternative to focus() that works with both lenses and views.
        When given a lens, it extracts the view from it; when given a view directly,
        it uses it as-is.

        Args:
            other: Either a lens or view to nest within this lens

        Returns:
            A new lens that composes this lens with the provided lens/view
        """
        ...


class BoundLens(Protocol[S, R]):
    """Protocol for a lens that has been bound to a specific data structure.

    A bound lens provides the same operations as a regular lens but without
    requiring the state parameter, since it's already bound to a specific instance.

    Bound lenses satisfy View (via zero-argument call) and can update via single-argument
    call, providing a convenient interface for repeated operations on the same state.

    Generic in ``S`` (bound data structure) and ``R`` (focused value).

    Examples:
        >>> bound = my_lens.bind(state)
        >>> bound()  # View: returns focused value
        >>> bound(new_value)  # Update: returns modified state
    """

    @overload
    def __call__(self, value: R) -> S: ...

    @overload
    def __call__(self) -> R: ...

    def __call__(self, value: R | _NOT_SET_TYPE = _NOT_SET) -> S | R:
        """Get or set the focused value in the bound state.

        When called with no arguments, returns the focused value.
        When called with one argument, sets the value and returns the modified state.

        Args:
            value: If provided, the new value to set

        Returns:
            The focused value (no args) or modified state (one arg)
        """
        ...

    def focus[B](self, where: Callable[[R], B]) -> BoundLens[S, B]:
        """Focus this bound lens on a deeper part of the data structure.

        Args:
            where: A function that extracts a value from the current focus

        Returns:
            A new bound lens focused on the result of the where function
        """
        ...

    def get(self) -> R:
        """Extract the focused value from the bound data structure.

        Returns:
            The focused value from the bound state
        """
        ...

    def set(self, value: R) -> S:
        """Set the focused value in the bound data structure.

        Args:
            value: The new value to set

        Returns:
            A new data structure with the value set
        """
        ...

    def apply(self, modifier: Modifier[R]) -> S:
        """Apply a modifier function to the focused value in the bound data structure.

        Args:
            modifier: A function that transforms the focused value

        Returns:
            A new data structure with the modified value set
        """
        ...

    def at(self, idxs: Any, *, args: ScatterArgs | None = None) -> BoundLens[S, R]:
        """Create a bound lens that slices the focused pytree at the given indices.

        Functionally equivalent to composing jax.tree.map and jax.array.at[idxs].get/set
        on the bound data structure.

        Args:
            idxs: The indices to slice the pytree at.
            args: Optional ScatterArgs controlling scatter behavior (mode,
                wrap_negative_indices, fill_value, indices_are_sorted, unique_indices).

        Returns:
            A new bound lens that slices the focused pytree at the given indices.
        """
        ...

    def merge[S2, R2](
        self: BoundLens[S2, R], other: Lens[S2, R2]
    ) -> BoundLens[S2, tuple[R, R2]]:
        """Merge this lens with another lens to access multiple values.

        Args:
            other: Another lens to merge with

        Returns:
            A new lens that accesses both focused values as a tuple
        """
        ...

    def nest[U](self, other: Lens[R, U]) -> BoundLens[S, U]:
        """Nest another lens or view within this bound lens.

        This provides an alternative to focus() that works with both lenses and views.
        When given a lens, it extracts the view from it; when given a view directly,
        it uses it as-is.

        Args:
            other: Either a lens or view to nest within this lens

        Returns:
            A new bound lens that composes this lens with the provided lens/view
        """
        ...


class BaseLens(Lens[S, R]):
    """Base class for lens implementations."""

    def focus[B](self, where: Callable[[R], B]) -> Lens[S, B]:
        return NestedLens(self, SimpleLens(view(where)))

    def apply[S2](self: BaseLens[S2, R], state: S2, modifier: Modifier[R]) -> S2:
        """Apply a modifier function to the focused value."""
        return self.set(state, modifier(self.get(state)))

    def bind[S2](self: BaseLens[S2, R], state: S2) -> BoundLens[S2, R]:
        """Bind this lens to a specific data structure."""
        return SimpleBoundLens(self, state)

    def at(self, idxs, *, args: ScatterArgs | None = None) -> Lens[S, R]:
        """Create a lens that slices the focused pytree at the given indices."""
        return IndexLens(self, idxs, args=args if args is not None else {})

    def merge[S2, R2](
        self: BaseLens[S2, R], other: Lens[S2, R2]
    ) -> Lens[S2, tuple[R, R2]]:
        return MergedLens(self, other)

    def nest[U](self, other: Lens[R, U]) -> Lens[S, U]:
        view_func = other.get if isinstance(other, Lens) else other
        return self.focus(view_func)

    @overload
    def __call__[S2](self: Lens[S2, R], state: S2, /, value: R) -> S2: ...

    @overload
    def __call__[S2](
        self: Lens[S2, R], state: S2, /, value: _NOT_SET_TYPE = _NOT_SET
    ) -> R: ...

    def __call__[S2](
        self: Lens[S2, R], state: S2, /, value: R | _NOT_SET_TYPE = _NOT_SET
    ) -> S2 | R:
        if value is _NOT_SET:
            return self.get(state)
        else:
            return self.set(state, value)


@dataclass
class SimpleLens(BaseLens[S, R]):
    """A simple lens implementation that uses traversal-based setting.

    This is the most basic lens implementation that works with any pytree structure
    supported by JAX.
    """

    _get: View[S, R] = field(static=True)

    @override
    def focus[B](self, where: Callable[[R], B]) -> Lens[S, B]:
        """Focus this lens on a deeper part of the data structure."""
        return SimpleLens(pipe(self._get, where))

    def get[S2](self: SimpleLens[S2, R], state: S2) -> R:
        """Get the focused value using the provided view function."""
        return self._get(state)

    def set[S2](self: SimpleLens[S2, R], state: S2, value: R) -> S2:
        """Set the focused value using traversal lens."""
        try:
            return _traversal_lens(self.get, cls=type(state)).set(state, value)
        except _LensTraversalError as e:
            raise ValueError(
                f"Cannot set value through this lens: {e}\n"
                "Hint: The focus function must return references to parts of the data, "
                "not computed values or literals."
            ) from e


@dataclass
class ConstLens(BaseLens[S, R]):
    """Lens that always returns a constant value; set is a no-op."""

    value: R

    def get[S2](self: ConstLens[S2, R], state: S2) -> R:
        return self.value

    def set[S2](self: ConstLens[S2, R], state: S2, value: R) -> S2:
        return state


def const_lens[R](value: R) -> ConstLens[Any, R]:
    """Create a lens that always returns the same value, ignoring input."""
    return ConstLens(value)


@dataclass
class MergedLens(BaseLens[S, tuple[R, R2]]):
    """A lens that merges two lenses to access multiple values.

    This lens combines two lenses into a single lens that accesses both focused values
    as a tuple. It allows you to work with multiple parts of a data structure simultaneously.
    """

    left: Lens[S, R] = field(static=True)
    right: Lens[S, R2] = field(static=True)

    def get[S2](self: MergedLens[S2, R, R2], state: S2) -> tuple[R, R2]:
        return self.left.get(state), self.right.get(state)

    def set[S2](self: MergedLens[S2, R, R2], state: S2, value: tuple[R, R2]) -> S2:
        state = self.left.set(state, value[0])
        return self.right.set(state, value[1])


@dataclass
class NestedLens(BaseLens[S, R2], Generic[S, R, R2]):
    """A lens that composes two lenses to access deeply nested data.

    This lens combines an outer lens (S -> A) with an inner lens (A -> B)
    to create a composite lens (S -> B). Operations are performed by first
    applying the outer lens, then the inner lens.
    """

    outer: Lens[S, R] = field(static=True)
    inner: Lens[R, R2] = field(static=True)

    def get[S2](self: NestedLens[S2, R, R2], state: S2) -> R2:
        """Get value by applying outer lens then inner lens."""
        return self.inner.get(self.outer.get(state))

    def set[S2](self: NestedLens[S2, R, R2], state: S2, value: R2) -> S2:
        """Set value by getting outer value, setting inner value, then setting outer."""
        inner = self.inner.set(self.outer.get(state), value)
        return self.outer.set(state, inner)


@dataclass
class SimpleBoundLens(BoundLens[S, R]):
    """A lens that has been bound to a specific data structure instance.

    This provides a convenient interface for repeatedly operating on the same
    data structure without having to pass it as a parameter each time.
    """

    lens: Lens[S, R] = field(static=True)
    target: S

    def focus[B](self, where: Callable[[R], B]) -> BoundLens[S, B]:
        """Focus deeper and maintain the binding to the same target."""
        return self.lens.focus(where).bind(self.target)

    def get(self) -> R:
        """Get the focused value from the bound target."""
        return self.lens.get(self.target)

    def set(self, value: R) -> S:
        """Set the focused value in the bound target."""
        return self.lens.set(self.target, value)

    def apply(self, modifier: Modifier[R]) -> S:
        """Apply a modifier to the focused value in the bound target."""
        return self.lens.set(self.target, modifier(self.lens.get(self.target)))

    def at(self, idxs, *, args: ScatterArgs | None = None) -> BoundLens[S, R]:
        """Create a slicing bound lens for the same target."""
        return self.lens.at(idxs, args=args).bind(self.target)

    def merge[S2, R2](
        self: SimpleBoundLens[S2, R], other: Lens[S2, R2]
    ) -> BoundLens[S2, tuple[R, R2]]:
        """Merge this bound lens with another lens to access multiple values."""
        return self.lens.merge(other).bind(self.target)

    def nest[U](self, other: Lens[R, U]) -> BoundLens[S, U]:
        """Nest another lens or view within this bound lens."""
        return self.lens.nest(other).bind(self.target)

    @overload
    def __call__(self, value: R) -> S: ...

    @overload
    def __call__(self, value: _NOT_SET_TYPE = _NOT_SET) -> R: ...

    def __call__(self, value: R | _NOT_SET_TYPE = _NOT_SET) -> S | R:
        if value is _NOT_SET:
            return self.get()
        else:
            return self.set(value)


@dataclass
class LambdaLens(BaseLens[S, R]):
    """A lens that uses custom getter and setter functions.

    This allows for more complex lens behavior that cannot be expressed
    with simple field access or traversal-based operations.
    """

    _get: View[S, R] = field(static=True)
    _set: Update[S, R] = field(static=True)

    def get[S2](self: LambdaLens[S2, R], state: S2) -> R:
        """Get the focused value using the custom getter function."""
        return self._get(state)

    def set[S2](self: LambdaLens[S2, R], state: S2, value: R) -> S2:
        """Set the focused value using the custom setter function."""
        return self._set(state, value)


@dataclass
class IndexLens(BaseLens[S, R]):
    """A lens that performs array indexing operations on the focused data.

    This lens wraps another lens and applies JAX array indexing operations
    to slice, index, or select specific elements from arrays in the focused data.
    """

    lens: Lens[S, R] = field(static=True)
    idxs: Array
    args: ScatterArgs = field(default_factory=lambda: {}, static=True)

    def focus[B](self, where: Callable[[R], B]) -> Lens[S, B]:
        """Focus deeper on the indexed data."""
        raise RuntimeError(
            "IndexLens cannot be focused further. Please reorder your lenses."
        )

    def get[S2](self: IndexLens[S2, R], state: S2) -> R:
        """Get values by applying array indexing to the focused data."""

        def _array_getter(scatter_args: ScatterArgs, arr: Array):
            return arr.at[self.idxs].get(**scatter_args)

        def _getter(arr: Array | HasScatterArgs):
            if isinstance(arr, Array):
                return _array_getter(self.args, arr)
            return tree_map(
                partial(_array_getter, {**arr.scatter_args, **self.args}), arr
            )

        return tree_map(
            _getter,
            self.lens.get(state),
            is_leaf=lambda x: isinstance(x, (Array, HasScatterArgs)),
        )

    def set[S2](self: IndexLens[S2, R], state: S2, value: R) -> S2:
        """Set values by applying array indexing to the focused data."""
        return self.lens.set(
            state, tree_scatter_set(self.lens.get(state), value, self.idxs, self.args)
        )

    def at(self, idxs, **extra_kwargs) -> Lens[S, R]:
        """Create a nested index lens for further slicing."""
        raise RuntimeError(
            "IndexLens cannot be sliced further. Please merge your lenses."
        )


TreeKey = GetAttrKey | DictKey | SequenceKey
TreePath = tuple[TreeKey, ...]


@dataclass
class TreePathView:
    """A view that follows a path of keys/attributes through a pytree."""

    path: TreePath

    def __call__(self, state: Any) -> Any:
        for key in self.path:
            match key:
                case SequenceKey(idx=key) | DictKey(key=key):
                    state = state[key]
                case GetAttrKey(name=key):
                    state = getattr(state, key)
                case _:
                    raise TypeError(f"Unknown path type: {type(key)}")
        return state


def all_where_lens[S, Target](
    obj: S,
    conditional: Callable[[Any], bool],
    *,
    target_cls: type[Target] | None = None,
) -> Lens[S, tuple[Target, ...]]:
    """Create a lens that focuses on all elements in a pytree satisfying a condition.

    Args:
        obj: An instance of the state type to infer the pytree structure
        conditional: A predicate function that tests each element
        target_cls: Optional type hint for the target type (currently unused)

    Returns:
        A lens that focuses on all elements satisfying the condition as a tuple.
    """
    leaves = jax.tree.leaves_with_path(obj, is_leaf=conditional)
    paths = [p for p, obj in leaves if conditional(obj)]

    def getter(state: S) -> tuple[Target, ...]:
        return tuple(f(state) for f in map(TreePathView, paths))

    return lens(getter)


def all_isinstance_lens[S, Target](
    obj: S, cls: type[Target]
) -> Lens[S, tuple[Target, ...]]:
    """Create a lens that focuses on all elements in a pytree of a specific type.

    Args:
        obj: An instance of the state type to infer the pytree structure
        cls: The type to filter elements by

    Returns:
        A lens that focuses on all elements of the specified type as a tuple.
    """
    return all_where_lens(obj, lambda x: isinstance(x, cls), target_cls=cls)


def _traversal_lens[S, R](
    where: Callable[[S], R], *, cls: type[S] | None = None
) -> Lens[S, R]:
    lifted = _lift_to_traversal(where)

    def getter(state: S) -> R:
        return _untyped_tree_lens_from_tree_traversal(
            lifted(_PathTraversal(obj=state))
        ).get(state)

    def setter(state: S, value: R) -> S:
        return _untyped_tree_lens_from_tree_traversal(
            lifted(_PathTraversal(obj=state))
        ).set(state, value)

    return LambdaLens(getter, setter)


def lens[S, R](where: Callable[[S], R], /, *, cls: type[S] | None = None) -> Lens[S, R]:
    """Create a lens from a getter function.

    This is the main factory function for creating lenses from getter functions
    that extract values from a data structure.

    Args:
        where: A function that extracts a value from the data structure
        cls: Optional class type hint (type inference only)

    Returns:
        A SimpleLens that can operate on the data structure

    Examples:
        >>> # Direct lens creation
        >>> position_lens = lens(lambda s: s.position)
        >>> velocity_lens = lens(lambda s: s.velocity)
    """
    _ = cls  # Intentionally unused, kept for API compatibility
    return SimpleLens(view(where))


def identity_lens[S](_cls: type[S], /) -> Lens[S, S]:
    """Create an identity lens for a type.

    An identity lens is a lens that focuses on the entire data structure.
    It's primarily useful as a starting point for composition using .focus().

    Args:
        _cls: The type to create an identity lens for (parameter name prefixed
              with underscore as the value itself is unused, only the type matters)

    Returns:
        A SimpleLens that acts as an identity function on the data structure

    Examples:
        >>> # Create an identity lens and compose it
        >>> State = identity_lens(SimState)
        >>> position_lens = State.focus(lambda s: s.position)
        >>> velocity_lens = State.focus(lambda s: s.velocity)
    """
    return SimpleLens(view(identity))


def update[S, R](
    where: Callable[[S], R], *, cls: type[S] | None = None
) -> Update[S, R]:
    """Create an update function from a getter function.

    Args:
        where: A function that extracts a value from the data structure
        cls: Optional class type hint (type inference only)

    Returns:
        A function that updates a value in the data structure
    """
    return lens(where, cls=cls).set


@overload
def bind[S](obj: S) -> BoundLens[S, S]: ...


@overload
def bind[S, R](obj: S, where: Callable[[S], R]) -> BoundLens[S, R]: ...


def bind[S, R](obj: S, where: Callable[[S], R] | None = None) -> BoundLens[S, R]:
    """Create a bound lens from a getter function and a data structure.

    This is a convenience function that creates a lens and immediately binds
    it to a specific data structure instance.

    Args:
        obj: The data structure to bind the lens to
        where: A function that extracts a value from the data structure

    Returns:
        A BoundLens that operates on the given object
    """
    if where is None:
        return SimpleBoundLens(lens(lambda x: x), obj)  # type: ignore[return-value]
    return SimpleBoundLens(lens(where), obj)


@dataclass
class _AttributeAccess:
    name: str


@dataclass
class _ItemAccess:
    key: Any


_LensPathComponent = _AttributeAccess | _ItemAccess


@dataclass
class _PathTraversal:
    obj: Any
    lens_path: tuple[_LensPathComponent, ...] = ()

    @property
    def current(self) -> Any:
        return _untyped_chained_lens_from_path_traversal(self).get(self.obj)

    def __len__(self) -> int:
        return len(self.current)

    def __iter__(self) -> Iterator[Any]:
        current = self.current
        if isinstance(current, Mapping):
            for key in current:
                yield key
        elif isinstance(current, Sequence):
            for idx in range(len(current)):
                yield self[idx]
        else:
            raise TypeError(f"Cannot iterate over object of type {type(current)}")

    def __contains__(self, item: Any) -> bool:
        return item in self.current

    # Comparison operators
    def __eq__(self, other: Any) -> bool:
        return self.current == other

    def __ne__(self, other: Any) -> bool:
        return self.current != other

    def __lt__(self, other: Any) -> bool:
        return self.current < other

    def __le__(self, other: Any) -> bool:
        return self.current <= other

    def __gt__(self, other: Any) -> bool:
        return self.current > other

    def __ge__(self, other: Any) -> bool:
        return self.current >= other

    # Arithmetic operators
    def __add__(self, other: Any) -> Any:
        return self.current + other

    def __sub__(self, other: Any) -> Any:
        return self.current - other

    def __mul__(self, other: Any) -> Any:
        return self.current * other

    def __truediv__(self, other: Any) -> Any:
        return self.current / other

    def __floordiv__(self, other: Any) -> Any:
        return self.current // other

    def __mod__(self, other: Any) -> Any:
        return self.current % other

    def __pow__(self, other: Any) -> Any:
        return self.current**other

    def __matmul__(self, other: Any) -> Any:
        return self.current @ other

    # Reflected arithmetic operators (for when traversal is on the right)
    def __radd__(self, other: Any) -> Any:
        return other + self.current

    def __rsub__(self, other: Any) -> Any:
        return other - self.current

    def __rmul__(self, other: Any) -> Any:
        return other * self.current

    def __rtruediv__(self, other: Any) -> Any:
        return other / self.current

    def __rfloordiv__(self, other: Any) -> Any:
        return other // self.current

    def __rmod__(self, other: Any) -> Any:
        return other % self.current

    def __rpow__(self, other: Any) -> Any:
        return other**self.current

    def __rmatmul__(self, other: Any) -> Any:
        return other @ self.current

    # Unary operators
    def __neg__(self) -> Any:
        return -self.current

    def __pos__(self) -> Any:
        return +self.current

    def __abs__(self) -> Any:
        return abs(self.current)

    def __invert__(self) -> Any:
        return ~self.current

    # Bitwise operators
    def __and__(self, other: Any) -> Any:
        return self.current & other

    def __or__(self, other: Any) -> Any:
        return self.current | other

    def __xor__(self, other: Any) -> Any:
        return self.current ^ other

    def __lshift__(self, other: Any) -> Any:
        return self.current << other

    def __rshift__(self, other: Any) -> Any:
        return self.current >> other

    # Reflected bitwise operators
    def __rand__(self, other: Any) -> Any:
        return other & self.current

    def __ror__(self, other: Any) -> Any:
        return other | self.current

    def __rxor__(self, other: Any) -> Any:
        return other ^ self.current

    def __rlshift__(self, other: Any) -> Any:
        return other << self.current

    def __rrshift__(self, other: Any) -> Any:
        return other >> self.current

    # Type conversion operators
    def __bool__(self) -> bool:
        return bool(self.current)

    def __int__(self) -> int:
        return int(self.current)

    def __float__(self) -> float:
        return float(self.current)

    def __complex__(self) -> complex:
        return complex(self.current)

    def __index__(self) -> int:
        return self.current.__index__()

    def __hash__(self) -> int:
        return hash(self.current)

    def __getattr__(self, name: str) -> _PathTraversal:
        if name in _IGNORE_ATTRIBUTES:
            return object.__getattribute__(self, name)
        field = inspect.getattr_static(self.current, name)
        if isinstance(field, lens_property):
            return field._fget(self)
        elif isinstance(field, property):
            assert field.fget is not None, "Property must have a getter"
            return field.fget(self)
        else:
            return _PathTraversal(
                obj=self.obj, lens_path=self.lens_path + (_AttributeAccess(name),)
            )

    def __getitem__(self, key: Any) -> _PathTraversal:
        return _PathTraversal(
            obj=self.obj, lens_path=self.lens_path + (_ItemAccess(key),)
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        method = self.current
        path = self.lens_path
        if inspect.ismethod(method):
            # if bound instance method
            if not inspect.isclass(method.__self__):
                path = path[:-1]
                func = method.__func__
                return func(_PathTraversal(self.obj, path), *args, **kwds)
            # if class method
            else:
                return method(*args, **kwds)
        # builtin methods (like dict.items, list.append)
        elif inspect.isbuiltin(method):
            # Special handling for dict methods that should return traversals for setting
            if hasattr(method, "__self__") and isinstance(method.__self__, Mapping):
                method_name = getattr(method, "__name__", "")
                parent_path = path[:-1]  # Path without the method name
                if method_name == "items":
                    # Return (key, traversal_to_value) pairs instead of (key, value)
                    parent = _PathTraversal(self.obj, parent_path)
                    return ((k, parent[k]) for k in method.__self__)
                elif method_name == "values":
                    # Return traversals to values instead of values
                    parent = _PathTraversal(self.obj, parent_path)
                    return (parent[k] for k in method.__self__)
            # For all other builtin methods, call directly
            return method(*args, **kwds)
        # any function
        elif inspect.isfunction(method):
            return method(*args, **kwds)

        # any other callable object (instance with __call__ method)
        elif hasattr(method, "__call__"):
            call_method = method.__call__
            if inspect.ismethod(call_method):
                func = call_method.__func__
                return func(_PathTraversal(self.obj, path), *args, **kwds)
        else:
            raise TypeError(f"Cannot call non-function/method: {method}")


@dataclass
class _TreeTraversal:
    treedef: Any
    children: tuple[_PathTraversal, ...]


# default lens interface provides more than necessary here
# this is only used privately within the implementation here
class _UntypedLens(Protocol):
    def get(self, obj: Any) -> Any: ...

    def set(self, obj: Any, value: Any) -> Any: ...


def _untyped_chained_lens_from_path_traversal(
    traversal: _PathTraversal,
) -> _UntypedLens:
    if len(traversal.lens_path) == 0:
        return _IdentityLens()
    lenses: list[_UntypedLens] = []
    for component in traversal.lens_path:
        if isinstance(component, _AttributeAccess):
            lenses.append(_AttributeLens(component.name))
        elif isinstance(component, _ItemAccess):
            lenses.append(_ItemLens(component.key))
        else:
            raise TypeError(f"Unknown lens path component: {component}")
    if len(lenses) == 1:
        return lenses[0]
    return ft.reduce(_ChainedLens, lenses)


def _untyped_tree_lens_from_tree_traversal(traversal: _TreeTraversal) -> _UntypedLens:
    children = tuple(
        _untyped_chained_lens_from_path_traversal(child) for child in traversal.children
    )
    return _TreeLens(treedef=traversal.treedef, children=children)


@dataclass
class _IdentityLens:
    def get(self, obj: Any) -> Any:
        return obj

    def set(self, obj: Any, value: Any) -> Any:
        return value


@dataclass
class _AttributeLens:
    name: str

    def get(self, obj: Any) -> Any:
        return getattr(obj, self.name)

    def set(self, obj: Any, value: Any) -> Any:
        if is_dataclass(type(obj)):
            return replace(obj, **{self.name: value})
        elif hasattr(obj, "_replace"):
            # NamedTuple support
            return obj._replace(**{self.name: value})
        else:
            new_obj = copy.copy(obj)
            setattr(new_obj, self.name, value)
            return new_obj


@dataclass
class _ItemLens:
    key: Any

    def get(self, obj: Any) -> Any:
        return obj[self.key]

    def set(self, obj: Any, value: Any) -> Any:
        if isinstance(obj, MutableSequence | MutableMapping):
            new_obj = copy.copy(obj)
            new_obj[self.key] = value
            return new_obj
        elif isinstance(obj, Mapping):
            # Try kwargs first (works for most mappings), fall back to positional arg
            new_mapping = {**obj, self.key: value}
            try:
                return obj.__class__(**new_mapping)
            except TypeError:
                # Some mappings like MappingProxyType take a single positional arg
                return obj.__class__(new_mapping)  # type: ignore[call-arg]
        elif isinstance(obj, Sequence):
            # Build list first, then pass to constructor
            items = list(obj)
            items[self.key] = value
            try:
                return obj.__class__(items)  # type: ignore[call-arg]
            except TypeError:
                # Fallback: try unpacking as separate arguments
                return obj.__class__(*items)
        else:
            raise TypeError(f"Unsupported object type for item lens: {type(obj)}")


@dataclass
class _ChainedLens:
    outer: _UntypedLens
    inner: _UntypedLens

    def get(self, obj: Any) -> Any:
        return self.inner.get(self.outer.get(obj))

    def set(self, obj: Any, value: Any) -> Any:
        return self.outer.set(obj, self.inner.set(self.outer.get(obj), value))


@dataclass
class _TreeLens:
    treedef: PyTreeDef
    children: tuple[_UntypedLens, ...]

    def get(self, obj: Any) -> Any:
        leaves = map(lambda child: child.get(obj), self.children)
        return jax.tree.unflatten(self.treedef, leaves)

    def set(self, obj: Any, value: Any) -> Any:
        leaves = self.treedef.flatten_up_to(value)
        for child, leaf in zip(self.children, leaves, strict=True):
            obj = child.set(obj, leaf)
        return obj


def _is_traversal(obj: Any):
    return isinstance(obj, _PathTraversal)


def _lift_to_traversal(
    view: Callable[[Any], Any],
) -> Callable[[_PathTraversal], _TreeTraversal]:
    def wrapper(traversal: _PathTraversal) -> _TreeTraversal:
        # Let user exceptions bubble through naturally
        with no_post_init():
            result = view(traversal)
        leaves, treedef = jax.tree.flatten(result, is_leaf=_is_traversal)
        bad = [leaf for leaf in leaves if not isinstance(leaf, _PathTraversal)]
        if bad:
            types = ", ".join(type(leaf).__name__ for leaf in bad)
            raise _LensTraversalError(
                "Focus function returned a computed value instead of a path into the data. "
                f"Use attribute access (x.field) or indexing (x[i]) to reference data. "
                f"Got: {types}"
            )
        return _TreeTraversal(treedef=treedef, children=tuple(leaves))

    return wrapper


class _LensTraversalError(Exception):
    pass


# Ignore the default attributes of _PathTraversal to avoid conflicts with getattr
_IGNORE_ATTRIBUTES = set(_PathTraversal.__dict__.keys())


class HasLensFields(metaclass=FieldMetaAccess):
    """Base class for dataclasses that support lens-enabled field access.

    Dataclasses inheriting from HasLensFields can use LensField[T] annotations
    to enable dual-mode field access:
    - Class access returns a Lens object for functional operations
    - Instance access returns the field value normally

    This class uses the FieldMetaAccess metaclass to intercept class attribute
    access and provide lens objects when appropriate.

    Example:
        >>> from kups.core.utils.jax import dataclass
        >>> from jax import Array
        >>>
        >>> @dataclass
        ... class State(HasLensFields):
        ...     position: LensField[Array]
        ...     velocity: LensField[Array]
        >>>
        >>> state = State(position=pos, velocity=vel)
        >>> pos_lens = State.position  # Returns Lens[State, Array]
        >>> current_pos = state.position  # Returns the Array value

    Note:
        HasLensFields itself cannot be instantiated. It must be subclassed.
    """

    def __new__(cls, *args, **kwargs):
        if cls is HasLensFields:
            raise TypeError(
                "Can't instantiate abstract class {}".format(HasLensFields.__name__)
            )
        return super(HasLensFields, cls).__new__(cls)


class LensField[B](abc.ABC):
    """Type annotation for lens-enabled fields in dataclasses.

    LensField provides a type-safe way to enable lens access on dataclass fields.
    When a dataclass inherits from HasLensFields, fields annotated with LensField[T]
    can be accessed both as regular attributes and as lenses.

    Type Parameters:
        B: The type of the field value

    Behavior:
        - **Class access** (e.g., `MyClass.field`): Returns a `Lens[MyClass, B]`
          that can be used for functional operations like `get()`, `set()`, `focus()`
        - **Instance access** (e.g., `obj.field`): Returns the actual field value
          of type `B`, behaving like a normal attribute

    Usage:
        For dataclasses deriving from HasLensFields, annotate fields with LensField[T]
        to enable lens access. Regular dataclasses without HasLensFields inheritance
        do not support lens access through field annotations.

    Examples:
        >>> from kups.core.utils.jax import dataclass
        >>> from kups.core.lens import LensField, HasLensFields
        >>> import jax.numpy as jnp
        >>> from jax import Array
        >>>
        >>> @dataclass
        >>> class Point(HasLensFields):
        ...     x: LensField[float]
        ...     y: LensField[Array]
        >>>
        >>> # Instance access - normal field behavior
        >>> point = Point(x=1.0, y=jnp.array([1.0, 2.0, 3.0]))
        >>> point.x  # Returns 1.0
        >>> point.y  # Returns Array([1., 2., 3.])
        >>>
        >>> # Class access - returns a lens
        >>> x_lens = Point.x  # Returns Lens[Point, float]
        >>> x_lens.get(point)  # Returns 1.0
        >>> new_point = x_lens.set(point, 5.0)  # Returns Point(x=5.0, y=...)
        >>>
        >>> # Compose with other lenses
        >>> doubled_x_lens = Point.x.focus(lambda x: x * 2)
        >>> doubled_x_lens.get(point)  # Returns 2.0
        >>>
        >>> # Works in JAX transformations
        >>> @jax.jit
        >>> def increment_y(p: Point) -> Point:
        ...     return Point.y.set(p, p.y + 1.0)

    Notes:
        - Only works with dataclasses that inherit from HasLensFields
        - The metaclass FieldMetaAccess intercepts class attribute access to return lenses
        - Compatible with JAX transformations when used with jax-compatible
          dataclasses (e.g., from `kups.core.utils.jax`)
        - Use `lens_field()` instead of `field()` for type-safe field definitions
          with default values or field options
    """

    def __new__(cls):
        if cls is LensField:
            raise TypeError(
                "Can't instantiate abstract class {}".format(LensField.__name__)
            )
        return super(LensField, cls).__new__(cls)

    @overload
    def __get__[A: HasLensFields](
        self, instance: None, owner: type[A]
    ) -> Lens[A, B]: ...

    @overload
    def __get__(self, instance: Any, owner: type[HasLensFields]) -> B: ...

    @abc.abstractmethod
    def __get__(self, instance, owner) -> Any: ...

    @abc.abstractmethod
    def __set__(self, instance: HasLensFields, value: B): ...

    @abc.abstractmethod
    def __set_name__(self, owner: HasLensFields, name: str) -> None: ...


@register_field_handler(LensField)
def _lens_field_handler(cls, name):
    """Field handler for LensField-annotated fields.

    Creates a Lens that focuses on the specified field of the given class.
    This handler is invoked when accessing a LensField-annotated attribute
    on a class (not an instance).

    Args:
        cls: The class containing the LensField
        name: The name of the field

    Returns:
        A Lens[cls, T] where T is the field type
    """
    return lens(partial(flip(getattr), name))


class lens_property[S, B]:
    """Decorator to create a lens-enabled property.

    This decorator allows properties to behave like LensField, returning a lens when
    accessed on the class and the property value when accessed on an instance.

    Type Parameters:
        B: The type of the property value

    Behavior:
        - **Class access** (e.g., `MyClass.prop`): Returns a `Lens[MyClass, B]`
          that can be used for functional operations like `get()`, `set()`, `focus()`
        - **Instance access** (e.g., `obj.prop`): Returns the property value of type `B`,
          behaving like a normal property

    Examples:
        >>> from kups.core.utils.jax import dataclass
        >>> from kups.core.lens import lens_property, HasLensFields
        >>>
        >>> @dataclass
        >>> class Temperature(HasLensFields):
        ...     _kelvin: float
        ...
        ...     @lens_property
        ...     def kelvin(self) -> float:
        ...         return self._kelvin
        ...
        ...     @lens_property
        ...     def celsius(self) -> float:
        ...         return self._kelvin - 273.15
        >>>
        >>> temp = Temperature(_kelvin=300.0)
        >>>
        >>> # Instance access returns values
        >>> temp.kelvin    # 300.0
        >>> temp.celsius   # 26.85
        >>>
        >>> # Class access returns lenses
        >>> kelvin_lens = Temperature.kelvin  # Lens[Temperature, float]
        >>> celsius_lens = Temperature.celsius  # Lens[Temperature, float]
        >>>
        >>> # Use lenses for functional updates
        >>> kelvin_lens.get(temp)  # 300.0
        >>> celsius_lens.get(temp)  # 26.85

    Notes:
        - Only works with classes that inherit from HasLensFields
        - The decorated function should be a simple getter (no parameters other than self)
        - Setting through the lens creates a new instance with the updated value
    """

    def __init__(self, fget: Callable[[S], B]):
        self._fget = fget
        self._name: str | None = None
        ft.update_wrapper(self, fget)  # type: ignore[arg-type]

    def __set_name__(self, owner: type[S], name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type[S]) -> Lens[S, B]: ...

    @overload
    def __get__(self, instance: S, owner: type[S]) -> B: ...

    def __get__(self, instance: S | None, owner: type[S]) -> Lens[S, B] | B:
        if instance is None:
            # Class access - return a lens
            return lens(self._fget)
        else:
            # Instance access - return the property value
            return self._fget(instance)
