# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""JAX utility functions and decorators for functional programming.

This module provides type-safe wrappers around JAX transformations and utilities
for working with PyTrees, including JIT compilation, vectorization, and custom
dataclass registration.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import threading
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    NotRequired,
    Protocol,
    Sequence,
    TypedDict,
    Union,
    cast,
    dataclass_transform,
    overload,
    runtime_checkable,
)

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.utils.ops import where_broadcast_last

StaticScalar = Union[
    np.bool_,
    np.number,  # NumPy scalar types
    bool,
    int,
    float,
    complex,  # Python scalar types
]

if TYPE_CHECKING:
    from kups.core.lens import LensField


@overload
def jit[C: Callable](
    fn: C,
    *,
    in_shardings: Any = ...,
    out_shardings: Any = ...,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Iterable[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    donate_argnames: str | Iterable[str] | None = ...,
    keep_unused: bool = ...,
    device: Any = ...,
    backend: str | None = ...,
    inline: bool = ...,
    abstracted_axes: Any = ...,
    compiler_options: dict[str, Any] | None = ...,
) -> C: ...


@overload
def jit[C: Callable](
    fn: None = None,
    *,
    in_shardings: Any = ...,
    out_shardings: Any = ...,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Iterable[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    donate_argnames: str | Iterable[str] | None = ...,
    keep_unused: bool = ...,
    device: Any = ...,
    backend: str | None = ...,
    inline: bool = ...,
    abstracted_axes: Any = ...,
    compiler_options: dict[str, Any] | None = ...,
) -> Callable[[C], C]: ...


def jit[C: Callable](
    fn: C | None = None,
    *,
    in_shardings: Any = None,
    out_shardings: Any = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: Any = None,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: Any = None,
    compiler_options: dict[str, Any] | None = None,
) -> C | Callable[[C], C]:
    """Type-preserving JIT compilation decorator for JAX functions.

    Sets up a function for just-in-time compilation with XLA. Wraps `jax.jit`
    while preserving function names and type signatures.

    Args:
        fn: Function to be jitted. Should be a pure function. The arguments and
            return value should be arrays, scalars, or (nested) standard Python
            containers (tuple/list/dict) thereof.
        in_shardings: Optional sharding specification for inputs. If provided,
            the positional arguments must have compatible shardings.
        out_shardings: Optional sharding specification for outputs. Has the same
            effect as applying `jax.lax.with_sharding_constraint` to the output.
        static_argnums: Int or collection of ints specifying which positional
            arguments to treat as static (trace- and compile-time constant).
            Static arguments should be hashable and immutable.
        static_argnames: String or collection of strings specifying which named
            arguments to treat as static (compile-time constant).
        donate_argnums: Collection of integers specifying which positional argument
            buffers can be overwritten by the computation and marked deleted in
            the caller. Useful for memory optimization.
        donate_argnames: String or collection of strings specifying which named
            arguments are donated to the computation.
        keep_unused: If False (default), arguments that JAX determines to be
            unused may be dropped from compiled executables. If True, unused
            arguments will not be pruned.
        device: Optional device to run the jitted function on.
        backend: Optional string representing the XLA backend: 'cpu', 'gpu', or 'tpu'.
        inline: If True, inline this function into enclosing jaxprs. Default False.
        abstracted_axes: Optional axis abstraction specification.
        compiler_options: Optional dictionary of compiler options.

    Returns:
        A wrapped version of the function, set up for just-in-time compilation
        with preserved type signature.

    Example:
        ```python
        @jit
        def add(x, y):
            return x + y

        @jit(static_argnames=("axis",))
        def sum_along_axis(x, axis):
            return jnp.sum(x, axis=axis)

        @jit(donate_argnums=(0,))
        def update_in_place(x, y):
            return x + y  # x's buffer can be reused
        ```
    """
    # Collect non-None kwargs to pass to jax.jit
    jit_kwargs: dict[str, Any] = {}
    if in_shardings is not None:
        jit_kwargs["in_shardings"] = in_shardings
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    if static_argnums is not None:
        jit_kwargs["static_argnums"] = static_argnums
    if static_argnames is not None:
        jit_kwargs["static_argnames"] = static_argnames
    if donate_argnums is not None:
        jit_kwargs["donate_argnums"] = donate_argnums
    if donate_argnames is not None:
        jit_kwargs["donate_argnames"] = donate_argnames
    if keep_unused:
        jit_kwargs["keep_unused"] = keep_unused
    if device is not None:
        jit_kwargs["device"] = device
    if backend is not None:
        jit_kwargs["backend"] = backend
    if inline:
        jit_kwargs["inline"] = inline
    if abstracted_axes is not None:
        jit_kwargs["abstracted_axes"] = abstracted_axes
    if compiler_options is not None:
        jit_kwargs["compiler_options"] = compiler_options

    def inner_jit(fun: C) -> C:
        # Post-init validation is enabled inside the jitted function so that
        # validation logic is traced and compiled into the program rather than
        # running on the host. Outside of jit, post-init is disabled to avoid
        # redundant host-side checks on the jitted outputs.
        def f_closed(*args, **kwargs):
            with enable_post_init():
                return fun(*args, **kwargs)

        f_closed.__name__ = getattr(
            fun, "__qualname__", getattr(fun, "__name__", "fun")
        )
        f_jitted = jax.jit(f_closed, **jit_kwargs)

        def f_result(*args, **kwargs):
            with no_post_init():
                return f_jitted(*args, **kwargs)

        return f_result  # type: ignore

    if fn is None:
        return inner_jit
    return inner_jit(fn)


def shard_map[C: Callable](
    f: C,
    /,
    *,
    out_specs: Any,
    in_specs: Any = None,
    mesh: jax.sharding.Mesh | None = None,
    axis_names: frozenset = frozenset(),
    check_vma: bool = True,
) -> C:
    """Map a function over shards of data for multi-device parallel computation.

    Wraps `jax.shard_map` for SPMD (Single Program Multiple Data) parallel
    execution across multiple devices. Each application of the function takes
    as input a shard of the mapped-over arguments and produces a shard of the
    output.

    Args:
        f: Callable to be mapped. Each instance of `f` takes as input a shard
            of the mapped-over arguments and produces a shard of the output.
        out_specs: A pytree with `PartitionSpec` instances as leaves, with a tree
            structure that is a tree prefix of the output of `f`. Each `PartitionSpec`
            represents how the corresponding output shards should be concatenated.
            Mentioning a mesh axis name at a position expresses concatenation; not
            mentioning a mesh axis name expresses a promise that outputs are equal
            along that mesh axis.
        in_specs: A pytree with `PartitionSpec` instances as leaves, with a tree
            structure that is a tree prefix of the args tuple to be mapped over.
            Each `PartitionSpec` represents how the corresponding argument should
            be sharded along the named axes of `mesh`. Mentioning a mesh axis name
            at a position expresses sharding; not mentioning an axis name expresses
            replication. If None, inputs will be treated as static. If `jax.sharding.Infer`,
            the in_specs are inferred from the argument types.
        mesh: A `jax.sharding.Mesh` representing the array of devices over which
            to shard the data and on which to execute instances of `f`. The names
            of the Mesh can be used in collective communication operations in `f`.
            If None, it will be inferred from context set via `jax.set_mesh`.
        axis_names: Set of axis names from `mesh` over which the function `f` is
            manual. If empty (default), `f` is manual over all mesh axes.
        check_vma: If True (default), enable additional validity checks and automatic
            differentiation optimizations. The validity checks concern whether any
            mesh axis names not mentioned in `out_specs` are consistent with how
            the outputs of `f` are replicated.

    Returns:
        A callable that applies the input function `f` across data sharded
        according to the `mesh` and `in_specs`.

    Example:
        ```python
        from jax.sharding import Mesh, PartitionSpec as P
        from jax.experimental import mesh_utils

        # Create a mesh of devices
        devices = mesh_utils.create_device_mesh((2, 2))
        mesh = Mesh(devices, axis_names=('x', 'y'))

        @partial(shard_map, mesh=mesh, in_specs=P('x', None), out_specs=P('x', None))
        def parallel_fn(x):
            return x * 2

        # x will be sharded along the first axis across 'x' devices
        result = parallel_fn(x)
        ```

    Note:
        Requires understanding of JAX's sharding model and mesh configuration.
        For an introduction to sharded data, refer to JAX's sharded computation
        documentation at https://docs.jax.dev/en/latest/notebooks/shard_map.html.
    """
    # Build kwargs, handling defaults
    shard_kwargs: dict[str, Any] = {"out_specs": out_specs}
    if in_specs is not None:
        shard_kwargs["in_specs"] = in_specs
    if mesh is not None:
        shard_kwargs["mesh"] = mesh
    if axis_names:
        shard_kwargs["axis_names"] = axis_names
    if not check_vma:
        shard_kwargs["check_vma"] = check_vma

    sharded_f = jax.shard_map(f, **shard_kwargs)  # type: ignore

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return sharded_f(*args, **kwargs)

    return wrapper  # type: ignore


@overload
def vectorize[C: Callable](
    pyfunc: None = None,
    *,
    excluded: frozenset[int] = ...,
    signature: str | None = ...,
) -> Callable[[C], C]: ...


@overload
def vectorize[C: Callable](
    pyfunc: C,
    *,
    excluded: frozenset[int] = ...,
    signature: str | None = ...,
) -> C: ...


def vectorize[C: Callable](
    pyfunc: C | None = None,
    *,
    excluded: frozenset[int] = frozenset(),
    signature: str | None = None,
) -> C | Callable[[C], C]:
    """Define a vectorized function with broadcasting.

    Wraps `jax.numpy.vectorize` for defining vectorized functions with broadcasting,
    in the style of NumPy's generalized universal functions. It allows defining
    functions that are automatically repeated across any leading dimensions, without
    the implementation needing to handle higher dimensional inputs.

    Unlike `numpy.vectorize`, this is syntactic sugar for an auto-batching
    transformation (`vmap`) rather than a Python loop, making it considerably
    more efficient.

    Args:
        pyfunc: Function to vectorize, or `None` when used as decorator with arguments.
        excluded: Optional set of integers representing positional arguments for
            which the function will not be vectorized. These will be passed directly
            to `pyfunc` unmodified.
        signature: Optional generalized universal function signature, e.g.,
            `"(m,n),(n)->(m)"` for vectorized matrix-vector multiplication. If
            provided, `pyfunc` will be called with (and expected to return) arrays
            with shapes given by the size of corresponding core dimensions. By
            default, `pyfunc` is assumed to take scalar arrays as input.

    Returns:
        Vectorized version of the given function that broadcasts over batch dimensions.

    Example:
        ```python
        @vectorize(signature="(3),(3)->()")
        def dot_product(a, b):
            return jnp.sum(a * b)

        # Now works on batches
        a = jnp.ones((10, 3))
        b = jnp.ones((10, 3))
        result = dot_product(a, b)  # shape: (10,)
        ```
    """

    def inner_vectorize(fun: C) -> C:
        vectorized = jnp.vectorize(fun, excluded=excluded, signature=signature)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return vectorized(*args, **kwargs)

        return wrapper  # type: ignore

    if pyfunc is None:
        return inner_vectorize

    return inner_vectorize(pyfunc)


@overload
def linearize[*T, R1, R2](
    fn: Callable[[*T], tuple[R1, R2]], *primals: *T, has_aux: Literal[True]
) -> tuple[R1, Callable[[*T], R1], R2]: ...


@overload
def linearize[*T, R1](
    fn: Callable[[*T], R1], *primals: *T, has_aux: Literal[False] = False
) -> tuple[R1, Callable[[*T], R1]]: ...


def linearize[*T, R1, R2](
    fn: Callable[[*T], R1] | Callable[[*T], tuple[R1, R2]],
    *primals: *T,
    has_aux: bool = False,
) -> tuple[R1, Callable[[*T], R1], R2] | tuple[R1, Callable[[*T], R1]]:
    """Linearize ``fn`` at ``primals``, returning the output and a JVP function.

    Type-preserving wrapper around ``jax.linearize``.

    Args:
        fn: Function to linearize.
        *primals: Points at which to linearize.
        has_aux: If ``True``, ``fn`` returns ``(output, aux)`` and aux is
            returned as a third element.

    Returns:
        ``(output, jvp_fn)`` or ``(output, jvp_fn, aux)`` when ``has_aux=True``.
    """
    if has_aux:
        out, jvp_fn, aux = jax.linearize(fn, *primals, has_aux=True)

        def jvp(*tangents: *T) -> R1:
            return jvp_fn(*tangents)

        return out, jvp, aux
    else:
        out, jvp_fn = jax.linearize(fn, *primals)

        def jvp(*tangents: *T) -> R1:
            return jvp_fn(*tangents)

        return out, jvp


class PyTreeDef[T](Protocol):
    """Typed protocol for JAX pytree structure definitions."""

    def flatten_up_to(self, x: T) -> list[Array]: ...
    def unflatten(self, x: Sequence[Array]) -> T: ...
    @property
    def num_leaves(self) -> int: ...


def tree_structure[T](
    x: T, is_leaf: Callable[[Any], bool] | None = None
) -> PyTreeDef[T]:
    """Return the pytree structure of ``x``.

    Args:
        x: Pytree to inspect.
        is_leaf: Optional predicate marking extra leaf types.

    Returns:
        A ``PyTreeDef`` that can flatten/unflatten matching pytrees.
    """
    return jax.tree_util.tree_structure(x, is_leaf=is_leaf)


def key_chain(rng: Array, shape: tuple[int, ...] = ()) -> Generator[Array, None, None]:
    """Generate an infinite sequence of PRNG keys with deterministic iteration.

    Creates a generator that produces an infinite stream of JAX PRNG keys by
    folding in incrementing counters. Useful for iterative algorithms that need
    reproducible randomness at each step.

    Args:
        rng: Initial JAX PRNG key.
        shape: Shape of key batches to generate. Default `()` yields scalar keys.

    Yields:
        JAX PRNG keys with the specified shape, incremented deterministically.

    Example:
        ```python
        key = jax.random.PRNGKey(0)
        keys = key_chain(key, shape=(5,))

        # Generate keys for each iteration
        k1 = next(keys)  # shape: (5,)
        k2 = next(keys)  # shape: (5,), different from k1
        ```
    """
    size = math.prod(shape)
    i = jnp.zeros((size,), dtype=int)
    key = jax.random.split(rng, size)

    @jit
    @jax.vmap
    def fold_keys(key: Array, i: Array) -> Array:
        return jax.random.fold_in(key, i)

    while True:
        yield fold_keys(key, i).reshape(shape)
        i += 1


def field[T](
    *,
    default: T | dataclasses._MISSING_TYPE = dataclasses.MISSING,
    default_factory: Callable[[], T] | dataclasses._MISSING_TYPE = dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool | dataclasses._MISSING_TYPE = dataclasses.MISSING,
    static: bool = False,
) -> T:
    """Create a dataclass field with JAX PyTree registration support.

    This is an enhanced version of `dataclasses.field` that adds a `static` parameter
    for controlling JAX PyTree registration. When `static=True`, the field is marked
    as static metadata and excluded from JAX transformations like `jit`, `grad`, and `vmap`.

    Args:
        default: Default value for the field. Cannot be used with `default_factory`.
        default_factory: Factory function to generate default values. Cannot be used with `default`.
        init: If `True`, include this field in the generated `__init__` method.
        repr: If `True`, include this field in the generated `__repr__` method.
        hash: If `True`, include this field in the generated `__hash__` method.
            If `None`, use the value of `compare`.
        compare: If `True`, include this field in comparison methods (`__eq__`, `__lt__`, etc.).
        metadata: Additional metadata dictionary for the field.
        kw_only: If `True`, make this field keyword-only in the `__init__` method.
        static: If `True`, mark this field as static for JAX PyTree registration.
            Static fields are not traced through JAX transformations and remain
            constant across function calls.

    Returns:
        A dataclass field configured with the specified parameters.

    Example:
        ```python
        @dataclass
        class Config:
            learning_rate: Array  # Dynamic field
            model_name: str = field(default="transformer", static=True)  # Static field
            weights: Array = field(default_factory=lambda: jnp.zeros((10,))) # Dynamic field

        config = Config()
        # Only learning_rate and weights are traced by JAX transformations
        # model_name remains constant as static metadata
        ```

    Note:
        Static fields are useful for configuration parameters, model hyperparameters,
        or any values that should remain constant during JAX transformations.
    """
    kwargs = locals()
    kwargs["metadata"] = {**(kwargs["metadata"] or {}), "static": kwargs.pop("static")}
    return dataclasses.field(**kwargs)


def lens_field[T](
    *,
    default: T | dataclasses._MISSING_TYPE = dataclasses.MISSING,
    default_factory: Callable[[], T] | dataclasses._MISSING_TYPE = dataclasses.MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
    kw_only=dataclasses.MISSING,
    static: bool = False,
) -> LensField[T]:
    """Create a field for use with LensField annotations.

    This is a type-safe wrapper around field() specifically for LensField[T] annotations.
    It returns the proper type for static type checkers while creating a regular dataclass
    field at runtime.

    Args:
        default: Default value for the field.
        default_factory: Factory function to generate default values.
        init (bool): Include field in __init__.
        repr (bool): Include field in __repr__.
        hash (bool): Include field in __hash__.
        compare (bool): Include field in comparison methods.
        metadata (dict): Additional metadata dictionary.
        kw_only (bool): Make this a keyword-only argument in __init__.
        static (bool): Mark field as static (JAX tree registration).

    Returns:
        A dataclass field that type checkers treat as LensField[T].

    Example:
        >>> from kups.core.lens import LensField, HasLensFields
        >>> from kups.core.utils.jax import dataclass, lens_field
        >>> from jax import Array
        >>>
        >>> @dataclass
        ... class Point(HasLensFields):
        ...     x: LensField[float] = lens_field(default=0.0)
        ...     y: LensField[Array] = lens_field(static=True)
    """
    return cast(
        LensField[T],
        field(
            default=default,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            metadata=metadata,
            kw_only=kw_only,
            static=static,
        ),
    )


@overload
def dataclass[T: type](
    cls: T,
    /,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    match_args: bool = ...,
    kw_only: bool = ...,
    slots: bool = ...,
    weakref_slot: bool = ...,
) -> T: ...


@overload
def dataclass[T: type](
    cls: None = None,
    /,
    *,
    init: bool = ...,
    repr: bool = ...,
    eq: bool = ...,
    order: bool = ...,
    unsafe_hash: bool = ...,
    frozen: bool = ...,
    match_args: bool = ...,
    kw_only: bool = ...,
    slots: bool = ...,
    weakref_slot: bool = ...,
) -> Callable[[T], T]: ...


@dataclass_transform(field_specifiers=(field, dataclasses.Field), frozen_default=True)
def dataclass[T: type](
    cls: T | None = None,
    /,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = True,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
    weakref_slot: bool = False,
) -> T | Callable[[T], T]:
    """Create a dataclass that works as a JAX PyTree.

    Combines Python's `@dataclass` with JAX's PyTree registration, enabling
    dataclasses to be used with JAX transformations like `jit`, `grad`, and `vmap`.
    Dataclasses are frozen by default for immutability (unlike standard dataclasses).

    Args:
        cls: Class to convert into a JAX-compatible dataclass, or None when used
            as a decorator with arguments.
        init: If True (default), generate an `__init__` method.
        repr: If True (default), generate a `__repr__` method.
        eq: If True (default), generate `__eq__` and `__ne__` methods.
        order: If True, generate `__lt__`, `__le__`, `__gt__`, and `__ge__` methods.
            Default is False.
        unsafe_hash: If True, generate a `__hash__` method even if `__eq__` is
            defined. Use with caution. Default is False.
        frozen: If True (default, unlike standard dataclasses), fields cannot be
            assigned after instance creation. This is the default because JAX
            transformations work best with immutable data structures.
        match_args: If True (default), generate `__match_args__` for use in
            `match` statements (Python 3.10+).
        kw_only: If True, all fields become keyword-only in `__init__`. Default
            is False.
        slots: If True, generate `__slots__` for memory efficiency. Default is
            False.
        weakref_slot: If True and `slots` is True, add a `__weakref__` slot.
            Default is False.

    Returns:
        A dataclass registered as a JAX PyTree, or a decorator if `cls` is None.

    Example:
        ```python
        @dataclass
        class Point:
            x: jax.Array
            y: jax.Array

        p = Point(jnp.array(1.0), jnp.array(2.0))
        jax.tree.map(lambda x: x * 2, p)  # Works with JAX transformations

        # With arguments
        @dataclass(frozen=False)
        class MutablePoint:
            x: jax.Array
            y: jax.Array
        ```

    Note:
        Unlike standard `dataclasses.dataclass`, this decorator defaults to
        `frozen=True`.
    """

    def make_dataclass(cls: T) -> T:
        # https://github.com/jax-ml/jax/pull/24664
        if "_jax_dataclass" in cls.__dict__:
            return cls
        dcls = dataclasses.dataclass(
            cls,
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            match_args=match_args,
            kw_only=kw_only,
            slots=slots,
            weakref_slot=weakref_slot,
        )  # type: ignore
        dcls._jax_dataclass = True
        jax.tree_util.register_dataclass(dcls)
        return dcls

    if cls is None:
        return make_dataclass
    return make_dataclass(cls)


_no_post_init = threading.local()


@contextmanager
def _set_post_init(active: bool):
    before_value = getattr(_no_post_init, "active", False)
    _no_post_init.active = active
    try:
        yield
    finally:
        _no_post_init.active = before_value


def no_post_init():
    return _set_post_init(True)


def enable_post_init():
    return _set_post_init(False)


def skip_post_init_if_disabled(post_init: Callable):
    """Skip ``__post_init__`` validation when inside a :func:`no_post_init` context.

    JAX dataclass containers like ``Table`` and ``Buffered`` validate
    invariants (unique keys, matching dimensions, …) in ``__post_init__``.
    During deserialization or lens-based structural updates the intermediate
    objects may temporarily violate those invariants, so validation is
    suppressed via the :func:`no_post_init` context manager.  Decorate a
    ``__post_init__`` with this function to opt into that suppression.
    """

    def wrapper(self, *args, **kwargs):
        if getattr(_no_post_init, "active", False):
            return
        return post_init(self, *args, **kwargs)

    return wrapper


ScatterModes = Literal["promise_in_bounds", "fill", "drop", "clip"]


class ScatterArgs(TypedDict):
    mode: NotRequired[ScatterModes]
    wrap_negative_indices: NotRequired[bool]
    fill_value: NotRequired[StaticScalar | None]
    indices_are_sorted: NotRequired[bool]
    unique_indices: NotRequired[bool]


@runtime_checkable
class HasScatterArgs(Protocol):
    @property
    def scatter_args(self) -> ScatterArgs: ...


def tree_scatter_set[T](item: T, value: T, idxs: Array, args: ScatterArgs) -> T:
    """Set values at indices in a pytree, respecting ``HasScatterArgs``.

    Traverses the pytree and applies ``arr.at[idxs].set(val)`` to each array
    leaf. Nodes implementing ``HasScatterArgs`` merge their ``scatter_args``
    (e.g. ``mode="drop"``) into the call before recursing into children.

    Args:
        item: Pytree to update.
        value: Pytree of replacement values (same structure as ``item``).
        idxs: Integer index array for the scatter operation.
        args: Scatter keyword args passed to ``Array.at[].set()``.

    Returns:
        Updated pytree with the same structure as ``item``.
    """

    def _is_leaf(x):
        return isinstance(x, HasScatterArgs)

    def _array_setter(scatter_args: ScatterArgs, arr: Array, val: Array):
        if getattr(idxs, "size", -1) == 0:
            return arr
        # Remove fill_value (not accepted by .at[].set) without mutating caller
        set_args: ScatterArgs = {
            k: v for k, v in scatter_args.items() if k != "fill_value"
        }  # type: ignore[assignment]
        return arr.at[idxs].set(val, **set_args)

    def _setter[L: Array | HasScatterArgs](arr: L, val: L):
        merged_args: ScatterArgs = (
            {**arr.scatter_args, **args} if isinstance(arr, HasScatterArgs) else args
        )
        if isinstance(arr, Array):
            assert isinstance(val, Array)
            return _array_setter(merged_args, arr, val)
        struc = tree_structure(arr, lambda x: x is not arr and _is_leaf(x))
        leaf1 = struc.flatten_up_to(arr)
        leaf2 = struc.flatten_up_to(val)  # type: ignore
        result = tree_scatter_set(leaf1, leaf2, idxs, merged_args)
        return jax.tree.unflatten(struc, result)

    return tree_map(_setter, item, value, is_leaf=_is_leaf)


def isin(a: Array, b: Array, max_item: int) -> Array:
    """Fast membership test for integer arrays using index-based lookup.

    Optimized alternative to `jnp.isin` for integer arrays with known maximum
    value. Uses array indexing instead of comparisons for better performance.

    Args:
        a: Query array of integers to test for membership.
        b: Reference array of integers to test membership against.
        max_item: Maximum possible value in both arrays (exclusive upper bound).

    Returns:
        Boolean array of same shape as `a`, where `True` indicates the element
        exists in `b`.

    Example:
        ```python
        a = jnp.array([1, 3, 5, 7])
        b = jnp.array([3, 5])
        result = isin(a, b, max_item=10)  # [False, True, True, False]
        ```
    """
    return (
        jnp.zeros(max_item, dtype=jnp.bool_)
        .at[b]
        .set(True, mode="drop")
        .at[a]
        .get(mode="fill", fill_value=False)
    )


def kahan_summation[T](*summands: T, compensate: T | None = None) -> tuple[T, T]:
    """Numerically stable summation using Kahan's compensated algorithm.

    Reduces floating-point accumulation errors when summing many numbers by
    tracking and compensating for rounding errors at each step. Works with
    arbitrary PyTree structures.

    The algorithm maintains an error compensation term that captures lost
    precision, significantly reducing numerical drift in iterative computations.

    Args:
        *summands: One or more PyTrees to sum together.
        compensate: Optional error compensation term from previous summation.

    Returns:
        Tuple of (sum, compensation) where compensation should be passed to
        subsequent calls for continued stability.

    Example:
        ```python
        # Single summation
        result, comp = kahan_summation(x, y, z)

        # Iterative summation with compensation
        total, comp = kahan_summation(a, b)
        total, comp = kahan_summation(total, c, compensate=comp)
        ```

    Reference:
        W. Kahan, "Further remarks on reducing truncation errors", 1965.
    """
    result = summands[0]
    if compensate is None:
        compensate = tree_map(jnp.zeros_like, result)
    assert compensate is not None

    def add(x: T, y: T) -> T:
        return tree_map(jnp.add, x, y)

    def sub(x: T, y: T) -> T:
        return tree_map(jnp.subtract, x, y)

    for summand in summands[1:]:
        y = sub(summand, compensate)
        t = add(result, y)
        compensate = sub(sub(t, result), y)
        result = t
    return result, compensate


@jax.custom_jvp
def non_differentiable[T](x: T) -> T:
    """Identity function that raises on differentiation.

    Use to mark values that must not be differentiated through. Any
    attempt to compute a JVP will raise ``NotImplementedError``.
    """
    return x


@non_differentiable.defjvp
def non_differentiable_jvp(primals, tangents):
    raise NotImplementedError("Function is not differentiable.")


class NotJaxCompatibleError(Exception):
    """Raised when a non-JAX-compatible function is called within a JAX transformation."""

    pass


def no_jax_tracing[C: Callable](fn: C) -> C:
    """Decorator to mark functions that should not be used within JAX transformations.

    Checks if any input pytree contains a JAX tracer. If so, raises NotJaxCompatibleError.
    Use this to prevent functions from being called inside jit, vmap, grad, etc.

    Args:
        fn: Function to protect from JAX tracing.

    Returns:
        Wrapped function that raises NotJaxCompatibleError if traced.

    Example:
        ```python
        @no_jax_tracing
        def load_data(path: str) -> Array:
            # This function reads from disk and shouldn't be traced
            return jnp.load(path)

        # Direct call works fine
        data = load_data("data.npy")

        # This will raise NotJaxCompatibleError
        @jit
        def process(path):
            return load_data(path)  # Error!
        ```
    """

    def _contains_tracer(x: Any) -> bool:
        leaves = jax.tree_util.tree_leaves(x)
        return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for arg in args:
            if _contains_tracer(arg):
                raise NotJaxCompatibleError(
                    f"Function '{fn.__name__}' cannot be called within a JAX "
                    f"transformation (jit, vmap, grad, etc.)."
                )
        for value in kwargs.values():
            if _contains_tracer(value):
                raise NotJaxCompatibleError(
                    f"Function '{fn.__name__}' cannot be called within a JAX "
                    f"transformation (jit, vmap, grad, etc.)."
                )
        return fn(*args, **kwargs)

    return wrapper  # type: ignore


@runtime_checkable
class SupportsTreeMatch(Protocol):
    """Protocol for pytree nodes that align themselves before mapping.

    When ``tree_map`` encounters a ``SupportsTreeMatch`` node, it calls
    ``__tree_match__`` to reconcile ``self`` with the corresponding nodes
    from the other input trees (e.g., merging label vocabularies in
    ``Index``). The returned tuple replaces the originals for the
    remainder of the map.
    """

    def __tree_match__[T](self: T, *others: T) -> tuple[T, ...]: ...


def tree_map[T, *S](
    fn: Callable,
    tree: T,
    *trees: *S,  # type: ignore[reportInvalidTypeVarUse]
    is_leaf: Callable[[Any], bool] | None = None,
) -> T:
    """Apply ``fn`` to every leaf of one or more pytrees, with label alignment.

    Extends ``jax.tree.map`` with support for ``SupportsTreeMatch`` nodes. Before
    ``fn`` is called, any node implementing ``__tree_match__`` is aligned
    across all input trees (e.g., ``Index`` objects merge their label
    vocabularies so integer indices become comparable).

    Nodes marked by ``is_leaf`` or ``SupportsTreeMatch`` are treated as leaves at
    the top level. If a ``SupportsTreeMatch`` node is *also* a pytree (not flagged
    by ``is_leaf``), its children are recursed into after alignment.

    Args:
        fn: Function applied to each aligned leaf.
        tree: Primary pytree (determines output structure).
        *trees: Additional pytrees with matching structure.
        is_leaf: Optional predicate for extra leaf types.

    Returns:
        Transformed pytree with the same structure as ``tree``.
    """
    _leaf_tree_def = jax.tree.structure(0)

    def _has_tree_match(x):
        return isinstance(x, SupportsTreeMatch)

    def _is_leaf(x):
        return _has_tree_match(x) or (bool(is_leaf) and is_leaf(x))

    def _fn(x, *other):
        if _has_tree_match(x) and other:
            x, *other = x.__tree_match__(*other)
        if (is_leaf and is_leaf(x)) or jax.tree.structure(x) == _leaf_tree_def:
            return fn(x, *other)
        struc = tree_structure(x, is_leaf=lambda y: y is not x and _is_leaf(y))
        x_l = struc.flatten_up_to(x)
        other_l = list(map(struc.flatten_up_to, other))
        new_l = tree_map(fn, x_l, *other_l, is_leaf=is_leaf)
        return struc.unflatten(new_l)

    return jax.tree_util.tree_map(_fn, tree, *trees, is_leaf=_is_leaf)


def tree_concat[T](*trees: T) -> T:
    """Concatenate pytrees along the leading axis.

    Args:
        *trees: Two or more pytrees with matching structure.

    Returns:
        Pytree with each leaf concatenated along axis 0.
    """
    return tree_map(lambda *x: jnp.concatenate(x), *trees)


def tree_stack[T](*trees: T) -> T:
    """Stack pytrees into a new leading dimension.

    Args:
        *trees: Two or more pytrees with matching structure.

    Returns:
        Pytree with each leaf stacked along a new leading axis.
    """
    return tree_map(lambda *x: jnp.stack(x), *trees)


def tree_zeros_like[T](tree: T) -> T:
    """Return a pytree of zeros with the same structure and dtypes as ``tree``."""
    return tree_map(jnp.zeros_like, tree)


def tree_where_broadcast_last[T](accept: Array, tree1: T, tree2: T) -> T:
    """Element-wise ``jnp.where`` over two pytrees, broadcasting ``accept`` on trailing dims.

    Args:
        accept: Boolean condition array, broadcast to match each leaf's shape.
        tree1: Pytree selected where ``accept`` is ``True``.
        tree2: Pytree selected where ``accept`` is ``False``.

    Returns:
        Pytree with the same structure, each leaf chosen per ``accept``.
    """
    return tree_map(lambda a, b: where_broadcast_last(accept, a, b), tree1, tree2)


def sequential_vmap_with_vjp[*P, R](func: Callable[[*P], R]) -> Callable[[*P], R]:
    """Create a sequentially vmapped function with custom VJP support.

    Wraps a function with sequential vmap (processes batch elements one at a time)
    and defines custom forward/backward passes for automatic differentiation.
    This is useful when the underlying function doesn't support standard vmap
    batching rules.

    Args:
        func: Function to be sequentially vmapped

    Returns:
        Vmapped function with proper VJP (vector-Jacobian product) support
    """
    vmap_call = jax.custom_vjp(jax.custom_batching.sequential_vmap(func))

    def f_fwd(*args: *P) -> tuple[R, tuple[*P]]:
        return vmap_call(*args), args

    def f_bwd(args: tuple[*P], g: R) -> tuple[*P]:
        @sequential_vmap_with_vjp
        def inner(g: R, *inner_args: *P) -> tuple[*P]:
            vjp_fn = jax.vjp(func, *inner_args)[1]
            return vjp_fn(g)

        return inner(g, *args)

    vmap_call.defvjp(f_fwd, f_bwd)
    return vmap_call


def is_traced(x: Array) -> bool:
    """Return ``True`` if ``x`` is a JAX tracer (inside a transformation)."""
    return isinstance(x, jax.core.Tracer)
