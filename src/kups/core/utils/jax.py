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
import importlib
import math
import threading
import warnings
from contextlib import AbstractContextManager, contextmanager
from copy import copy
from itertools import count
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
    Union,
    cast,
    dataclass_transform,
    no_type_check,
    overload,
    runtime_checkable,
)

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from typing_extensions import TypedDict

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
def jit[C: Callable[..., Any]](
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
def jit[C: Callable[..., Any]](
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


def jit[C: Callable[..., Any]](
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
    """Type-preserving `jax.jit` that runs dataclass post-init checks in the trace.

    Usable bare (``@jit``) or with arguments (``@jit(static_argnames=...)``).
    The remaining arguments are passed through to `jax.jit`; see its
    documentation for details.

    Args:
        fn: Pure function to compile, or None when used with arguments.
        in_shardings: Sharding specification for inputs.
        out_shardings: Sharding specification for outputs.
        static_argnums: Positional arguments treated as compile-time constants.
        static_argnames: Named arguments treated as compile-time constants.
        donate_argnums: Positional arguments whose buffers may be reused.
        donate_argnames: Named arguments whose buffers may be reused.
        keep_unused: Keep arguments that JAX finds unused.
        device: Device to run the compiled function on.
        backend: XLA backend: 'cpu', 'gpu' or 'tpu'.
        inline: Inline this function into enclosing jaxprs.
        abstracted_axes: Axis abstraction specification.
        compiler_options: Compiler options.

    Returns:
        The compiled function with the signature of ``fn``, or a decorator if
        ``fn`` is None.
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
        def f_closed(*args: Any, **kwargs: Any):
            with enable_post_init():
                return fun(*args, **kwargs)

        f_closed.__name__ = getattr(
            fun, "__qualname__", getattr(fun, "__name__", "fun")
        )
        f_jitted = jax.jit(f_closed, **jit_kwargs)

        def f_result(*args: Any, **kwargs: Any):
            with no_post_init():
                return f_jitted(*args, **kwargs)

        # pyrefly: ignore[bad-return]
        return f_result

    if fn is None:
        return inner_jit
    return inner_jit(fn)


def shard_map[C: Callable[..., Any]](
    f: C,
    /,
    *,
    out_specs: Any,
    in_specs: Any = None,
    mesh: jax.sharding.Mesh | None = None,
    axis_names: frozenset[str] = frozenset(),
    check_vma: bool = True,
) -> C:
    """Type-preserving `jax.shard_map` for SPMD execution across devices.

    Each application of ``f`` maps a shard of the inputs to a shard of the
    output. See https://docs.jax.dev/en/latest/notebooks/shard_map.html.

    Args:
        f: Callable applied to each shard.
        out_specs: Pytree prefix of the output with `PartitionSpec` leaves;
            a named mesh axis concatenates along it, an unnamed one promises
            the outputs are equal along it.
        in_specs: Pytree prefix of the arguments with `PartitionSpec` leaves;
            a named mesh axis shards along it, an unnamed one replicates. With
            `jax.sharding.Infer` the specs come from the argument types.
        mesh: Devices to shard over; inferred from `jax.set_mesh` if None.
        axis_names: Mesh axes over which ``f`` is manual; all if empty.
        check_vma: Check that outputs are replicated over the mesh axes not
            named in ``out_specs``, and enable related autodiff optimizations.

    Returns:
        ``f`` applied across data sharded according to ``mesh`` and ``in_specs``.
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

    sharded_f = jax.shard_map(f, **shard_kwargs)

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any):
        return sharded_f(*args, **kwargs)

    # pyrefly: ignore[bad-return]
    return wrapper


@overload
def vectorize[C: Callable[..., Any]](
    pyfunc: None = None,
    *,
    excluded: frozenset[int] = ...,
    signature: str | None = ...,
) -> Callable[[C], C]: ...


@overload
def vectorize[C: Callable[..., Any]](
    pyfunc: C,
    *,
    excluded: frozenset[int] = ...,
    signature: str | None = ...,
) -> C: ...


def vectorize[C: Callable[..., Any]](
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
        def wrapper(*args: Any, **kwargs: Any):
            return vectorized(*args, **kwargs)

        # pyrefly: ignore[bad-return]
        return wrapper

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


@no_type_check
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


def enable_unrolled_threefry() -> None:
    """Lower Threefry-2x32 random bits without a rolled loop on CPU.

    JAX keeps a rolled loop on CPU to save compile time; the unrolled
    lowering, the default on other backends, is much faster per draw but
    compiles slower, so it pays off for long simulations rather than many
    short programs. Applies process-wide to programs lowered afterwards;
    random numbers are unchanged. Warns if the private JAX symbols moved.
    """
    try:
        from jax._src.interpreters import mlir

        try:
            threefry = importlib.import_module("jax._src.random.threefry2x32")
        except ImportError:  # JAX < 0.11
            threefry = importlib.import_module("jax._src.prng")
        mlir.register_lowering(
            threefry.threefry2x32_p,
            threefry._threefry2x32_lowering_rule,
            platform="cpu",
        )
    except (ImportError, AttributeError):  # pragma: no cover - JAX internals moved
        warnings.warn(
            "Could not register the unrolled Threefry lowering on CPU; "
            "random draws fall back to JAX's slower rolled loop.",
            stacklevel=2,
        )


def astype_floating[T](tree: T, dtype: DTypeLike) -> T:
    """Cast the floating-point leaves of ``tree`` to ``dtype``."""
    return jax.tree.map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        tree,
    )


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
    key = jax.random.split(rng, size)
    for i in count():
        # fold_in consumes 32 bits; wrap on the host without a device increment.
        yield _fold_keys(key, np.uint32(i % 2**32), shape=shape)


@jit(static_argnames="shape")
def _fold_keys(key: Array, i: np.uint32, shape: tuple[int, ...]) -> Array:
    return jax.vmap(jax.random.fold_in, in_axes=(0, None))(key, i).reshape(shape)


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
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool | dataclasses._MISSING_TYPE = dataclasses.MISSING,
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
def dataclass[T](
    cls: type[T] | None = None,
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
) -> type[T] | Callable[[type[T]], type[T]]:
    """Create a dataclass that works as a JAX PyTree.

    Combines `dataclasses.dataclass` with PyTree registration so instances pass
    through `jit`, `grad` and `vmap`. Unlike the standard decorator, classes are
    frozen by default. The keyword arguments match `dataclasses.dataclass`.

    Args:
        cls: Class to convert, or None when used as a decorator with arguments.
        init: Generate `__init__`.
        repr: Generate `__repr__`.
        eq: Generate `__eq__` and `__ne__`.
        order: Generate `__lt__`, `__le__`, `__gt__` and `__ge__`.
        unsafe_hash: Generate `__hash__` even if `__eq__` is defined.
        frozen: Forbid field assignment after creation.
        match_args: Generate `__match_args__`.
        kw_only: Make all fields keyword-only in `__init__`.
        slots: Generate `__slots__`.
        weakref_slot: Add a `__weakref__` slot when `slots` is True.

    Returns:
        A dataclass registered as a JAX PyTree, or a decorator if `cls` is None.
    """

    @no_type_check
    def make_dataclass(cls: type[T]) -> type[T]:
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
        )
        setattr(dcls, "_jax_dataclass", True)
        # All static fields land in aux_data so sibling subclasses pinning
        # the same field to different `init=False` defaults (e.g.
        # PeriodicCell vs VacuumCell) produce distinct treedefs; the
        # unflatten skips `init=False` fields and lets their class-level
        # default re-apply via `__init__`.
        all_fields = dataclasses.fields(dcls)
        data_fields = tuple(
            f.name for f in all_fields if f.init and not f.metadata.get("static", False)
        )
        meta_fields = tuple(
            f.name for f in all_fields if f.metadata.get("static", False)
        )
        init_meta_fields = tuple(
            f.name for f in all_fields if f.init and f.metadata.get("static", False)
        )

        def _flatten_with_keys(
            x: T,
            _data: tuple[str, ...] = data_fields,
            _meta: tuple[str, ...] = meta_fields,
        ):
            children = tuple(
                (jax.tree_util.GetAttrKey(n), getattr(x, n)) for n in _data
            )
            aux = tuple(getattr(x, n) for n in _meta)
            return children, aux

        def _flatten(x: T):
            return (
                tuple(getattr(x, n) for n in data_fields),
                tuple(getattr(x, n) for n in meta_fields),
            )

        def _unflatten(
            meta: tuple[object, ...],
            data: Iterable[Any],
            _cls: type = dcls,
            _data: tuple[str, ...] = data_fields,
            _meta: tuple[str, ...] = meta_fields,
            _init_meta: frozenset[str] = frozenset(init_meta_fields),
            _post_init: Callable[[Any], None] | None = getattr(
                dcls, "_unflatten_post_init", None
            ),
        ) -> T:
            kwargs = dict(zip(_data, data, strict=True))
            kwargs.update(
                (n, v) for n, v in zip(_meta, meta, strict=True) if n in _init_meta
            )
            if _post_init is None:
                return _cls(**kwargs)
            with no_post_init():
                obj = _cls(**kwargs)
            _post_init(obj)
            return obj

        jax.tree_util.register_pytree_with_keys(
            dcls, _flatten_with_keys, _unflatten, flatten_func=_flatten
        )
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


def no_post_init() -> AbstractContextManager[None]:
    """Context manager that disables ``__post_init__`` validation of decorated dataclasses."""
    return _set_post_init(True)


def enable_post_init() -> AbstractContextManager[None]:
    """Context manager that re-enables ``__post_init__`` validation inside :func:`no_post_init`."""
    return _set_post_init(False)


def skip_post_init_if_disabled(post_init: Callable[..., None]):
    """Skip ``__post_init__`` validation when inside a :func:`no_post_init` context.

    JAX dataclass containers like ``Table`` and ``Buffered`` validate
    invariants (unique keys, matching dimensions, …) in ``__post_init__``.
    During deserialization or lens-based structural updates the intermediate
    objects may temporarily violate those invariants, so validation is
    suppressed via the :func:`no_post_init` context manager.  Decorate a
    ``__post_init__`` with this function to opt into that suppression.
    """

    def wrapper(self: object, *args: Any, **kwargs: Any):
        if getattr(_no_post_init, "active", False):
            return
        return post_init(self, *args, **kwargs)

    return wrapper


ScatterModes = Literal["promise_in_bounds", "fill", "drop", "clip"]


class ScatterArgs(TypedDict, closed=True):
    """Keyword arguments forwarded to JAX scatter updates such as ``.at[idxs].set``."""

    mode: NotRequired[ScatterModes]
    wrap_negative_indices: NotRequired[bool]
    fill_value: NotRequired[StaticScalar | None]
    indices_are_sorted: NotRequired[bool]
    unique_indices: NotRequired[bool]


@runtime_checkable
class HasScatterArgs(Protocol):
    """Pytree node that sets the scatter arguments used when updating its leaves."""

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

    def _is_leaf(x: object) -> bool:
        return isinstance(x, HasScatterArgs)

    def _array_setter(scatter_args: ScatterArgs, arr: Array, val: Array) -> Array:
        if getattr(idxs, "size", -1) == 0:
            return arr
        # Remove fill_value (not accepted by .at[].set) without mutating caller
        set_args = copy(scatter_args)
        if "fill_value" in set_args:
            del set_args["fill_value"]
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
        # pyrefly: ignore[bad-argument-type]
        leaf2 = struc.flatten_up_to(val)
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


def drop_nonfinite_compensation[T](compensate: T) -> T:
    """Zero the compensation terms that are not finite.

    Once a running sum reaches an infinity the compensation update yields NaN or
    an infinity, which would poison later additions. Exact dtypes pass through.

    Args:
        compensate: Compensation PyTree from a compensated-summation step.

    Returns:
        The compensation with every non-finite entry replaced by zero.
    """

    def drop(c: Any) -> Any:
        if not jnp.issubdtype(jnp.result_type(c), jnp.inexact):
            return c
        return jnp.where(jnp.isfinite(c), c, jnp.zeros_like(c))

    return tree_map(drop, compensate)


def kahan_summation[T](*summands: T, compensate: T | None = None) -> tuple[T, T]:
    """Numerically stable summation using Kahan's compensated algorithm.

    Reduces floating-point accumulation errors when summing many numbers by
    tracking and compensating for rounding errors at each step. Works with
    arbitrary PyTree structures.

    The algorithm maintains an error compensation term that captures lost
    precision, significantly reducing numerical drift in iterative computations.

    A step whose compensation is not finite drops it, so an infinite sum stays
    infinite instead of becoming NaN.

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
        compensate = drop_nonfinite_compensation(sub(sub(t, result), y))
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
def non_differentiable_jvp(primals: tuple[Any, ...], tangents: tuple[Any, ...]):
    """Raise ``NotImplementedError``, since :func:`non_differentiable` has no derivative."""
    raise NotImplementedError("Function is not differentiable.")


class NotJaxCompatibleError(Exception):
    """Raised when a non-JAX-compatible function is called within a JAX transformation."""

    pass


def no_jax_tracing[C: Callable[..., Any]](fn: C) -> C:
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

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        for arg in args:
            if is_traced(arg):
                raise NotJaxCompatibleError(
                    f"Function '{fn.__name__}' cannot be called within a JAX "
                    f"transformation (jit, vmap, grad, etc.)."
                )
        for value in kwargs.values():
            if is_traced(value):
                raise NotJaxCompatibleError(
                    f"Function '{fn.__name__}' cannot be called within a JAX "
                    f"transformation (jit, vmap, grad, etc.)."
                )
        return fn(*args, **kwargs)

    # pyrefly: ignore[bad-return]
    return wrapper


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
    fn: Callable[..., Any],
    tree: T,
    *trees: *S,
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

    def _has_tree_match(x: object) -> bool:
        return isinstance(x, SupportsTreeMatch)

    def _is_leaf(x: object) -> bool:
        return _has_tree_match(x) or (bool(is_leaf) and is_leaf(x))

    def _fn(x: Any, *other: Any):
        if _has_tree_match(x) and other:
            matched = x.__tree_match__(*other)
            x = matched[0]
            other = matched[1:]
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


def tree_copy[T](tree: T) -> T:
    """Return a copy of the pytree with the same structure and values as ``tree``."""
    return tree_map(jnp.copy, tree)


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


@no_type_check
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


def skip_on_cpu[T, R](
    pred: Array, fn: Callable[[T], R], otherwise: Callable[[T], R], operand: T
) -> R:
    """``lax.cond`` on CPU; elsewhere always ``fn``.

    Accelerator conditionals synchronize with the host and copy their operands,
    which costs more than the skipped work.

    Args:
        pred: Scalar boolean selecting ``fn`` on CPU.
        fn: Function applied when ``pred`` holds, and always off CPU.
        otherwise: Cheaper equivalent of ``fn`` whenever ``pred`` is false.
        operand: Argument of either function.

    Returns:
        The selected function's result.
    """
    if jax.default_backend() == "cpu":
        return jax.lax.cond(pred, fn, otherwise, operand)
    return fn(operand)


def is_traced(x: object) -> bool:
    """Return ``True`` if any leaf of the pytree ``x`` is a JAX tracer."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(x))
