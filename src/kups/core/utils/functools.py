# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Functional programming utilities for composing and transforming functions."""

from typing import Any, Callable, Literal, overload


def compose[**P, B, C](fst: Callable[[B], C], snd: Callable[P, B]) -> Callable[P, C]:
    """Chain two functions together, applying them right-to-left.

    Returns a new function that applies `snd` first, then `fst` to the result.

    Example:
        ```python
        def add_one(x): return x + 1
        def double(x): return x * 2
        add_then_double = compose(double, add_one)
        add_then_double(5)  # (5 + 1) * 2 = 12
        # Returns: 12
        ```
    """

    def composed(*args: P.args, **kwargs: P.kwargs) -> C:
        return fst(snd(*args, **kwargs))

    return composed


def identity[A](x: A) -> A:
    """Return the input unchanged.

    Useful as a default function or placeholder in generic code.

    Example:
        ```python
        identity(42)
        # Returns: 42
        identity([1, 2, 3])
        # Returns: [1, 2, 3]
        ```
    """
    return x


def curry[A, B, C](f: Callable[[A, B], C]) -> Callable[[A], Callable[[B], C]]:
    """Convert a two-argument function into a chain of one-argument functions.

    Useful for partial application - fixing some arguments now, others later.

    Example:
        ```python
        def add(a, b): return a + b
        add_curried = curry(add)
        add_5 = add_curried(5)  # Fix first argument
        add_5(3)  # Supply second argument
        # Returns: 8
        add_5(10)  # Reuse with different second argument
        # Returns: 15
        ```
    """

    def curried(a: A) -> Callable[[B], C]:
        def inner(b: B) -> C:
            return f(a, b)

        return inner

    return curried


def uncurry[A, B, C](f: Callable[[A], Callable[[B], C]]) -> Callable[[A, B], C]:
    """Convert a curried function back to a two-argument function.

    Inverse of curry - takes nested single-argument functions and makes them
    accept both arguments at once.

    Example:
        ```python
        def curried_add(a): return lambda b: a + b
        add = uncurry(curried_add)
        add(5, 3)
        # Returns: 8
        ```
    """

    def uncurried(a: A, b: B) -> C:
        return f(a)(b)

    return uncurried


def pack_args[*A, B](f: Callable[[*A], B]) -> Callable[[tuple[*A]], B]:
    """Convert a function that takes multiple arguments to one that takes a tuple.

    Useful when you have data in tuples but functions that expect unpacked arguments.

    Example:
        ```python
        def add(a, b, c): return a + b + c
        add_packed = pack_args(add)
        add_packed((1, 2, 3))
        # Returns: 6
        ```
    """

    def packed(args: tuple[*A]) -> B:
        return f(*args)

    return packed


def unpack_args[*A, B](f: Callable[[tuple[*A]], B]) -> Callable[[*A], B]:
    """Convert a function that takes a tuple to one that takes multiple arguments.

    Inverse of pack_args.

    Example:
        ```python
        def sum_tuple(args): return sum(args)
        sum_unpacked = unpack_args(sum_tuple)
        sum_unpacked(1, 2, 3)
        # Returns: 6
        ```
    """

    def unpacked(*args: *A) -> B:
        return f(args)

    return unpacked


def constant[B](value: B) -> Callable[[Any], B]:
    """Create a function that always returns the same value, ignoring its input.

    Useful for providing default values or placeholder functions.

    Example:
        ```python
        always_5 = constant(5)
        always_5("anything")
        # Returns: 5
        always_5(100)
        # Returns: 5
        list(map(constant(0), [1, 2, 3]))
        # Returns: [0, 0, 0]
        ```
    """

    def const(_: Any) -> B:
        return value

    return const


def flip[A, B, C](f: Callable[[A, B], C]) -> Callable[[B, A], C]:
    """Swap the order of a function's first two arguments.

    Useful when you have a function but need arguments in the opposite order.

    Example:
        ```python
        def divide(a, b): return a / b
        divide(10, 2)  # 10 / 2
        # Returns: 5.0
        divide_flipped = flip(divide)
        divide_flipped(10, 2)  # 2 / 10
        # Returns: 0.2
        ```
    """

    def flipped(b: B, a: A) -> C:
        return f(a, b)

    return flipped


def pipe[**P, B, C](f: Callable[P, B], g: Callable[[B], C]) -> Callable[P, C]:
    """Chain two functions together, applying them left-to-right.

    Returns a new function that applies ``f`` first, then ``g`` to the result.

    Example:
        ```python
        def add_one(x): return x + 1
        def double(x): return x * 2
        add_then_double = pipe(add_one, double)
        add_then_double(5)  # (5 + 1) * 2 = 12
        # Returns: 12
        ```
    """
    return flip(compose)(f, g)


@overload
def typed[A, R](f: Callable[[A], R], a: type[A], /) -> Callable[[A], R]: ...


@overload
def typed[A, B, R](
    f: Callable[[A, B], R], a: type[A], b: type[B], /
) -> Callable[[A, B], R]: ...


@overload
def typed[A, B, C, R](
    f: Callable[[A, B, C], R], a: type[A], b: type[B], c: type[C], /
) -> Callable[[A, B, C], R]: ...


def typed[R](f: Callable[..., R], *types: type[Any]) -> Callable[..., R]:
    """Add runtime type checking to a function.

    Wraps a function to validate argument types at runtime, raising TypeError
    if types don't match.

    Example:
        ```python
        def add(a, b): return a + b
        safe_add = typed(add, int, int)
        safe_add(5, 3)
        # Returns: 8
        safe_add(5, "3")  # Raises TypeError
        # TypeError: Expected type <class 'int'>, got <class 'str'>
        ```
    """

    def wrapper(*args: Any) -> R:
        if len(args) != len(types):
            raise TypeError("Argument count does not match type count")
        for arg, typ in zip(args, types):
            if not isinstance(arg, typ):
                raise TypeError(f"Expected type {typ}, got {type(arg)}")
        return f(*args)

    return wrapper


@overload
def select_nth[A](n: Literal[0]) -> Callable[[A, *tuple[Any, ...]], A]: ...
@overload
def select_nth[A](n: Literal[1]) -> Callable[[Any, A, *tuple[Any, ...]], A]: ...
@overload
def select_nth[A](n: Literal[2]) -> Callable[[Any, Any, A, *tuple[Any, ...]], A]: ...
@overload
def select_nth[A](
    n: Literal[3],
) -> Callable[[Any, Any, Any, A, *tuple[Any, ...]], A]: ...
@overload
def select_nth[A](
    n: Literal[4],
) -> Callable[[Any, Any, Any, Any, A, *tuple[Any, ...]], A]: ...
@overload
def select_nth(n: int) -> Callable[..., Any]: ...
def select_nth(n: int) -> Callable[..., Any]:
    """Return a function that picks the ``n``-th positional argument.

    Args:
        n: Zero-based index of the argument to select.

    Returns:
        A callable ``(*args) -> args[n]``.
    """

    def fn(*args):
        return args[n]

    return fn
