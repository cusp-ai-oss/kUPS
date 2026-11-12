# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit-aware wrappers for jax transforms.

Most jax transforms handle quantities on their own because quantities are pytrees.
The wrappers here cover what jax alone gets wrong: derivatives need cotangents
relabelled to dim(out)/dim(arg), and vectorize must strip labels before
``jnp.vectorize`` converts the operands. ``relabel`` asserts a dimension the type
algebra cannot compute.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, overload

import jax
import jax.numpy as jnp
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike

from kups.core.units._base import active_base, require_base
from kups.core.units._core import (
    AnyQuantity,
    Quantity,
    attach_plain,
    checked_dim,
    lift,
    lower,
    rebase,
    shape_of,
    sym,
    tree_flatten,
    tree_leaves,
    tree_map,
)
from kups.core.units._precision import check_dtype, rebase_factor
from kups.core.units._typed import (
    E_A,
    E_K,
    E_KG,
    E_M,
    E_S,
    F_A,
    F_K,
    F_KG,
    F_M,
    F_S,
    G_A,
    G_K,
    G_KG,
    G_M,
    G_S,
)
from kups.core.units._unit import ONE, Dim, DimensionError, ddiv, dmul
from kups.core.utils.jax import vectorize as _vectorize

_AUX = TypeVar("_AUX")

# jax.vjp is overloaded on a literal has_aux; this binding takes it as a bool.
_jax_vjp: Callable[
    ..., tuple[object, Callable[[object], tuple[object, ...]], *tuple[object, ...]]
] = jax.vjp

if TYPE_CHECKING:
    from kups.core.units._base import BaseSystem


def _label(x: object) -> tuple[Dim, BaseSystem]:
    """The dimension and base of one leaf; a plain leaf is dimensionless."""
    return (x.dim, x.bs) if isinstance(x, Quantity) else (ONE, active_base())


def _relabel(
    ct: AnyQuantity | ArrayLike, primal: object, odim: Dim, obs: BaseSystem
) -> object:
    """Relabel one cotangent leaf as ``dim(out)/dim(primal)`` in the primal's base."""
    lo = lower(ct)  # jax returns it labelled like the primal
    if not isinstance(primal, Quantity):
        return lift(lo, odim, obs)  # w.r.t. a plain array it still has dim(out)
    fac = rebase_factor(obs, primal.bs, odim, primal.dtype)
    return lift(lo if fac == 1.0 else lo * fac, ddiv(odim, primal.dim), primal.bs)


def relabel(x: object, dim: Dim, bs: BaseSystem | None = None) -> Any:
    """Reinterpret magnitudes as a different dimension, leaving the numbers as-is.

    Args:
      x: a value or a pytree of values; every leaf is relabelled.
      dim: the dimension to assign; ``units.ONE`` gives plain arrays back.
      bs: the base system to use, or None to keep each leaf's own.

    Returns:
      ``x`` with every leaf labelled ``dim``.

    Raises:
      DimensionError: if ``dim`` is not a dimension vector, or the magnitudes'
        dtype cannot carry it.
    """
    checked_dim(dim, "relabel")
    if bs is not None:
        require_base(bs, "the base to relabel into")

    def one(v: AnyQuantity | ArrayLike) -> object:
        """Relabel one leaf; plain magnitudes are attached as by ``from_dim``."""
        if not isinstance(v, Quantity):
            return attach_plain(v, dim, bs or active_base(), "relabel")
        into = bs or v.bs
        lo = lower(v)
        check_dtype(jnp.result_type(lo), dim, into, "relabel")
        return lift(lo, dim, into)

    return tree_map(one, x)


def _leaf_name(x: object) -> str:
    """Describe what one output leaf measures, for error messages."""
    return sym(x.dim, x.bs) if isinstance(x, Quantity) else "a plain array"


def _out_label(name: str, out: object) -> tuple[Dim, BaseSystem]:
    """Find the one dimension and base shared by a differentiated output.

    Plain leaves count as dimensionless.
    """
    leaves = tree_leaves(out)
    dims = [x.dim if isinstance(x, Quantity) else ONE for x in leaves]
    for i, d in enumerate(dims[1:], 1):
        if d != dims[0]:
            raise DimensionError(
                f"units.{name}: the output measures both {_leaf_name(leaves[0])} and "
                f"{_leaf_name(leaves[i])}, so one cotangent cannot label the primal's: "
                f"the leaves' contributions add up. Differentiate the parts "
                f"separately, attach the unit to a plain leaf, or divide the odd "
                f"leaves by a reference quantity."
            )
    qs = [x for x in leaves if isinstance(x, Quantity)]
    for t in qs[1:]:
        if t.bs != qs[0].bs:
            raise DimensionError(
                f"units.{name}: the output measures {sym(qs[0].dim, qs[0].bs)} in "
                f"base {qs[0].bs.name} and in base {t.bs.name}, so its leaves' "
                f"magnitudes are in different units and the contributions they add up "
                f"to are not one number. Build the whole output in one base "
                f"(q.rebase()), or differentiate the parts separately."
            )
    return (qs[0].dim, qs[0].bs) if qs else (ONE, active_base())


def _cotangent(
    name: str, cotangents: object, out: object, obs: BaseSystem
) -> tuple[object, Dim]:
    """Seed jax with a cotangent's magnitudes in the output's base and labels.

    Returns the seed and the cotangent's dimension, which the pulled-back
    cotangents carry on top of ``dim(out) / dim(primal)``.
    """
    leaves = tree_leaves(cotangents)
    for x in leaves[1:]:
        if _label(x)[0] != _label(leaves[0])[0]:
            raise DimensionError(
                f"units.{name}: the cotangent measures both {_leaf_name(leaves[0])} "
                f"and {_leaf_name(x)}, so the pulled-back cotangents would add up "
                f"different dimensions. Seed one dimension at a time."
            )

    def seed(c: AnyQuantity | ArrayLike, o: object) -> object:
        mag = lower(rebase(c, obs)) if isinstance(c, Quantity) else c
        return lift(mag, o.dim, o.bs) if isinstance(o, Quantity) else mag

    return tree_map(seed, cotangents, out), _label(leaves[0])[0] if leaves else ONE


class Pullback(Protocol[E_M, E_KG, E_S, E_A, E_K]):
    """A one-primal ``vjp_fn`` whose result has dimension ``dim(ct) * E``."""

    @overload
    def __call__(
        self: Pullback[0, 0, 0, 0, 0], ct: ArrayLike, /
    ) -> tuple[jax.Array]: ...
    @overload
    def __call__(
        self, ct: ArrayLike, /
    ) -> tuple[Quantity[E_M, E_KG, E_S, E_A, E_K]]: ...
    @overload
    def __call__(
        self, ct: Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K], /
    ) -> tuple[jax.Array]: ...
    @overload
    def __call__(
        self, ct: Quantity[G_M, G_KG, G_S, G_A, G_K], /
    ) -> tuple[Quantity[G_M + E_M, G_KG + E_KG, G_S + E_S, G_A + E_A, G_K + E_K]]: ...


class Pullback2(Protocol[E_M, E_KG, E_S, E_A, E_K, F_M, F_KG, F_S, F_A, F_K]):
    """A two-primal ``vjp_fn`` whose results have ``dim(ct) * E`` and ``dim(ct) * F``."""

    @overload
    def __call__(
        self: Pullback2[0, 0, 0, 0, 0, F_M, F_KG, F_S, F_A, F_K], ct: ArrayLike, /
    ) -> tuple[jax.Array, Quantity[F_M, F_KG, F_S, F_A, F_K]]: ...
    @overload
    def __call__(
        self: Pullback2[E_M, E_KG, E_S, E_A, E_K, 0, 0, 0, 0, 0], ct: ArrayLike, /
    ) -> tuple[Quantity[E_M, E_KG, E_S, E_A, E_K], jax.Array]: ...
    @overload
    def __call__(
        self, ct: ArrayLike, /
    ) -> tuple[
        Quantity[E_M, E_KG, E_S, E_A, E_K], Quantity[F_M, F_KG, F_S, F_A, F_K]
    ]: ...
    @overload
    def __call__(
        self, ct: Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K], /
    ) -> tuple[
        jax.Array, Quantity[F_M - E_M, F_KG - E_KG, F_S - E_S, F_A - E_A, F_K - E_K]
    ]: ...
    @overload
    def __call__(
        self, ct: Quantity[-F_M, -F_KG, -F_S, -F_A, -F_K], /
    ) -> tuple[
        Quantity[E_M - F_M, E_KG - F_KG, E_S - F_S, E_A - F_A, E_K - F_K], jax.Array
    ]: ...
    @overload
    def __call__(
        self, ct: Quantity[G_M, G_KG, G_S, G_A, G_K], /
    ) -> tuple[
        Quantity[G_M + E_M, G_KG + E_KG, G_S + E_S, G_A + E_A, G_K + E_K],
        Quantity[G_M + F_M, G_KG + F_KG, G_S + F_S, G_A + F_A, G_K + F_K],
    ]: ...


@overload
def vjp(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        Quantity[F_M, F_KG, F_S, F_A, F_K],
    ],
    primal: Quantity[E_M, E_KG, E_S, E_A, E_K],
    /,
    *,
    has_aux: Literal[False] = ...,
) -> tuple[
    Quantity[F_M, F_KG, F_S, F_A, F_K],
    Pullback[F_M - E_M, F_KG - E_KG, F_S - E_S, F_A - E_A, F_K - E_K],
]: ...
@overload
def vjp(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        tuple[Quantity[F_M, F_KG, F_S, F_A, F_K], _AUX],
    ],
    primal: Quantity[E_M, E_KG, E_S, E_A, E_K],
    /,
    *,
    has_aux: Literal[True],
) -> tuple[
    Quantity[F_M, F_KG, F_S, F_A, F_K],
    Pullback[F_M - E_M, F_KG - E_KG, F_S - E_S, F_A - E_A, F_K - E_K],
    _AUX,
]: ...
@overload
def vjp(
    f: Callable[[Quantity[E_M, E_KG, E_S, E_A, E_K]], jax.Array],
    primal: Quantity[E_M, E_KG, E_S, E_A, E_K],
    /,
    *,
    has_aux: Literal[False] = ...,
) -> tuple[jax.Array, Pullback[-E_M, -E_KG, -E_S, -E_A, -E_K]]: ...
@overload
def vjp(
    f: Callable[
        [
            Quantity[E_M, E_KG, E_S, E_A, E_K],
            Quantity[G_M, G_KG, G_S, G_A, G_K],
        ],
        Quantity[F_M, F_KG, F_S, F_A, F_K],
    ],
    primal: Quantity[E_M, E_KG, E_S, E_A, E_K],
    second: Quantity[G_M, G_KG, G_S, G_A, G_K],
    /,
    *,
    has_aux: Literal[False] = ...,
) -> tuple[
    Quantity[F_M, F_KG, F_S, F_A, F_K],
    Pullback2[
        F_M - E_M,
        F_KG - E_KG,
        F_S - E_S,
        F_A - E_A,
        F_K - E_K,
        F_M - G_M,
        F_KG - G_KG,
        F_S - G_S,
        F_A - G_A,
        F_K - G_K,
    ],
]: ...
@overload
def vjp[R](
    f: Callable[..., R], *primals: object, has_aux: Literal[False] = ...
) -> tuple[R, Callable[[R | ArrayLike], tuple[Any, ...]]]: ...
@overload
def vjp[R, A](
    f: Callable[..., tuple[R, A]], *primals: object, has_aux: Literal[True]
) -> tuple[R, Callable[[R | ArrayLike], tuple[Any, ...]], A]: ...
def vjp(
    f: Callable[..., object], *primals: object, has_aux: bool = False
) -> tuple[object, ...]:
    """Reverse-mode derivative as a reusable function, with units.

    Args:
      f: the function to differentiate; every output leaf must share one dimension.
      *primals: the point to differentiate at.
      has_aux: whether ``f`` returns ``(output, aux)``.

    Returns:
      ``(out, vjp_fn)``, or ``(out, vjp_fn, aux)`` when ``has_aux``. ``vjp_fn``
      takes an output cotangent, plain magnitudes or a quantity of any dimension,
      and returns one cotangent per primal, of dimension
      ``dim(ct) * dim(out) / dim(primal)``.

    Raises:
      DimensionError: if the output's leaves, or the cotangent's, do not share one
        dimension, or the output's leaves do not share one base.
    """
    out, vjp_fn, aux = _vjp(f, primals, has_aux, "vjp")
    return (out, vjp_fn, aux) if has_aux else (out, vjp_fn)


def _vjp(
    f: Callable[..., object], primals: tuple[object, ...], has_aux: bool, name: str
) -> tuple[object, Callable[[object], tuple[object, ...]], object]:
    """``vjp`` returning ``(out, vjp_fn, aux)``, with aux None unless ``has_aux``.

    ``name`` is the public caller, for errors.
    """
    res = _jax_vjp(f, *primals, has_aux=has_aux)
    out, raw = res[0], res[1]
    odim, obs = _out_label(name, out)

    def vjp_fn(cotangents: object) -> tuple[object, ...]:
        """Pull back an output cotangent to one cotangent per primal."""
        seed, cdim = _cotangent(name, cotangents, out, obs)
        rdim = dmul(cdim, odim)
        return tuple(
            tree_map(lambda c, p: _relabel(c, p, rdim, obs), ct, primal)
            for ct, primal in zip(raw(seed), primals, strict=True)
        )

    return out, vjp_fn, res[2] if has_aux else None


def _check_argnums(
    name: str, argnums: int | tuple[int, ...], idx: tuple[int, ...], nargs: int
) -> None:
    """Check argnums against the positional arguments passed."""
    for i in idx:
        if not -nargs <= i < nargs:
            raise TypeError(
                f"units.{name}: differentiating with respect to argnums={argnums} "
                f"requires at least {-i if i < 0 else i + 1} positional arguments to "
                f"be passed by the caller, but got only {nargs}."
            )
    norm = [i % nargs for i in idx]
    if len(set(norm)) != len(norm):
        raise ValueError(
            f"units.{name}: argnums={argnums} names the same argument twice; "
            f"differentiate with respect to each argument once."
        )


def _vjp_grad[**P](
    f: Callable[P, object],
    argnums: int | tuple[int, ...],
    with_value: bool,
    has_aux: bool,
) -> Callable[P, object]:
    """Build a unit-aware grad or value_and_grad."""
    idx = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    name = "value_and_grad" if with_value else "grad"
    call: Callable[..., object] = f

    @functools.wraps(f)
    def run(*args: P.args, **kwargs: P.kwargs) -> object:
        _check_argnums(name, argnums, idx, len(args))
        for i in idx:
            for leaf in tree_leaves(args[i]):
                dt = jnp.result_type(lower(cast(AnyQuantity | ArrayLike, leaf)))
                if not jnp.issubdtype(dt, jnp.inexact):
                    raise TypeError(
                        f"units.{name} requires real- or complex-valued inputs, "
                        f"got {dt.name} for argument {i}"
                    )

        def sub(*sel: object) -> object:
            full = list(args)
            for i, v in zip(idx, sel, strict=True):
                full[i] = v
            return call(*full, **kwargs)

        out, pull, aux = _vjp(sub, tuple(args[i] for i in idx), has_aux, name)
        shape = shape_of(cast(AnyQuantity | ArrayLike, out))
        if shape != ():
            raise TypeError(f"units.{name} needs a scalar output, got shape {shape}")
        dt = jnp.result_type(lower(cast(AnyQuantity | ArrayLike, out)))
        if not jnp.issubdtype(dt, jnp.floating):
            raise TypeError(
                f"units.{name} needs a real floating output, got {dt.name}; use "
                "units.vjp for a complex or integer one"
            )
        gs = pull(
            jnp.ones((), jnp.result_type(lower(cast(AnyQuantity | ArrayLike, out))))
        )
        g = gs[0] if isinstance(argnums, int) else gs
        val = (out, aux) if has_aux else out
        return (val, g) if with_value else (g, aux) if has_aux else g

    return run


@overload
def grad(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        Quantity[E_M, E_KG, E_S, E_A, E_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[[Quantity[E_M, E_KG, E_S, E_A, E_K]], jax.Array]: ...
@overload
def grad(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        Quantity[F_M, F_KG, F_S, F_A, F_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[
    [Quantity[E_M, E_KG, E_S, E_A, E_K]],
    Quantity[F_M - E_M, F_KG - E_KG, F_S - E_S, F_A - E_A, F_K - E_K],
]: ...
@overload
def grad[**P](
    f: Callable[P, object],
    argnums: int | tuple[int, ...] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[P, Any]: ...
@overload
def grad[**P, A](
    f: Callable[P, tuple[object, A]],
    argnums: int | tuple[int, ...] = ...,
    *,
    has_aux: Literal[True],
) -> Callable[P, tuple[Any, A]]: ...
def grad[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = 0, has_aux: bool = False
) -> Callable[P, object]:
    """Gradient of a scalar function, with units.

    Args:
      f: function returning a scalar quantity (or scalar array).
      argnums: index or indices to differentiate with respect to.
      has_aux: whether ``f`` returns ``(output, aux)``.

    Returns:
      A function returning ``d f / d arg``, or ``(d f / d arg, aux)`` when ``has_aux``.
    """
    return _vjp_grad(f, argnums, False, has_aux)


@overload
def value_and_grad(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        Quantity[E_M, E_KG, E_S, E_A, E_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[
    [Quantity[E_M, E_KG, E_S, E_A, E_K]],
    tuple[Quantity[E_M, E_KG, E_S, E_A, E_K], jax.Array],
]: ...
@overload
def value_and_grad(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        Quantity[F_M, F_KG, F_S, F_A, F_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[
    [Quantity[E_M, E_KG, E_S, E_A, E_K]],
    tuple[
        Quantity[F_M, F_KG, F_S, F_A, F_K],
        Quantity[
            F_M - E_M,
            F_KG - E_KG,
            F_S - E_S,
            F_A - E_A,
            F_K - E_K,
        ],
    ],
]: ...
@overload
def value_and_grad[**P, R](
    f: Callable[P, R],
    argnums: int | tuple[int, ...] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[P, tuple[R, Any]]: ...
@overload
def value_and_grad[**P, R, A](
    f: Callable[P, tuple[R, A]],
    argnums: int | tuple[int, ...] = ...,
    *,
    has_aux: Literal[True],
) -> Callable[P, tuple[tuple[R, A], Any]]: ...
def value_and_grad[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = 0, has_aux: bool = False
) -> Callable[P, object]:
    """``grad`` that also returns the function value.

    Args:
      f: function returning a scalar quantity (or scalar array).
      argnums: index or indices to differentiate with respect to.
      has_aux: whether ``f`` returns ``(output, aux)``.

    Returns:
      A function returning ``(value, gradient)``, or ``((value, aux), gradient)`` when
      ``has_aux``.
    """
    return _vjp_grad(f, argnums, True, has_aux)


def _jac(
    mode: Literal["rev", "fwd"],
    f: Callable[..., object],
    argnums: int | tuple[int, ...],
) -> Callable[..., object]:
    """Wrap ``jax.jacrev`` or ``jax.jacfwd`` to label the Jacobian per leaf."""
    idx = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    jaxfn = jax.jacrev if mode == "rev" else jax.jacfwd

    @functools.wraps(f)
    def run(*args: object, **kwargs: object) -> object:
        _check_argnums("hessian", argnums, idx, len(args))
        prims: list[list[AnyQuantity | ArrayLike]] = []
        trees: list[PyTreeDef] = []
        for i in idx:
            leaves, tree = tree_flatten(args[i])
            prims.append(leaves)
            trees.append(tree)
        mags = [[lower(x) for x in leaves] for leaves in prims]
        meta: list[tuple[PyTreeDef, list[tuple[Dim, BaseSystem]]]] = []

        def sub(*ms: list[jax.Array]) -> list[object]:
            """Run ``f`` on magnitudes, lifting arguments and lowering outputs."""
            full = list(args)
            for i, tree, ps, m in zip(idx, trees, prims, ms, strict=True):
                full[i] = tree.unflatten(
                    [lift(x, *_label(p)) for x, p in zip(m, ps, strict=True)]
                )
            out = f(*full, **kwargs)
            leaves, tree = tree_flatten(out)
            labels = [_label(x) for x in leaves]
            meta[:] = [(tree, labels)]
            return [lower(x) for x in leaves]

        jac = jaxfn(sub, argnums=tuple(range(len(idx))))(*mags)
        otree, labels = meta[0]
        rows: list[object] = []
        for row, (odim, obs) in zip(jac, labels, strict=True):
            per_arg = [
                trees[k].unflatten(
                    [
                        _relabel(c, p, odim, obs)
                        for c, p in zip(row[k], prims[k], strict=True)
                    ]
                )
                for k in range(len(idx))
            ]
            rows.append(per_arg[0] if isinstance(argnums, int) else tuple(per_arg))
        return otree.unflatten(rows)

    return run


@overload
def hessian(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        Quantity[2 * E_M, 2 * E_KG, 2 * E_S, 2 * E_A, 2 * E_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[[Quantity[E_M, E_KG, E_S, E_A, E_K]], jax.Array]: ...
@overload
def hessian(
    f: Callable[
        [Quantity[E_M, E_KG, E_S, E_A, E_K]],
        Quantity[F_M, F_KG, F_S, F_A, F_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[
    [Quantity[E_M, E_KG, E_S, E_A, E_K]],
    Quantity[
        F_M - 2 * E_M,
        F_KG - 2 * E_KG,
        F_S - 2 * E_S,
        F_A - 2 * E_A,
        F_K - 2 * E_K,
    ],
]: ...
@overload
def hessian[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = ...
) -> Callable[P, Any]: ...
def hessian[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = 0
) -> Callable[P, object]:
    """Hessian, computed as a forward-mode Jacobian of a reverse-mode one.

    Args:
      f: the function to differentiate twice.
      argnums: index or indices to differentiate with respect to.

    Returns:
      A function returning the Hessian, of dimension ``dim(out)/dim(arg)^2`` and
      shape ``out.shape + arg.shape + arg.shape``.
    """
    return _jac("fwd", _jac("rev", f, argnums), argnums)


@overload
def vectorize[**P, R](
    fn: None = None,
    *,
    signature: str | None = ...,
    excluded: frozenset[int] = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def vectorize[**P, R](
    fn: Callable[P, R],
    *,
    signature: str | None = ...,
    excluded: frozenset[int] = ...,
) -> Callable[P, R]: ...
def vectorize[**P, R](
    fn: Callable[P, R] | None = None,
    *,
    signature: str | None = None,
    excluded: frozenset[int] = frozenset(),
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Vectorize a function written for one element, keeping units.

    ``jnp.vectorize`` converts every operand with ``jnp.asarray``, which a quantity
    refuses, so the operands' magnitudes are vectorized and their labels reattached
    around each call of ``fn``.

    Args:
      fn: the function to vectorize, written for one element, or None to return a
        decorator.
      signature: a gufunc signature, e.g. ``"(),(n),(n)->(n)"``; None treats every
        operand as a scalar.
      excluded: indices of positional arguments to pass through untouched.

    Returns:
      A function taking the same arguments with leading batch axes allowed; keyword
      arguments are passed to ``fn`` untouched.

    Raises:
      TypeError: if ``excluded`` holds anything but indices, or ``fn`` returns
        a different number of results than the signature names.
      ValueError: if the signature does not parse or an operand/result mismatches it.
    """
    if fn is None:
        return functools.partial(vectorize, signature=signature, excluded=excluded)
    # fn is called with rebuilt positional arguments, which ParamSpec cannot express
    untyped: Callable[..., R] = fn

    @functools.wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        """Apply ``fn`` over the operands' batch axes."""
        labels = [_label(a) for a in args]
        mags = [
            a if i in excluded else lower(cast(ArrayLike, a))
            for i, a in enumerate(args)
        ]
        meta: list[tuple[PyTreeDef, list[tuple[Dim, BaseSystem]]]] = []

        def call(*xs: object) -> object:
            """Call ``fn`` on one element's labelled operands; return magnitudes."""
            full = [
                x if i in excluded else lift(cast(ArrayLike, x), *labels[i])
                for i, x in enumerate(xs)
            ]
            leaves, out_tree = tree_flatten(untyped(*full, **kwargs))
            meta[:] = [(out_tree, [_label(x) for x in leaves])]
            return out_tree.unflatten([lower(x) for x in leaves])

        out = _vectorize(call, excluded=excluded, signature=signature)(*mags)
        out_tree, out_labels = meta[0]
        return out_tree.unflatten(
            [
                lift(m, *lab)
                for m, lab in zip(out_tree.flatten_up_to(out), out_labels, strict=True)
            ]
        )

    return run
