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
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import jax
import jax.numpy as jnp
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike

from kups.core.units._base import active_base
from kups.core.units._core import (
    AnyQuantity,
    Quantity,
    _dim,
    _lift,
    _lower,
    _rebase,
    _shape,
    _sym,
    is_leaf,
    tree_map,
)
from kups.core.units._precision import _check_dtype, _rebase_factor
from kups.core.units._unit import ONE, Dim, DimensionError, ddiv, exps
from kups.core.utils.jax import vectorize as _vectorize

_AUX = TypeVar("_AUX")

# jax.vjp is overloaded on a literal has_aux; this binding takes it as a bool.
_jax_vjp: Callable[
    ..., tuple[object, Callable[[object], tuple[object, ...]], *tuple[object, ...]]
] = jax.vjp

if TYPE_CHECKING:
    from kups.core.units._base import BaseSystem
    from kups.core.units._typed import (
        _E_A,
        _E_K,
        _E_KG,
        _E_M,
        _E_S,
        _F_A,
        _F_K,
        _F_KG,
        _F_M,
        _F_S,
        _G_A,
        _G_K,
        _G_KG,
        _G_M,
        _G_S,
    )


def _plain_leaf[T](v: AnyQuantity | T) -> jax.Array | T:
    """Lower one leaf to plain numbers; a plain leaf passes unchanged."""
    return _lower(v) if isinstance(v, Quantity) else v


def _label(x: object) -> tuple[Dim, BaseSystem]:
    """The dimension and base of one leaf; a plain leaf is dimensionless."""
    return (x.dim, x.bs) if isinstance(x, Quantity) else (ONE, active_base())


def _relabel(
    ct: AnyQuantity | ArrayLike, primal: object, odim: Dim, obs: BaseSystem
) -> object:
    """Relabel one cotangent leaf as ``dim(out)/dim(primal)`` in the primal's base."""
    lo = _lower(ct)  # jax returns it labelled like the primal
    if not isinstance(primal, Quantity):
        return _lift(lo, odim, obs)  # w.r.t. a plain array it still has dim(out)
    fac = _rebase_factor(obs, primal.bs, odim, primal.dtype)
    return _lift(lo if fac == 1.0 else lo * fac, ddiv(odim, primal.dim), primal.bs)


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
    _dim(dim, "relabel")

    def one(v: AnyQuantity | ArrayLike) -> object:
        """Relabel one leaf."""
        into = bs or (v.bs if isinstance(v, Quantity) else active_base())
        lo = _plain_leaf(v)
        _check_dtype(jnp.result_type(lo), dim, into, "relabel")
        return _lift(lo, dim, into)

    return tree_map(one, x)


def _seed(x: AnyQuantity | ArrayLike, want: object, what: str) -> object:
    """Label a tangent or cotangent leaf like the leaf ``want``, rebasing if needed."""
    if not isinstance(want, Quantity):
        if not isinstance(x, Quantity):
            return x
        raise DimensionError(
            f"kups.core.units: this {what} is {_sym(x.dim, x.bs)} where the value it "
            "belongs to "
            f"is a plain array, and jax accepts no other label there. Divide by a "
            f"reference quantity, or relabel it (units.relabel(x, units.ONE))."
        )
    if not isinstance(x, Quantity):
        return _lift(x, want.dim, want.bs)
    if x.dim != want.dim:
        raise DimensionError(
            f"kups.core.units: this {what} is {_sym(x.dim, x.bs)} where the value it "
            "belongs to "
            f"is {_sym(want.dim, want.bs)}, and jax accepts no other label there: a "
            f"{what} lives in the same space as its own value.\n"
            f"  {_sym(x.dim, x.bs)}: {exps(x.dim)}\n"
            f"  {_sym(want.dim, want.bs)}: {exps(want.dim)}\n"
            f"Relabel it (units.relabel(x, units.dim_of(unit))) if the numbers are "
            f"already the ones you want."
        )
    return x if x.bs == want.bs else _rebase(x, want.bs)


def _seeded(xs: object, ref: object, what: str) -> object:
    """Run ``_seed`` on every leaf of a tangent or cotangent tree."""
    return tree_map(lambda x, r: _seed(x, r, what), xs, ref)


def _leaf_name(x: object) -> str:
    """Describe what one output leaf measures, for error messages."""
    return _sym(x.dim, x.bs) if isinstance(x, Quantity) else "a plain array"


def _out_label(name: str, out: object) -> tuple[Dim, BaseSystem]:
    """Find the one dimension and base shared by a differentiated output.

    Plain leaves count as dimensionless.
    """
    leaves = jax.tree.leaves(out, is_leaf=is_leaf)
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
                f"units.{name}: the output measures {_sym(qs[0].dim, qs[0].bs)} in "
                f"base {qs[0].bs.name} and in base {t.bs.name}, so its leaves' "
                f"magnitudes are in different units and the contributions they add up "
                f"to are not one number. Build the whole output in one base "
                f"(q.rebase()), or differentiate the parts separately."
            )
    return (qs[0].dim, qs[0].bs) if qs else (ONE, active_base())


@overload
def vjp(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    ],
    primal: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    /,
    *,
    has_aux: Literal[False] = ...,
) -> tuple[
    Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    Callable[
        [Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K] | ArrayLike],
        tuple[
            Quantity[
                _F_M - _E_M,
                _F_KG - _E_KG,
                _F_S - _E_S,
                _F_A - _E_A,
                _F_K - _E_K,
            ]
        ],
    ],
]: ...
@overload
def vjp(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        tuple[Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K], _AUX],
    ],
    primal: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    /,
    *,
    has_aux: Literal[True],
) -> tuple[
    Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    Callable[
        [Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K] | ArrayLike],
        tuple[
            Quantity[
                _F_M - _E_M,
                _F_KG - _E_KG,
                _F_S - _E_S,
                _F_A - _E_A,
                _F_K - _E_K,
            ]
        ],
    ],
    _AUX,
]: ...
@overload
def vjp(
    f: Callable[[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]], jax.Array],
    primal: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    /,
    *,
    has_aux: Literal[False] = ...,
) -> tuple[
    jax.Array,
    Callable[[ArrayLike], tuple[Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]]],
]: ...
@overload
def vjp(
    f: Callable[
        [
            Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
            Quantity[_G_M, _G_KG, _G_S, _G_A, _G_K],
        ],
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    ],
    primal: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    second: Quantity[_G_M, _G_KG, _G_S, _G_A, _G_K],
    /,
    *,
    has_aux: Literal[False] = ...,
) -> tuple[
    Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    Callable[
        [Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K] | ArrayLike],
        tuple[
            Quantity[
                _F_M - _E_M,
                _F_KG - _E_KG,
                _F_S - _E_S,
                _F_A - _E_A,
                _F_K - _E_K,
            ],
            Quantity[
                _F_M - _G_M,
                _F_KG - _G_KG,
                _F_S - _G_S,
                _F_A - _G_A,
                _F_K - _G_K,
            ],
        ],
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
      returns one cotangent per primal, each labelled ``dim(out)/dim(primal)``.

    Raises:
      DimensionError: if the output's leaves do not share one dimension and base,
        or a cotangent leaf's dimension differs from the output's.
    """
    out, vjp_fn, aux = _vjp(f, primals, has_aux)
    return (out, vjp_fn, aux) if has_aux else (out, vjp_fn)


def _vjp(
    f: Callable[..., object], primals: tuple[object, ...], has_aux: bool
) -> tuple[object, Callable[[object], tuple[object, ...]], object]:
    """``vjp`` returning ``(out, vjp_fn, aux)``, with aux None unless ``has_aux``."""
    res = _jax_vjp(f, *primals, has_aux=has_aux)
    out, raw = res[0], res[1]
    odim, obs = _out_label("vjp", out)

    def vjp_fn(cotangents: object) -> tuple[object, ...]:
        """Pull back an output cotangent to one cotangent per primal."""
        cts = raw(_seeded(cotangents, out, "cotangent"))
        return tuple(
            tree_map(lambda c, p: _relabel(c, p, odim, obs), ct, primal)
            for ct, primal in zip(cts, primals, strict=True)
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

        def sub(*sel: object) -> object:
            full = list(args)
            for i, v in zip(idx, sel, strict=True):
                full[i] = v
            return call(*full, **kwargs)

        out, pull, aux = _vjp(sub, tuple(args[i] for i in idx), has_aux)
        shape = _shape(cast(AnyQuantity | ArrayLike, out))
        if shape != ():
            raise TypeError(f"units.{name} needs a scalar output, got shape {shape}")
        gs = pull(jnp.ones((), jnp.result_type(_plain_leaf(out))))
        g = gs[0] if isinstance(argnums, int) else gs
        val = (out, aux) if has_aux else out
        return (val, g) if with_value else (g, aux) if has_aux else g

    return run


@overload
def grad(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]], jax.Array]: ...
@overload
def grad(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[
    [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
    Quantity[_F_M - _E_M, _F_KG - _E_KG, _F_S - _E_S, _F_A - _E_A, _F_K - _E_K],
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
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[
    [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
    tuple[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K], jax.Array],
]: ...
@overload
def value_and_grad(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    ],
    argnums: Literal[0] = ...,
    has_aux: Literal[False] = ...,
) -> Callable[
    [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
    tuple[
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
        Quantity[
            _F_M - _E_M,
            _F_KG - _E_KG,
            _F_S - _E_S,
            _F_A - _E_A,
            _F_K - _E_K,
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
    name: Literal["jacrev", "jacfwd"],
    f: Callable[..., object],
    argnums: int | tuple[int, ...],
) -> Callable[..., object]:
    """Build a unit-aware jacrev or jacfwd."""
    idx = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    jaxfn = jax.jacrev if name == "jacrev" else jax.jacfwd

    @functools.wraps(f)
    def run(*args: object, **kwargs: object) -> object:
        _check_argnums(name, argnums, idx, len(args))
        prims: list[list[object]] = []
        trees: list[PyTreeDef] = []
        for i in idx:
            leaves, tree = jax.tree.flatten(args[i], is_leaf=is_leaf)
            prims.append(leaves)
            trees.append(tree)
        mags = [[_plain_leaf(x) for x in leaves] for leaves in prims]
        meta: list[tuple[PyTreeDef, list[tuple[Dim, BaseSystem]]]] = []

        def sub(*ms: list[jax.Array]) -> list[object]:
            """Run ``f`` on magnitudes, lifting arguments and lowering outputs."""
            full = list(args)
            for i, tree, ps, m in zip(idx, trees, prims, ms, strict=True):
                full[i] = tree.unflatten(
                    [_lift(x, *_label(p)) for x, p in zip(m, ps, strict=True)]
                )
            out = f(*full, **kwargs)
            leaves, tree = jax.tree.flatten(out, is_leaf=is_leaf)
            labels = [_label(x) for x in leaves]
            meta[:] = [(tree, labels)]
            return [_plain_leaf(x) for x in leaves]

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
def jacrev(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]], jax.Array]: ...
@overload
def jacrev(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[
    [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
    Quantity[_F_M - _E_M, _F_KG - _E_KG, _F_S - _E_S, _F_A - _E_A, _F_K - _E_K],
]: ...
@overload
def jacrev[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = ...
) -> Callable[P, Any]: ...
def jacrev[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = 0
) -> Callable[P, object]:
    """Reverse-mode Jacobian, with units.

    Args:
      f: the function to differentiate; its output may be any shape and any pytree.
      argnums: index or indices to differentiate with respect to.

    Returns:
      A function returning the Jacobian; each leaf has dimension
      ``dim(out leaf)/dim(arg leaf)`` and shape ``out.shape + arg.shape``.
    """
    return _jac("jacrev", f, argnums)


@overload
def jacfwd(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]], jax.Array]: ...
@overload
def jacfwd(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[
    [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
    Quantity[_F_M - _E_M, _F_KG - _E_KG, _F_S - _E_S, _F_A - _E_A, _F_K - _E_K],
]: ...
@overload
def jacfwd[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = ...
) -> Callable[P, Any]: ...
def jacfwd[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = 0
) -> Callable[P, object]:
    """Forward-mode Jacobian, with units.

    Args:
      f: the function to differentiate; its output may be any shape and any pytree.
      argnums: index or indices to differentiate with respect to.

    Returns:
      A function returning the Jacobian, labelled like ``units.jacrev``'s.
    """
    return _jac("jacfwd", f, argnums)


@overload
def hessian(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[2 * _E_M, 2 * _E_KG, 2 * _E_S, 2 * _E_A, 2 * _E_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]], jax.Array]: ...
@overload
def hessian(
    f: Callable[
        [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
        Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K],
    ],
    argnums: Literal[0] = ...,
) -> Callable[
    [Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]],
    Quantity[
        _F_M - 2 * _E_M,
        _F_KG - 2 * _E_KG,
        _F_S - 2 * _E_S,
        _F_A - 2 * _E_A,
        _F_K - 2 * _E_K,
    ],
]: ...
@overload
def hessian[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = ...
) -> Callable[P, Any]: ...
def hessian[**P](
    f: Callable[P, object], argnums: int | tuple[int, ...] = 0
) -> Callable[P, object]:
    """Hessian, computed as ``units.jacfwd`` of ``units.jacrev``.

    Args:
      f: the function to differentiate twice.
      argnums: index or indices to differentiate with respect to.

    Returns:
      A function returning the Hessian, of dimension ``dim(out)/dim(arg)^2`` and
      shape ``out.shape + arg.shape + arg.shape``.
    """
    return jacfwd(jacrev(f, argnums), argnums)


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
            a if i in excluded else _lower(cast(ArrayLike, a))
            for i, a in enumerate(args)
        ]
        meta: list[tuple[PyTreeDef, list[tuple[Dim, BaseSystem]]]] = []

        def call(*xs: object) -> object:
            """Call ``fn`` on one element's labelled operands; return magnitudes."""
            full = [
                x if i in excluded else _lift(cast(ArrayLike, x), *labels[i])
                for i, x in enumerate(xs)
            ]
            leaves, out_tree = jax.tree.flatten(
                untyped(*full, **kwargs), is_leaf=is_leaf
            )
            meta[:] = [(out_tree, [_label(x) for x in leaves])]
            return out_tree.unflatten([_plain_leaf(x) for x in leaves])

        out = _vectorize(call, excluded=excluded, signature=signature)(*mags)
        out_tree, out_labels = meta[0]
        return out_tree.unflatten(
            [
                _lift(m, *lab)
                for m, lab in zip(out_tree.flatten_up_to(out), out_labels, strict=True)
            ]
        )

    return run
