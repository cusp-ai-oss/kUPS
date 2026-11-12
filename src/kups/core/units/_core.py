# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Core quantity types and the operations behind their methods.

A dimension is five rational exponents over ``(m, kg, s, A, K)``; ``mol`` is
dimensionless with scale ``N_A``. Magnitudes are canonicalised to the base active
at construction, and a dimensionless result decays to a plain ``jax.Array``. A
quantity is a pytree with the magnitudes as its single leaf and the dimension and
base as metadata.

Methods share the helpers here; dtype checks and scale factors live in
``units._precision``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction as Frac
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Never,
    cast,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike, DTypeLike
from typing_extensions import TypeIs

from kups.core.units._base import BaseSystem, active_base, require_base
from kups.core.units._precision import (
    base_scale,
    cast_advice,
    check_dtype,
    rebase_factor,
)
from kups.core.units._typed import E_A, E_K, E_KG, E_M, E_S
from kups.core.units._unit import (
    AXES,
    ONE,
    AnyUnit,
    Dim,
    DimensionError,
    Unit,
    ddiv,
    dmul,
    dpow,
    exponent,
    exps,
    require_unit,
)
from kups.core.utils.jax import tree_map as _tree_map

if TYPE_CHECKING:
    from shape_extensions import Int

    from kups.core.units._typed import F_A, F_K, F_KG, F_M, F_S, K
    from kups.core.units._unit import Dimensionless

_Op = Callable[..., jax.Array]
type _Nested = float | list[_Nested]
_Index = (
    int
    | slice
    | EllipsisType
    | ArrayLike
    | tuple[int | slice | EllipsisType | ArrayLike | None, ...]
    | None
)


_NO_ARRAY = (
    "kups.core.units: a quantity does not convert to a plain array implicitly. Use "
    "q.to(unit) for magnitudes in a chosen unit, or q.value for magnitudes in "
    "q's base. For comparisons, use units.testing.assert_allclose(a, b, unit)."
)
_NO_EQ = (
    "kups.core.units: == and != are not defined on quantities, because an "
    "elementwise result would masquerade as a Python bool. Compare magnitudes "
    "(a.to(unit) == b.to(unit)) or use < <= > >=."
)


class OddExponentRoot:
    """Static type of ``sqrt`` on a dimension with an odd exponent.

    Supports no operations, so using the result is a type error. At runtime the
    root is a quantity with half-integer exponents; for that, write
    ``x ** Fraction(1, 2)``.
    """


@dataclasses.dataclass(frozen=True, eq=False)
class Quantity(Generic[E_M, E_KG, E_S, E_A, E_K]):
    """Magnitudes in the units of base ``bs`` for dimension ``dim``.

    The dimension exponents are type parameters, so a checker with symbolic
    integer support (pyrefly) tracks them through arithmetic.

    Attributes:
      v: the magnitudes, stored as given; use ``q.value`` for an array.
      dim: dimension exponents over ``(m, kg, s, A, K)``.
      bs: the base the magnitudes are expressed in.
    """

    v: ArrayLike
    dim: Dim
    bs: BaseSystem

    # no ``__jax_array__``: it would break ``plain_array * quantity``
    __array_ufunc__: ClassVar[None] = None  # ``np.array(...) * q`` reaches __rmul__

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> Never:
        """Refuse conversion to a numpy or jax array."""
        del dtype, copy
        raise DimensionError(_NO_ARRAY)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the magnitudes."""
        return jnp.shape(self.v)

    @property
    def ndim(self) -> int:
        """Rank of the magnitudes."""
        return jnp.ndim(self.v)

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Dtype of the magnitudes."""
        return jnp.result_type(self.v)

    @property
    def size(self) -> int:
        """Number of magnitudes."""
        return math.prod(self.shape)

    @property
    def sharding(self) -> jax.sharding.Sharding:
        """Sharding of the magnitudes."""
        return cast("jax.Array", self.v).sharding

    def devices(self) -> set[jax.Device]:
        """Devices the magnitudes are on."""
        return cast("jax.Array", self.v).devices()

    def __format__(self, spec: str) -> str:
        """Magnitudes formatted by ``spec``, then the unit symbol: ``"1.000 Å"``."""
        sym = self.bs.symbol(self.dim)
        body = format(self.v, spec) if spec else str(self.v)
        return f"{body} {sym}" if sym else body

    def __repr__(self) -> str:
        sym = self.bs.symbol(self.dim)
        return f"{self.v!s} {sym}" if sym else str(self.v)

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of a 0-d quantity")
        return self.shape[0]

    @property
    def value(self) -> jax.Array:
        """Raw magnitudes, in the quantity's own base."""
        return lower(self)

    @property
    def unit(self) -> Unit[E_M, E_KG, E_S, E_A, E_K]:
        """One unit of this quantity's dimension, sized by its base.

        ``q.to(q.unit)`` is ``q.value``; a catalog unit of that size is preferred.
        """
        return self.bs.unit(self.dim)

    def __add__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Add magnitudes of the same dimension."""
        if not operand(other):
            return NotImplemented
        return _add(self, other)

    def __radd__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Add magnitudes of the same dimension, reflected."""
        if not operand(other):
            return NotImplemented
        return _add(other, self)

    def __sub__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Subtract magnitudes of the same dimension."""
        if not operand(other):
            return NotImplemented
        return _sub(self, other)

    def __rsub__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Subtract magnitudes of the same dimension, reflected."""
        if not operand(other):
            return NotImplemented
        return _sub(other, self)

    def __neg__(self) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Negate the magnitudes, keeping dimension and base."""
        return lift(-lower(self), self.dim, self.bs)

    def __abs__(self) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Elementwise absolute value, keeping dimension and base."""
        return apply(jnp.abs, self)

    def __lt__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> jax.Array:
        """Elementwise less-than; returns a plain boolean array."""
        if not operand(other):
            return NotImplemented
        return _compare(self, other, operator.lt)

    def __le__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> jax.Array:
        """Elementwise less-or-equal; returns a plain boolean array."""
        if not operand(other):
            return NotImplemented
        return _compare(self, other, operator.le)

    def __gt__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> jax.Array:
        """Elementwise greater-than; returns a plain boolean array."""
        if not operand(other):
            return NotImplemented
        return _compare(self, other, operator.gt)

    def __ge__(
        self,
        other: Quantity[E_M, E_KG, E_S, E_A, E_K] | Unit[E_M, E_KG, E_S, E_A, E_K],
    ) -> jax.Array:
        """Elementwise greater-or-equal; returns a plain boolean array."""
        if not operand(other):
            return NotImplemented
        return _compare(self, other, operator.ge)

    # == and != always raise, hence Never
    def __eq__(self, other: object) -> Never:
        """Refused: an elementwise result would masquerade as a Python bool."""
        raise DimensionError(_NO_EQ)

    def __ne__(self, other: object) -> Never:
        """Refused: an elementwise result would masquerade as a Python bool."""
        raise DimensionError(_NO_EQ)

    __hash__ = object.__hash__  # defining __eq__ clears it

    def __float__(self) -> float:
        """Magnitude of a 0-d quantity as a Python float, in its own base."""
        return float(_concrete(self, "float(q)"))

    def __int__(self) -> int:
        """Magnitude of a 0-d quantity as a Python int, in its own base."""
        return int(_concrete(self, "int(q)"))

    def __bool__(self) -> bool:
        """Truth value of the magnitudes, with jax's array semantics."""
        return bool(lower(self))

    def __getitem__(self, idx: _Index) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Index the magnitudes, keeping dimension and base."""
        return lift(lower(self)[idx], self.dim, self.bs)

    def __iter__(self) -> Iterator[Quantity[E_M, E_KG, E_S, E_A, E_K]]:
        """Iterate over the leading axis; unrolled so it works inside a trace."""
        return _unrolled(self)

    # mod keeps the shared dimension; floor division cancels it
    def __mod__(
        self, other: Quantity[E_M, E_KG, E_S, E_A, E_K]
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Elementwise remainder of two operands of the same dimension."""
        if not operand(other):
            return NotImplemented
        return _mod(self, other)

    def __rmod__(
        self, other: Quantity[E_M, E_KG, E_S, E_A, E_K]
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Elementwise remainder, reflected."""
        if not operand(other):
            return NotImplemented
        return _mod(other, self)

    def __floordiv__(self, other: Quantity[E_M, E_KG, E_S, E_A, E_K]) -> jax.Array:
        """Elementwise floor division; a count is dimensionless."""
        if not operand(other):
            return NotImplemented
        return _floordiv(self, other)

    def __rfloordiv__(self, other: Quantity[E_M, E_KG, E_S, E_A, E_K]) -> jax.Array:
        """Elementwise floor division, reflected."""
        if not operand(other):
            return NotImplemented
        return _floordiv(other, self)

    # the first overload of each pair is the cancelling case: a plain array
    @overload
    def __mul__(self, other: Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K]) -> jax.Array: ...
    @overload
    def __mul__(
        self, other: Quantity[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]: ...
    @overload
    def __mul__(self, other: Dimensionless) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    @overload
    def __mul__(self, other: Unit[-E_M, -E_KG, -E_S, -E_A, -E_K]) -> jax.Array: ...
    @overload
    def __mul__(
        self, other: Unit[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]: ...
    @overload
    def __mul__(self, other: ArrayLike) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    def __mul__(
        self, other: AnyQuantity | AnyUnit | ArrayLike
    ) -> jax.Array | AnyQuantity:
        """Multiply; the dimensions multiply too."""
        if not operand(other):
            return NotImplemented
        return _mul(self, other)

    @overload
    def __rmul__(self, other: Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K]) -> jax.Array: ...
    @overload
    def __rmul__(
        self, other: Quantity[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]: ...
    @overload
    def __rmul__(self, other: Dimensionless) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    @overload
    def __rmul__(self, other: Unit[-E_M, -E_KG, -E_S, -E_A, -E_K]) -> jax.Array: ...
    @overload
    def __rmul__(
        self, other: Unit[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]: ...
    @overload
    def __rmul__(self, other: ArrayLike) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    def __rmul__(
        self, other: AnyQuantity | AnyUnit | ArrayLike
    ) -> jax.Array | AnyQuantity:
        """Multiply, reflected; the dimensions multiply too."""
        if not operand(other):
            return NotImplemented
        return _mul(other, self)

    @overload
    def __truediv__(self, other: Quantity[E_M, E_KG, E_S, E_A, E_K]) -> jax.Array: ...
    @overload
    def __truediv__(
        self, other: Quantity[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M - F_M,
        E_KG - F_KG,
        E_S - F_S,
        E_A - F_A,
        E_K - F_K,
    ]: ...
    @overload
    def __truediv__(
        self, other: Dimensionless
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    @overload
    def __truediv__(self, other: Unit[E_M, E_KG, E_S, E_A, E_K]) -> jax.Array: ...
    @overload
    def __truediv__(
        self, other: Unit[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M - F_M,
        E_KG - F_KG,
        E_S - F_S,
        E_A - F_A,
        E_K - F_K,
    ]: ...
    @overload
    def __truediv__(self, other: ArrayLike) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    def __truediv__(
        self, other: AnyQuantity | AnyUnit | ArrayLike
    ) -> jax.Array | AnyQuantity:
        """Divide; the dimensions divide too."""
        if not operand(other):
            return NotImplemented
        return _div(self, other)

    @overload
    def __rtruediv__(self, other: Quantity[E_M, E_KG, E_S, E_A, E_K]) -> jax.Array: ...
    @overload
    def __rtruediv__(
        self, other: Quantity[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        F_M - E_M,
        F_KG - E_KG,
        F_S - E_S,
        F_A - E_A,
        F_K - E_K,
    ]: ...
    @overload
    def __rtruediv__(
        self, other: Dimensionless
    ) -> Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K]: ...
    @overload
    def __rtruediv__(self, other: Unit[E_M, E_KG, E_S, E_A, E_K]) -> jax.Array: ...
    @overload
    def __rtruediv__(
        self, other: Unit[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        F_M - E_M,
        F_KG - E_KG,
        F_S - E_S,
        F_A - E_A,
        F_K - E_K,
    ]: ...
    @overload
    def __rtruediv__(
        self, other: ArrayLike
    ) -> Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K]: ...
    def __rtruediv__(
        self, other: AnyQuantity | AnyUnit | ArrayLike
    ) -> jax.Array | AnyQuantity:
        """Divide, reflected; the dimensions divide too."""
        if not operand(other):
            return NotImplemented
        return _div(other, self)

    @overload
    def __matmul__(
        self, other: Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K]
    ) -> jax.Array: ...
    @overload
    def __matmul__(
        self, other: Quantity[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]: ...
    @overload
    def __matmul__(self, other: ArrayLike) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    def __matmul__(
        self, other: AnyQuantity | AnyUnit | ArrayLike
    ) -> jax.Array | AnyQuantity:
        """Matrix product; the dimensions multiply."""
        if not operand(other):
            return NotImplemented
        return _matmul(self, other)

    @overload
    def __rmatmul__(
        self, other: Quantity[-E_M, -E_KG, -E_S, -E_A, -E_K]
    ) -> jax.Array: ...
    @overload
    def __rmatmul__(
        self, other: Quantity[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]: ...
    @overload
    def __rmatmul__(self, other: ArrayLike) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    def __rmatmul__(
        self, other: AnyQuantity | AnyUnit | ArrayLike
    ) -> jax.Array | AnyQuantity:
        """Matrix product, reflected; the dimensions multiply."""
        if not operand(other):
            return NotImplemented
        return _matmul(other, self)

    @overload
    def __pow__(self, e: Literal[0]) -> jax.Array: ...
    @overload
    def __pow__(
        self, e: Int[K]
    ) -> Quantity[E_M * K, E_KG * K, E_S * K, E_A * K, E_K * K]: ...
    @overload
    def __pow__(self, e: float | Frac) -> Any: ...  # not an integer exponent
    def __pow__(self, e: int | float | Frac) -> jax.Array | AnyQuantity:
        """Raise to a static rational power; the dimension is raised with it."""
        return _pow(self, e)

    def to(self, unit: Unit[E_M, E_KG, E_S, E_A, E_K]) -> jax.Array:
        """Magnitudes measured in ``unit``, as a plain array."""
        return _to(self, unit)

    def rebase(
        self, bs: BaseSystem | None = None
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Re-express the quantity in another base (the active one by default)."""
        return rebase(self, bs)

    def item(self) -> float:
        """The single magnitude of a one-element quantity, in its own base."""
        return _concrete(self, "q.item()").item()

    def tolist(self) -> _Nested:
        """The magnitudes as nested Python lists, in the quantity's own base."""
        return _concrete(self, "q.tolist()").tolist()

    def astype(self, dtype: DTypeLike) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Cast the magnitudes to another dtype."""
        dt = jnp.result_type(dtype)
        if self.dim != ONE and not jnp.issubdtype(dt, jnp.floating):
            raise DimensionError(
                f"q.astype: {dt.name} magnitudes cannot carry "
                f"{self.bs.symbol(self.dim)!r}; "
                "unit conversions require floating magnitudes. Use "
                "q.to(unit).astype(dtype) or q.value.astype(dtype) for unlabelled "
                "magnitudes."
            )
        return lift(jnp.astype(lower(self), dtype), self.dim, self.bs)

    # named overloads catch odd exponents, which ``//`` would floor
    @overload
    def sqrt(self: Acceleration) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Action) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Charge) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Compressibility) -> OddExponentRoot: ...
    @overload
    def sqrt(self: CoulombConstant) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Current) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Density) -> OddExponentRoot: ...
    @overload
    def sqrt(self: DipoleMoment) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Energy) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Entropy) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Force) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Frequency) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Length) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Mass) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Momentum) -> OddExponentRoot: ...
    @overload
    def sqrt(self: NumberDensity) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Permittivity) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Power) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Pressure) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Stiffness) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Temperature) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Time) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Velocity) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Voltage) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Volume) -> OddExponentRoot: ...
    @overload
    def sqrt(self: Wavenumber) -> OddExponentRoot: ...
    @overload
    def sqrt(
        self: Quantity[E_M, E_KG, E_S, E_A, E_K],
    ) -> Quantity[E_M // 2, E_KG // 2, E_S // 2, E_A // 2, E_K // 2]: ...
    @overload
    def sqrt(self) -> Any: ...
    def sqrt(self) -> Any:
        """Square root; halves every dimension exponent."""
        return sqrt_impl(self)

    if not TYPE_CHECKING:
        # runtime error, hidden so the checker still flags the expression

        def __rpow__(self, other):
            """Refused: a quantity exponent has no fixed dimension."""
            return _pow(other, self)


# a single-leaf pytree; the labels are metadata compared by value
# jax's native dataclass registration, not kups's dataclass, for flatten speed
jax.tree_util.register_dataclass(Quantity, data_fields=["v"], meta_fields=["dim", "bs"])

type AnyQuantity = Quantity[int, int, int, int, int]


def is_quantity(x: object) -> TypeIs[AnyQuantity]:
    """Whether ``x`` is a quantity, concrete or traced.

    Args:
      x: any value.

    Returns:
      True for a quantity; False for anything else, including Units.
    """
    return isinstance(x, Quantity)


# ``is_leaf=units.is_leaf`` makes ``jax.tree.map`` pass quantities whole
is_leaf = is_quantity


def tree_flatten(x: object) -> tuple[list[Any], PyTreeDef]:
    """Flatten a pytree, treating each quantity as a single leaf.

    Args:
      x: any pytree.

    Returns:
      The leaves (each quantity one leaf) and the treedef that reassembles them.
    """
    return jax.tree.flatten(x, is_leaf=is_leaf)


def tree_leaves(x: object) -> list[Any]:
    """The leaves of a pytree, with each quantity as one leaf.

    Args:
      x: any pytree.

    Returns:
      The leaf list.
    """
    return jax.tree.leaves(x, is_leaf=is_leaf)


def tree_map[T, *S](f: Callable[..., object], x: T, *rest: *S) -> T:
    """Map over a pytree's leaves, passing each quantity to ``f`` whole.

    Args:
      f: the function applied to each leaf.
      x: the pytree whose structure the result has.
      *rest: further pytrees of the same structure, whose leaves ``f`` also takes.

    Returns:
      A pytree with ``x``'s structure holding ``f``'s results.
    """
    return _tree_map(f, x, *rest, is_leaf=is_leaf)


def lift(x: ArrayLike, dim: Dim, bs: BaseSystem) -> Any:
    """Attach a dimension to magnitudes; a dimensionless result stays an array."""
    return x if dim == ONE else Quantity(x, dim, bs)


def lower(q: AnyQuantity | ArrayLike) -> jax.Array:
    """Extract magnitudes as an array; plain values pass."""
    v = q.v if isinstance(q, Quantity) else q
    return v if isinstance(v, (jax.Array, jax.core.Tracer)) else jnp.asarray(v)


def dim_of(x: object) -> Dim:
    """Dimension of any value.

    Args:
      x: any value; a non-kups.core.units value counts as dimensionless.

    Returns:
      Its dimension exponents.
    """
    return x.dim if isinstance(x, (Unit, Quantity)) else ONE


def require_dim(x: object, unit: AnyUnit) -> None:
    """Check that a value has the same dimension as ``unit``.

    Args:
      x: a quantity, or any other value (treated as dimensionless).
      unit: the unit whose dimension the value must match.

    Raises:
      DimensionError: if the dimensions differ.
    """
    got = dim_of(x)
    if got == unit.dim:
        return
    raise DimensionError(
        f"kups.core.units: expected a value in {unit.name}, got one in "
        f"{active_base().symbol(got) or 'no unit at all'}\n"
        f"  expected {exps(unit.dim)}\n  received {exps(got)}"
    )


type Measured = jax.Array | Quantity[int, int, int, int, int]
"""A quantity of any dimension, or a plain array of magnitudes."""


def magnitudes(x: Measured) -> jax.Array:
    """Magnitudes of ``x`` in its own base.

    A plain array is returned unchanged, read as magnitudes in the active base.

    Args:
      x: a quantity, or the magnitudes of one.

    Returns:
      The raw array.
    """
    return x.value if is_quantity(x) else x


def magnitudes_of[T](tree: T) -> T:
    """Magnitudes of every quantity leaf of ``tree``, in its own base.

    Args:
      tree: any pytree, whose leaves may carry dimensions.

    Returns:
      The same structure with plain magnitudes.
    """
    return tree_map(magnitudes, tree)


def base_of(x: object) -> BaseSystem:
    """The base system a quantity's magnitudes are expressed in.

    Args:
      x: a quantity, concrete or traced.

    Returns:
      The base system.

    Raises:
      TypeError: if ``x`` is not a quantity.
    """
    if not isinstance(x, Quantity):
        raise TypeError(
            f"kups.core.units: {x!r} is not a quantity. Use units.active_base() for "
            "the base "
            "a plain array would be attached in."
        )
    return x.bs


def checked_dim(dim: object, what: str) -> Dim:
    """Check that ``dim`` is a dimension vector; ``what`` names the caller."""
    if (
        isinstance(dim, tuple)
        and len(dim) == len(AXES)
        and all(isinstance(v, int | Frac) for v in dim)
    ):
        return dim
    raise DimensionError(
        f"units.{what}: {dim!r} is not a dimension. A dimension is five rational "
        f"exponents over {AXES}; use units.dim_of(unit)."
    )


def dim_str(dim: Dim) -> str:
    """Format a dimension as ``axis=exponent`` text.

    Args:
      dim: dimension vector.

    Returns:
      A string like ``"m=2 kg=1 s=-2 A=0 K=0"`` (an energy).

    Raises:
      DimensionError: if ``dim`` is not a dimension vector.
    """
    return " ".join(
        f"{a}={v}" for a, v in zip(AXES, checked_dim(dim, "dim_str"), strict=True)
    )


def from_dim(
    magnitudes: ArrayLike | AnyQuantity, dim: Dim, bs: BaseSystem | None = None
) -> Any:
    """Attach a dimension to plain magnitudes.

    Args:
      magnitudes: the numbers, as anything ``jnp.asarray`` accepts.
      dim: the dimension they measure, e.g. ``units.dim_of(u.eV)``.
      bs: the base they are expressed in, or None for the active one.

    Returns:
      The quantity, or a plain jax.Array when ``dim`` is dimensionless.

    Raises:
      TypeError: if ``bs`` is not a BaseSystem.
      DimensionError: if ``dim`` is not a dimension vector, if the magnitudes are
        complex or integral, or if they already carry a dimension.
    """
    checked_dim(dim, "from_dim")
    if bs is not None:
        require_base(bs, "the base the magnitudes are in")
    if isinstance(magnitudes, Quantity):
        raise DimensionError(
            "units.from_dim: these magnitudes are "
            f"{sym(magnitudes.dim, magnitudes.bs)} "
            "already, and "
            "from_dim reads plain numbers. Take q.value for the magnitudes in q's own "
            "base."
        )
    v = jnp.asarray(magnitudes)
    if jnp.issubdtype(v.dtype, jnp.complexfloating):
        raise DimensionError(
            f"kups.core.units: complex magnitudes are not supported ({v.dtype}); take "
            "jnp.real/jnp.imag before attaching a dimension"
        )
    v = jnp.astype(v, v.dtype)  # drops weak typing, as from_unit does
    into = bs or active_base()
    check_dtype(v.dtype, dim, into, "from_dim")
    return lift(v, dim, into)


def joint(*xs: object) -> BaseSystem:
    """Common base of the quantity operands, or the active base if they disagree."""
    bs = None
    for x in xs:
        if isinstance(x, Quantity):
            if bs is None:
                bs = x.bs
            elif x.bs != bs:
                return active_base()
    return active_base() if bs is None else bs


def _unit_mag(unit: AnyUnit, bs: BaseSystem) -> float:
    """One ``unit`` as a magnitude in ``bs``."""
    return unit.scale / base_scale(bs, unit.dim)


def mag[T](
    x: AnyQuantity | AnyUnit | T, bs: BaseSystem
) -> tuple[jax.Array | float | T, Dim]:
    """Magnitudes of ``x`` in base ``bs``, and its dimension."""
    if isinstance(x, Unit):
        return _unit_mag(x, bs), x.dim
    if not isinstance(x, Quantity):
        return x, ONE
    lo = lower(x)
    if x.bs == bs:
        return lo, x.dim
    f = rebase_factor(x.bs, bs, x.dim, x.dtype)
    if f == 1.0:
        return lo, x.dim
    return lo * f, x.dim


def from_unit(x: ArrayLike, unit: AnyUnit) -> jax.Array | AnyQuantity:
    """Canonicalize ``x * unit`` into the active base (the ``5 * u.kcal`` path).

    Args:
      x: magnitudes, in units of ``unit``.
      unit: the unit to attach.

    Returns:
      The quantity, in the active base; a plain array if ``unit`` is dimensionless.

    Raises:
      DimensionError: if the magnitudes are complex.
    """
    bs = active_base()
    v = jnp.asarray(x)
    if jnp.issubdtype(v.dtype, jnp.complexfloating):
        raise DimensionError(
            f"kups.core.units: complex magnitudes are not supported ({v.dtype}); take "
            "jnp.real/jnp.imag before attaching a unit"
        )
    # an explicit dtype drops weak typing, matching promotion in a trace
    v = jnp.astype(
        v, v.dtype if jnp.issubdtype(v.dtype, jnp.floating) else jnp.result_type(float)
    )
    f = unit.scale / base_scale(bs, unit.dim)
    if f == 1.0:
        return lift(v, unit.dim, bs)
    return lift(v * f, unit.dim, bs)


def _to(q: object, unit: AnyUnit) -> jax.Array:
    """Measure ``q`` in ``unit``, as a plain array."""
    if not isinstance(q, Quantity):
        raise TypeError(f"kups.core.units: {q!r} is not a quantity")
    require_unit(unit, "the unit to convert to")
    if q.dim != unit.dim:
        raise DimensionError(
            f"cannot express {q.bs.symbol(q.dim) or '1'!r} in {unit!r}: "
            f"dimensions differ\n  value {exps(q.dim)}\n  unit  {exps(unit.dim)}"
        )
    out = lower(q)
    b = base_scale(q.bs, q.dim)
    if unit.scale == 0.0 or not math.isfinite(unit.scale):
        raise DimensionError(
            f"kups.core.units: {unit!r} has no representable size: its exact size is "
            f"outside the float range (scale {unit.scale!r}). "
            f"Measure in a unit closer to the magnitudes' own size."
        )
    if b == unit.scale:
        return out  # rescaling would only lose bits
    if not jnp.issubdtype(q.dtype, jnp.floating):
        raise DimensionError(
            f"kups.core.units: {jnp.dtype(q.dtype).name} magnitudes cannot be measured "
            "in "
            f"{unit!r}: rescaling requires floating magnitudes; "
            f"{cast_advice(q.dtype)}."
        )
    if b == 1.0:
        return out / unit.scale
    if unit.scale == 1.0:
        return out * b
    return out * (b / unit.scale)


def sym(d: Dim, bs: BaseSystem) -> str:
    """A dimension as ``bs``'s unit symbol, for error messages."""
    s = bs.symbol(d)
    return repr(s) if s else "a dimensionless value"


def _mismatch(verb: str, dx: Dim, dy: Dim, bs: BaseSystem) -> DimensionError:
    return DimensionError(
        f"cannot {verb} {sym(dx, bs)} and {sym(dy, bs)}: dimensions differ\n"
        f"  {sym(dx, bs)}: {exps(dx)}\n  {sym(dy, bs)}: {exps(dy)}\n"
        "Attach a unit to the plain operand (x * u.m) or take a ratio first."
    )


def _not_complex[T](out: T, d: Dim, bs: BaseSystem) -> T:
    """Reject a complex result that would be labelled with dimension ``d``."""
    dt = getattr(out, "dtype", None)
    if d != ONE and dt is not None and jnp.issubdtype(dt, jnp.complexfloating):
        raise DimensionError(
            "kups.core.units: complex magnitudes are not supported "
            f"({jnp.dtype(dt).name}); "
            f"this operation would label them {bs.symbol(d) or '1'!r}. Take "
            "jnp.real/jnp.imag of the plain operand before combining them."
        )
    return out


def _binop(
    x: AnyQuantity | AnyUnit | ArrayLike,
    y: AnyQuantity | AnyUnit | ArrayLike,
    op: _Op,
    dimop: Callable[[Dim, Dim], Dim] | str,
) -> Any:
    """Apply ``op`` to magnitudes and combine dimensions with ``dimop``.

    A string ``dimop`` means "require equal dimensions" and names the verb for
    the error.
    """
    bs = joint(x, y)
    lx, dx = mag(x, bs)
    ly, dy = mag(y, bs)
    if isinstance(dimop, str):
        if dx != dy:
            raise _mismatch(dimop, dx, dy, bs)
        d = dx
    else:
        d = dimop(dx, dy)
    return lift(_not_complex(op(lx, ly), d, bs), d, bs)


_add = functools.partial(_binop, op=operator.add, dimop="add")
_sub = functools.partial(_binop, op=operator.sub, dimop="subtract")
_mul = functools.partial(_binop, op=operator.mul, dimop=dmul)
_div = functools.partial(_binop, op=operator.truediv, dimop=ddiv)
_matmul = functools.partial(_binop, op=jnp.matmul, dimop=dmul)


def _compare(
    x: AnyQuantity | AnyUnit | ArrayLike, y: AnyQuantity | AnyUnit | ArrayLike, op: _Op
) -> jax.Array:
    """Compare magnitudes of the same dimension, giving a plain boolean array."""
    mags, _, _ = agree([x, y], "compare")
    return op(*mags)


def _pow(x: AnyQuantity | ArrayLike, e: float | Frac | AnyQuantity) -> Any:
    if isinstance(e, (jax.core.Tracer, Quantity)):
        kind = "a quantity" if isinstance(e, Quantity) else "a traced value"
        raise DimensionError(
            "kups.core.units: the exponent of a quantity must be a static int or "
            f"Fraction, not {kind} ({e!r}); the result's dimension depends on it"
        )
    ef = exponent(e)
    bs = joint(x)
    lx, dx = mag(x, bs)
    return lift(lx ** _jnp_exponent(ef), dpow(dx, ef), bs)


def _jnp_exponent(ef: Frac) -> int | float:
    """A static exponent as jnp takes it; an int keeps ``integer_pow``."""
    return int(ef) if ef.denominator == 1 else float(ef)


def _unrolled(
    x: Quantity[E_M, E_KG, E_S, E_A, E_K],
) -> Iterator[Quantity[E_M, E_KG, E_S, E_A, E_K]]:
    """Iterate over ``x``'s leading axis; unrolled so it works inside a trace."""
    if not x.shape:
        raise TypeError("iteration over a 0-d quantity")
    return iter([x[i] for i in range(x.shape[0])])


def _mod(
    x: Quantity[E_M, E_KG, E_S, E_A, E_K],
    y: Quantity[E_M, E_KG, E_S, E_A, E_K],
) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
    """Elementwise remainder of two operands of the same dimension."""
    mags, dim, bs = agree([x, y], "take the remainder of")
    return lift(jnp.mod(*mags), dim, bs)


def _floordiv(x: AnyQuantity, y: AnyQuantity) -> jax.Array:
    """Elementwise floor division of two operands of the same dimension."""
    mags, _, _ = agree([x, y], "floor-divide")
    return jnp.floor_divide(*mags)


_ARRAYLIKE = (jax.Array, np.ndarray, np.number, np.bool_, int, float, complex)


@overload
def apply(
    fn: Callable[..., Any],
    x: Quantity[E_M, E_KG, E_S, E_A, E_K],
    *,
    dim: None = ...,
) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
@overload
def apply(
    fn: Callable[..., Any],
    x: object,
    *args: object,
    dim: Dim | None = ...,
    **kwargs: object,
) -> Any: ...
def apply(
    fn: Callable[..., Any],
    x: object,
    *args: object,
    dim: Dim | None = None,
    **kwargs: object,
) -> Any:
    """Apply a function to quantities' magnitudes and relabel the result.

    Args:
      fn: function on magnitudes, e.g. ``jnp.flip``.
      x: a quantity or plain array.
      *args: further arguments for ``fn``; quantities among them are lowered too.
      dim: result dimension, or None for the product of the operands' dimensions;
        ``units.ONE`` requests a plain array.
      **kwargs: keyword arguments for ``fn``; quantities among them are lowered too.

    Returns:
      The result in the operands' base, or a plain jax.Array when dimensionless.

    Raises:
      DimensionError: if ``dim`` is not a dimension vector, or a dimensioned
        result is not floating.
    """
    if dim is not None:
        checked_dim(dim, "apply")
    ops = (x, *args, *kwargs.values())
    bs = joint(*ops)
    mags = [mag(o, bs) for o in ops]
    n = len(args)
    lo = [m for m, _ in mags]
    out = fn(lo[0], *lo[1 : n + 1], **dict(zip(kwargs, lo[n + 1 :], strict=True)))
    d = functools.reduce(dmul, (dd for _, dd in mags), ONE) if dim is None else dim
    if d != ONE and not isinstance(out, _ARRAYLIKE):
        raise DimensionError(
            f"units.apply: {getattr(fn, '__name__', fn)!r} returned "
            f"{type(out).__name__}, not one array, so there is no single result to "
            "label. Apply a function that returns one of the outputs instead."
        )
    if d == ONE:
        return lift(out, d, bs)
    dt = jnp.result_type(out)
    if not jnp.issubdtype(dt, jnp.floating):
        raise DimensionError(
            f"units.apply: {getattr(fn, '__name__', fn)!r} returned {dt.name} "
            f"magnitudes, which kups.core.units will not label {bs.symbol(d)!r}: an "
            "index, a "
            f"count or a mask measures nothing. Pass dim=units.dim_of(1.0) for an "
            f"unlabelled result, or call fn on q.to(unit)."
        )
    return lift(out, d, bs)


def agree(
    xs: Sequence[AnyQuantity | AnyUnit | ArrayLike], verb: str
) -> tuple[list[ArrayLike], Dim, BaseSystem]:
    """Lower same-dimension operands into one base: (magnitudes, dimension, base).

    ``verb`` names the operation in errors.
    """
    bs = joint(*xs)
    mags: list[ArrayLike] = []
    dim: Dim | None = None
    for x in xs:
        lo, d = mag(x, bs)
        mags.append(lo)
        if dim is None:
            dim = d
        elif d != dim:
            raise _mismatch(verb, dim, d, bs)
    return mags, ONE if dim is None else dim, bs


def operand(x: object) -> bool:
    """Whether ``x`` can be the other operand of an operator."""
    return isinstance(x, (Unit, Quantity, jax.Array, int, float, complex)) or hasattr(
        x, "__array__"
    )


def _concrete(q: AnyQuantity, what: str) -> jax.Array:
    """Concrete magnitudes in ``q``'s own base; refuses traced values."""
    lo = lower(q)
    if isinstance(lo, jax.core.Tracer):
        raise DimensionError(
            f"kups.core.units: {what} needs a concrete magnitude, and this quantity is "
            "an "
            "argument of the traced function or is derived from one. Hoist the "
            "number out of the traced region (RC = float(CUTOFF.to(u.angstrom))), "
            "or keep the comparison in the array world."
        )
    return lo


def rebase(
    q: Quantity[E_M, E_KG, E_S, E_A, E_K], bs: BaseSystem | None = None
) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
    """Re-express a quantity in another base (the active one by default)."""
    if not isinstance(q, Quantity):
        raise TypeError(f"kups.core.units: {q!r} is not a quantity")
    if bs is not None:
        require_base(bs, "the base to rebase to")
    bs = bs or active_base()
    out, _ = mag(q, bs)
    return lift(out, q.dim, bs)


def sqrt_impl(x: AnyQuantity | ArrayLike) -> Any:
    """Square root of the magnitudes and the dimension."""
    bs = joint(x)
    lx, dim = mag(x, bs)
    return lift(jnp.sqrt(lx), dpow(dim, Frac(1, 2)), bs)


# dimension aliases
type Area = Quantity[2, 0, 0, 0, 0]
type Volume = Quantity[3, 0, 0, 0, 0]
type Velocity = Quantity[1, 0, -1, 0, 0]
type Acceleration = Quantity[1, 0, -2, 0, 0]
type Momentum = Quantity[1, 1, -1, 0, 0]
type Density = Quantity[-3, 1, 0, 0, 0]
type Wavenumber = Quantity[-1, 0, 0, 0, 0]
type Stiffness = Quantity[0, 1, -2, 0, 0]
type Action = Quantity[2, 1, -1, 0, 0]
type Entropy = Quantity[2, 1, -2, 0, -1]
type Permittivity = Quantity[-3, -1, 4, 2, 0]
type CoulombConstant = Quantity[3, 1, -4, -2, 0]
type DipoleMoment = Quantity[1, 0, 1, 1, 0]
type NumberDensity = Quantity[-3, 0, 0, 0, 0]
type Compressibility = Quantity[1, -1, 2, 0, 0]
type Charge = Quantity[0, 0, 1, 1, 0]
type Current = Quantity[0, 0, 0, 1, 0]
type Energy = Quantity[2, 1, -2, 0, 0]
type Force = Quantity[1, 1, -2, 0, 0]
type Length = Quantity[1, 0, 0, 0, 0]
type Mass = Quantity[0, 1, 0, 0, 0]
type Power = Quantity[2, 1, -3, 0, 0]
type Pressure = Quantity[-1, 1, -2, 0, 0]
type Temperature = Quantity[0, 0, 0, 0, 1]
type Time = Quantity[0, 0, 1, 0, 0]
type Frequency = Quantity[0, 0, -1, 0, 0]
type Voltage = Quantity[2, 1, -3, -1, 0]
