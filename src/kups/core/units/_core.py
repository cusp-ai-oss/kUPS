# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Core quantity types and the operations behind their methods.

A dimension is five rational exponents over ``(m, kg, s, A, K)``; ``mol`` is
dimensionless with scale ``N_A``. Magnitudes are canonicalised to the base active
at construction, and a dimensionless result decays to a plain ``jax.Array``. A
quantity is a pytree with the magnitudes as its single leaf and the dimension and
base as metadata.

Methods and array functions share the helpers here; dtype and scale-factor checks
live in ``units._precision``.
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
    TypeVar,
    cast,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike, DTypeLike
from typing_extensions import TypedDict, TypeIs

from kups.core.units._base import BaseSystem, active_base, require_base
from kups.core.units._precision import (
    base_scale,
    cast_advice,
    check_dtype,
    checked_cast,
    other_base,
    rebase_factor,
    require_normal,
    widen,
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
_Axis = int | Sequence[int] | None
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
    def T(self) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """The magnitudes with their axes reversed, keeping dimension and base."""
        return self.transpose()

    @property
    def unit(self) -> Unit[E_M, E_KG, E_S, E_A, E_K]:
        """One unit of this quantity's dimension, sized by its base.

        ``q.to(q.unit)`` is ``q.value``; a catalog unit of that size is preferred.
        """
        return self.bs.unit(self.dim)

    @property
    def at(self) -> _At[Quantity[E_M, E_KG, E_S, E_A, E_K]]:
        """``q.at[idx]``: indexed reads and updates that keep dimension and base."""
        return _At(self)

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
        """Cast the magnitudes to another dtype, refusing one x64 being off rules out."""
        dt = jnp.result_type(dtype)
        if self.dim != ONE and not jnp.issubdtype(dt, jnp.floating):
            raise DimensionError(
                f"q.astype: {dt.name} magnitudes cannot carry "
                f"{self.bs.symbol(self.dim)!r}; "
                "unit conversions require floating magnitudes. Use "
                "q.to(unit).astype(dtype) or q.value.astype(dtype) for unlabelled "
                "magnitudes."
            )
        return lift(checked_cast(lower(self), dtype, "q.astype"), self.dim, self.bs)

    def _rearranged(
        self, fn: Callable[..., jax.Array], *args: object, **kwargs: object
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``fn`` of the magnitudes, which moves values without changing them.

        Not apply, which refuses integer magnitudes with a dimension: these keep
        whatever magnitudes the quantity holds, as indexing does.
        """
        return lift(fn(lower(self), *args, **kwargs), self.dim, self.bs)

    # signatures follow jnp
    def transpose(
        self, axes: Sequence[int] | None = None
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Permute axes, keeping dimension and base."""
        return self._rearranged(jnp.transpose, axes)

    def reshape(self, shape: int | Sequence[int]) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Reshape the magnitudes, keeping dimension and base."""
        return self._rearranged(jnp.reshape, shape)

    def ravel(self) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """The magnitudes flattened, keeping dimension and base."""
        return self.reshape(-1)

    def repeat(
        self,
        repeats: ArrayLike,
        axis: int | None = None,
        *,
        total_repeat_length: int | None = None,
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.repeat`` of the magnitudes, keeping dimension and base."""
        return self._rearranged(
            jnp.repeat, repeats, axis=axis, total_repeat_length=total_repeat_length
        )

    def copy(self) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """A copy of the magnitudes, keeping dimension and base."""
        return self._rearranged(jnp.copy)

    def squeeze(self, axis: _Axis = None) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.squeeze`` of the magnitudes, keeping dimension and base."""
        return self._rearranged(jnp.squeeze, axis)

    def swapaxes(self, axis1: int, axis2: int) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.swapaxes`` of the magnitudes, keeping dimension and base."""
        return self._rearranged(jnp.swapaxes, axis1, axis2)

    def take(
        self,
        indices: ArrayLike,
        axis: int | None = None,
        *,
        mode: str | None = None,
        fill_value: Quantity[E_M, E_KG, E_S, E_A, E_K] | None = None,
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.take`` of the magnitudes, keeping dimension and base."""
        lx, fill, dim, bs = static_fill(self, fill_value, "q.take")
        return lift(jnp.take(lx, indices, axis, mode=mode, fill_value=fill), dim, bs)

    def sort(
        self, axis: int = -1, *, stable: bool = True, descending: bool = False
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.sort`` of the magnitudes, keeping dimension and base."""
        return self._rearranged(jnp.sort, axis, stable=stable, descending=descending)

    def diagonal(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.diagonal`` of the magnitudes, keeping dimension and base."""
        return self._rearranged(jnp.diagonal, offset, axis1, axis2)

    def trace(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        *,
        dtype: DTypeLike | None = None,
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.trace`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.trace, self, offset, axis1, axis2, dtype=dtype)

    def cumsum(
        self, axis: int | None = None, *, dtype: DTypeLike | None = None
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.cumsum`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.cumsum, self, axis, dtype=dtype)

    def std(
        self, axis: _Axis = None, *, ddof: int = 0, keepdims: bool = False
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """``jnp.std`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.std, self, axis=axis, ddof=ddof, keepdims=keepdims)

    def var(
        self, axis: _Axis = None, *, ddof: int = 0, keepdims: bool = False
    ) -> Quantity[2 * E_M, 2 * E_KG, 2 * E_S, 2 * E_A, 2 * E_K]:
        """Variance over an axis; the result carries the squared dimension."""
        return apply(
            jnp.var,
            self,
            axis=axis,
            ddof=ddof,
            keepdims=keepdims,
            dim=dpow(self.dim, Frac(2)),
        )

    def prod(
        self,
        axis: _Axis = None,
        dtype: DTypeLike | None = None,
        *,
        keepdims: bool = False,
    ) -> Any:
        """Product over an axis; the dimension is raised to the number of factors."""
        return apply(
            jnp.prod,
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            dim=dpow(self.dim, Frac(_factors(self, axis))),
        )

    def argmin(self, axis: int | None = None, *, keepdims: bool = False) -> jax.Array:
        """``jnp.argmin`` of the magnitudes; indices carry no units."""
        return apply(jnp.argmin, self, axis=axis, keepdims=keepdims, dim=ONE)

    def argmax(self, axis: int | None = None, *, keepdims: bool = False) -> jax.Array:
        """``jnp.argmax`` of the magnitudes; indices carry no units."""
        return apply(jnp.argmax, self, axis=axis, keepdims=keepdims, dim=ONE)

    def argsort(
        self, axis: int = -1, *, stable: bool = True, descending: bool = False
    ) -> jax.Array:
        """``jnp.argsort`` of the magnitudes; indices carry no units."""
        return apply(
            jnp.argsort, self, axis, stable=stable, descending=descending, dim=ONE
        )

    def nonzero(
        self,
        *,
        size: int | None = None,
        fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
    ) -> tuple[jax.Array, ...]:
        """``jnp.nonzero`` of the magnitudes; indices carry no units."""
        return apply(jnp.nonzero, self, size=size, fill_value=fill_value, dim=ONE)

    def all(self, axis: _Axis = None, *, keepdims: bool = False) -> jax.Array:
        """``jnp.all`` of the magnitudes; a mask carries no units."""
        return apply(jnp.all, self, axis=axis, keepdims=keepdims, dim=ONE)

    def any(self, axis: _Axis = None, *, keepdims: bool = False) -> jax.Array:
        """``jnp.any`` of the magnitudes; a mask carries no units."""
        return apply(jnp.any, self, axis=axis, keepdims=keepdims, dim=ONE)

    @overload
    def dot(
        self, other: Quantity[F_M, F_KG, F_S, F_A, F_K]
    ) -> Quantity[
        E_M + F_M,
        E_KG + F_KG,
        E_S + F_S,
        E_A + F_A,
        E_K + F_K,
    ]: ...
    @overload
    def dot(self, other: ArrayLike) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
    def dot(self, other: object) -> jax.Array | AnyQuantity:
        """``jnp.dot``; the dimensions multiply."""
        return apply(jnp.dot, self, other)

    def sum(
        self, axis: _Axis = None, *, keepdims: bool = False
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Sum over an axis, keeping the dimension."""
        return apply(jnp.sum, self, axis=axis, keepdims=keepdims)

    def mean(
        self, axis: _Axis = None, *, keepdims: bool = False
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Mean over an axis, keeping the dimension."""
        return apply(jnp.mean, self, axis=axis, keepdims=keepdims)

    def min(
        self, axis: _Axis = None, *, keepdims: bool = False
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Smallest magnitude over an axis, keeping the dimension."""
        reduce_axis("q.min", axis)
        return apply(jnp.min, self, axis=axis, keepdims=keepdims)

    def max(
        self, axis: _Axis = None, *, keepdims: bool = False
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Largest magnitude over an axis, keeping the dimension."""
        reduce_axis("q.max", axis)
        return apply(jnp.max, self, axis=axis, keepdims=keepdims)

    def norm(
        self,
        ord: float | str | None = None,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Vector or matrix norm, keeping the operand's dimension."""
        return norm_impl(self, ord, axis, keepdims, "q.norm")

    def clip(
        self,
        min: Quantity[E_M, E_KG, E_S, E_A, E_K] | None = None,
        max: Quantity[E_M, E_KG, E_S, E_A, E_K] | None = None,
    ) -> Quantity[E_M, E_KG, E_S, E_A, E_K]:
        """Clamp magnitudes between two bounds of the same dimension."""
        return clip_impl(self, min, max)

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


def _promote_arg(x: object) -> object:
    """``x``'s contribution to dtype promotion, or None."""
    if isinstance(x, Quantity):
        return x.dtype
    if isinstance(x, Unit) or not (hasattr(x, "dtype") or isinstance(x, int | float)):
        return None
    return x


def operand_dtype(
    xs: Sequence[object], dtype: DTypeLike | None = None
) -> np.dtype[np.generic] | None:
    """The dtype a Unit operand's size is held in; None if there is no Unit.

    ``dtype`` is the result's, when the caller fixes one.
    """
    if dtype is not None:
        return jnp.result_type(dtype)
    if not any(isinstance(x, Unit) for x in xs):
        return None
    args = [a for a in map(_promote_arg, xs) if a is not None]
    return jnp.result_type(*args, float)


def _unit_mag(unit: AnyUnit, bs: BaseSystem, dtype: DTypeLike | None = None) -> float:
    """One ``unit`` as a magnitude in ``bs``, checked to be normal in ``dtype``."""
    f = unit.scale / base_scale(bs, unit.dim)
    if f == 1.0:
        return f
    dtype = jnp.result_type(float) if dtype is None else jnp.dtype(dtype)
    require_normal(
        f,
        dtype,
        f"one {unit} is {f!r} in base {bs.name}",
        f"so the magnitudes it scales would be destroyed. Instead "
        f"{widen(dtype)}work in a base where {unit} is O(1){other_base(bs)}.",
    )
    return f


def mag[T](
    x: AnyQuantity | AnyUnit | T, bs: BaseSystem, dtype: DTypeLike | None = None
) -> tuple[jax.Array | float | T, Dim]:
    """Magnitudes of ``x`` in base ``bs``, and its dimension."""
    if isinstance(x, Unit):
        return _unit_mag(x, bs, dtype), x.dim
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
      DimensionError: if the magnitudes are complex, or if the unit's factor in
        the active base is not a normal number of their dtype.
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
    f = _unit_mag(unit, bs, v.dtype)
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
    f = b / unit.scale
    # a side of size 1 leaves the other applied alone, so that number is checked too
    applied = unit.scale if b == 1.0 else b if unit.scale == 1.0 else f
    for factor in (f,) if applied == f else (f, applied):
        require_normal(
            factor,
            q.dtype,
            f"expressing {q.bs.symbol(q.dim) or '1'!r} in {unit!r} needs a factor "
            f"of {factor!r}",
            f"so the magnitudes would come back as zero or nan. Instead "
            f"{widen(q.dtype)}measure in a unit closer to the magnitudes' own size.",
        )
    if b == 1.0:
        return out / unit.scale
    if unit.scale == 1.0:
        return out * b
    return out * f


def sym(d: Dim, bs: BaseSystem) -> str:
    """A dimension as ``bs``'s unit symbol, for error messages."""
    s = bs.symbol(d)
    return repr(s) if s else "a dimensionless value"


def _measures(t: AnyQuantity) -> str:
    """What ``t`` measures, with exponents, for error messages."""
    return f"{t.bs.symbol(t.dim)!r} {exps(t.dim)}"


def _mismatch(verb: str, dx: Dim, dy: Dim, bs: BaseSystem) -> DimensionError:
    return DimensionError(
        f"cannot {verb} {sym(dx, bs)} and {sym(dy, bs)}: dimensions differ\n"
        f"  {sym(dx, bs)}: {exps(dx)}\n  {sym(dy, bs)}: {exps(dy)}\n"
        "Attach a unit to the plain operand (x * u.m) or take a ratio first."
    )


def shape_of(x: AnyQuantity | ArrayLike) -> tuple[int, ...]:
    """Shape of a quantity or a plain array, traced or not."""
    return x.shape if isinstance(x, Quantity) else jnp.shape(x)


def no_index(name: str, what: str, x: object) -> None:
    """Reject a quantity passed where an index or a mask belongs."""
    if not isinstance(x, Quantity):
        return
    raise DimensionError(
        f"units.{name}: {what} must be a plain integer or boolean array, not "
        f"{_measures(x)}; it selects rather than measures"
    )


def reduce_axis(name: str, axis: object) -> None:
    """Reject a quantity or Unit where a reduction expects an axis."""
    if not isinstance(axis, (Quantity, Unit)):
        return
    raise DimensionError(
        f"{name}(axis=...) reduces over an axis, and you passed a quantity "
        f"as the axis. For an elementwise bound use units.minimum(x, y), "
        f"units.maximum(x, y) or x.clip(lo, hi)."
    )


def _factors(x: AnyQuantity | ArrayLike, axis: _Axis) -> int:
    """Number of elements a reduction over ``axis`` combines (None: all)."""
    shape = shape_of(x)
    axes = (
        range(len(shape))
        if axis is None
        else (axis,)
        if isinstance(axis, int)
        else axis
    )
    return math.prod(shape[a] for a in axes)


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
    dt = operand_dtype((x, y))
    lx, dx = mag(x, bs, dt)
    ly, dy = mag(y, bs, dt)
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
    dt = operand_dtype(ops)
    mags = [mag(o, bs, dt) for o in ops]
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
    xs: Sequence[AnyQuantity | AnyUnit | ArrayLike],
    verb: str,
    dtype: DTypeLike | None = None,
) -> tuple[list[ArrayLike], Dim, BaseSystem]:
    """Lower same-dimension operands into one base: (magnitudes, dimension, base).

    ``verb`` names the operation in errors; ``dtype`` is the result's, if given.
    """
    bs = joint(*xs)
    dt = operand_dtype(xs, dtype)
    mags: list[ArrayLike] = []
    dim: Dim | None = None
    for x in xs:
        lo, d = mag(x, bs, dt)
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


def clip_impl(
    x: ArrayLike | AnyQuantity,
    lo: ArrayLike | AnyQuantity | None,
    hi: ArrayLike | AnyQuantity | None,
) -> Any:
    """Clamp ``x`` between bounds of its dimension; a None bound is absent."""
    bounds = [b for b in (lo, hi) if b is not None]
    mags, dim, bs = agree([x, *bounds], "clip")
    rest = iter(mags[1:])
    low = None if lo is None else next(rest)
    high = None if hi is None else next(rest)
    return lift(jnp.clip(mags[0], low, high), dim, bs)


@overload
def norm_impl(
    x: Quantity[E_M, E_KG, E_S, E_A, E_K],
    ord: float | str | None,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    name: str,
) -> Quantity[E_M, E_KG, E_S, E_A, E_K]: ...
@overload
def norm_impl(
    x: ArrayLike,
    ord: float | str | None,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    name: str,
) -> jax.Array: ...
def norm_impl(
    x: AnyQuantity | ArrayLike,
    ord: float | str | None,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    name: str,
) -> AnyQuantity | jax.Array:
    """Vector or matrix norm; ``ord=0`` (a count) is refused for a quantity.

    ``name`` is the caller's spelling, for the error.
    """
    if ord == 0 and isinstance(x, Quantity):
        raise DimensionError(
            f"{name}: ord=0 counts the nonzero elements of {_measures(x)} "
            f"rather than measuring them, so the count carries no dimension. Write "
            f"jnp.count_nonzero(x.value)."
        )
    return apply(jnp.linalg.norm, x, ord=ord, axis=axis, keepdims=keepdims)


# q.at[idx]

_Q = TypeVar("_Q", bound="AnyQuantity")


class _AtOpts(TypedDict, closed=True):
    mode: str | None
    indices_are_sorted: bool
    unique_indices: bool
    wrap_negative_indices: bool


def _at_opts(
    mode: str | None,
    indices_are_sorted: bool,
    unique_indices: bool,
    wrap_negative_indices: bool,
) -> _AtOpts:
    """Index options of jax's ``at[...]`` methods, as keywords."""
    return {
        "mode": mode,
        "indices_are_sorted": indices_are_sorted,
        "unique_indices": unique_indices,
        "wrap_negative_indices": wrap_negative_indices,
    }


def static_fill(
    x: AnyQuantity, fill: object, what: str
) -> tuple[jax.Array, float | None, Dim, BaseSystem]:
    """``x``'s magnitudes, and ``fill`` as a float in ``x``'s base.

    jax hashes a gather's fill, so a labelled fill is rebased on its concrete value
    rather than inside the trace.
    """
    bs = x.bs
    lx, dim = mag(x, bs)
    if fill is None:
        return jnp.asarray(lx), None, dim, bs
    if isinstance(fill, Quantity):
        lo = lower(fill)
        if isinstance(lo, jax.core.Tracer):
            raise DimensionError(
                f"kups.core.units: {what}(mode='fill') needs a fill value jax can "
                "hash, and this one is traced. Hoist the number out of the traced "
                "region (FILL = float(fill.to(unit))), or scatter with mode='drop' "
                "and mask the result instead."
            )
        f = 1.0 if fill.bs == bs else rebase_factor(fill.bs, bs, fill.dim, fill.dtype)
        v, d = float(lo) * f, fill.dim
    elif isinstance(fill, Unit):
        v, d = _unit_mag(fill, bs, operand_dtype((x, fill))), fill.dim
    else:
        v, d = 0.0, ONE
    if d != dim:
        raise _mismatch("fill", dim, d, bs)
    return jnp.asarray(lx), v, dim, bs


@dataclasses.dataclass(frozen=True)
class _AtRef(Generic[_Q]):
    """The slice ``x.at[idx]`` of a quantity.

    Each method lowers ``x`` and its operand to plain numbers, calls jax's
    ``at[idx]`` method, and puts ``x``'s label back. An operand must share
    ``x``'s dimension and is rebased with it.
    """

    x: _Q
    idx: _Index

    def _lowered(
        self, v: AnyQuantity | AnyUnit | ArrayLike, verb: str
    ) -> tuple[jax.Array, jax.Array, Callable[[jax.Array], _Q]]:
        """Plain numbers for ``x`` and ``v`` in ``x``'s base, and the relabeller."""
        bs = self.x.bs
        dt = operand_dtype((self.x, v))
        lx, dim = mag(self.x, bs, dt)
        lv, dv = mag(v, bs, dt)
        if dv != dim:
            raise _mismatch(verb, dim, dv, bs)
        return jnp.asarray(lx), jnp.asarray(lv), lambda out: lift(out, dim, bs)

    def get(
        self,
        mode: str | None = None,
        fill_value: _Q | AnyUnit | None = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        wrap_negative_indices: bool = True,
    ) -> _Q:
        """Read the slice.

        Args:
          mode: out-of-range index handling, as jax's gather takes it.
          fill_value: what ``mode="fill"`` reads, of ``x``'s dimension.
          indices_are_sorted: whether the indices are sorted.
          unique_indices: whether every index appears at most once.
          wrap_negative_indices: whether a negative index counts from the end.

        Returns:
          The slice, with ``x``'s label.

        Raises:
          DimensionError: if ``fill_value``'s dimension is not ``x``'s, or its
            magnitude is traced.
        """
        lx, fill, dim, bs = static_fill(self.x, fill_value, "at[...].get")
        opts = _at_opts(mode, indices_are_sorted, unique_indices, wrap_negative_indices)
        return lift(lx.at[self.idx].get(fill_value=fill, **opts), dim, bs)

    def set(
        self,
        v: _Q | AnyUnit,
        mode: str | None = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        wrap_negative_indices: bool = True,
    ) -> _Q:
        """Replace the slice.

        Args:
          v: the replacement, measuring what ``x`` does.
          mode: out-of-range index handling, as jax's scatter takes it.
          indices_are_sorted: whether the indices are sorted.
          unique_indices: whether every index appears at most once.
          wrap_negative_indices: whether a negative index counts from the end.

        Returns:
          A new quantity with the slice replaced, with ``x``'s label.

        Raises:
          DimensionError: if ``v``'s dimension is not ``x``'s.
        """
        lx, lv, lift = self._lowered(v, "assign")
        opts = _at_opts(mode, indices_are_sorted, unique_indices, wrap_negative_indices)
        return lift(lx.at[self.idx].set(lv, **opts))

    def add(
        self,
        v: _Q | AnyUnit,
        mode: str | None = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        wrap_negative_indices: bool = True,
    ) -> _Q:
        """Add into the slice, accumulating duplicate indices.

        Args:
          v: the addend, of ``x``'s dimension.
          mode: out-of-range index handling, as jax's scatter takes it.
          indices_are_sorted: whether the indices are sorted.
          unique_indices: whether every index appears at most once.
          wrap_negative_indices: whether a negative index counts from the end.

        Returns:
          A new quantity with ``v`` added into the slice, with ``x``'s label.

        Raises:
          DimensionError: if ``v``'s dimension is not ``x``'s.
        """
        lx, lv, lift = self._lowered(v, "add")
        opts = _at_opts(mode, indices_are_sorted, unique_indices, wrap_negative_indices)
        return lift(lx.at[self.idx].add(lv, **opts))

    def multiply(
        self,
        v: ArrayLike,
        mode: str | None = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        wrap_negative_indices: bool = True,
    ) -> _Q:
        """Scale the slice by a dimensionless factor.

        Args:
          v: a dimensionless factor.
          mode: out-of-range index handling, as jax's scatter takes it.
          indices_are_sorted: whether the indices are sorted.
          unique_indices: whether every index appears at most once.
          wrap_negative_indices: whether a negative index counts from the end.

        Returns:
          A new quantity with the slice scaled.

        Raises:
          DimensionError: if ``v`` carries a dimension (an array holds one).
        """
        bs = self.x.bs
        dt = operand_dtype((self.x, v))
        lx, dx = mag(self.x, bs, dt)
        lx = jnp.asarray(lx)
        lv, dv = mag(v, bs, dt)
        if dv != ONE:
            raise DimensionError(
                "kups.core.units: q.at[...].multiply needs a dimensionless factor, "
                "got "
                f"{bs.symbol(dv)!r} {exps(dv)}; scaling part of an array would "
                f"leave it with two dimensions. Multiply the whole quantity."
            )
        opts = _at_opts(mode, indices_are_sorted, unique_indices, wrap_negative_indices)
        return lift(lx.at[self.idx].multiply(lv, **opts), dx, bs)


@dataclasses.dataclass(frozen=True)
class _At(Generic[_Q]):
    """``x.at[idx]``: indexed reads and updates that keep a quantity's label."""

    x: _Q

    def __getitem__(self, idx: _Index) -> _AtRef[_Q]:
        """The slice at ``idx``, ready to read or update."""
        return _AtRef(self.x, idx)


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
