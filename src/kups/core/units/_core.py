# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Core quantity types and the operations behind their methods.

A dimension is five rational exponents over ``(m, kg, s, A, K)``; ``mol`` is
dimensionless with scale ``N_A``. Magnitudes are canonicalised to the base active
at construction, and a dimensionless result decays to a plain ``jax.Array``. A
quantity is a pytree with the magnitudes as its single leaf and the dimension and
base as metadata.

Methods and array functions share the helpers here; dtype and range checks live
in ``units._precision``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from collections.abc import Callable, Iterator, Sequence
from fractions import Fraction as Frac
from typing import (
    TYPE_CHECKING,
    Annotated,
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
from jax.typing import ArrayLike
from typing_extensions import TypeIs

from kups.core.units._base import BaseSystem, _require_base, active_base
from kups.core.units._precision import (
    _X64_HINT,
    _base_scale,
    _cast_advice,
    _check_canonical,
    _check_dtype,
    _check_host,
    _check_rescaled,
    _checked_cast,
    _host,
    _other_base,
    _precision_warning,
    _rebase_factor,
    _require_normal,
    _widen,
)
from kups.core.units._typed import _E_A, _E_K, _E_KG, _E_M, _E_S
from kups.core.units._unit import (
    AXES,
    ONE,
    AnyUnit,
    Dim,
    DimensionError,
    Unit,
    _require_unit,
    ddiv,
    dmul,
    dpow,
    exponent,
    exps,
)
from kups.core.units.catalog import CATALOG, eV
from kups.core.utils.jax import tree_map as _tree_map

if TYPE_CHECKING:
    from shape_extensions import Int

    from kups.core.units._typed import _F_A, _F_K, _F_KG, _F_M, _F_S, _K
    from kups.core.units._unit import Dimensionless

_Op = Callable[..., Any]


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


@dataclasses.dataclass(frozen=True, eq=False)
class Quantity(Generic[_E_M, _E_KG, _E_S, _E_A, _E_K]):
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

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        """Refuse conversion to a numpy or jax array."""
        del dtype, copy
        raise DimensionError(_NO_ARRAY)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Never:
        """Refuse a pydantic field annotated with a quantity that names no dimension.

        Dimension aliases such as ``units.Length`` carry their own schema hook, so
        only a bare ``Quantity`` annotation lands here.

        Raises:
          DimensionError: always.
        """
        del source, handler
        raise DimensionError(
            f"units.pydantic: {cls.__name__} names no dimension. Annotate with "
            "units.Length, units.Energy, or another dimension alias."
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the magnitudes."""
        return jnp.shape(self.v)

    @property
    def ndim(self) -> int:
        """Rank of the magnitudes."""
        return jnp.ndim(self.v)

    @property
    def dtype(self) -> Any:
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

    @property
    def _committed(self) -> bool:
        # jit reads this to decide whether an in_sharding conflicts
        return getattr(self.v, "_committed", True)

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
        return _lower(self)

    @property
    def T(self) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """The magnitudes with their axes reversed, keeping dimension and base."""
        return self.transpose()

    @property
    def unit(self) -> Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """One unit of this quantity's dimension, sized by its base.

        ``q.to(q.unit)`` is ``q.value``; a catalog unit of that size is preferred.
        """
        bs, dim = self.bs, self.dim
        scale = bs.scale(dim)
        for cand in CATALOG:
            if cand.dim == dim and cand.scale == scale:
                return cand
        return Unit(bs.symbol(dim) or "1", dim, scale, bs.factors(dim))

    @property
    def at(self) -> _At[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]]:
        """``q.at[idx]``: indexed reads and updates that keep dimension and base."""
        return _At(self)

    def __add__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Add magnitudes of the same dimension."""
        if not _operand(other):
            return NotImplemented
        return _add(self, other)

    def __radd__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Add magnitudes of the same dimension, reflected."""
        if not _operand(other):
            return NotImplemented
        return _add(other, self)

    def __sub__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Subtract magnitudes of the same dimension."""
        if not _operand(other):
            return NotImplemented
        return _sub(self, other)

    def __rsub__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Subtract magnitudes of the same dimension, reflected."""
        if not _operand(other):
            return NotImplemented
        return _sub(other, self)

    def __neg__(self) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Negate the magnitudes, keeping dimension and base."""
        return _lift(-_lower(self), self.dim, self.bs)

    def __abs__(self) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Elementwise absolute value, keeping dimension and base."""
        return apply(jnp.abs, self)

    def __lt__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> jax.Array:
        """Elementwise less-than; returns a plain boolean array."""
        if not _operand(other):
            return NotImplemented
        return _compare(self, other, operator.lt)

    def __le__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> jax.Array:
        """Elementwise less-or-equal; returns a plain boolean array."""
        if not _operand(other):
            return NotImplemented
        return _compare(self, other, operator.le)

    def __gt__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> jax.Array:
        """Elementwise greater-than; returns a plain boolean array."""
        if not _operand(other):
            return NotImplemented
        return _compare(self, other, operator.gt)

    def __ge__(
        self,
        other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
        | Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> jax.Array:
        """Elementwise greater-or-equal; returns a plain boolean array."""
        if not _operand(other):
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
        return bool(_lower(self))

    def __getitem__(self, idx: Any) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Index the magnitudes, keeping dimension and base."""
        return _lift(_lower(self)[idx], self.dim, self.bs)

    def __iter__(self) -> Iterator[Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]]:
        """Iterate over the leading axis; unrolled so it works inside a trace."""
        return _unrolled(self)

    # mod keeps the shared dimension; floor division cancels it
    def __mod__(
        self, other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Elementwise remainder of two operands of the same dimension."""
        if not _operand(other):
            return NotImplemented
        return _mod(self, other)

    def __rmod__(
        self, other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Elementwise remainder, reflected."""
        if not _operand(other):
            return NotImplemented
        return _mod(other, self)

    def __floordiv__(
        self, other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float
    ) -> jax.Array:
        """Elementwise floor division; a count is dimensionless."""
        if not _operand(other):
            return NotImplemented
        return _floordiv(self, other)

    def __rfloordiv__(
        self, other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float
    ) -> jax.Array:
        """Elementwise floor division, reflected."""
        if not _operand(other):
            return NotImplemented
        return _floordiv(other, self)

    # the first overload of each pair is the cancelling case: a plain array
    @overload
    def __mul__(
        self, other: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]
    ) -> jax.Array: ...
    @overload
    def __mul__(
        self, other: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def __mul__(
        self, other: Dimensionless
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    @overload
    def __mul__(self, other: Unit[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]) -> jax.Array: ...
    @overload
    def __mul__(
        self, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def __mul__(self, other: ArrayLike) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __mul__(self, other: Any) -> Any:
        """Multiply; the dimensions multiply too."""
        if not _operand(other):
            return NotImplemented
        return _mul(self, other)

    @overload
    def __rmul__(
        self, other: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]
    ) -> jax.Array: ...
    @overload
    def __rmul__(
        self, other: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def __rmul__(
        self, other: Dimensionless
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    @overload
    def __rmul__(
        self, other: Unit[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]
    ) -> jax.Array: ...
    @overload
    def __rmul__(
        self, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def __rmul__(self, other: ArrayLike) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __rmul__(self, other: Any) -> Any:
        """Multiply, reflected; the dimensions multiply too."""
        if not _operand(other):
            return NotImplemented
        return _mul(other, self)

    @overload
    def __truediv__(
        self, other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
    ) -> jax.Array: ...
    @overload
    def __truediv__(
        self, other: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M - _F_M,
        _E_KG - _F_KG,
        _E_S - _F_S,
        _E_A - _F_A,
        _E_K - _F_K,
    ]: ...
    @overload
    def __truediv__(
        self, other: Dimensionless
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    @overload
    def __truediv__(self, other: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]) -> jax.Array: ...
    @overload
    def __truediv__(
        self, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M - _F_M,
        _E_KG - _F_KG,
        _E_S - _F_S,
        _E_A - _F_A,
        _E_K - _F_K,
    ]: ...
    @overload
    def __truediv__(
        self, other: ArrayLike
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __truediv__(self, other: Any) -> Any:
        """Divide; the dimensions divide too."""
        if not _operand(other):
            return NotImplemented
        return _div(self, other)

    @overload
    def __rtruediv__(
        self, other: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]
    ) -> jax.Array: ...
    @overload
    def __rtruediv__(
        self, other: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _F_M - _E_M,
        _F_KG - _E_KG,
        _F_S - _E_S,
        _F_A - _E_A,
        _F_K - _E_K,
    ]: ...
    @overload
    def __rtruediv__(
        self, other: Dimensionless
    ) -> Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]: ...
    @overload
    def __rtruediv__(self, other: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]) -> jax.Array: ...
    @overload
    def __rtruediv__(
        self, other: Unit[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _F_M - _E_M,
        _F_KG - _E_KG,
        _F_S - _E_S,
        _F_A - _E_A,
        _F_K - _E_K,
    ]: ...
    @overload
    def __rtruediv__(
        self, other: ArrayLike
    ) -> Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]: ...
    def __rtruediv__(self, other: Any) -> Any:
        """Divide, reflected; the dimensions divide too."""
        if not _operand(other):
            return NotImplemented
        return _div(other, self)

    @overload
    def __matmul__(
        self, other: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]
    ) -> jax.Array: ...
    @overload
    def __matmul__(
        self, other: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def __matmul__(
        self, other: ArrayLike
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __matmul__(self, other: Any) -> Any:
        """Matrix product; the dimensions multiply."""
        if not _operand(other):
            return NotImplemented
        return _matmul(self, other)

    @overload
    def __rmatmul__(
        self, other: Quantity[-_E_M, -_E_KG, -_E_S, -_E_A, -_E_K]
    ) -> jax.Array: ...
    @overload
    def __rmatmul__(
        self, other: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def __rmatmul__(
        self, other: ArrayLike
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def __rmatmul__(self, other: Any) -> Any:
        """Matrix product, reflected; the dimensions multiply."""
        if not _operand(other):
            return NotImplemented
        return _matmul(other, self)

    @overload
    def __pow__(self, e: Literal[0]) -> jax.Array: ...
    @overload
    def __pow__(
        self, e: Int[_K]
    ) -> Quantity[_E_M * _K, _E_KG * _K, _E_S * _K, _E_A * _K, _E_K * _K]: ...
    @overload
    def __pow__(self, e: float | Frac) -> Any: ...  # not an integer exponent
    def __pow__(self, e: Any) -> Any:
        """Raise to a static rational power; the dimension is raised with it."""
        return _pow(self, e)

    def to(self, unit: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]) -> jax.Array:
        """Magnitudes measured in ``unit``, as a plain array."""
        return _to(self, unit)

    def rebase(
        self, bs: BaseSystem | None = None
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Re-express the quantity in another base (the active one by default)."""
        return _rebase(self, bs)

    def item(self) -> Any:
        """The single magnitude of a one-element quantity, in its own base."""
        return _concrete(self, "q.item()").item()

    def tolist(self) -> Any:
        """The magnitudes as nested Python lists, in the quantity's own base."""
        return _concrete(self, "q.tolist()").tolist()

    def astype(self, dtype: Any) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Cast the magnitudes to another dtype, refusing casts that corrupt them."""
        bs = _joint(self)
        lx, dx = _mag(self, bs, _operand_dtype((self,), dtype))
        dt = jnp.result_type(dtype)
        if dx != ONE and not jnp.issubdtype(dt, jnp.floating):
            raise DimensionError(
                f"q.astype: {dt.name} magnitudes cannot carry {bs.symbol(dx)!r}; "
                "unit conversions require floating magnitudes. Use "
                "q.to(unit).astype(dtype) or q.value.astype(dtype) for unlabelled "
                "magnitudes."
            )
        return _lift(_checked_cast(lx, dtype, bs), dx, bs)

    # signatures follow jnp
    def transpose(self, axes: Any = None) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Permute axes, keeping dimension and base."""
        return apply(jnp.transpose, self, axes)

    def reshape(self, shape: Any) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Reshape the magnitudes, keeping dimension and base."""
        return apply(jnp.reshape, self, shape)

    def ravel(self) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """The magnitudes flattened, keeping dimension and base."""
        return self.reshape(-1)

    def repeat(
        self,
        repeats: ArrayLike,
        axis: int | None = None,
        *,
        total_repeat_length: int | None = None,
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.repeat`` of the magnitudes, keeping dimension and base."""
        return apply(
            jnp.repeat,
            self,
            repeats,
            axis=axis,
            total_repeat_length=total_repeat_length,
        )

    def copy(self) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """A copy of the magnitudes, keeping dimension and base."""
        # not apply(jnp.copy, ...): apply refuses integer magnitudes with a dimension
        return _lift(jnp.copy(_lower(self)), self.dim, self.bs)

    def squeeze(self, axis: Any = None) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.squeeze`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.squeeze, self, axis)

    def swapaxes(
        self, axis1: int, axis2: int
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.swapaxes`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.swapaxes, self, axis1, axis2)

    def take(
        self,
        indices: ArrayLike,
        axis: int | None = None,
        *,
        mode: Any = None,
        fill_value: Any = None,
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.take`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.take, self, indices, axis, mode=mode, fill_value=fill_value)

    def sort(
        self, axis: int = -1, *, stable: bool = True, descending: bool = False
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.sort`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.sort, self, axis, stable=stable, descending=descending)

    def diagonal(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.diagonal`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.diagonal, self, offset, axis1, axis2)

    def trace(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1, *, dtype: Any = None
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.trace`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.trace, self, offset, axis1, axis2, dtype=dtype)

    def cumsum(
        self, axis: Any = None, *, dtype: Any = None
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.cumsum`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.cumsum, self, axis, dtype=dtype)

    def std(
        self, axis: Any = None, *, ddof: int = 0, keepdims: bool = False
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """``jnp.std`` of the magnitudes, keeping dimension and base."""
        return apply(jnp.std, self, axis=axis, ddof=ddof, keepdims=keepdims)

    def var(
        self, axis: Any = None, ddof: int = 0, keepdims: bool = False
    ) -> Quantity[2 * _E_M, 2 * _E_KG, 2 * _E_S, 2 * _E_A, 2 * _E_K]:
        """Variance over an axis; the result carries the squared dimension."""
        return apply(
            jnp.var,
            self,
            axis=axis,
            ddof=ddof,
            keepdims=keepdims,
            dim=dpow(self.dim, Frac(2)),
        )

    def prod(self, axis: Any = None, dtype: Any = None, keepdims: bool = False) -> Any:
        """Product over an axis; the dimension is raised to the number of factors."""
        return apply(
            jnp.prod,
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            dim=dpow(self.dim, Frac(_factors(self, axis))),
        )

    # indices, counts, and masks carry no units
    def argmin(self, axis: Any = None, *, keepdims: bool = False) -> jax.Array:
        """``jnp.argmin`` of the magnitudes; indices carry no units."""
        return apply(jnp.argmin, self, axis=axis, keepdims=keepdims, dim=ONE)

    def argmax(self, axis: Any = None, *, keepdims: bool = False) -> jax.Array:
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
        self, *, size: int | None = None, fill_value: Any = None
    ) -> tuple[jax.Array, ...]:
        """``jnp.nonzero`` of the magnitudes; indices carry no units."""
        return apply(jnp.nonzero, self, size=size, fill_value=fill_value, dim=ONE)

    def all(self, axis: Any = None, *, keepdims: bool = False) -> jax.Array:
        """``jnp.all`` of the magnitudes; a mask carries no units."""
        return apply(jnp.all, self, axis=axis, keepdims=keepdims, dim=ONE)

    def any(self, axis: Any = None, *, keepdims: bool = False) -> jax.Array:
        """``jnp.any`` of the magnitudes; a mask carries no units."""
        return apply(jnp.any, self, axis=axis, keepdims=keepdims, dim=ONE)

    @overload
    def dot(
        self, other: Quantity[_F_M, _F_KG, _F_S, _F_A, _F_K]
    ) -> Quantity[
        _E_M + _F_M,
        _E_KG + _F_KG,
        _E_S + _F_S,
        _E_A + _F_A,
        _E_K + _F_K,
    ]: ...
    @overload
    def dot(self, other: ArrayLike) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    def dot(self, other: Any) -> Any:
        """``jnp.dot``; the dimensions multiply."""
        return apply(jnp.dot, self, other)

    def sum(
        self, axis: Any = None, keepdims: bool = False
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Sum over an axis, keeping the dimension."""
        return apply(jnp.sum, self, axis=axis, keepdims=keepdims)

    def mean(
        self, axis: Any = None, keepdims: bool = False
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Mean over an axis, keeping the dimension."""
        return apply(jnp.mean, self, axis=axis, keepdims=keepdims)

    def min(
        self, axis: Any = None, keepdims: bool = False
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Smallest magnitude over an axis, keeping the dimension."""
        _reduce_axis("min", axis)
        return apply(jnp.min, self, axis=axis, keepdims=keepdims)

    def max(
        self, axis: Any = None, keepdims: bool = False
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Largest magnitude over an axis, keeping the dimension."""
        _reduce_axis("max", axis)
        return apply(jnp.max, self, axis=axis, keepdims=keepdims)

    def norm(
        self, ord: Any = None, axis: Any = None, keepdims: bool = False
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Vector or matrix norm, keeping the operand's dimension."""
        return _norm_impl(self, ord, axis, keepdims)

    def clip(
        self,
        lo: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float | None = None,
        hi: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K] | float | None = None,
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]:
        """Clamp magnitudes between two bounds of the same dimension."""
        bounds = [b for b in (lo, hi) if b is not None]
        mags, dim, bs = _agree([self, *bounds], "clip")
        rest = iter(mags[1:])
        low = None if lo is None else next(rest)
        high = None if hi is None else next(rest)
        return _lift(jnp.clip(mags[0], low, high), dim, bs)

    # named overloads catch odd-exponent dimensions, which ``//`` would floor
    @overload
    def sqrt(self: Acceleration) -> Any: ...
    @overload
    def sqrt(self: Action) -> Any: ...
    @overload
    def sqrt(self: Charge) -> Any: ...
    @overload
    def sqrt(self: Compressibility) -> Any: ...
    @overload
    def sqrt(self: CoulombConstant) -> Any: ...
    @overload
    def sqrt(self: Current) -> Any: ...
    @overload
    def sqrt(self: Density) -> Any: ...
    @overload
    def sqrt(self: DipoleMoment) -> Any: ...
    @overload
    def sqrt(self: Energy) -> Any: ...
    @overload
    def sqrt(self: Entropy) -> Any: ...
    @overload
    def sqrt(self: Force) -> Any: ...
    @overload
    def sqrt(self: Frequency) -> Any: ...
    @overload
    def sqrt(self: Length) -> Any: ...
    @overload
    def sqrt(self: Mass) -> Any: ...
    @overload
    def sqrt(self: Momentum) -> Any: ...
    @overload
    def sqrt(self: NumberDensity) -> Any: ...
    @overload
    def sqrt(self: Permittivity) -> Any: ...
    @overload
    def sqrt(self: Power) -> Any: ...
    @overload
    def sqrt(self: Pressure) -> Any: ...
    @overload
    def sqrt(self: Stiffness) -> Any: ...
    @overload
    def sqrt(self: Temperature) -> Any: ...
    @overload
    def sqrt(self: Time) -> Any: ...
    @overload
    def sqrt(self: Velocity) -> Any: ...
    @overload
    def sqrt(self: Voltage) -> Any: ...
    @overload
    def sqrt(self: Volume) -> Any: ...
    @overload
    def sqrt(self: Wavenumber) -> Any: ...
    @overload
    def sqrt(
        self: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> Quantity[_E_M // 2, _E_KG // 2, _E_S // 2, _E_A // 2, _E_K // 2]: ...
    @overload
    def sqrt(self) -> Any: ...
    def sqrt(self) -> Any:
        """Square root; halves every dimension exponent."""
        return _sqrt_impl(self)

    if not TYPE_CHECKING:
        # runtime error, hidden so the checker still flags the expression

        def __rpow__(self, other):
            """Refused: a quantity exponent has no fixed dimension."""
            return _pow(other, self)


# a single-leaf pytree; the labels are metadata compared by value
# jax's native dataclass registration, not kups's dataclass, for flatten speed
jax.tree_util.register_dataclass(Quantity, data_fields=["v"], meta_fields=["dim", "bs"])

type AnyQuantity = Quantity[Any, Any, Any, Any, Any]


def is_leaf(x: object) -> bool:
    """Whether a tree walk should stop at ``x``.

    Pass ``is_leaf=units.is_leaf`` to ``jax.tree.map`` so ``fn`` gets whole
    quantities rather than their raw magnitudes.

    Args:
      x: a node of a pytree.

    Returns:
      True for a quantity.
    """
    return isinstance(x, Quantity)


def tree_map(f: Callable[..., Any], x: Any, *rest: Any) -> Any:
    """Map over a pytree's leaves, passing each quantity to ``f`` whole.

    Args:
      f: the function applied to each leaf.
      x: the pytree whose structure the result has.
      *rest: further pytrees of the same structure, whose leaves ``f`` also takes.

    Returns:
      A pytree with ``x``'s structure holding ``f``'s results.
    """
    return _tree_map(f, x, *rest, is_leaf=is_leaf)


def _lift(x: Any, dim: Dim, bs: BaseSystem) -> Any:
    """Attach a dimension to magnitudes; a dimensionless result stays an array."""
    return x if dim == ONE else Quantity(x, dim, bs)


def _lower(q: Any) -> jax.Array:
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


def is_quantity(x: object) -> TypeIs[AnyQuantity]:
    """Whether ``x`` is a quantity, concrete or traced.

    Args:
      x: any value.

    Returns:
      True for a quantity; False for anything else, including Units.
    """
    return isinstance(x, Quantity)


def base_of(x: Any) -> BaseSystem:
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


def _dim(dim: Any, what: str) -> Dim:
    """Check that ``dim`` is a dimension vector; ``what`` names the caller."""
    if (
        isinstance(dim, tuple)
        and len(dim) == len(AXES)
        and all(isinstance(v, int | Frac) for v in dim)
    ):
        return dim
    raise DimensionError(
        f"units.{what}: {dim!r} is not a dimension. A dimension is five rational "
        f"exponents over {AXES}; use units.dim_of(unit) or units.dim_from_str."
    )


def dim_str(dim: Dim) -> str:
    """Format a dimension as text that ``dim_from_str`` can read back.

    Args:
      dim: dimension vector.

    Returns:
      A string like ``"m=2 kg=1 s=-2 A=0 K=0"`` (an energy).

    Raises:
      DimensionError: if ``dim`` is not a dimension vector.
    """
    return " ".join(f"{a}={v}" for a, v in zip(AXES, _dim(dim, "dim_str"), strict=True))


def dim_from_str(s: str) -> Dim:
    """Parse a dimension from the text format ``dim_str`` writes.

    Args:
      s: whitespace-separated ``axis=exponent`` terms, any order; missing axes are 0.

    Returns:
      The dimension vector, with integral exponents as ints so it compares equal
      to ``units.dim_of`` of the same dimension.

    Raises:
      DimensionError: if a term is malformed, repeats an axis, or is not rational.
    """
    got: dict[str, Frac] = {}
    for term in s.split():
        axis, sep, power = term.partition("=")
        if not sep or axis not in AXES or axis in got:
            raise DimensionError(
                f"units.dim_from_str: {term!r} in {s!r} is not one of the five "
                f"{AXES} written once each as axis=exponent, e.g. "
                f"{dim_str(dim_of(eV))!r}"
            )
        try:
            got[axis] = Frac(power)
        except (ValueError, ZeroDivisionError) as err:
            raise DimensionError(
                f"units.dim_from_str: {power!r} in {s!r} is not a rational exponent; "
                "a half power is written 1/2"
            ) from err
    return tuple(
        int(v) if v.denominator == 1 else v for v in (got.get(a, Frac(0)) for a in AXES)
    )


def from_dim(magnitudes: Any, dim: Dim, bs: BaseSystem | None = None) -> Any:
    """Attach a dimension to plain magnitudes.

    Args:
      magnitudes: the numbers, as anything ``jnp.asarray`` accepts.
      dim: the dimension they measure, e.g. from ``dim_from_str``.
      bs: the base they are expressed in, or None for the active one.

    Returns:
      The quantity, or a plain jax.Array when ``dim`` is dimensionless.

    Raises:
      TypeError: if ``bs`` is not a BaseSystem.
      DimensionError: if ``dim`` is not a dimension vector, if the magnitudes are
        complex or integral, or if they already carry a dimension.
    """
    _dim(dim, "from_dim")
    if bs is not None:
        _require_base(bs, "the base the magnitudes are in")
    if isinstance(magnitudes, Quantity):
        raise DimensionError(
            "units.from_dim: these magnitudes are "
            f"{_sym(magnitudes.dim, magnitudes.bs)} "
            "already, and "
            "from_dim reads plain numbers. Take q.value for the magnitudes in q's own "
            "base, or units.relabel(q, dim) to read them as something else."
        )
    src = _host(magnitudes)
    v = jnp.asarray(magnitudes)
    if src is not None:  # jax narrows a float64 host input when x64 is off
        _check_host(src, 1.0, v.dtype, "reading these magnitudes")
    if jnp.issubdtype(v.dtype, jnp.complexfloating):
        raise DimensionError(
            f"kups.core.units: complex magnitudes are not supported ({v.dtype}); take "
            "jnp.real/jnp.imag before attaching a dimension"
        )
    into = bs or active_base()
    _check_dtype(v.dtype, dim, into, "from_dim")
    return _lift(v, dim, into)


def _joint(*xs: Any) -> BaseSystem:
    """Common base of the quantity operands, or the active base if they disagree."""
    bs = None
    for x in xs:
        if isinstance(x, Quantity):
            if bs is None:
                bs = x.bs
            elif x.bs != bs:
                return active_base()
    return active_base() if bs is None else bs


def _promote_arg(x: Any) -> Any:
    """``x``'s contribution to dtype promotion, or None."""
    if isinstance(x, Quantity):
        return x.dtype
    if isinstance(x, Unit) or not (hasattr(x, "dtype") or isinstance(x, int | float)):
        return None
    return x


def _operand_dtype(xs: Sequence[Any], dtype: Any = None) -> Any:
    """The dtype a Unit operand's size is held in; None if there is no Unit."""
    if dtype is not None:
        return jnp.result_type(dtype)
    if not any(isinstance(x, Unit) for x in xs):
        return None
    args = [a for a in map(_promote_arg, xs) if a is not None]
    return jnp.result_type(*args, float)


def _unit_mag(unit: AnyUnit, bs: BaseSystem, dtype: Any = None) -> float:
    """One ``unit`` as a magnitude in ``bs``, checked to be normal in ``dtype``."""
    b = _base_scale(bs, unit.dim)
    f = unit.scale if b == 1.0 else unit.scale / b
    if f == 1.0:
        return f
    dtype = jnp.result_type(float) if dtype is None else jnp.dtype(dtype)
    _require_normal(
        f,
        dtype,
        f"one {unit} is {f!r} in base {bs.name}",
        f"so using it as an operand or a fill would destroy the magnitudes. Instead "
        f"{_widen(dtype)}work in a base where {unit} is O(1){_other_base(bs)}.",
    )
    return f


def _mag(x: Any, bs: BaseSystem, dtype: Any = None) -> tuple[Any, Dim]:
    """Magnitudes of ``x`` in base ``bs``, and its dimension."""
    if isinstance(x, Unit):
        return _unit_mag(x, bs, dtype), x.dim
    if not isinstance(x, Quantity):
        return x, ONE
    lo = _lower(x)
    if x.bs == bs:
        return lo, x.dim
    f = _rebase_factor(x.bs, bs, x.dim, x.dtype)
    if f == 1.0:
        return lo, x.dim
    out = lo * f
    _check_rescaled(
        lo,
        out,
        f"re-expressing {x.bs.symbol(x.dim) or '1'!r} from base "
        f"{x.bs.name} in base {bs.name}",
        bs,
        f,
    )
    return out, x.dim


def from_unit(x: Any, unit: Any) -> Any:
    """Canonicalize ``x * unit`` into the active base (the ``5 * u.kcal`` path).

    Args:
      x: magnitudes, in units of ``unit``.
      unit: the unit to attach.

    Returns:
      The quantity, in the active base; a plain array if ``unit`` is dimensionless.

    Raises:
      DimensionError: if the magnitudes are complex, or if canonicalizing them
        would turn a finite nonzero magnitude into zero or infinity.
    """
    bs = active_base()
    src = _host(x)
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
    b = _base_scale(bs, unit.dim)
    f = unit.scale if b == 1.0 else unit.scale / b
    if src is not None:  # host check: exact, no device round trip
        _check_host(src, f, v.dtype, f"attaching {unit} in base {bs.name}")
    if f == 1.0:
        return _lift(v, unit.dim, bs)
    _precision_warning(unit, bs, f, v.dtype)
    _require_normal(
        f,
        v.dtype,
        f"attaching {unit} in base {bs.name} needs a factor of {f!r}",
        f"so the magnitudes would be destroyed. Enable {_X64_HINT}, or work in a "
        f"base where {unit} is O(1){_other_base(bs)}.",
    )
    out = v * f
    _check_canonical(v, out, unit, bs, f)
    return _lift(out, unit.dim, bs)


def _to(q: Any, unit: AnyUnit) -> jax.Array:
    """Measure ``q`` in ``unit``; every quantity-to-array conversion ends here."""
    if not isinstance(q, Quantity):
        raise TypeError(f"kups.core.units: {q!r} is not a quantity")
    _require_unit(unit, "the unit to convert to")
    if q.dim != unit.dim:
        raise DimensionError(
            f"cannot express {q.bs.symbol(q.dim) or '1'!r} in {unit!r}: "
            f"dimensions differ\n  value {exps(q.dim)}\n  unit  {exps(unit.dim)}"
        )
    out = _lower(q)
    b = _base_scale(q.bs, q.dim)
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
            f"{_cast_advice(q.dtype)}."
        )
    f = b / unit.scale
    _require_normal(
        f,
        q.dtype,
        f"expressing {q.bs.symbol(q.dim) or '1'!r} in {unit!r} needs a factor of {f!r}",
        "so the magnitudes would come back as zero or nan. Set jax_enable_x64, or "
        "measure in a unit closer to the magnitudes' own size.",
    )
    if b == 1.0:
        res = out / unit.scale
    elif unit.scale == 1.0:
        res = out * b
    else:
        res = out * f  # single rescale, no unchecked intermediate
    _check_rescaled(
        out, res, f"expressing {q.bs.symbol(q.dim) or '1'!r} in {unit!r}", q.bs, f
    )
    return res


def _sym(d: Dim, bs: BaseSystem) -> str:
    """A dimension as ``bs``'s unit symbol, for error messages."""
    s = bs.symbol(d)
    return repr(s) if s else "a dimensionless value"


def _measures(t: AnyQuantity) -> str:
    """What ``t`` measures, with exponents, for error messages."""
    return f"{t.bs.symbol(t.dim)!r} {exps(t.dim)}"


def _mismatch(verb: str, dx: Dim, dy: Dim, bs: BaseSystem) -> DimensionError:
    return DimensionError(
        f"cannot {verb} {_sym(dx, bs)} and {_sym(dy, bs)}: dimensions differ\n"
        f"  {_sym(dx, bs)}: {exps(dx)}\n  {_sym(dy, bs)}: {exps(dy)}\n"
        "Attach a unit to the plain operand (x * u.m) or take a ratio first."
    )


def _shape(x: Any) -> tuple[int, ...]:
    """Shape of a quantity or a plain array, traced or not."""
    return x.shape if isinstance(x, Quantity) else jnp.shape(x)


def _no_index(name: str, what: str, x: Any) -> None:
    """Reject a quantity passed where an index or a mask belongs."""
    if not isinstance(x, Quantity):
        return
    raise DimensionError(
        f"units.{name}: {what} must be a plain integer or boolean array, not "
        f"{_measures(x)}; it selects rather than measures"
    )


def _reduce_axis(name: str, axis: Any) -> None:
    """Reject a quantity or Unit where a reduction expects an axis."""
    if not isinstance(axis, (Quantity, Unit)):
        return
    raise DimensionError(
        f"q.{name}(axis=...) reduces over an axis, and you passed a quantity "
        f"as the axis. For an elementwise bound use units.minimum(x, y), "
        f"units.maximum(x, y) or x.clip(lo, hi)."
    )


def _factors(x: Any, axis: Any) -> int:
    """Number of elements a reduction over ``axis`` combines (None: all)."""
    shape = _shape(x)
    axes = (
        range(len(shape))
        if axis is None
        else (axis,)
        if isinstance(axis, int)
        else axis
    )
    return math.prod(shape[a] for a in axes)


def _not_complex(out: Any, d: Dim, bs: BaseSystem) -> Any:
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


def _binop(x: Any, y: Any, op: _Op, dimop: Callable[..., Dim] | str) -> Any:
    """Apply ``op`` to magnitudes and combine dimensions with ``dimop``.

    A string ``dimop`` means "require equal dimensions" and names the verb for
    the error.
    """
    bs = _joint(x, y)
    dt = _operand_dtype((x, y))
    lx, dx = _mag(x, bs, dt)
    ly, dy = _mag(y, bs, dt)
    if isinstance(dimop, str):
        if dx != dy:
            raise _mismatch(dimop, dx, dy, bs)
        d = dx
    else:
        d = dimop(dx, dy)
    return _lift(_not_complex(op(lx, ly), d, bs), d, bs)


_add = functools.partial(_binop, op=operator.add, dimop="add")
_sub = functools.partial(_binop, op=operator.sub, dimop="subtract")
_mul = functools.partial(_binop, op=operator.mul, dimop=dmul)
_div = functools.partial(_binop, op=operator.truediv, dimop=ddiv)
_matmul = functools.partial(_binop, op=jnp.matmul, dimop=dmul)


def _compare(x: Any, y: Any, op: _Op) -> jax.Array:
    """Compare magnitudes of the same dimension, giving a plain boolean array."""
    bs = _joint(x, y)
    dt = _operand_dtype((x, y))
    lx, dx = _mag(x, bs, dt)
    ly, dy = _mag(y, bs, dt)
    if dx != dy:
        raise _mismatch("compare", dx, dy, bs)
    return op(lx, ly)


def _pow(x: Any, e: Any) -> Any:
    if isinstance(e, (jax.core.Tracer, Quantity)):
        raise DimensionError(
            "kups.core.units: the exponent of a quantity must be a static int or "
            "Fraction, "
            f"not a traced value ({e!r}); the result's dimension depends on it"
        )
    ef = exponent(e)
    bs = _joint(x)
    lx, dx = _mag(x, bs)
    return _lift(lx ** _jnp_exponent(ef), dpow(dx, ef), bs)


def _jnp_exponent(ef: Frac) -> int | float:
    """A static exponent as jnp takes it; an int keeps ``integer_pow``."""
    return int(ef) if ef.denominator == 1 else float(ef)


def _unrolled(x: Any) -> Iterator[Any]:
    """Iterate over ``x``'s leading axis; unrolled so it works inside a trace."""
    if not x.shape:
        raise TypeError("iteration over a 0-d quantity")
    return iter([x[i] for i in range(x.shape[0])])


def _mod(x: Any, y: Any) -> Any:
    """Elementwise remainder of two operands of the same dimension."""
    mags, dim, bs = _agree([x, y], "take the remainder of")
    return _lift(jnp.mod(*mags), dim, bs)


def _floordiv(x: Any, y: Any) -> jax.Array:
    """Elementwise floor division of two operands of the same dimension."""
    mags, _, _ = _agree([x, y], "floor-divide")
    return jnp.floor_divide(*mags)


# helpers shared with units._array


@overload
def apply(
    fn: Callable[..., Any],
    x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
    *,
    dim: None = ...,
) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
@overload
def apply(
    fn: Callable[..., Any], x: Any, *args: Any, dim: Dim | None = ..., **kwargs: Any
) -> Any: ...
def apply(
    fn: Callable[..., Any], x: Any, *args: Any, dim: Dim | None = None, **kwargs: Any
) -> Any:
    """Apply a function to quantities' magnitudes and relabel the result.

    Args:
      fn: function on magnitudes, e.g. ``jnp.flip``.
      x: a quantity or plain array; the result carries its base.
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
        _dim(dim, "apply")
    ops = (x, *args, *kwargs.values())
    bs = _joint(*ops)
    dt = _operand_dtype(ops)
    mags = [_mag(o, bs, dt) for o in ops]
    n = len(args)
    lo = [m for m, _ in mags]
    out = fn(lo[0], *lo[1 : n + 1], **dict(zip(kwargs, lo[n + 1 :], strict=True)))
    d = functools.reduce(dmul, (dd for _, dd in mags), ONE) if dim is None else dim
    dt = jnp.result_type(out) if hasattr(out, "dtype") else None
    if d != ONE and dt is not None and not jnp.issubdtype(dt, jnp.floating):
        raise DimensionError(
            f"units.apply: {getattr(fn, '__name__', fn)!r} returned {dt.name} "
            f"magnitudes, which kups.core.units will not label {bs.symbol(d)!r}: an "
            "index, a "
            f"count or a mask measures nothing. Pass dim=units.dim_of(1.0) for an "
            f"unlabelled result, or call fn on q.to(unit)."
        )
    return _lift(out, d, bs)


def _zeroish(x: Any) -> bool:
    """Whether ``x`` is a plain Python zero, which fits any unit."""
    return isinstance(x, int | float) and x == 0


def _agree(xs: Sequence[Any], verb: str) -> tuple[list[Any], Dim, BaseSystem]:
    """Lower same-dimension operands into one base: (magnitudes, dimension, base).

    A plain zero skips the dimension check; ``verb`` names the operation in errors.
    """
    bs = _joint(*xs)
    dt = _operand_dtype(xs)
    mags: list[Any] = []
    dim: Dim | None = None
    for x in xs:
        lo, d = _mag(x, bs, dt)
        mags.append(lo)
        if d == ONE and _zeroish(x):
            continue
        if dim is None:
            dim = d
        elif d != dim:
            raise _mismatch(verb, dim, d, bs)
    return mags, ONE if dim is None else dim, bs


def _operand(x: Any) -> bool:
    """Whether ``x`` can be the other operand of an operator."""
    return (
        isinstance(x, (Unit, jax.Array, int, float, complex, bool))
        or isinstance(x, Quantity)
        or hasattr(x, "__array__")
    )


def _concrete(q: Any, what: str) -> jax.Array:
    """Concrete magnitudes in ``q``'s own base; refuses traced values."""
    lo = _lower(q)
    if isinstance(lo, jax.core.Tracer):
        raise DimensionError(
            f"kups.core.units: {what} needs a concrete magnitude, and this quantity is "
            "an "
            "argument of the traced function or is derived from one. Hoist the "
            "number out of the traced region (RC = float(CUTOFF.to(u.angstrom))), "
            "or keep the comparison in the array world."
        )
    return lo


def _rebase(q: Any, bs: BaseSystem | None = None) -> Any:
    """Re-express a quantity in another base (the active one by default)."""
    if not isinstance(q, Quantity):
        raise TypeError(f"kups.core.units: {q!r} is not a quantity")
    if bs is not None:
        _require_base(bs, "the base to rebase to")
    bs = bs or active_base()
    lo = _lower(q)
    f = _rebase_factor(q.bs, bs, q.dim, q.dtype)
    if f == 1.0:
        return _lift(lo, q.dim, bs)
    out = lo * f
    _check_rescaled(
        lo,
        out,
        f"rebasing {q.bs.symbol(q.dim) or '1'!r} from base {q.bs.name} to base "
        f"{bs.name}",
        bs,
        f,
    )
    return _lift(out, q.dim, bs)


def _sqrt_impl(x: Any) -> Any:
    """Square root of the magnitudes and the dimension."""
    bs = _joint(x)
    lx, dim = _mag(x, bs)
    return _lift(jnp.sqrt(lx), dpow(dim, Frac(1, 2)), bs)


def _norm_impl(x: Any, ord: Any, axis: Any, keepdims: bool) -> Any:
    """Vector or matrix norm; ``ord=0`` (a count) is refused for a quantity."""
    if ord == 0 and isinstance(x, Quantity):
        raise DimensionError(
            f"units.norm: ord=0 counts the nonzero elements of {_measures(x)} "
            f"rather than measuring them, so the count carries no dimension. Write "
            f"jnp.count_nonzero(x.value)."
        )
    return apply(jnp.linalg.norm, x, ord=ord, axis=axis, keepdims=keepdims)


# q.at[idx]

_Q = TypeVar("_Q", bound="AnyQuantity")


def _at_opts(
    mode: Any,
    indices_are_sorted: bool,
    unique_indices: bool,
    wrap_negative_indices: bool,
) -> dict[str, Any]:
    """Index options of jax's ``at[...]`` methods, as keywords."""
    return {
        "mode": mode,
        "indices_are_sorted": indices_are_sorted,
        "unique_indices": unique_indices,
        "wrap_negative_indices": wrap_negative_indices,
    }


@dataclasses.dataclass(frozen=True)
class _AtRef(Generic[_Q]):
    """The slice ``x.at[idx]`` of a quantity.

    Each method lowers ``x`` and its operand to plain numbers, calls jax's
    ``at[idx]`` method, and puts ``x``'s label back. An operand must share
    ``x``'s dimension and is rebased with it.
    """

    x: _Q
    idx: Any

    def _lowered(self, v: Any, verb: str) -> tuple[Any, Any, Callable[[Any], Any]]:
        """Plain numbers for ``x`` and ``v`` (None if absent), and the relabeller."""
        mags, dim, bs = _agree([self.x] if v is None else [self.x, v], verb)
        return mags[0], None if v is None else mags[1], lambda out: _lift(out, dim, bs)

    def get(
        self,
        mode: Any = None,
        fill_value: _Q | float | None = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        wrap_negative_indices: bool = True,
    ) -> _Q:
        """Read the slice.

        Args:
          mode: out-of-range index handling, as jax's gather takes it.
          fill_value: what ``mode="fill"`` reads: a value of ``x``'s dimension
            or a plain zero.
          indices_are_sorted: whether the indices are sorted.
          unique_indices: whether every index appears at most once.
          wrap_negative_indices: whether a negative index counts from the end.

        Returns:
          The slice, with ``x``'s label.

        Raises:
          DimensionError: if ``fill_value``'s dimension is not ``x``'s, or its
            magnitude is traced.
        """
        # jax hashes the fill into the gather, so a labelled fill must lower to a
        # static scalar; a plain one already is
        plain = not isinstance(fill_value, (Quantity, Unit))
        lx, lv, lift = self._lowered(fill_value, "fill")
        fill = fill_value
        if not plain:
            if isinstance(lv, jax.core.Tracer):
                raise DimensionError(
                    "kups.core.units: at[...].get(mode='fill') needs a fill value jax "
                    "can hash, "
                    "and this one is traced. Hoist the number out of the traced region "
                    "(FILL = float(fill.to(unit))), or scatter with mode='drop' and "
                    "mask the result instead."
                )
            fill = float(lv)
        opts = _at_opts(mode, indices_are_sorted, unique_indices, wrap_negative_indices)
        return lift(lx.at[self.idx].get(fill_value=fill, **opts))

    def set(
        self,
        v: _Q | AnyUnit | float,
        mode: Any = None,
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
        v: _Q | ArrayLike,
        mode: Any = None,
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
        mode: Any = None,
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
        bs = _joint(self.x, v)
        dt = _operand_dtype((self.x, v))
        lx, dx = _mag(self.x, bs, dt)
        lv, dv = _mag(v, bs, dt)
        if dv != ONE:
            raise DimensionError(
                "kups.core.units: q.at[...].multiply needs a dimensionless factor, "
                "got "
                f"{bs.symbol(dv)!r} {exps(dv)}; scaling part of an array would "
                f"leave it with two dimensions. Multiply the whole quantity."
            )
        opts = _at_opts(mode, indices_are_sorted, unique_indices, wrap_negative_indices)
        return _lift(lx.at[self.idx].multiply(lv, **opts), dx, bs)


@dataclasses.dataclass(frozen=True)
class _At(Generic[_Q]):
    """``x.at[idx]``: indexed reads and updates that keep a quantity's label."""

    x: _Q

    def __getitem__(self, idx: Any) -> _AtRef[_Q]:
        """The slice at ``idx``, ready to read or update."""
        return _AtRef(self.x, idx)


class _Dimension:
    """Pydantic schema hook carried by a dimension alias at runtime.

    It imports ``kups.core.units.pydantic`` only when pydantic builds a schema, so
    importing the core never loads pydantic.
    """

    __slots__ = ("dim",)

    def __init__(self, dim: Dim):
        """Store the alias's exponents."""
        self.dim = dim

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        """Build the validator and serializer schema for this dimension."""
        del source, handler
        from kups.core.units.pydantic import _core_schema

        return _core_schema(self.dim)

    def __get_pydantic_json_schema__(self, schema: Any, handler: Any) -> Any:
        """Build the JSON schema for the accepted input forms."""
        del schema, handler
        from kups.core.units.pydantic import _json_schema

        return _json_schema(self.dim)


# dimension aliases: parameterised for the type checker; at runtime pydantic reads
# the exponents from the _Dimension metadata, since it skips
# __get_pydantic_core_schema__ on a parameterised generic dataclass
if TYPE_CHECKING:
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

else:
    type Acceleration = Annotated[Quantity, _Dimension((1, 0, -2, 0, 0))]
    type Action = Annotated[Quantity, _Dimension((2, 1, -1, 0, 0))]
    type Area = Annotated[Quantity, _Dimension((2, 0, 0, 0, 0))]
    type Charge = Annotated[Quantity, _Dimension((0, 0, 1, 1, 0))]
    type Compressibility = Annotated[Quantity, _Dimension((1, -1, 2, 0, 0))]
    type CoulombConstant = Annotated[Quantity, _Dimension((3, 1, -4, -2, 0))]
    type Current = Annotated[Quantity, _Dimension((0, 0, 0, 1, 0))]
    type Density = Annotated[Quantity, _Dimension((-3, 1, 0, 0, 0))]
    type DipoleMoment = Annotated[Quantity, _Dimension((1, 0, 1, 1, 0))]
    type Energy = Annotated[Quantity, _Dimension((2, 1, -2, 0, 0))]
    type Entropy = Annotated[Quantity, _Dimension((2, 1, -2, 0, -1))]
    type Force = Annotated[Quantity, _Dimension((1, 1, -2, 0, 0))]
    type Frequency = Annotated[Quantity, _Dimension((0, 0, -1, 0, 0))]
    type Length = Annotated[Quantity, _Dimension((1, 0, 0, 0, 0))]
    type Mass = Annotated[Quantity, _Dimension((0, 1, 0, 0, 0))]
    type Momentum = Annotated[Quantity, _Dimension((1, 1, -1, 0, 0))]
    type NumberDensity = Annotated[Quantity, _Dimension((-3, 0, 0, 0, 0))]
    type Permittivity = Annotated[Quantity, _Dimension((-3, -1, 4, 2, 0))]
    type Power = Annotated[Quantity, _Dimension((2, 1, -3, 0, 0))]
    type Pressure = Annotated[Quantity, _Dimension((-1, 1, -2, 0, 0))]
    type Stiffness = Annotated[Quantity, _Dimension((0, 1, -2, 0, 0))]
    type Temperature = Annotated[Quantity, _Dimension((0, 0, 0, 0, 1))]
    type Time = Annotated[Quantity, _Dimension((0, 0, 1, 0, 0))]
    type Velocity = Annotated[Quantity, _Dimension((1, 0, -1, 0, 0))]
    type Voltage = Annotated[Quantity, _Dimension((2, 1, -3, -1, 0))]
    type Volume = Annotated[Quantity, _Dimension((3, 0, 0, 0, 0))]
    type Wavenumber = Annotated[Quantity, _Dimension((-1, 0, 0, 0, 0))]
