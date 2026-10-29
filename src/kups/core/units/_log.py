# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Logarithms of quantities, measured against a reference unit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from kups.core.units._core import Quantity
from kups.core.units._unit import DimensionError, Unit
from kups.core.utils.jax import dataclass, field

if TYPE_CHECKING:
    from kups.core.units._typed import _E_A, _E_K, _E_KG, _E_M, _E_S


@dataclass
class Log[U: Unit[Any, Any, Any, Any, Any]]:
    """Natural logs ``log(x / ref)`` of a quantity ``x``.

    The logs are the only pytree leaf; ``ref`` is static, so tables and lenses
    index, scatter and concatenate a ``Log`` like a plain array.

    Attributes:
        log: The stored logs, a plain array.
        ref: The unit the logs are measured against.
    """

    log: jax.Array
    ref: U = field(static=True)

    @overload
    @staticmethod
    def of(
        x: Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K],
        ref: Unit[_E_M, _E_KG, _E_S, _E_A, _E_K],
    ) -> Log[Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]]: ...
    @overload
    @staticmethod
    def of(x: Any, ref: Any) -> Log[Any]: ...
    @staticmethod
    def of(x: Any, ref: Any) -> Log[Any]:
        """Take the logs of ``x`` against ``ref``.

        Args:
            x: Quantity of ``ref``'s dimension.
            ref: The unit the logs are measured against.

        Returns:
            ``log(x / ref)`` with ``ref`` kept alongside.

        Raises:
            DimensionError: If ``x`` and ``ref`` differ in dimension.
        """
        ratio = x / ref
        if isinstance(ratio, Quantity):
            raise DimensionError(f"Log.of: {ratio.unit} is not dimensionless")
        return Log(jnp.log(ratio), ref)

    @staticmethod
    def from_logs[V: Unit[Any, Any, Any, Any, Any]](log: ArrayLike, ref: V) -> Log[V]:
        """Wrap logs that are already measured against ``ref``.

        Args:
            log: The logs ``log(x / ref)``.
            ref: The unit they are measured against.

        Returns:
            The wrapped logs.
        """
        return Log(jnp.asarray(log), ref)

    @overload
    def exp(
        self: Log[Unit[_E_M, _E_KG, _E_S, _E_A, _E_K]],
    ) -> Quantity[_E_M, _E_KG, _E_S, _E_A, _E_K]: ...
    @overload
    def exp(self) -> Any: ...
    def exp(self) -> Any:
        """The linear values ``exp(log) * ref``.

        Returns:
            The quantity the logs encode.
        """
        return jnp.exp(self.log) * self.ref

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the stored logs."""
        return self.log.shape

    def __getitem__(self, idx: Any) -> Log[U]:
        """Index the stored logs, keeping the reference."""
        return Log(self.log[idx], self.ref)
