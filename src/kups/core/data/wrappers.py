# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from kups.core.data.batched import Batched
from kups.core.data.index import Index, SupportsSorting
from kups.core.utils.jax import dataclass

I = TypeVar("I", bound=SupportsSorting, covariant=True)
TData = TypeVar("TData", covariant=True)
TCache = TypeVar("TCache", covariant=True)


@dataclass
class WithIndices(Batched, Generic[I, TData]):
    """Data paired with an :class:`Index` selecting a subset of elements.

    Attributes:
        indices: Index array mapping entries to labeled elements.
        data: Associated data for the selected elements.
    """

    indices: Index[I]
    data: TData

    def map_data[NewData](
        self, f: Callable[[TData], NewData]
    ) -> WithIndices[I, NewData]:
        """Apply ``f`` to ``data``, keeping the same indices."""
        return WithIndices(self.indices, f(self.data))


@dataclass
class WithCache(Generic[TData, TCache]):
    """Data paired with an associated cache.

    Attributes:
        data: Primary data.
        cache: Cached auxiliary values derived from or associated with `data`.
    """

    data: TData
    cache: TCache
