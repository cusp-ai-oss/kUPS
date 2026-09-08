# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Fixed-capacity slot layout for streaming batches.

A *slotted* batch has ``n_slots`` systems, each owning ``capacity`` contiguous
rows of the flat particle table: slot ``s`` owns rows
``[s * capacity, (s + 1) * capacity)``. Because the rows are contiguous, the
flat ``(n_rows, ...)`` view and the slotted ``(n_slots, capacity, ...)`` view
of any particle-level pytree are pure reshapes of each other.

Rows a slot does not use carry the out-of-bounds sentinel in their
``system`` :class:`~kups.core.data.index.Index` (the same convention as
:class:`~kups.core.data.buffered.Buffered`), so neighbor lists and potentials
ignore them. :attr:`SlotLayout.slot_of_row` is the *static* row-to-slot map
that stays valid for every row regardless of occupation.
"""

from __future__ import annotations

import jax
import numpy as np
from jax import Array

from kups.core.data.index import Index
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field


def _is_index(x: object) -> bool:
    return isinstance(x, Index)


@dataclass
class SlotLayout:
    """Static geometry of a slotted batch.

    Attributes:
        n_slots: Number of systems in the batch.
        capacity: Particle rows owned by each slot.
    """

    n_slots: int = field(static=True)
    capacity: int = field(static=True)

    def __post_init__(self) -> None:
        if self.n_slots < 1 or self.capacity < 1:
            raise ValueError("n_slots and capacity must be positive.")

    @property
    def n_rows(self) -> int:
        """Total number of particle rows, ``n_slots * capacity``."""
        return self.n_slots * self.capacity

    @property
    def keys(self) -> tuple[SystemId, ...]:
        """System keys of the slots, ``SystemId(0), ..., SystemId(n_slots - 1)``."""
        return tuple(SystemId(i) for i in range(self.n_slots))

    @property
    def slot_of_row(self) -> Index[SystemId]:
        """Static row-to-slot map over all ``n_rows`` rows (never OOB)."""
        return Index.integer(
            np.repeat(np.arange(self.n_slots), self.capacity),
            n=self.n_slots,
            label=SystemId,
            max_count=self.capacity,
        )

    def to_slots[T](self, tree: T) -> T:
        """Reshape every leaf from ``(n_rows, ...)`` to ``(n_slots, capacity, ...)``."""
        return jax.tree.map(
            lambda x: x.reshape((self.n_slots, self.capacity) + x.shape[1:]),
            tree,
            is_leaf=_is_index,
        )

    def from_slots[T](self, tree: T) -> T:
        """Inverse of :meth:`to_slots`."""
        return jax.tree.map(
            lambda x: x.reshape((self.n_rows,) + x.shape[2:]), tree, is_leaf=_is_index
        )

    def system_index(self, valid: Array) -> Index[SystemId]:
        """Flat ``system`` index: the owning slot where ``valid``, OOB elsewhere.

        Args:
            valid: Boolean occupation, flat ``(n_rows,)`` or slotted
                ``(n_slots, capacity)``.
        """
        return self.slot_of_row.apply_mask(valid.reshape(self.n_rows))
