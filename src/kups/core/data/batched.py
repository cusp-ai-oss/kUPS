# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from typing import Self

import jax
import numpy as np

from kups.core.lens import BoundLens, bind


class Batched:
    """Mixin that validates consistent leading batch dimension across pytree leaves."""

    def __post_init__(self):
        """Validate that all array leaves share the same leading dimension.

        Raises:
            ValueError: If arrays have inconsistent or missing leading dimensions.
        """
        try:
            leaves = jax.tree.leaves(self)
            shapes = tuple(x.shape for x in leaves)
        except AttributeError:
            return
        if not isinstance(leaves[0], jax.Array | np.ndarray):
            return
        if len(shapes) > 0:
            reference = shapes[0]
            if len(reference) == 0:
                raise ValueError(
                    f"Batched cannot be initialized with no leading dimension. Got shapes: {shapes}."
                )
            errors = []
            for i, shape in enumerate(shapes[1:]):
                if len(shape) == 0 or shape[0] != reference[0]:
                    errors.append((i + 1, shape))
            if len(errors) > 0:
                msg = "Inconsistent shapes in batched data: "
                msg += f"Expected all leaves to have the same first dimension size {reference[0]}.\n"
                msg += "The following leaves have different shapes: "
                for idx, shape in errors:
                    msg += f"Leaf {idx} has shape {shape}, expected {reference}. "
                raise ValueError(msg)

    def __len__(self) -> int:
        """Return the batch size (leading dimension size).

        Returns:
            The size of the batch dimension across all arrays
        """
        return jax.tree.leaves(self)[0].shape[0]

    @property
    def size(self) -> int:
        """The batch size (same as len()).

        Returns:
            The size of the batch dimension across all arrays
        """
        return len(self)


class Sliceable(Batched):
    """Batched dataclass with ``.at`` slicing and ``__getitem__`` support.

    Provides lens-based ``.at(index)`` for get/set and direct indexing
    via ``self[index]``.
    """

    def at[S](self: S, index, **kwargs) -> BoundLens[S, S]:
        """Return a bound lens focused on ``index``."""
        return bind(self).at(index, **kwargs)

    def __getitem__(self, index) -> Self:
        """Gather entries at ``index``."""
        return self.at(index).get()
