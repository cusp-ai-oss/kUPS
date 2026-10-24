# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Random sampling utilities."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from kups.core.data import Index

type Key = Array
type Shape = tuple[int, ...]
type DType = jax.numpy.dtype
type Sampler = Callable[[Key, Shape, DType], Array]


def sample_like[PyTree](sampler: Sampler, key: Key, tree: PyTree) -> PyTree:
    """Generate random arrays matching the structure and shapes of a PyTree.

    Args:
        sampler: Function ``(key, shape, dtype) -> Array`` that produces samples.
        key: JAX PRNG key (split internally for each leaf).
        tree: Template PyTree; each leaf's shape and dtype are matched.

    Returns:
        PyTree with the same structure as ``tree``, filled with random samples.

    Example:
        ```python
        import jax.random as jr
        template = {"a": jnp.zeros((3,)), "b": jnp.zeros((2, 4))}
        result = sample_like(jr.normal, jr.PRNGKey(0), template)
        # result["a"].shape == (3,), result["b"].shape == (2, 4)
        ```
    """
    leaves, treedef = jax.tree.flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten(
        [sampler(key, ref.shape, ref.dtype) for key, ref in zip(keys, leaves)]
    )


def segment_chi_squared(
    key: Key,
    segment: Index[Any],
    dof: Array,
    slots_per_row: int = 3,
    dtype: DTypeLike = float,
) -> Array:
    """Per-segment ``χ²(dof)`` samples, exact and without loops.

    Draws ``slots_per_row`` standard normals per row and sums the squares of the
    first ``dof[s]`` slots of segment ``s``. ``jax.random.chisquare`` samples by
    rejection in data-dependent while loops, a host round trip per iteration on GPU.

    Args:
        key: PRNG key.
        segment: Segment of each row.
        dof: Degrees of freedom per segment, at most ``slots_per_row`` times the
            segment's row count. ``dof <= 0`` gives 0.
        slots_per_row: Normals per row (3 for momenta).
        dtype: Float dtype of the samples.

    Returns:
        One sample per segment, shape ``(num_labels,)``.
    """
    ids = segment.raw_indices
    rows = jnp.arange(ids.shape[0])
    counts = segment.counts.data
    order = jnp.argsort(ids, stable=True)
    start = (jnp.cumsum(counts) - counts).at[ids[order]].get(mode="fill", fill_value=0)
    rank = jnp.zeros_like(rows).at[order].set(rows - start)
    slot = rank[:, None] * slots_per_row + jnp.arange(slots_per_row)
    limit = dof.at[ids].get(mode="fill", fill_value=0)[:, None]
    squares = jax.random.normal(key, slot.shape, dtype) ** 2
    return segment.sum_over(jnp.where(slot < limit, squares, 0.0).sum(-1)).data
