# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.capacity import Capacity
from kups.core.utils.jax import dataclass, jit
from kups.core.utils.segment import bincount


@dataclass
class SubselectResult:
    """Paired scatter/gather indices for sub-selecting elements by category.

    Attributes:
        scatter_idxs: Target indices for scattering gathered elements.
        gather_idxs: Source indices for gathering matching elements.
    """

    scatter_idxs: Array
    gather_idxs: Array

    def __iter__(self):
        """Yield ``(scatter_idxs, gather_idxs)`` for unpacking."""
        yield self.scatter_idxs
        yield self.gather_idxs


def argsort_indices(x: Array) -> Array:
    """Stable argsort of a 1-D integer ``x`` with indices in ``x``'s dtype.

    Before JAX 0.10, ``jnp.argsort`` always sorts a payload of the default integer.

    Args:
        x: 1-D integer array.

    Returns:
        The permutation that sorts ``x``, in ``x``'s dtype.
    """
    rows = jnp.arange(x.size, dtype=x.dtype)
    return jax.lax.sort((x, rows), num_keys=1, is_stable=True)[1]


def offsets_from_counts(counts: Array) -> Array:
    """Compute cumulative offsets from count arrays.

    Converts an array of counts into cumulative offsets, useful for
    indexing into segmented data structures.

    Args:
        counts: Array of count values.

    Returns:
        Array of cumulative offsets starting from 0.
    """
    if counts.size == 0:
        return jnp.array([], dtype=counts.dtype)
    return jnp.cumulative_sum(counts, include_initial=True, dtype=counts.dtype)[:-1]


@jit(static_argnames=("num_segments", "is_sorted"))
def subselect(
    target_ids: Array,
    segment_ids: Array,
    *,
    output_buffer_size: Capacity[int],
    num_segments: int,
    is_sorted: bool = False,
) -> SubselectResult:
    """Find indices for elements belonging to target segments.

    This function efficiently finds all occurrences of elements from specified
    target segment IDs within a segmented array, returning gather and scatter
    indices for data manipulation operations.

    Args:
        target_ids: Array of segment IDs to search for.
        segment_ids: Array mapping each element to its segment ID.
        output_buffer_size: Capacity object controlling output buffer size.
        num_segments: Total number of segments in the data.
        is_sorted: Whether segment_ids is already sorted by segment.

    Returns:
        SubselectResult containing scatter_idxs and gather_idxs for indexing
        operations, in ``segment_ids``' dtype.
    """
    dtype = segment_ids.dtype
    num_occurrences = bincount(segment_ids, length=num_segments)
    target_num_occ = num_occurrences.at[target_ids].get(mode="fill", fill_value=0)
    total_occurrences = jnp.sum(target_num_occ, dtype=dtype)
    output_buffer_size = output_buffer_size.generate_assertion(total_occurrences)
    # Compute gather indices for extracting matching elements
    gather_idxs = jnp.arange(output_buffer_size.size, dtype=dtype) + jnp.repeat(
        offsets_from_counts(num_occurrences)[target_ids]
        - offsets_from_counts(target_num_occ),
        target_num_occ,
        total_repeat_length=output_buffer_size.size,
    )
    # JAX cannot deal with indexing of zero-length arrays
    if gather_idxs.size == 0:
        empty = jnp.array([], dtype=dtype)
        return SubselectResult(empty, empty)
    if not is_sorted:
        gather_idxs = argsort_indices(segment_ids)[gather_idxs]
    # Compute scatter indices for organizing results
    scatter_idxs = jnp.repeat(
        jnp.arange(len(target_ids), dtype=dtype),
        target_num_occ,
        total_repeat_length=output_buffer_size.size,
    )
    mask = jnp.arange(output_buffer_size.size, dtype=dtype) < total_occurrences
    scatter_idxs = jnp.where(mask, scatter_idxs, len(target_ids))
    gather_idxs = jnp.where(mask, gather_idxs, len(segment_ids))
    return SubselectResult(scatter_idxs, gather_idxs)
