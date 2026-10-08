# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Array operations with broadcasting utilities.

Provides helper functions for array dimension expansion, conditional selection
with automatic broadcasting, and axis-specific padding.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import tree_map
from jax.typing import ArrayLike


def expand_last_dims(operand: Array, other: Array | tuple[int, ...]) -> Array:
    """Expand trailing dimensions of ``operand`` to match ``other``'s rank.

    Appends size-1 dimensions so that ``operand`` can broadcast against an
    array (or shape tuple) with more axes.

    Args:
        operand: Input array to expand.
        other: Reference array or shape tuple whose rank is the target.

    Returns:
        ``operand`` with ``len(other) - operand.ndim`` trailing size-1 dims.

    Raises:
        AssertionError: If ``operand`` already has more dimensions than ``other``.

    Example:
        ```python
        x = jnp.array([1, 2, 3])          # (3,)
        expand_last_dims(x, (2, 3, 4))    # (3, 1, 1)
        ```
    """
    if isinstance(other, Array):
        other = other.shape
    to_expand = len(other) - len(operand.shape)
    assert to_expand >= 0, "Operand has more dimensions than the target shape."
    return jnp.expand_dims(operand, axis=tuple(range(len(operand.shape), len(other))))


def where_broadcast_last(
    condition: Array, x: Array | ArrayLike, y: Array | ArrayLike
) -> Array:
    """Element-wise ``jnp.where`` with ``condition`` broadcast on trailing dims.

    Expands ``condition`` to match the shapes of ``x`` and ``y`` before
    selecting, so a lower-rank condition naturally broadcasts over trailing
    feature dimensions.

    Args:
        condition: Boolean array used for selection.
        x: Values selected where ``condition`` is ``True``.
        y: Values selected where ``condition`` is ``False``.

    Returns:
        Array with shape ``broadcast(x, y)``.

    Example:
        ```python
        cond = jnp.array([True, False])         # (2,)
        x = jnp.array([[1, 2], [3, 4]])         # (2, 2)
        y = jnp.array([[5, 6], [7, 8]])         # (2, 2)
        where_broadcast_last(cond, x, y)         # [[1, 2], [7, 8]]
        ```
    """
    if x is y and isinstance(x, Array):
        return x
    x, y = tree_map(jnp.asarray, (x, y))
    x, y = jnp.broadcast_arrays(x, y)
    expanded_condition = expand_last_dims(condition, x)
    return jnp.where(expanded_condition, x, y)


def pad_axis(operand: Array, to_pad: tuple[int, int], axis: int) -> Array:
    """Pad a single axis of an array with zeros.

    Args:
        operand: Array to pad.
        to_pad: ``(before, after)`` padding widths for the target axis.
        axis: Axis index to pad.

    Returns:
        Padded array with the same dtype as ``operand``.
    """
    padding = [(0, 0)] * operand.ndim
    padding[axis] = to_pad
    return jnp.pad(operand, padding)


def select_n(which: Array, *cands: Array) -> Array:
    """Like ``jax.lax.select_n`` but short-circuits when all candidates are identical.

    At trace time, if every candidate is the *same* tracer (``is`` check),
    the selection is a no-op and the single candidate is returned directly,
    avoiding an unnecessary ``select_n`` primitive in the jaxpr.

    Args:
        which: Integer array indexing into ``cands``.
        *cands: Candidate arrays to select from.

    Returns:
        The selected array, or ``cands[0]`` directly if all candidates
        are the same object.
    """
    if all(c is cands[0] for c in cands[1:]):
        return cands[0]
    return jax.lax.select_n(which, *cands)


_FIRST_TRUE_BLOCK = 32


def first_true_indices(mask: Array, size: int, fill_value: int) -> Array:
    """Positions of the first ``size`` true entries of a 1D ``mask``.

    Equivalent to ``jnp.where(mask, size=size, fill_value=fill_value)[0]``,
    which lowers to a prefix sum over the whole mask. On CPU, a few positions
    are found with one top-k over negated positions instead; long masks are
    first reduced to per-block occupancy, so that only the first ``size``
    occupied blocks are searched.

    Args:
        mask: Boolean array of shape ``(n,)``.
        size: Number of positions to return.
        fill_value: Value used once fewer than ``size`` entries are true.

    Returns:
        Integer array of shape ``(size,)`` with increasing positions.
    """
    n = mask.shape[0]
    if jax.default_backend() != "cpu" or not 0 < size <= min(n, 8) or n >= 2**24:
        return jnp.where(mask, size=size, fill_value=fill_value)[0]
    block = _FIRST_TRUE_BLOCK
    if n > 8 * block:
        n_blocks = -(-n // block)
        blocks = pad_axis(mask, (0, n_blocks * block - n), axis=0)
        blocks = blocks.reshape(n_blocks, block)
        occupied = first_true_indices(blocks.any(axis=1), size, n_blocks)
        rows = blocks.at[occupied].get(mode="fill", fill_value=False)
        local = first_true_indices(rows.reshape(-1), size, size * block)
        found = occupied.at[local // block].get(mode="fill", fill_value=0) * block
        return jnp.where(local < size * block, found + local % block, fill_value)
    # Positions below 2**24 are exact in float32.
    keys = jnp.where(mask, -jnp.arange(n, dtype=jnp.float32), -jnp.inf)
    values, found = jax.lax.top_k(keys, size)
    return jnp.where(values > -jnp.inf, found.astype(int), fill_value)
