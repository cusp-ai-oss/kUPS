# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Array operations with broadcasting utilities.

Provides helper functions for array dimension expansion, conditional selection
with automatic broadcasting, and axis-specific padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from kups.core.utils.units import Measured


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


@overload
def where_broadcast_last[T: Measured](
    condition: Array, x: T, y: T | ArrayLike
) -> T: ...
@overload
def where_broadcast_last[T: Measured](
    condition: Array, x: T | ArrayLike, y: T
) -> T: ...
@overload
def where_broadcast_last(condition: Array, x: ArrayLike, y: ArrayLike) -> Array: ...


def where_broadcast_last(condition: Array, x: Any, y: Any) -> Any:
    """Element-wise ``jnp.where`` with ``condition`` broadcast on trailing dims.

    Expands ``condition`` to match the shapes of ``x`` and ``y`` before
    selecting, so a lower-rank condition naturally broadcasts over trailing
    feature dimensions. Quantity operands keep their dimension; a quantity
    meeting a plain operand reads it as magnitudes in the quantity's base.

    Args:
        condition: Boolean array used for selection.
        x: Values selected where ``condition`` is ``True``.
        y: Values selected where ``condition`` is ``False``.

    Returns:
        Values with shape ``broadcast(x, y)``, measuring what ``x``/``y`` do.

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
    x, y = _match_structure(x, y)

    def _where(a: ArrayLike, b: ArrayLike) -> Array:
        a, b = jnp.broadcast_arrays(jnp.asarray(a), jnp.asarray(b))
        return jnp.where(expand_last_dims(condition, a), a, b)

    return jax.tree.map(_where, x, y)


def _match_structure(x: Any, y: Any) -> tuple[Any, Any]:
    """Give a plain operand the structure of a labelled single-leaf one.

    A quantity is a pytree node over its magnitudes, so a plain operand meeting
    one is read as magnitudes in the quantity's base.

    Args:
        x: First operand, a pytree such as a quantity or a plain array.
        y: Second operand, a pytree such as a quantity or a plain array.

    Returns:
        ``(x, y)`` with a plain leaf operand rewrapped in the other's single-leaf
        structure; unchanged when the structures already match or neither applies.
    """
    xt, yt = jax.tree.structure(x), jax.tree.structure(y)
    if xt == yt:
        return x, y
    if jax.tree_util.treedef_is_leaf(yt) and xt.num_leaves == 1:
        return x, xt.unflatten([y])
    if jax.tree_util.treedef_is_leaf(xt) and yt.num_leaves == 1:
        return yt.unflatten([x]), y
    return x, y


def segment_max(data: Array, segment_ids: Array, num_segments: int) -> Array:
    """Per-segment maximum, ``jax.ops.segment_max`` with out-of-range ids dropped.

    One segment reduces with ``jnp.max``: a one-segment ``segment_max`` is a fully
    contended atomic on GPU.

    Args:
        data: Values, segmented along the leading axis.
        segment_ids: Segment of each row of ``data``.
        num_segments: Number of segments.

    Returns:
        Maximum per segment, shape ``(num_segments, *data.shape[1:])``. Empty
        segments hold the dtype's lowest value (``-inf`` for floats, ``False``
        for booleans).
    """
    if num_segments != 1:
        return jax.ops.segment_max(data, segment_ids, num_segments, mode="drop")
    if jnp.issubdtype(data.dtype, jnp.floating):
        lowest = -jnp.inf
    elif data.dtype == jnp.bool_:
        lowest = False
    else:
        lowest = jnp.iinfo(data.dtype).min
    valid = expand_last_dims(segment_ids == 0, data)
    return jnp.max(jnp.where(valid, data, lowest), axis=0, initial=lowest)[None]


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


def select_n[T](which: Array, *cands: T) -> T:
    """Like ``jax.lax.select_n`` but short-circuits when all candidates are identical.

    At trace time, if every candidate is the *same* tracer (``is`` check),
    the selection is a no-op and the single candidate is returned directly,
    avoiding an unnecessary ``select_n`` primitive in the jaxpr.

    Args:
        which: Integer array indexing into ``cands``.
        *cands: Candidate arrays or quantities to select from, all measuring the
            same thing.

    Returns:
        The selected candidate, or ``cands[0]`` directly if all candidates
        are the same object.
    """
    if all(c is cands[0] for c in cands[1:]):
        return cands[0]
    return jax.tree.map(lambda *c: jax.lax.select_n(which, *c), *cands)


# Block width when reducing long masks to per-block occupancy.
_FIRST_TRUE_BLOCK = 32
# Top-k beats the prefix sum only for a few positions.
_FIRST_TRUE_MAX_SIZE = 8
# Positions below this are exact in float32.
_FIRST_TRUE_MAX_N = 2**24


def first_true_indices(mask: Array, size: int, fill_value: int) -> Array:
    """Positions of the first ``size`` true entries of a 1D ``mask``.

    Equivalent to ``jnp.where(mask, size=size, fill_value=fill_value)[0]``,
    which lowers to a slow scan on accelerators; there, masks shorter than
    ``2**24`` use ``compact_indices``. On CPU, a few positions are found with
    one top-k over negated positions instead; long masks are first reduced to
    per-block occupancy, so that only the first ``size`` occupied blocks are
    searched.

    Args:
        mask: Boolean array of shape ``(n,)``.
        size: Number of positions to return.
        fill_value: Value used once fewer than ``size`` entries are true.

    Returns:
        Integer array of shape ``(size,)`` with increasing positions.
    """
    n = mask.shape[0]
    if jax.default_backend() != "cpu" and n < _FIRST_TRUE_MAX_N:
        found = compact_indices(mask, size)
        return jnp.where(found < n, found, fill_value).astype(int)
    if not 0 < size <= min(n, _FIRST_TRUE_MAX_SIZE) or n >= _FIRST_TRUE_MAX_N:
        return jnp.where(mask, size=size, fill_value=fill_value)[0]
    block = _FIRST_TRUE_BLOCK
    if n > _FIRST_TRUE_MAX_SIZE * block:
        n_blocks = -(-n // block)
        blocks = pad_axis(mask, (0, n_blocks * block - n), axis=0)
        blocks = blocks.reshape(n_blocks, block)
        occupied = first_true_indices(blocks.any(axis=1), size, n_blocks)
        rows = blocks.at[occupied].get(mode="fill", fill_value=False)
        local = first_true_indices(rows.reshape(-1), size, size * block)
        found = occupied.at[local // block].get(mode="fill", fill_value=0) * block
        return jnp.where(local < size * block, found + local % block, fill_value)
    keys = jnp.where(mask, -jnp.arange(n, dtype=jnp.float32), -jnp.inf)
    values, found = jax.lax.top_k(keys, size)
    return jnp.where(values > -jnp.inf, found.astype(int), fill_value)


def _inclusive_sum(x: Array) -> Array:
    """Inclusive prefix sum of a 1D float array via blocked triangular products."""
    n = x.shape[0]
    width = min(32, max(n, 1))
    blocks = jnp.pad(x, (0, -n % width)).reshape(-1, width)
    upper = jnp.triu(jnp.ones((width, width), x.dtype))
    # Exact float32 products: the default GPU precision rounds to TF32.
    inner = jnp.matmul(blocks, upper, precision=jax.lax.Precision.HIGHEST)
    if blocks.shape[0] > 1:
        totals = inner[:, -1]
        inner = inner + (_inclusive_sum(totals) - totals)[:, None]
    return inner.reshape(-1)[:n]


def compact_indices(mask: Array, size: int) -> Array:
    """Positions of the first ``size`` true entries of a 1D ``mask``.

    Equivalent to ``jnp.nonzero(mask, size=size, fill_value=len(mask))[0]``,
    but the prefix sum uses nested blocked triangular matrix products, which
    are faster than a scan. Positions are exact below ``2**24``.

    Args:
        mask: Boolean array of shape ``(n,)``.
        size: Number of positions to return.

    Returns:
        Integer array of shape ``(size,)``, padded with ``n``.
    """
    n = mask.shape[0]
    position = _inclusive_sum(mask.astype(jnp.float32)).astype(jnp.int32) - 1
    return (
        jnp.full(size, n, jnp.int32)
        .at[jnp.where(mask, position, size)]
        .set(jnp.arange(n, dtype=jnp.int32), mode="drop")
    )
