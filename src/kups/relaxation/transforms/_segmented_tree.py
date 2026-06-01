# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-segment tree operations for system-aware Optax transforms.

When several independent systems are flattened into one
[Table[ParticleId, ...]][kups.core.data.table.Table], the tree-global
reductions used by Optax transforms (``optax.tree.vdot``,
``jax.tree.reduce(maximum, ...)``) collapse all systems into a single
scalar — the bug behind cusp-ai-oss/kUPS#94 for batched relaxation.

This module provides three system-aware helpers that the kUPS-native
transforms need to operate per-segment instead, all built directly on
existing relational primitives:

* [tree_vdot][kups.relaxation.transforms._segmented_tree.tree_vdot] —
  per-segment inner product, summed across pytree leaves.
* [tree_segment_max][kups.relaxation.transforms._segmented_tree.tree_segment_max]
  — per-segment row-wise maximum, taken across pytree leaves.
* [tree_segment_norm][kups.relaxation.transforms._segmented_tree.tree_segment_norm]
  — per-segment L2 norm across pytree leaves (sqrt of the per-segment
  inner product with itself).
* [tree_scale_per_row][kups.relaxation.transforms._segmented_tree.tree_scale_per_row]
  — multiply each row of every leaf by its segment's entry in a
  ``Table[K, Array]``.

The reductions are built on
[Index.sum_over][kups.core.data.index.Index.sum_over] /
[Index.max_over][kups.core.data.index.Index.max_over] (segment-wise
reductions returning a [Table[K, Array]][kups.core.data.table.Table]),
and the broadcast on ``table[index]`` (foreign-key lookup back to
per-row shape).

The ``reduce_index`` argument is a pytree prefix of the operand pytree
whose leaves are ``Index[K]`` (treated as pytree leaves), describing
how each leaf's leading axis partitions into segments.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data.index import Index, SupportsSorting
from kups.core.data.table import Table
from kups.core.lens import bind
from kups.core.typing import PyTree
from kups.core.utils.jax import tree_structure


def _is_index(x: Any) -> bool:
    return isinstance(x, Index)


def _layout_and_leaves(
    index_prefix: PyTree, *trees: PyTree
) -> tuple[list[Index[SupportsSorting]], tuple[list[Array], ...]]:
    structure = tree_structure(index_prefix, is_leaf=_is_index)
    idx_leaves: list[Index[SupportsSorting]] = []
    tree_leaves = tuple(list[Array]() for _ in trees)
    for idx_leaf, *subtrees in zip(
        structure.flatten_up_to(index_prefix),
        *[structure.flatten_up_to(t) for t in trees],
        strict=True,
    ):
        leaves = [jax.tree.leaves(t) for t in subtrees]
        idx_leaves.extend([idx_leaf] * len(leaves[0]))
        for i, leaf_list in enumerate(leaves):
            tree_leaves[i].extend(leaf_list)
    expected = len(idx_leaves)
    actual = [len(leaves) for leaves in tree_leaves]
    if any(n != expected for n in actual):
        n_slots = len(structure.flatten_up_to(index_prefix))
        mismatches = "; ".join(
            f"tree[{i}] expanded to {n} array leaves"
            for i, n in enumerate(actual)
            if n != expected
        )
        raise ValueError(
            f"All trees must share the same sub-structure under each index_prefix "
            f"leaf. index_prefix has {n_slots} Index slot(s) and the first tree "
            f"contributes {expected} array leaves under them, but {mismatches}. "
            "Check that every tree you pass alongside index_prefix (e.g. updates, "
            "params, momenta) has matching pytree structure below each Index leaf."
        )
    return idx_leaves, tree_leaves


def tree_vdot(
    a: PyTree, b: PyTree, reduce_index: PyTree
) -> Table[SupportsSorting, Array]:
    """Per-segment inner product across a pytree.

    For each leaf, contracts every axis except the leading row axis to
    yield one dot-product per row, reduces those per-row values per
    segment via [Index.sum_over][kups.core.data.index.Index.sum_over],
    and sums the resulting tables across pytree leaves so the final
    value at segment ``s`` is the inner product of every aligned row of
    every leaf assigned to ``s``.

    Args:
        a: Pytree of arrays. Each leaf's leading axis indexes rows.
        b: Pytree of arrays with the same structure and leaf shapes as ``a``.
        reduce_index: Pytree prefix of ``a`` whose leaves are ``Index[K]``
            (treated as pytree leaves) describing how each leaf's leading
            axis partitions into segments.

    Returns:
        ``Table[K, Array]`` with data shape ``(n_segments,)`` — one
        inner-product per segment, summed across all leaves.
    """

    def _trailing_dot(a: Array, b: Array) -> Array:
        """Per-row dot product: contract every axis but the leading one."""
        return jax.vmap(jnp.vdot, in_axes=(0, 0), out_axes=0)(a, b)

    layout, (a_leaves, b_leaves) = _layout_and_leaves(reduce_index, a, b)
    return functools.reduce(
        lambda x, y: x + y,
        (
            idx.sum_over(_trailing_dot(la, lb))
            for la, lb, idx in zip(a_leaves, b_leaves, layout, strict=True)
        ),
    )


def tree_segment_max(
    tree: PyTree, reduce_index: PyTree
) -> Table[SupportsSorting, Array]:
    """Per-segment maximum across all leaves of ``tree``.

    For each leaf, any trailing axes beyond the leading row axis are
    collapsed via ``jnp.max``, yielding one scalar per row. The per-row
    values are then reduced per segment with
    [Index.max_over][kups.core.data.index.Index.max_over], and the
    resulting ``Table[K, Array]`` are combined across leaves with an
    elementwise ``jnp.maximum`` so the final value at segment ``s`` is
    the maximum over every row of every leaf assigned to ``s``.

    Args:
        tree: Pytree of arrays. Each leaf's leading axis indexes rows.
        reduce_index: Pytree prefix of ``tree`` whose leaves are
            ``Index[K]`` (treated as pytree leaves) describing how each
            leaf's leading axis partitions into segments.

    Returns:
        ``Table[K, Array]`` with data shape ``(n_segments,)`` — one
        scalar per segment.
    """

    def _trailing_max(leaf: Array) -> Array:
        if leaf.ndim > 1:
            leaf = jnp.max(leaf, axis=tuple(range(1, leaf.ndim)))
        return leaf

    layout, (leaves,) = _layout_and_leaves(reduce_index, tree)
    return functools.reduce(
        lambda x, y: bind(x, lambda x: x.data).set(jnp.maximum(x.data, y.data)),
        (
            idx.max_over(_trailing_max(leaf))
            for leaf, idx in zip(leaves, layout, strict=True)
        ),
    )


def tree_segment_norm(
    tree: PyTree, reduce_index: PyTree
) -> Table[SupportsSorting, Array]:
    """Per-segment L2 norm across all leaves of ``tree``.

    Equivalent to ``tree_vdot(tree, tree, reduce_index).map_data(jnp.sqrt)``:
    for each segment ``s`` returns the Euclidean norm of the concatenation
    of every row of every leaf assigned to ``s``.

    Args:
        tree: Pytree of arrays. Each leaf's leading axis indexes rows.
        reduce_index: Pytree prefix of ``tree`` whose leaves are
            ``Index[K]`` (treated as pytree leaves).

    Returns:
        ``Table[K, Array]`` with data shape ``(n_segments,)``.
    """
    return tree_vdot(tree, tree, reduce_index).map_data(jnp.sqrt)


def tree_scale_per_row[P](
    tree: P,
    scale: Table[SupportsSorting, Array],
    reduce_index: PyTree,
) -> P:
    """Multiply each row of every leaf by a (possibly per-segment) scalar.

    Args:
        tree: Pytree of arrays.
        scale: ``Table[K, Array]`` of shape ``(n_segments,)``.
        reduce_index: Pytree of ``Index[K]`` matching ``tree``.
            Each leaf's row at position ``r`` is multiplied by ``scale[layout_leaf.indices[r]]``.

    Returns:
        Pytree with the same structure as ``tree``.
    """
    layout, (leaves,) = _layout_and_leaves(reduce_index, tree)
    scaled = [
        leaf
        * scale[idx].reshape(scale[idx].shape + (1,) * (leaf.ndim - scale[idx].ndim))
        for leaf, idx in zip(leaves, layout, strict=True)
    ]
    return jax.tree.unflatten(jax.tree.structure(tree), scaled)


def tree_clip_per_row[P](
    tree: P,
    limit: Table[SupportsSorting, Array],
    reduce_index: PyTree,
) -> P:
    """Per-component clip of every leaf to ``±limit[segment]``.

    For each leaf row at position ``r`` (segment ``layout_leaf.indices[r]``),
    every element is clamped to ``[-limit[r], limit[r]]``. Pass ``inf`` for
    any segment whose rows should be left untouched.

    Args:
        tree: Pytree of arrays.
        limit: ``Table[K, Array]`` of shape ``(n_segments,)`` of non-negative
            per-segment clip bounds.
        reduce_index: Pytree of ``Index[K]`` matching ``tree``.

    Returns:
        Pytree with the same structure as ``tree``.
    """
    layout, (leaves,) = _layout_and_leaves(reduce_index, tree)
    clipped: list[Array] = []
    for leaf, idx in zip(leaves, layout, strict=True):
        per_row = limit[idx]
        broadcast = per_row.reshape(per_row.shape + (1,) * (leaf.ndim - per_row.ndim))
        clipped.append(jnp.clip(leaf, -broadcast, broadcast))
    return jax.tree.unflatten(jax.tree.structure(tree), clipped)


def tree_where_per_row[P](
    mask: Table[SupportsSorting, Array],
    a: P,
    b: P,
    reduce_index: PyTree,
) -> P:
    """Per-component select between ``a`` and ``b`` based on a per-segment mask.

    For each leaf row at position ``r`` (segment ``layout_leaf.indices[r]``),
    every element is taken from ``a`` if ``mask[r]`` is truthy and from
    ``b`` otherwise.

    Args:
        mask: ``Table[K, Array]`` of shape ``(n_segments,)``, used as a
            per-segment boolean.
        a: Pytree of arrays returned where the mask is true.
        b: Pytree of arrays with the same structure and leaf shapes as ``a``,
            returned where the mask is false.
        reduce_index: Pytree of ``Index[K]`` matching ``a`` and ``b``.

    Returns:
        Pytree with the same structure as ``a``.
    """
    layout, (a_leaves, b_leaves) = _layout_and_leaves(reduce_index, a, b)
    out: list[Array] = []
    for la, lb, idx in zip(a_leaves, b_leaves, layout, strict=True):
        per_row = mask[idx]
        broadcast = per_row.reshape(per_row.shape + (1,) * (la.ndim - per_row.ndim))
        out.append(jnp.where(broadcast, la, lb))
    return jax.tree.unflatten(jax.tree.structure(a), out)
