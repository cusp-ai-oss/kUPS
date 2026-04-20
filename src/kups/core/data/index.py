# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, Sequence, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.capacity import Capacity
from kups.core.utils.jax import (
    ScatterArgs,
    dataclass,
    field,
    is_traced,
    isin,
    skip_post_init_if_disabled,
)
from kups.core.utils.subselect import subselect

type PyTree = Any

if TYPE_CHECKING:
    from kups.core.data.table import Table


class SupportsDunderLT(Protocol):
    def __lt__(self, other, /) -> bool: ...


class SupportsDunderGT(Protocol):
    def __gt__(self, other, /) -> bool: ...


type SupportsSorting = SupportsDunderLT | SupportsDunderGT


@dataclass
class Index[Key: SupportsSorting]:
    """JAX-compatible foreign-key column referencing a set of unique keys.

    An ``Index[Key]`` stores a static key vocabulary (``keys``) and a JAX
    integer array of positions into that vocabulary (``indices``).  This
    makes the column compatible with ``jax.jit`` while preserving
    categorical / relational semantics.

    Attributes:
        keys: Unique sorted key vocabulary (stored as a static pytree field).
        indices: Integer JAX array of positions into ``keys``.
        max_count: Optional upper bound on occurrences per key.
    """

    keys: tuple[Key, ...] = field(static=True)
    indices: Array
    max_count: int | None = field(static=True, default=None)
    _cls: type[Key] | None = field(static=True, default=None, kw_only=True)

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(object)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.indices.shape

    @property
    def size(self) -> int:
        return self.indices.size

    @property
    def ndim(self) -> int:
        return self.indices.ndim

    @property
    def T(self) -> "Index[Key]":
        return self.transpose()

    @property
    def cls(self) -> type[Key]:
        if self._cls is not None:
            return self._cls
        if len(self.keys) > 0:
            return type(self.keys[0])
        raise ValueError("Key class cannot be inferred.")

    @skip_post_init_if_disabled
    def __post_init__(self):
        if not isinstance(self.indices, Array):
            return
        assert jnp.issubdtype(self.indices.dtype, jnp.integer)
        np_data = np.empty(len(self.keys), dtype=object)
        np_data[:] = self.keys
        np_sorted, order = np.unique(np_data, return_inverse=True)
        if (np_sorted != np_data).any():
            raise ValueError("Keys must be unique and sorted.")
        object.__setattr__(self, "_cls", self.cls)

    @classmethod
    def new[T: SupportsSorting](
        cls, data: Sequence[T] | np.ndarray, *, max_count: int | None = None
    ) -> Index[T]:
        """Create an ``Index`` from a sequence of keys.

        Args:
            data: Input keys (any shape supported by ``np.asarray``).
            max_count: Optional upper bound on occurrences per key.
                If provided, validated against the actual data.

        Returns:
            A new ``Index`` with deduplicated sorted keys and
            integer-encoded data preserving the original shape.
        """
        np_data = np.asarray(data, dtype=object)
        unique_keys, values = np.unique(np_data, return_inverse=True)
        if max_count is not None and np_data.size > 0:
            _, counts = np.unique(values, return_counts=True)
            assert int(counts.max()) <= max_count, (
                f"max_count={max_count} exceeded: key appears {int(counts.max())} times"
            )
        key_tuple = tuple(unique_keys.tolist())
        return Index(
            key_tuple,
            jnp.asarray(values).reshape(np_data.shape),
            max_count,
            _cls=type(key_tuple[0]) if key_tuple else None,
        )

    @classmethod
    def arange[T: SupportsSorting](
        cls,
        n: int,
        label: Callable[[int], T] = int,
        max_count: int | None = None,
    ) -> Index[T]:
        """Create an ``Index`` with ``n`` unique keys, each appearing once.

        Equivalent to ``Index.integer(np.arange(n), n=n, label=label)``.

        Args:
            n: Number of keys (and elements).
            label: Callable mapping ``int`` to key values (default: ``int``).
            max_count: Optional upper bound on occurrences per key.
        """
        return cls.integer(np.arange(n), n=n, label=label, max_count=max_count)

    @classmethod
    def integer[T: SupportsSorting](
        cls,
        ids: Array | np.ndarray | Sequence[int],
        *,
        n: int | None = None,
        label: Callable[[int], T] = int,
        max_count: int | None = None,
    ) -> Index[T]:
        """Create an ``Index`` with keys ``label(0), ..., label(n-1)``.

        Args:
            ids: Integer array of key indices, shape arbitrary.
            n: Number of unique keys. If ``None``, inferred as
                ``int(ids.max()) + 1``.
            label: Callable mapping ``int`` to key values (default: ``int``).
            max_count: Optional upper bound on occurrences per key.
        """
        if not isinstance(ids, Array):
            ids = np.asarray(ids)
        if n is None:
            n = int(ids.max()) + 1
        return Index(
            tuple(label(i) for i in range(n)),
            jnp.asarray(ids),
            max_count,
            _cls=type(label(0)),
        )

    @classmethod
    def zeros[T: SupportsSorting](
        cls,
        shape: int | tuple[int, ...],
        *,
        label: Callable[[int], T] = int,
        max_count: int | None = None,
    ) -> Index[T]:
        """Create an ``Index`` filled with a single key (the zeroth).

        Args:
            shape: Shape of the resulting index array.
            label: Callable mapping ``int`` to key values (default: ``int``).
            max_count: Optional upper bound on occurrences per key.
        """
        return cls.integer(
            np.zeros(shape, dtype=int), n=1, label=label, max_count=max_count
        )

    @cached_property
    def value(self) -> np.ndarray:
        """Decoded numpy array of key values (same shape as ``indices``)."""
        return np.asarray(self.keys, dtype=object)[np.asarray(self.indices)]

    def __array__(self) -> np.ndarray:
        """Support ``np.asarray(index)`` by returning decoded keys."""
        return self.value

    def __iter__(self):
        """Iterate over elements, yielding scalar ``Index`` per entry."""
        return (
            Index(self.keys, i, max_count=self.max_count, _cls=self.cls)
            for i in self.indices
        )

    def __len__(self):
        """Number of elements along the first axis."""
        return len(self.indices)

    def __getitem__(self, item) -> Index[Key]:
        """Index into the underlying integer array, preserving keys."""
        return self._forward_to_data("__getitem__", item)

    @property
    def num_labels(self):
        """Number of unique keys in the vocabulary."""
        return len(self.keys)

    def transpose(self, *axes: int) -> Index[Key]:
        """Transpose the index array, preserving keys."""
        return self._forward_to_data("transpose", *axes)

    def ravel(self) -> Index[Key]:
        """Flatten to 1-D, preserving keys."""
        return self._forward_to_data("ravel")

    def reshape(self, *args, **kwargs) -> Index[Key]:
        """Reshape the index array, preserving keys."""
        return self._forward_to_data("reshape", *args, **kwargs)

    def _forward_to_data(self, name: str, *args, **kwargs) -> Index[Key]:
        return Index(
            self.keys,
            getattr(self.indices, name)(*args, **kwargs),
            self.max_count,
            _cls=self.cls,
        )

    @property
    def scatter_args(self) -> ScatterArgs:
        """Scatter args using ``len(keys)`` as OOB fill value."""
        return {"mode": "fill", "fill_value": len(self.keys)}

    @property
    def counts(self) -> Table[Key, Array]:
        """Number of occurrences of each key, as ``Table[Key, Array]``."""
        from kups.core.data.table import Table

        return Table(
            self.keys, jnp.bincount(self.indices.ravel(), length=len(self.keys))
        )

    @property
    def valid_mask(self) -> Array:
        """Boolean mask: ``True`` where the index is within bounds (not OOB)."""
        return self.indices < len(self.keys)

    def apply_mask(self, mask: Array) -> Index[Key]:
        """Set entries where ``mask`` is ``False`` to the OOB sentinel.

        Args:
            mask: Boolean array broadcastable to ``self.indices``.

        Returns:
            New ``Index`` with masked-out entries replaced by ``len(keys)``.
        """
        return Index(
            self.keys,
            jnp.where(mask, self.indices, len(self.keys)),
            max_count=self.max_count,
            _cls=self._cls,
        )

    def select_per_label(self, indices: Array) -> Array:
        """For each key, pick the k-th occurrence (relative to absolute index).

        Args:
            indices: Integer array of shape ``(len(keys),)`` with per-key
                relative indices. Values are taken modulo the key count.

        Returns:
            Integer array of shape ``(len(keys),)`` with absolute positions.
            Keys with zero occurrences return ``len(self)`` (OOB sentinel).
        """
        label_counts = self.counts.data
        indices = indices % jnp.maximum(label_counts, 1)
        sorted_indices = jnp.argsort(self.indices, stable=True)
        offsets = jnp.cumulative_sum(label_counts, include_initial=True)[:-1]
        result = sorted_indices.at[offsets + indices].get(
            mode="fill", fill_value=len(self)
        )
        return jnp.where(label_counts == 0, len(self), result)

    def where_rectangular(self, target: Index[Key], max_count: int) -> Array:
        """For each target key, find all positions padded to ``max_count``.

        Args:
            target: Index with keys to search for (must be a subset of ``self.keys``).
            max_count: Maximum number of positions per key.

        Returns:
            Integer array of shape ``(len(target), max_count)`` with positions
            for each target key. Excess entries filled with ``len(self)`` (OOB).
        """
        target_ids = target.indices_in(self.keys)

        @jax.vmap
        def _find(label: Array) -> Array:
            return jnp.where(
                self.indices == label, size=max_count, fill_value=len(self)
            )[0]

        return _find(target_ids)

    def where_flat(
        self, target: Index[Key], /, capacity: Capacity[int] | None = None
    ) -> Array:
        """Find all positions matching any target key, flat.

        Args:
            target: Index with keys to search for (must be a subset of ``self.keys``).
            capacity: Capacity controlling output buffer size.

        Returns:
            Integer array of matching positions. Excess entries are filled
            with ``len(self)`` (OOB sentinel).
        """
        target_ids = target.indices_in(self.keys)
        mask = isin(self.indices, target_ids, len(self.keys))
        required = mask.sum()
        if is_traced(required):
            assert capacity is not None, "Capacity must be set during tracing."
            size = capacity.generate_assertion(required).size
        else:
            size = None
        return jnp.where(mask, size=size, fill_value=len(self))[0]

    def indices_in(
        self, tokens: tuple[Key, ...], *, allow_missing: bool = False
    ) -> Array:
        """Map this array's elements to indices in a target key tuple.

        For each element in ``self``, returns the index of its key in
        ``tokens``. Useful for re-indexing into a different key ordering
        (e.g., mapping per-atom species to potential parameters).

        Args:
            tokens: Target key tuple whose elements must be a superset of
                ``self.keys`` (each key must appear exactly once).
            allow_missing: If ``True``, keys in ``self`` that are absent from
                ``tokens`` are mapped to the OOB sentinel instead of raising.

        Returns:
            A JAX integer array (same shape as ``self.indices``) containing the
            corresponding indices into ``tokens``.

        Raises:
            AssertionError: If any key in ``self`` is missing from ``tokens``.
        """
        if self.keys == tokens:
            return self.indices
        if self.keys == tokens[: len(self.keys)]:
            return jnp.where(self.indices == len(self.keys), len(tokens), self.indices)
        keys1 = np.asarray(self.keys, dtype=object)
        keys2 = np.asarray(tokens, dtype=object)
        mask = keys1[:, None] == keys2
        missing = keys1[~(mask.sum(-1) == 1).astype(bool)]
        if not allow_missing and len(missing) > 0:
            raise ValueError(f"Keys in self not found in target: {missing.tolist()}")
        if len(tokens) == 0:
            return jnp.full_like(self.indices, fill_value=0)
        idx_map = jnp.asarray(np.argmax(mask, axis=-1))
        return idx_map.at[self.indices].get(mode="fill", fill_value=len(tokens))

    def update_labels(
        self, labels: tuple[Key, ...], *, allow_missing: bool = False
    ) -> Index[Key]:
        """Re-index into a new key tuple, preserving logical mapping.

        Args:
            labels: New key tuple (must be a superset of ``self.keys``).
            allow_missing: If ``True``, keys in ``self`` that are absent from
                ``labels`` are mapped to the OOB sentinel instead of raising.

        Returns:
            New ``Index`` with ``labels`` and remapped indices.
        """
        return Index(
            labels,
            self.indices_in(labels, allow_missing=allow_missing),
            max_count=self.max_count,
            _cls=self.cls,
        )

    def subselect(
        self, needle: Index[Key], /, capacity: Capacity[int], is_sorted: bool = False
    ) -> IndexSubselectResult[Key]:
        """Find elements matching target keys, returning key-level indices.

        Args:
            needle: Index with target keys (must be a subset of ``self.keys``).
            capacity: Capacity controlling output buffer size.
            is_sorted: Whether ``self.indices`` is already sorted by key.

        Returns:
            :class:`IndexSubselectResult` with scatter/gather as Index[Key].
        """
        result = subselect(
            needle.indices_in(self.keys),
            self.indices,
            output_buffer_size=capacity,
            num_segments=len(self.keys),
            is_sorted=is_sorted,
        )
        return IndexSubselectResult(
            scatter=Index(
                needle.keys,
                needle.indices.at[result.scatter_idxs].get(**needle.scatter_args),
                _cls=needle.cls,
            ),
            gather=Index(
                self.keys,
                self.indices.at[result.gather_idxs].get(**self.scatter_args),
                _cls=self.cls,
            ),
        )

    def isin(self, other: Index[Key]) -> Array:
        """Test whether each element's key appears in ``other``'s keys.

        Args:
            other: Index whose keys define the membership set.

        Returns:
            Boolean JAX array (same shape as ``self.indices``).
        """
        all_keys = _merge_keys(self.keys, other.keys)
        lh_idx = self.indices_in(all_keys)
        rh_idx = other.indices_in(all_keys)
        max_item = len(all_keys)
        return isin(lh_idx, rh_idx, max_item)

    def __str__(self) -> str:
        keys_str = _format_keys(self.keys)
        return f"Index({self.cls.__name__}, keys={keys_str}, shape={self.shape})"

    def __repr__(self) -> str:
        keys_str = _format_keys(self.keys)
        try:
            strings = np.asarray(self.keys + ("OOB",))[np.asarray(self.indices)]
            arr_str = str(strings)
            return f"Index({arr_str}, cls={self.cls.__name__}, keys={keys_str}, shape={self.shape}, max_count={self.max_count})"
        except jax.errors.TracerArrayConversionError:
            return (
                f"Index(cls={self.cls.__name__}, keys={keys_str}, data={self.indices})"
            )

    def __tree_match__(self, *others: Index[Key]) -> tuple[Index[Key], ...]:
        all_indices: list[Index[Key]] = [self, *others]
        assert all(i.cls is self.cls for i in others), (
            f"cls mismatch: {[i.cls for i in all_indices]}"
        )
        new_keys = _merge_keys(*[t.keys for t in all_indices])
        max_count = max(
            (i.max_count for i in all_indices if i.max_count is not None), default=None
        )
        return tuple(
            Index(new_keys, i.indices_in(new_keys), max_count=max_count, _cls=i.cls)
            for i in all_indices
        )

    def to_cls[T: int, T2: SupportsSorting](
        self: Index[T], new: Callable[[int], T2] | type[T2]
    ) -> Index[T2]:
        """Convert keys to a different sentinel type.

        Args:
            new: Type or callable mapping each integer key to the new type.

        Returns:
            New ``Index`` with converted keys, same indices and max_count.
        """
        assert len(self.keys) > 0 or isinstance(new, type)
        new_keys = tuple(map(new, self.keys))
        cls = new if isinstance(new, type) else None
        return Index(new_keys, self.indices, max_count=self.max_count, _cls=cls)

    def sum_over(self, array: Array) -> Table[Key, Array]:
        """Sums ``array`` values grouped by this index via segment sum.

        Args:
            array: Array with leading dimension matching ``self.indices``.

        Returns:
            A ``Table`` mapping each key to its summed values.
        """
        from kups.core.data.table import Table

        return Table(
            self.keys,
            jax.ops.segment_sum(array, self.indices, self.num_labels, mode="drop"),
            _cls=self._cls,
        )

    @staticmethod
    def find[L: SupportsSorting](obj: PyTree, cls: type[L]) -> Index[L]:
        """Extracts the unique ``Index`` of a given type from a pytree.

        Args:
            obj: Pytree to search for ``Index`` leaves.
            cls: Key type to match against ``Index.cls``.

        Returns:
            The single ``Index`` leaf whose ``cls`` matches.

        Raises:
            AssertionError: If there is not exactly one matching ``Index``.
        """
        leaves = jax.tree.leaves(obj, is_leaf=lambda x: isinstance(x, Index))
        valid_leaves = [x for x in leaves if isinstance(x, Index) and x.cls is cls]
        assert len(valid_leaves) > 0, f"No Index[{cls.__name__}] found in pytree."
        assert len(valid_leaves) < 2, f"Multiple Index[{cls.__name__}] found in pytree."
        return valid_leaves[0]

    @overload
    @staticmethod
    def combine[L1: SupportsSorting](
        idx1: Index[L1],
        /,
    ) -> Index[tuple[L1]]: ...
    @overload
    @staticmethod
    def combine[L1: SupportsSorting, L2: SupportsSorting](
        idx1: Index[L1],
        idx2: Index[L2],
        /,
    ) -> Index[tuple[L1, L2]]: ...
    @overload
    @staticmethod
    def combine[L1: SupportsSorting, L2: SupportsSorting, L3: SupportsSorting](
        idx1: Index[L1],
        idx2: Index[L2],
        idx3: Index[L3],
        /,
    ) -> Index[tuple[L1, L2, L3]]: ...
    @overload
    @staticmethod
    def combine[
        L1: SupportsSorting,
        L2: SupportsSorting,
        L3: SupportsSorting,
        L4: SupportsSorting,
    ](
        idx1: Index[L1],
        idx2: Index[L2],
        idx3: Index[L3],
        idx4: Index[L4],
        /,
    ) -> Index[tuple[L1, L2, L3, L4]]: ...
    @staticmethod
    def combine(*indices: Index) -> Index:
        """Combines multiple indices into a single index of tuple keys.

        The key set is the full Cartesian product of all input key sets.

        Args:
            *indices: Indices to combine. Must all have the same shape.

        Returns:
            An ``Index`` whose keys are the Cartesian product of the
            input keys (as sorted tuples) and whose indices encode the
            per-element combination via mixed-radix encoding.

        Raises:
            AssertionError: If no indices are provided or shapes differ.

        Example::

            >>> idx1 = Index.new([ParticleId(0), ParticleId(1), ParticleId(0)])
            >>> idx2 = Index.new([SystemId(0), SystemId(0), SystemId(1)])
            >>> combined = Index.combine(idx1, idx2)
            >>> combined.keys  # 2 * 2 = 4 entries
            ((ParticleId(0), SystemId(0)), (ParticleId(0), SystemId(1)),
             (ParticleId(1), SystemId(0)), (ParticleId(1), SystemId(1)))
        """
        from itertools import product

        assert len(indices) > 0, "At least one index required."
        shape = indices[0].indices.shape
        assert all(idx.indices.shape == shape for idx in indices), (
            "Index shapes must match."
        )
        all_keys = tuple(sorted(product(*(idx.keys for idx in indices))))
        flat_key = jnp.zeros(indices[0].indices.size, dtype=jnp.int32)
        stride = 1
        for idx in reversed(indices):
            flat_key = flat_key + idx.indices.ravel() * stride
            stride *= idx.num_labels
        return Index(all_keys, flat_key.reshape(shape))

    @overload
    @staticmethod
    def concatenate[K: SupportsSorting](
        *indices: Index[K],
        shift_keys: Literal[False] = ...,
    ) -> Index[K]: ...
    @overload
    @staticmethod
    def concatenate[K: int](  # type: ignore[overload-overlap]
        *indices: Index[K],
        shift_keys: Literal[True],
    ) -> Index[K]: ...
    @staticmethod
    def concatenate(  # type: ignore[misc]
        *indices: Index,
        shift_keys: bool = False,
    ) -> Index:
        """Concatenate Index objects.

        Args:
            *indices: One or more Index objects to concatenate.
            shift_keys: If False (default), keys are merged (deduplicated,
                sorted) and indices remapped into the combined key space.
                If True, each input's keys are offset-shifted to be disjoint;
                requires integer sentinel keys (e.g. ``ParticleId``,
                ``SystemId``).

        Returns:
            A single concatenated Index.
        """
        assert len(indices) > 0, "At least 1 set of indices must be provided."
        assert all(i.cls == indices[0].cls for i in indices)

        if not shift_keys:
            new_keys = _merge_keys(*[i.keys for i in indices])
            parts = [i.indices_in(new_keys) if i.keys else i.indices for i in indices]
            new_indices = jnp.concatenate(parts) if parts else jnp.zeros(0, dtype=int)
            mc = 0
            for i in indices:
                if i.max_count is None:
                    mc = None
                    break
                mc += i.max_count
            return Index(new_keys, new_indices, mc, _cls=indices[0].cls)

        return _concatenate_shifted(*indices)  # type: ignore[arg-type]

    @overload
    @staticmethod
    def match[K: SupportsSorting](
        __i0: Index[K], __i1: Index[K], /
    ) -> tuple[Array, Array]: ...
    @overload
    @staticmethod
    def match[K: SupportsSorting](
        __i0: Index[K], __i1: Index[K], __i2: Index[K], /
    ) -> tuple[Array, Array, Array]: ...
    @overload
    @staticmethod
    def match[K: SupportsSorting](
        *indices: Index[K],
    ) -> tuple[Array, ...]: ...
    @staticmethod
    def match[K: SupportsSorting](*indices: Index[K]) -> tuple[Array, ...]:  # type: ignore[misc]
        """Remap multiple Index objects into a shared key space.

        Merges the key vocabularies of all inputs and returns the integer
        index arrays remapped into the combined key ordering. This is
        useful when two or more Index objects need element-wise comparison
        or alignment (e.g., matching species across systems).

        Args:
            *indices: One or more Index objects to align.

        Returns:
            A tuple of integer JAX arrays (one per input), each indexing
            into the merged key tuple.
        """
        all_keys = _merge_keys(*[i.keys for i in indices])
        return tuple(i.indices_in(all_keys) for i in indices)


@dataclass
class IndexSubselectResult[Key: SupportsSorting]:
    """Key-level scatter/gather result from :meth:`Index.subselect`.

    For each match, ``scatter`` gives the matched key from the needle side
    and ``gather`` gives the matched key from the self (haystack) side.
    For valid entries both carry the same key value, encoded in their
    respective key spaces.

    Attributes:
        scatter: Index[Key] -- matched key per entry from needle.
        gather: Index[Key] -- matched key per entry from self.
    """

    scatter: Index[Key]
    gather: Index[Key]

    def __iter__(self):
        yield self.scatter
        yield self.gather


def _format_keys(keys: tuple[Any, ...]) -> str:
    """Format keys as ranges when contiguous (e.g. '0-99') or as a tuple."""
    if len(keys) <= 1:
        return str(keys)
    try:
        ints = [int(k) for k in keys]
    except (TypeError, ValueError):
        return (
            str(keys)
            if len(keys) <= 10
            else f"({keys[0]}, ..., {keys[-1]}) [{len(keys)}]"
        )
    if ints == list(range(ints[0], ints[0] + len(ints))):
        return f"{ints[0]}-{ints[-1]}"
    if len(keys) <= 10:
        return str(keys)
    return f"({keys[0]}, ..., {keys[-1]}) [{len(keys)}]"


def _concatenate_shifted[Key: int](*indices: Index[Key]) -> Index[Key]:
    total_keys = sum(len(idx.keys) for idx in indices)
    all_keys: list[Key] = []
    parts: list[Array] = []
    off = 0
    for idx in indices:
        all_keys.extend(idx.cls(int(k) + off) for k in idx.keys)
        oob = idx.indices >= len(idx.keys)
        parts.append(jnp.where(oob, total_keys, idx.indices + off))
        off += len(idx.keys)
    mc = max((i.max_count for i in indices if i.max_count is not None), default=None)
    result_keys = tuple(all_keys)
    assert all(int(a) < int(b) for a, b in zip(result_keys, result_keys[1:])), (
        f"Keys must be strictly increasing after offset shift, got {result_keys}"
    )
    return Index(result_keys, jnp.concatenate(parts), mc, _cls=indices[0].cls)


def unify_keys_by_cls[T](tree: T) -> T:
    """Unify ``Index`` keys across a pytree by key class.

    Groups Index leaves by ``cls``, merges their key vocabularies,
    and updates each Index to use the shared keys via
    :meth:`Index.update_labels`.

    Args:
        tree: Arbitrary pytree containing ``Index`` leaves.

    Returns:
        Same-structured pytree with unified Index keys per cls.
    """
    vals, tree_def = jax.tree.flatten(tree, is_leaf=lambda x: isinstance(x, Index))
    shared_cls: dict[type, list[tuple[int, Index]]] = defaultdict(list)
    for i, v in enumerate(vals):
        if isinstance(v, Index):
            shared_cls[v.cls].append((i, v))
    for items in shared_cls.values():
        indices = [v for _, v in items]
        all_keys = _merge_keys(*[v.keys for v in indices])
        matched = [v.update_labels(all_keys) for v in indices]
        for (i, _), idx in zip(items, matched):
            vals[i] = idx
    return jax.tree.unflatten(tree_def, vals)


def _merge_keys[Key: SupportsSorting](
    *tokens: tuple[Key, ...],
) -> tuple[Key, ...]:
    """Return the sorted union of multiple key tuples."""
    return tuple(sorted(set(t for ts in tokens for t in ts)))


find_index = Index.find
