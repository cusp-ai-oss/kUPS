# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    Self,
    Sequence,
    TypeVar,
    no_type_check,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.data.batched import Batched
from kups.core.data.index import (
    Index,
    SupportsSorting,
    _format_keys,
)
from kups.core.lens import BoundLens, bind
from kups.core.utils.jax import (
    ScatterArgs,
    dataclass,
    field,
    no_post_init,
    skip_post_init_if_disabled,
    tree_map,
    tree_where_broadcast_last,
)

TKey = TypeVar("TKey", covariant=True, bound=SupportsSorting)
TData = TypeVar("TData", covariant=True)


class SupportsAdd[T](Protocol):
    def __add__(self, other: T) -> Self: ...


class SupportsSub[T](Protocol):
    def __sub__(self, other: T) -> Self: ...


class SupportsMul[T](Protocol):
    def __mul__(self, other: T) -> Self: ...


class SupportsTrueDiv[T](Protocol):
    def __truediv__(self, other: T) -> Self: ...


class SupportsGt[T](Protocol):
    def __gt__(self, other: T) -> Self: ...


class SupportsFloorDiv[T](Protocol):
    def __floordiv__(self, other: T) -> Self: ...


@dataclass
class Table(Batched, Generic[TKey, TData]):
    """Entity-relation table with a primary key column and a data column.

    A ``Table[TKey, TData]`` is a keyed data container analogous to a database
    table.  ``keys`` is the primary key column (unique, sorted) and ``data``
    is the value column — a pytree of arrays whose leaves share a leading
    dimension equal to ``len(keys)``.

    **Accessing data:**

    - ``.data`` gives the raw value pytree, aligned to this table's own keys.
      Use for operations within a single key space.
    - ``table[index]`` where ``index: Index[TKey]`` performs a foreign-key
      lookup — gathering rows by key, analogous to a SQL JOIN.  Use this
      whenever data must be broadcast across key spaces::

          # system → particle: broadcast system data to each particle
          per_particle = systems[particles.data.system]

          # particle → edge: broadcast particle data to each edge
          per_edge = particles[edges.indices]

      Any ``Index[TKey]`` leaf inside another table's ``data`` acts as a
      foreign key referencing this table's primary keys.

    Attributes:
        keys: Primary key column — unique sorted tuple, one entry per row.
        data: Value column — pytree of arrays with leading dimension ``len(keys)``.
    """

    keys: tuple[TKey, ...] = field(static=True)
    data: TData
    _cls: type[TKey] | None = field(static=True, default=None, kw_only=True)

    @property
    def cls(self) -> type[TKey]:
        """Key type, always available even for empty tables."""
        if self._cls is not None:
            return self._cls
        if len(self.keys) > 0:
            return type(self.keys[0])
        raise ValueError("Key type cannot be inferred from empty Table.")

    @skip_post_init_if_disabled
    def __post_init__(self):
        n = len(self.keys)
        assert len(np.unique(np.asarray(self.keys, dtype=object))) == n, (
            "Primary keys must be unique"
        )

        def _validate_and_broadcast(leaf):
            if isinstance(leaf, (Index, Table)):
                return leaf
            dim = leaf.shape[0]
            if dim != n and dim != 1:
                raise ValueError(
                    f"Leaf has size {dim} along axis 0, "
                    f"expected {n} (or 1 for broadcasting)."
                )
            if dim == 1:
                shape = (n,) + leaf.shape[1:]
                return jnp.broadcast_to(leaf, shape)
            return leaf

        updated_data = tree_map(
            _validate_and_broadcast,
            self.data,
            is_leaf=lambda x: isinstance(x, (Index, Table)),
        )
        assert all(type(x) is self.cls for x in self.keys), (
            f"Key type mismatch: expected {self.cls.__name__}, "
            f"got {set(type(x).__name__ for x in self.keys)}."
        )
        object.__setattr__(self, "_cls", self.cls)
        object.__setattr__(self, "data", updated_data)

    def __str__(self) -> str:
        keys_str = _format_keys(self.keys)
        return f"Table({self.cls.__name__}, keys={keys_str}, data={self.data})"

    @classmethod
    def arange[D, Label: SupportsSorting](
        cls, data: D, *, label: type[Label] = int
    ) -> Table[Label, D]:
        """Create a ``Table`` with keys ``label(0), label(1), ..., label(n-1)``."""
        leaves = jax.tree.leaves(data)
        n_items = leaves[0].shape[0]
        index = tuple(map(label, range(n_items)))
        return Table(index, data, _cls=label)

    def at[L: SupportsSorting, D](
        self: Table[L, D], index: Index[L], *, args: ScatterArgs | None = None
    ) -> BoundLens[D, D]:
        """Return a bound lens focused on entries selected by ``index``."""
        return bind(self.data).at(index.indices_in(self.keys), args=args)

    def update[D](
        self: Table[TKey, D], index: Index[TKey], data: D, **kwargs
    ) -> Table[TKey, D]:
        """Write ``data`` into rows selected by ``index``."""
        return (
            bind(self, lambda x: x.data)
            .at(index.indices_in(self.keys), **kwargs)
            .set(data)
        )

    def subset(self, index: Index[TKey]) -> Self:
        """Extract a subset of rows, re-keying as ``(0, 1, ...)``.

        Args:
            index: Rows to extract (must reference ``self.keys``).

        Returns:
            New container with freshly numbered keys.
        """
        new_data = self[index]
        new_idx = tuple(map(self.cls, range(len(index))))
        with no_post_init():
            result = bind(self, lambda x: (x.keys, x.data)).set((new_idx, new_data))
        return result

    def update_if[D, L: SupportsSorting](
        self: Table[TKey, D],
        accept: Table[L, Array],
        indices: Index[TKey],
        new_data: D,
    ) -> Table[TKey, D]:
        """Conditionally update rows based on a per-element accept mask.

        The accept mask is resolved against the indices found in both
        ``self[indices]`` and ``new_data``, and the union of the two is
        used to select which entries to write.

        Args:
            accept: Per-key boolean acceptance indexed by ``L``.
            indices: Target slot positions in ``self``.
            new_data: Proposed replacement data (same structure as subset).

        Returns:
            Updated container with accepted entries written.
        """
        target_cls = accept.cls
        current_data = self[indices]
        self_idx = Index.find(current_data, target_cls)
        data_idx = Index.find(new_data, target_cls)
        mask = (
            accept.at(self_idx, args={"mode": "fill", "fill_value": False}).get()
            | accept.at(data_idx, args={"mode": "fill", "fill_value": False}).get()
        )
        to_write = tree_where_broadcast_last(mask, new_data, current_data)
        return self.update(indices, to_write)

    @overload
    def __getitem__[L1: SupportsSorting, L2: SupportsSorting, L3: SupportsSorting, D](
        self: Table[TKey, Table[L1, Table[L2, Table[L3, D]]]],
        index: tuple[Index[TKey], Index[L1], Index[L2], Index[L3]],
    ) -> D: ...
    @overload
    def __getitem__[L1: SupportsSorting, L2: SupportsSorting, D](
        self: Table[TKey, Table[L1, Table[L2, D]]],
        index: tuple[Index[TKey], Index[L1], Index[L2]],
    ) -> D: ...
    @overload
    def __getitem__[L: SupportsSorting, D](
        self: Table[TKey, Table[L, D]], index: tuple[Index[TKey], Index[L]]
    ) -> D: ...
    @overload
    def __getitem__(self, index: Index[TKey]) -> TData: ...
    @no_type_check
    def __getitem__(self, index):
        """Retrieve data entries selected by ``index``."""
        if isinstance(index, tuple):
            result = self
            for idx in index:
                result = result[idx]
            return result
        idx = index.indices_in(self.keys)
        return bind(self.data).at((idx,)).get()

    def __len__(self) -> int:
        """Number of entries along the leading axis."""
        return len(self.keys)

    @property
    def size(self) -> int:
        """Number of entries along the leading axis (same as ``len()``)."""
        return len(self)

    def __contains__[L: SupportsSorting](self: Table[L, TData], key: L) -> bool:
        """Check whether ``key`` exists in the table."""
        return key in self.keys

    def map_data[L: SupportsSorting, D, T](
        self: Table[L, D], fn: Callable[[D], T]
    ) -> Table[L, T]:
        """Apply ``fn`` to ``data``, keeping the same keys."""
        return Table(self.keys, fn(self.data), _cls=self._cls)

    def set_data[L: SupportsSorting, D, T](self: Table[L, D], data: T) -> Table[L, T]:
        """Replace ``data``, keeping the same keys."""
        return Table(self.keys, data, _cls=self._cls)

    @staticmethod
    @overload
    def join[L: SupportsSorting, D, T1](
        base: Table[L, D], other: Table[L, T1], /
    ) -> Table[L, tuple[D, T1]]: ...
    @staticmethod
    @overload
    def join[L: SupportsSorting, D, T1, T2](
        base: Table[L, D], o1: Table[L, T1], o2: Table[L, T2], /
    ) -> Table[L, tuple[D, T1, T2]]: ...
    @staticmethod
    @overload
    def join[L: SupportsSorting, D, T1, T2, T3](
        base: Table[L, D],
        o1: Table[L, T1],
        o2: Table[L, T2],
        o3: Table[L, T3],
        /,
    ) -> Table[L, tuple[D, T1, T2, T3]]: ...
    @staticmethod
    @overload
    def join[L: SupportsSorting, D, T1, T2, T3, T4](
        base: Table[L, D],
        o1: Table[L, T1],
        o2: Table[L, T2],
        o3: Table[L, T3],
        o4: Table[L, T4],
        /,
    ) -> Table[L, tuple[D, T1, T2, T3, T4]]: ...
    @staticmethod
    @overload
    def join[L: SupportsSorting, D, T1, T2, T3, T4, T5](
        base: Table[L, D],
        o1: Table[L, T1],
        o2: Table[L, T2],
        o3: Table[L, T3],
        o4: Table[L, T4],
        o5: Table[L, T5],
        /,
    ) -> Table[L, tuple[D, T1, T2, T3, T4, T5]]: ...
    @staticmethod
    def join(base: Table, *others: Table) -> Table:
        """Join multiple ``Table`` objects on matching keys into tuple data.

        Performs a SQL-style ``JOIN`` on key equality. All arguments must
        share the same key set. If keys appear in a different order, the
        ``others`` are reindexed to match ``base``'s key ordering before
        their data is combined.

        Args:
            base: The reference ``Table`` whose key ordering is preserved.
            *others: One or more additional ``Table`` objects to join.
                Must have exactly the same key set as ``base``.

        Returns:
            A new ``Table`` with the same keys as ``base`` and data
            ``(base.data, others[0].data, ...)``.

        Raises:
            ValueError: If fewer than one ``other`` is provided, or if
                any argument's key set differs from ``base``.

        Example::

            >>> species = Table(("H", "O"), jnp.array([1, 8]))
            >>> masses  = Table(("O", "H"), jnp.array([16.0, 1.0]))
            >>> joined  = Table.join(species, masses)
            >>> joined.data  # (array([1, 8]), array([1.0, 16.0]))
        """
        if not others:
            raise ValueError("merge_tables requires at least two arguments")
        base, *others = Table.broadcast(base, *others)  # type: ignore

        def _align(other: Table, i: int):
            if other.keys == base.keys:
                return other.data
            if set(other.keys) != set(base.keys):
                raise ValueError(
                    f"Key set mismatch at argument {i + 1}: "
                    f"expected {set(base.keys)}, got {set(other.keys)}"
                )
            idx = Index.new(list(base.keys))
            return other[idx]

        aligned = [_align(o, i) for i, o in enumerate(others)]
        return Table(base.keys, (base.data, *aligned), _cls=base._cls)

    def __iter__(self):
        def data_slice(i: int) -> TData:
            return tree_map(lambda x: x[i], self.data)

        yield from ((key, data_slice(i)) for i, key in enumerate(self.keys))

    def slice(
        self, start: int = 0, end: int | None = None, step: int = 1
    ) -> Table[TKey, TData]:
        """Slice along the leading axis, preserving the corresponding keys.

        Args:
            start: Start index (default 0).
            end: End index (default ``None`` = end of array).
            step: Step size (default 1).
        """
        s = slice(start, end, step)
        data = tree_map(lambda x: x[s], self.data)
        index = self.keys[s]
        return Table(index, data, _cls=self.cls)

    @staticmethod
    def broadcast_to[L: SupportsSorting, D](
        source: Table[L, D], target: Table[L, Any]
    ) -> Table[L, D]:
        """Broadcast ``source`` to match the size of ``target``.

        Convenience wrapper around ``Table.broadcast(source, target)[0]``.
        ``source`` must either already have the same length as ``target``
        or have length 1 (in which case it is repeated).

        Args:
            source: The ``Table`` to broadcast.
            target: The ``Table`` whose size is matched.

        Returns:
            A ``Table`` with the same data as ``source``, broadcast to
            ``len(target)`` entries.
        """
        return Table.broadcast(source, target)[0]

    def __tree_match__(
        self, *others: Table[TKey, TData]
    ) -> tuple[Table[TKey, TData], ...]:
        return Table.broadcast(self, *others)

    @overload
    def __add__(
        self: Table[TKey, SupportsAdd[TData]], other: Table[TKey, TData]
    ) -> Table[TKey, TData]: ...
    @overload
    def __add__[D](
        self: Table[TKey, SupportsAdd[D]], other: D
    ) -> Table[TKey, TData]: ...
    def __add__(self, other) -> Table[TKey, TData]:
        return _table_operator(lambda a, b: a + b, self, other)

    @overload
    def __sub__(
        self: Table[TKey, SupportsSub[TData]], other: Table[TKey, TData]
    ) -> Table[TKey, TData]: ...
    @overload
    def __sub__[D](
        self: Table[TKey, SupportsSub[D]], other: D
    ) -> Table[TKey, TData]: ...
    def __sub__(self, other) -> Table[TKey, TData]:
        return _table_operator(lambda a, b: a - b, self, other)

    @overload
    def __mul__(
        self: Table[TKey, SupportsMul[TData]], other: Table[TKey, TData]
    ) -> Table[TKey, TData]: ...
    @overload
    def __mul__[D](
        self: Table[TKey, SupportsMul[D]], other: D
    ) -> Table[TKey, TData]: ...
    def __mul__(self, other) -> Table[TKey, TData]:
        return _table_operator(lambda a, b: a * b, self, other)

    @overload
    def __truediv__(
        self: Table[TKey, SupportsTrueDiv[TData]], other: Table[TKey, TData]
    ) -> Table[TKey, TData]: ...
    @overload
    def __truediv__[D](
        self: Table[TKey, SupportsTrueDiv[D]], other: D
    ) -> Table[TKey, TData]: ...
    def __truediv__(self, other) -> Table[TKey, TData]:
        return _table_operator(lambda a, b: a / b, self, other)

    @overload
    def __floordiv__(
        self: Table[TKey, SupportsFloorDiv[TData]], other: Table[TKey, TData]
    ) -> Table[TKey, TData]: ...
    @overload
    def __floordiv__[D](
        self: Table[TKey, SupportsFloorDiv[D]], other: D
    ) -> Table[TKey, TData]: ...
    def __floordiv__(self, other) -> Table[TKey, TData]:
        return _table_operator(lambda a, b: a // b, self, other)

    @overload
    def __gt__(
        self: Table[TKey, SupportsGt[TData]], other: Table[TKey, TData]
    ) -> Table[TKey, TData]: ...
    @overload
    def __gt__[D](self: Table[TKey, SupportsGt[D]], other: D) -> Table[TKey, TData]: ...
    def __gt__(self, other) -> Table[TKey, TData]:
        return _table_operator(lambda a, b: a > b, self, other)

    @staticmethod
    @overload
    def broadcast[L: SupportsSorting, D1](
        item1: Table[L, D1], /
    ) -> tuple[Table[L, D1]]: ...
    @staticmethod
    @overload
    def broadcast[L: SupportsSorting, D1, D2](
        item1: Table[L, D1], item2: Table[L, D2], /
    ) -> tuple[Table[L, D1], Table[L, D2]]: ...
    @staticmethod
    @overload
    def broadcast[L: SupportsSorting, D1, D2, D3](
        item1: Table[L, D1], item2: Table[L, D2], item3: Table[L, D3], /
    ) -> tuple[Table[L, D1], Table[L, D2], Table[L, D3]]: ...
    @staticmethod
    @overload
    def broadcast[L: SupportsSorting, D1, D2, D3, D4](
        item1: Table[L, D1],
        item2: Table[L, D2],
        item3: Table[L, D3],
        item4: Table[L, D4],
        /,
    ) -> tuple[Table[L, D1], Table[L, D2], Table[L, D3], Table[L, D4]]: ...
    @staticmethod
    @overload
    def broadcast[L: SupportsSorting](
        *items: Table[L, Any],
    ) -> tuple[Table[L, Any], ...]: ...
    @staticmethod
    def broadcast(
        *items: Table,
    ) -> tuple[Table, ...]:
        """Broadcast ``Table`` containers to a common leading-axis size.

        Analogous to NumPy broadcasting: all inputs must share the same
        key type, and each must either have the maximum size among inputs
        or have exactly size 1. Size-1 tables are expanded by repeating
        their single entry along the leading axis. Requires integer-based
        keys (e.g. ``SystemId``) so that the expanded key range
        ``0 .. max_size-1`` can be generated.

        Args:
            *items: One or more ``Table`` objects to broadcast. All must
                share the same key type. Each must have length equal to
                the maximum or length 1.

        Returns:
            A tuple of ``Table`` objects (one per input), all with the
            same leading-axis size and ``arange`` keys.

        Raises:
            AssertionError: If key types differ, sizes are not
                broadcastable, or keys of full-size tables are not
                ``arange``-style.

        Example::

            >>> scalars = Table.arange(jnp.array([1.0]), label=SystemId)
            >>> vectors = Table.arange(jnp.array([1, 2, 3]), label=SystemId)
            >>> s, v = Table.broadcast(scalars, vectors)
            >>> len(s)  # 3, was broadcast from 1
        """
        cls = items[0].cls
        assert all(cls is i.cls for i in items), (
            f"Key type mismatch: expected all {cls.__name__}, "
            f"got {[i.cls.__name__ for i in items if i.cls is not cls]}"
        )
        max_size = max(len(i) for i in items)

        if all(len(i) == max_size for i in items):
            return items

        sizes = [len(i) for i in items]
        assert all(s in (max_size, 1) for s in sizes), (
            f"Cannot broadcast Table sizes {sizes}: all must be {max_size} or 1"
        )
        assert issubclass(cls, int), (
            f"Broadcasting requires int-based keys, got {cls.__name__}"
        )
        assert all(
            len(i) == 1 or i.keys == tuple(map(cls, range(max_size))) for i in items
        ), (
            f"All full-size Table objects must have arange keys (0..{max_size - 1}) for broadcast"
        )

        new_index = tuple(map(cls, range(max_size)))
        result = [
            bind(i, lambda x: (x.keys, x.data)).set(
                (
                    new_index,
                    tree_map(
                        lambda y: jnp.broadcast_to(y, (max_size, *y.shape[1:])), i.data
                    ),
                )
            )
            for i in items
        ]
        return tuple(result)

    @staticmethod
    @overload
    def union[L1: SupportsSorting, D1](
        item1: Sequence[Table[L1, D1]],
        /,
    ) -> Table[L1, D1]: ...
    @staticmethod
    @overload
    def union[L1: SupportsSorting, D1, L2: SupportsSorting, D2](
        item1: Sequence[Table[L1, D1]],
        item2: Sequence[Table[L2, D2]],
        /,
    ) -> tuple[Table[L1, D1], Table[L2, D2]]: ...
    @staticmethod
    @overload
    def union[
        L1: SupportsSorting,
        D1,
        L2: SupportsSorting,
        D2,
        L3: SupportsSorting,
        D3,
    ](
        item1: Sequence[Table[L1, D1]],
        item2: Sequence[Table[L2, D2]],
        item3: Sequence[Table[L3, D3]],
        /,
    ) -> tuple[Table[L1, D1], Table[L2, D2], Table[L3, D3]]: ...
    @staticmethod
    @overload
    def union[
        L1: SupportsSorting,
        D1,
        L2: SupportsSorting,
        D2,
        L3: SupportsSorting,
        D3,
        L4: SupportsSorting,
        D4,
    ](
        item1: Sequence[Table[L1, D1]],
        item2: Sequence[Table[L2, D2]],
        item3: Sequence[Table[L3, D3]],
        item4: Sequence[Table[L4, D4]],
        /,
    ) -> tuple[Table[L1, D1], Table[L2, D2], Table[L3, D3], Table[L4, D4]]: ...
    @staticmethod
    def union(*groups: Sequence[Table]) -> tuple[Table, ...] | Table:
        """Concatenate multiple ``Table`` sequences (SQL ``UNION ALL``).

        Each positional argument is a sequence of ``Table`` objects that
        share the same key type and schema. Tables within each sequence
        are concatenated along the leading axis. Integer-based sentinel
        keys (e.g. ``SystemId``, ``ParticleId``) are offset-shifted per
        source so that the resulting keys are globally unique. Leaf
        ``Index`` objects nested inside ``data`` are similarly remapped.

        When multiple groups are given, leaf ``Index`` keys are first
        aligned across corresponding tables via ``Table.match`` so that
        cross-references (e.g. particles pointing at system ids) remain
        consistent after concatenation.

        Args:
            *groups: One or more sequences of ``Table`` objects. All
                sequences must have the same length (i.e. the same
                number of sources to merge). Each sequence represents
                one "column" of the database being unioned.

        Returns:
            A single ``Table`` if one group is provided, otherwise a
            tuple of ``Table`` objects (one per group).

        Raises:
            AssertionError: If group lengths differ or duplicate key
                types appear across groups.

        Example::

            >>> p0 = Table.arange(jnp.array([1, 8]), label=ParticleId)
            >>> p1 = Table.arange(jnp.array([6, 7]), label=ParticleId)
            >>> merged = Table.union([p0, p1])
            >>> len(merged)  # 4
            >>> merged.keys  # (ParticleId(0), ..., ParticleId(3))
        """
        n = len(groups[0])
        assert all(len(g) == n for g in groups), "All groups must have the same length"
        groups = tuple(zip(*map(lambda x: Table.match(*x), zip(*groups))))

        def _key_type(group: Sequence[Table]) -> type | None:
            for item in group:
                if item._cls is not None:
                    return item._cls
                if item.keys:
                    return type(item.keys[0])
            return None

        offsets: dict[type, list[int]] = {}
        for group in groups:
            lt = _key_type(group)
            if lt is None:
                continue
            assert lt not in offsets, f"Duplicate key type {lt.__name__} across groups"
            acc = 0
            offsets[lt] = []
            for item in group:
                offsets[lt].append(acc)
                acc += len(item)

        def _shift_key(key, off: int):
            return type(key)(int(key) + off) if isinstance(key, int) else key

        def _concat_leaves(*leaves):
            if isinstance(leaves[0], Index):
                if leaves[0].cls in offsets:
                    return Index.concatenate(*leaves, shift_keys=True)
                return Index.concatenate(*leaves)
            return jnp.concatenate(leaves, axis=0)

        is_index = lambda x: isinstance(x, Index)  # noqa: E731
        results: list[Table] = []
        for group in groups:
            lt = _key_type(group)
            if lt is None:
                results.append(group[0])
                continue
            merged_index = tuple(
                _shift_key(key, offsets[lt][i])
                for i, item in enumerate(group)
                for key in item.keys
            )
            merged_data = jax.tree.map(
                _concat_leaves, *(item.data for item in group), is_leaf=is_index
            )
            results.append(Table(merged_index, merged_data, _cls=lt))

        return results[0] if len(results) == 1 else tuple(results)

    @staticmethod
    @overload
    def match[L1: SupportsSorting, D1](
        group1: Table[L1, D1], /
    ) -> tuple[Table[L1, D1]]: ...
    @staticmethod
    @overload
    def match[L1: SupportsSorting, D1, L2: SupportsSorting, D2](
        group1: Table[L1, D1], group2: Table[L2, D2], /
    ) -> tuple[Table[L1, D1], Table[L2, D2]]: ...
    @staticmethod
    @overload
    def match[
        L1: SupportsSorting,
        D1,
        L2: SupportsSorting,
        D2,
        L3: SupportsSorting,
        D3,
    ](
        group1: Table[L1, D1],
        group2: Table[L2, D2],
        group3: Table[L3, D3],
        /,
    ) -> tuple[Table[L1, D1], Table[L2, D2], Table[L3, D3]]: ...
    @staticmethod
    @overload
    def match[
        L1: SupportsSorting,
        D1,
        L2: SupportsSorting,
        D2,
        L3: SupportsSorting,
        D3,
        L4: SupportsSorting,
        D4,
    ](
        group1: Table[L1, D1],
        group2: Table[L2, D2],
        group3: Table[L3, D3],
        group4: Table[L4, D4],
        /,
    ) -> tuple[Table[L1, D1], Table[L2, D2], Table[L3, D3], Table[L4, D4]]: ...
    @staticmethod
    def match(*groups: Table) -> tuple[Table, ...] | Table:
        """Align leaf ``Index`` keys across multiple ``Table`` containers.

        Ensures that all ``Index`` leaves of the same key type share an
        identical key vocabulary. For each key type present across the
        inputs, the key tuples are merged (deduplicated, sorted) and
        every ``Index`` leaf of that type is updated to use the shared
        vocabulary via ``Index.update_labels``.

        This is typically called before operations that require
        element-wise comparison of indices across tables (e.g. before
        ``Table.union``).

        Args:
            *groups: One or more ``Table`` objects whose leaf indices
                should be aligned.

        Returns:
            A tuple of ``Table`` objects with unified ``Index`` keys,
            or a single ``Table`` if only one input is given.
        """
        indices = {g.cls: g.keys for g in groups}

        def traversal(x):
            if not isinstance(x, Index):
                return x
            if x.cls in indices:
                return x.update_labels(indices[x.cls])
            return x

        result = tree_map(traversal, groups, is_leaf=lambda x: isinstance(x, Index))
        return result

    @staticmethod
    @overload
    def transform[L: SupportsSorting, D1, R](
        fn: Callable[[D1], R],
    ) -> Callable[[Table[L, D1]], Table[L, R]]: ...
    @staticmethod
    @overload
    def transform[L: SupportsSorting, D1, D2, R](
        fn: Callable[[D1, D2], R],
    ) -> Callable[[Table[L, D1], Table[L, D2]], Table[L, R]]: ...
    @staticmethod
    @overload
    def transform[L: SupportsSorting, D1, D2, D3, R](
        fn: Callable[[D1, D2, D3], R],
    ) -> Callable[[Table[L, D1], Table[L, D2], Table[L, D3]], Table[L, R]]: ...
    @staticmethod
    @overload
    def transform[L: SupportsSorting, D1, D2, D3, D4, R](
        fn: Callable[[D1, D2, D3, D4], R],
    ) -> Callable[
        [Table[L, D1], Table[L, D2], Table[L, D3], Table[L, D4]], Table[L, R]
    ]: ...
    @staticmethod
    def transform(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Lift a function on raw data to operate on ``Table`` containers.

        Returns a wrapper that unpacks ``.data`` from each ``Table``
        argument, calls ``fn``, and re-wraps the result in a new
        ``Table`` with the same keys. All inputs must share identical
        keys.

        Args:
            fn: A callable ``(D1, D2, ...) -> R`` operating on the raw
                data payloads of the tables.

        Returns:
            A callable ``(Table[L, D1], Table[L, D2], ...) -> Table[L, R]``
            that applies ``fn`` element-wise over the data.

        Example::

            >>> double = Table.transform(lambda x: x * 2)
            >>> t = Table.arange(jnp.array([1, 2, 3]), label=SystemId)
            >>> double(t).data  # array([2, 4, 6])
        """

        def wrapped(*args: Table[Any, Any]) -> Table[Any, Any]:
            assert all(a.keys == args[0].keys for a in args)
            return Table(args[0].keys, fn(*[a.data for a in args]))

        return wrapped


def _table_operator(op, self, other):
    if isinstance(other, Table):
        assert self.keys == other.keys
        return bind(self, lambda x: x.data).set(op(self.data, other.data))
    else:
        # TODO: This should probably only be allowed when broadcasting?
        return bind(self, lambda x: x.data).set(op(self.data, other))
