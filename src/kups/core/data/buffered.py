# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar, cast, overload

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data.index import Index, SupportsSorting
from kups.core.data.table import Table
from kups.core.lens import bind
from kups.core.utils.jax import dataclass, field, skip_post_init_if_disabled, tree_map
from kups.core.utils.ops import pad_axis

if TYPE_CHECKING:
    from kups.core.typing import HasSystemIndex

TLabel = TypeVar("TLabel", covariant=True, bound=SupportsSorting)
TData = TypeVar("TData", covariant=True)


def system_view(x: HasSystemIndex):
    """Default view function: extracts the ``system`` :class:`Index` leaf.

    Used as the default ``view`` argument for :class:`Buffered` and
    :func:`add_buffers` when data implements ``HasSystemIndex``.
    """
    return x.system


@dataclass
class Buffered(Table[TLabel, TData], Generic[TLabel, TData]):
    """:class:`Table` with buffer management for row occupation.

    A ``Buffered[TKey, TData]`` IS-A :class:`Table` where some rows may be
    unoccupied (soft-deleted).  The ``view`` function extracts an
    :class:`Index` leaf from the data whose :attr:`~Index.valid_mask` serves
    as the occupation mask: rows with OOB sentinel indices are considered
    unoccupied.

    On construction, all leaves *except* the viewed leaf are sanitized:
    plain array leaves are zeroed and other :class:`Index` leaves get an
    OOB sentinel for unoccupied rows.

    Attributes:
        view: Static callable that extracts the authoritative Index leaf
            from the data.
    """

    view: Callable[[TData], Index] = field(static=True)

    @property
    def occupation(self) -> Array:
        """Boolean mask derived from the viewed Index leaf's valid_mask."""
        return self.view(self.data).valid_mask

    @property
    def num_occupied(self) -> Array:
        """Number of occupied slots."""
        return self.occupation.sum()

    @skip_post_init_if_disabled
    def __post_init__(self):
        super().__post_init__()
        mask = self.occupation
        try:
            viewed_leaf = self.view(self.data)
        except AttributeError as e:
            raise ValueError(
                "Could not get occupation index with view. "
                "Most likely the user Buffered an item without .system "
                "attribute and did not overwrite `view`."
            ) from e
        assert isinstance(viewed_leaf, Index), (
            f"View must point towards an Index. Got {viewed_leaf}."
        )

        def _sanitize(leaf):
            if isinstance(leaf, Index):
                if leaf is viewed_leaf:
                    return leaf  # source of truth — do not sanitize
                oob_sentinel = len(leaf.keys)
                data = jnp.where(mask, leaf.indices, oob_sentinel)
                return Index(leaf.keys, data, leaf.max_count, _cls=leaf._cls)
            expand_axes = tuple(i for i in range(leaf.ndim) if i != 0)
            return jnp.where(
                jnp.expand_dims(mask, axis=expand_axes),
                leaf,
                jnp.zeros_like(leaf),
            )

        sanitized = jax.tree.map(
            _sanitize, self.data, is_leaf=lambda x: isinstance(x, Index)
        )
        object.__setattr__(self, "data", sanitized)

    def select_free(self, n: int) -> Index[TLabel]:
        """Return an ``Index`` referencing ``n`` unoccupied slots.

        Args:
            n: Number of free slots to select.

        Returns:
            ``Index`` of shape ``(n,)`` into this buffer's labels.
            If fewer than ``n`` free slots exist, excess entries use
            the OOB sentinel (``len(index)``).
        """
        data = jnp.where(~self.occupation, size=n, fill_value=len(self.keys))[0]
        return Index(self.keys, data)

    @classmethod
    def arange[L: SupportsSorting, D](
        cls,
        data: D,
        *,
        num_occupied: int | None = None,
        label: Callable[[int], L] = int,
        view: Callable[[D], Index] = system_view,
    ) -> Buffered[L, D]:
        """Create a ``Buffered`` with integer labels ``(0, 1, ..., n-1)``.

        The ``view`` function extracts the authoritative Index leaf from
        ``data``. If ``num_occupied`` is less than ``n``, the viewed leaf
        is masked so that trailing entries have OOB sentinel values.

        Args:
            data: Pytree of arrays with a common leading dimension.
            num_occupied: Number of leading slots marked as occupied.
                Defaults to all slots occupied.
            label: Callable mapping ``int`` to label values.
            view: Callable extracting the authoritative Index leaf.

        Returns:
            ``Buffered`` with labels ``(0, 1, ..., n-1)``.
        """
        n = jax.tree.leaves(data)[0].shape[0]
        n_occ = n if num_occupied is None else num_occupied
        index = tuple(map(label, range(n)))
        if n_occ < n:
            mask = jnp.arange(n) < n_occ
            viewed = view(data)
            masked = viewed.apply_mask(mask)
            data = bind(data, view).set(masked)
        return Buffered(
            index, data, view, _cls=label if isinstance(label, type) else None
        )

    @classmethod
    def full[D](
        cls, table: Table[TLabel, D], *, view: Callable[[D], Index] = system_view
    ) -> Buffered[TLabel, D]:
        """Create a fully-occupied ``Buffered`` from a ``Table``.

        Args:
            table: Source table.
            view: Callable extracting the authoritative Index leaf.

        Returns:
            ``Buffered`` with all rows marked as occupied.
        """
        return Buffered(table.keys, table.data, view)

    @classmethod
    def pad[L: int, D](
        cls,
        table: Table[L, D],
        num_free: int,
        *,
        view: Callable[[D], Index] = system_view,
    ) -> Buffered[L, D]:
        """Convert a ``Table`` to a ``Buffered`` with extra free rows.

        All original entries are marked as occupied. New labels are
        consecutive integers starting after the last existing label.
        Zero-padded data has OOB indices in the viewed Index leaf.

        Args:
            table: Source table (fully occupied in the result).
            num_free: Number of unoccupied rows to append.
            view: Callable extracting the authoritative Index leaf.

        Returns:
            ``Buffered`` with ``len(table) + num_free`` rows.
        """
        n = len(table)
        L_t = table.cls
        new_idx = tuple(map(L_t, range(n, num_free + n)))
        new_data = jax.tree.map(lambda x: pad_axis(x, (0, num_free), 0), table.data)
        # Ensure the viewed leaf has OOB for padded entries.
        if num_free > 0:
            mask = jnp.arange(n + num_free) < n
            viewed = view(new_data)
            masked = viewed.apply_mask(mask)
            new_data = bind(new_data, view).set(masked)
        return Buffered(table.keys + new_idx, new_data, view)

    def update[D](
        self: Buffered[TLabel, D], index: Index[TLabel], data: D, **kwargs
    ) -> Buffered[TLabel, D]:
        """Update rows, returning ``Buffered``.

        The viewed Index leaf in ``data`` must carry correct validity
        (OOB sentinel for unoccupied rows).
        """
        return cast(Buffered[TLabel, D], super().update(index, data, **kwargs))  # type: ignore

    def update_if[D, L: SupportsSorting](
        self: Buffered[TLabel, D],
        accept: Table[L, Array],
        indices: Index[TLabel],
        new_data: D,
    ) -> Buffered[TLabel, D]:
        """Conditionally update rows, returning ``Buffered``."""
        return cast(Buffered[TLabel, D], super().update_if(accept, indices, new_data))  # type: ignore


_BufferGroup = tuple[Table, int] | tuple[Table, int, Callable]
"""Either ``(Table, num_free)`` or ``(Table, num_free, view)``."""


# 2-tuple overloads: data must have HasSystemIndex, uses system_view by default
@overload
def add_buffers[L1: int, D1: HasSystemIndex](
    group1: tuple[Table[L1, D1], int],
    /,
) -> tuple[Buffered[L1, D1]]: ...
@overload
def add_buffers[L1: int, D1: HasSystemIndex, L2: int, D2: HasSystemIndex](
    group1: tuple[Table[L1, D1], int],
    group2: tuple[Table[L2, D2], int],
    /,
) -> tuple[Buffered[L1, D1], Buffered[L2, D2]]: ...
@overload
def add_buffers[
    L1: int,
    D1: HasSystemIndex,
    L2: int,
    D2: HasSystemIndex,
    L3: int,
    D3: HasSystemIndex,
](
    group1: tuple[Table[L1, D1], int],
    group2: tuple[Table[L2, D2], int],
    group3: tuple[Table[L3, D3], int],
    /,
) -> tuple[Buffered[L1, D1], Buffered[L2, D2], Buffered[L3, D3]]: ...
@overload
def add_buffers[
    L1: int,
    D1: HasSystemIndex,
    L2: int,
    D2: HasSystemIndex,
    L3: int,
    D3: HasSystemIndex,
    L4: int,
    D4: HasSystemIndex,
](
    group1: tuple[Table[L1, D1], int],
    group2: tuple[Table[L2, D2], int],
    group3: tuple[Table[L3, D3], int],
    group4: tuple[Table[L4, D4], int],
    /,
) -> tuple[Buffered[L1, D1], Buffered[L2, D2], Buffered[L3, D3], Buffered[L4, D4]]: ...
# 3-tuple overloads: explicit view, no HasSystemIndex bound
@overload
def add_buffers[L1: int, D1](
    group1: tuple[Table[L1, D1], int, Callable[[D1], Index]],
    /,
) -> tuple[Buffered[L1, D1]]: ...
@overload
def add_buffers[L1: int, D1, L2: int, D2](
    group1: tuple[Table[L1, D1], int, Callable[[D1], Index]],
    group2: tuple[Table[L2, D2], int, Callable[[D2], Index]],
    /,
) -> tuple[Buffered[L1, D1], Buffered[L2, D2]]: ...
@overload
def add_buffers[
    L1: int,
    D1,
    L2: int,
    D2,
    L3: int,
    D3,
](
    group1: tuple[Table[L1, D1], int, Callable[[D1], Index]],
    group2: tuple[Table[L2, D2], int, Callable[[D2], Index]],
    group3: tuple[Table[L3, D3], int, Callable[[D3], Index]],
    /,
) -> tuple[Buffered[L1, D1], Buffered[L2, D2], Buffered[L3, D3]]: ...
@overload
def add_buffers[
    L1: int,
    D1,
    L2: int,
    D2,
    L3: int,
    D3,
    L4: int,
    D4,
](
    group1: tuple[Table[L1, D1], int, Callable[[D1], Index]],
    group2: tuple[Table[L2, D2], int, Callable[[D2], Index]],
    group3: tuple[Table[L3, D3], int, Callable[[D3], Index]],
    group4: tuple[Table[L4, D4], int, Callable[[D4], Index]],
    /,
) -> tuple[Buffered[L1, D1], Buffered[L2, D2], Buffered[L3, D3], Buffered[L4, D4]]: ...


def add_buffers(
    *groups: _BufferGroup,
) -> tuple[Buffered, ...] | Buffered:
    """Convert ``Table`` containers to ``Buffered`` with extra free rows.

    Each argument is either a ``(table, num_free)`` pair (uses
    ``system_view``) or a ``(table, num_free, view)`` triple.

    Args:
        *groups: ``(Table, num_free)`` or ``(Table, num_free, view)``.

    Returns:
        Tuple of ``Buffered`` containers, one per input group.
    """

    def _normalize(g: _BufferGroup) -> tuple[Table, int, Callable]:
        if len(g) == 2:
            return (g[0], g[1], system_view)
        return g  # type: ignore[return-value]

    normalized = [_normalize(g) for g in groups]

    def generate_padding(item: tuple[Table, int, Callable]) -> Table:
        idx, pad, _view = item
        empty = tree_map(
            lambda x: jnp.zeros_like(x, shape=(pad, *x.shape[1:])), idx.data
        )
        padding = Table.arange(empty, label=idx.cls)
        return padding

    def to_buffered(item: tuple[tuple[Table, int, Callable], Table]) -> Buffered:
        (inp, _, view_fn), result = item
        n_occ = len(inp)
        mask = jnp.arange(len(result)) < n_occ
        viewed = view_fn(result.data)
        masked = viewed.apply_mask(mask)
        new_data = bind(result.data, view_fn).set(masked)
        return Buffered(result.keys, new_data, view_fn)

    tables = [g for g, _, _ in normalized]
    padded = list(map(generate_padding, normalized))
    # Set keys for padded tables to match their sizes.
    keys = {t.cls: t.keys for t in padded}
    padded = jax.tree.map(
        lambda x: (
            x.update_labels(keys[x.cls], allow_missing=True)
            if isinstance(x, Index) and x.cls in keys
            else x
        ),
        padded,
        is_leaf=lambda x: isinstance(x, Index),
    )
    unbuffered_result = Table.union(*zip(tables, padded))
    buffered = tuple(map(to_buffered, zip(normalized, unbuffered_result)))
    if len(buffered) == 1:
        return buffered
    return buffered
