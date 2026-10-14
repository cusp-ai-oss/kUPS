# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import itertools as it
from collections.abc import Sequence
from typing import Literal, cast


def split_sequence[S: Sequence[object]](
    seq: S, sizes: Sequence[int], *, align: Literal["left"] | Literal["right"] = "left"
) -> tuple[S, ...]:
    """Split a sequence into parts according to sizes.

    Args
    - seq: the sequence to split (e.g., tuple/list of vars or values)
    - sizes: lengths of leading parts; the final part gets the remainder
    - align: when "left", allocate sizes from the left; when "right", allocate
        from the right so the last parts have the requested sizes if possible.
    """
    if align == "left":
        offsets = tuple(it.accumulate((0, *sizes)))
    elif align == "right":
        # Work backwards from the end: each size specifies how many elements
        # the corresponding part gets, but we allocate from right to left
        cumulative_from_end = tuple(it.accumulate(reversed(sizes)))[::-1]
        # Ensure offsets are non-negative to handle oversized splits correctly
        offsets = (0, *(max(0, len(seq) - size) for size in cumulative_from_end))
    else:
        raise ValueError(f"Unknown align: {align}")
    slices = tuple(slice(a, b) for a, b in zip(offsets, offsets[1:] + (None,)))
    return tuple(cast(S, seq[s]) for s in slices)
