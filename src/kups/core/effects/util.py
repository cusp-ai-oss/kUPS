# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import itertools as it
from collections.abc import Sequence
from typing import cast


def split_sequence[S: Sequence[object]](seq: S, sizes: Sequence[int]) -> tuple[S, ...]:
    """Split a sequence into consecutive parts.

    Args:
        seq: The sequence to split (e.g. a tuple or list of vars or values).
        sizes: Lengths of the leading parts; the final part gets the remainder.
    """
    offsets = tuple(it.accumulate((0, *sizes)))
    slices = tuple(slice(a, b) for a, b in zip(offsets, offsets[1:] + (None,)))
    return tuple(cast(S, seq[s]) for s in slices)
