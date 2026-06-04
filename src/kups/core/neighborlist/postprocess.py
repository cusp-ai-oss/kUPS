# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Postprocessors for the neighbor list pipeline.

Postprocessors run after compaction and can transform the final ``Edges``
using both the edges and the prepared pipeline context. They are the right
place for graph-level output shaping that should not be coupled to row
compaction itself.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from kups.core.data import Index
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import PipelineContext, Postprocessor
from kups.core.utils.jax import dataclass, field


@dataclass
class MirrorPairEdges(Postprocessor[Literal[2]]):
    """Append reversed pair edges for undirected graph outputs.

    The default mirrors only self-graph update calls selected by
    ``ctx.queried_keys``. Full self-neighbor calls already emit both directions
    before compaction, while ``queried_keys`` calls operate on affected ids in
    the already-updated ``keys`` table and are deduplicated by
    ``QueriedKeysDedupMask``. Their reverse edges are restored after compaction.

    Attributes:
        only_when_queried_keys: When ``True``, no-op unless ``ctx.queried_keys``
            is active; ``ctx.queries`` is not involved. Set to ``False`` for
            pipelines whose selector emits only one direction even in full
            calls.
    """

    only_when_queried_keys: bool = field(default=True, static=True)

    def __call__(
        self, edges: Edges[Literal[2]], ctx: PipelineContext
    ) -> Edges[Literal[2]]:
        if self.only_when_queried_keys and ctx.queried_keys is None:
            return edges

        indices = edges.indices.indices
        mirrored_indices = jnp.concatenate([indices, indices[:, ::-1]], axis=0)
        mirrored_shifts = jnp.concatenate([edges.shifts, -edges.shifts], axis=0)
        return Edges(Index(edges.indices.keys, mirrored_indices), mirrored_shifts)
