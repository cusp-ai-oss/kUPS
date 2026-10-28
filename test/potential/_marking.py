# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor lists whose patches leave marks on the state.

A test follows the patch of a list through input construction and potentials by
the marks that the final patch leaves, one decimal digit per patch in the order
the patches apply.
"""

from typing import Any, Literal

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from kups.core.data import Index, Table, WithIndices
from kups.core.neighborlist import (
    Edges,
    EmptyNeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.patch import Accept, ExplicitPatch, Patch, WithPatch
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


def record_mark(
    state: dict[str, Any], mark: ArrayLike, accept: Accept
) -> dict[str, Any]:
    """Append ``mark`` as a decimal digit to the marks of the accepting systems."""
    marks = state["marks"]
    return {**state, "marks": jnp.where(accept.data, 10 * marks + mark, marks)}


def applied_marks(patch: Patch[Any], n_systems: int = 1) -> Array:
    """Marks that ``patch`` leaves on an unmarked state when every system accepts."""
    state = {"marks": jnp.zeros(n_systems, dtype=int)}
    accept = Table.arange(jnp.ones(n_systems, dtype=bool), label=SystemId)
    return patch(state, accept)["marks"]


@dataclass
class MarkingNeighborList:
    """List without edges whose patch records ``mark``, as a list that stores a
    cache on the state would."""

    mark: int

    def __call__(
        self,
        keys: Table[ParticleId, NeighborListPoints],
        systems: Table[SystemId, NeighborListSystems],
        *,
        queries: Table[ParticleId, NeighborListPoints] | None = None,
        queried_keys: Index[ParticleId] | None = None,
    ) -> WithPatch[Edges[Literal[2]], Patch[Any]]:
        edges = EmptyNeighborList[Literal[2]](degree=2)(keys, systems).data
        return WithPatch(edges, ExplicitPatch(self.mark, record_mark))


@dataclass
class MarkingProbe:
    """Graph probe whose lists record which graph read them."""

    particles: WithIndices[ParticleId, Any]
    neighborlist_before: MarkingNeighborList
    neighborlist_after: MarkingNeighborList
