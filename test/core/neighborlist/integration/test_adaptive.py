# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end checks for the adaptive cutoff neighbor list.

The adaptive object dispatches to a concrete implementation on each call; these
tests run it to confirm the dispatched implementation produces correct edges and
that the choices agree with each other on the same fixture.
"""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from kups.core.capacity import FixedCapacity
from kups.core.cell import OrthogonalFrame, PeriodicCell
from kups.core.data import Table
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    NeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.neighborlist.adaptive import AdaptiveNeighborList
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId

from .._builders import EvalState, make_lh, make_systems, valid_edge_set


def _fixture(n: int = 30, cutoff: float = 3.5):
    rng = np.random.default_rng(0)
    L = 12.0
    positions = jnp.asarray(rng.uniform(1.0, L - 1.0, size=(n, 3)))
    cell = PeriodicCell(OrthogonalFrame(jnp.array([L, L, L])[None]))
    lh = make_lh(positions, jnp.zeros(n, dtype=int))
    systems, _ = make_systems(cell, jnp.array([cutoff]))
    params = UniversalNeighborlistParameters(
        avg_edges=64, avg_candidates=64, avg_image_candidates=64, cells=64
    )
    state = EvalState(particles=lh, systems=systems, neighborlist_params=params)
    return state, lh, systems, cutoff


def _run(
    nl: NeighborList[Literal[2]],
    lh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> set[tuple[int, int]]:
    result = jax.jit(as_result_function(nl))(keys=lh, systems=systems)
    result.raise_assertion()
    return valid_edge_set(result.value, lh.size)


class TestAdaptiveNeighborListEndToEnd:
    def test_auto_matches_forced_dense(self):
        """The factory returns an adaptive object whose AUTO dispatch (dense for
        the small fixture) matches a forced ``DenseNearestNeighborList``."""
        state, lh, systems, cutoff = _fixture()
        nl = AdaptiveNeighborList.from_state(state, cutoff)
        assert isinstance(nl, AdaptiveNeighborList)

        ref = DenseNearestNeighborList(
            avg_candidates=FixedCapacity(900),
            avg_edges=FixedCapacity(900),
            avg_image_candidates=FixedCapacity(900),
            cutoff=cutoff,
        )
        assert _run(nl, lh, systems) == _run(ref, lh, systems)

    def test_seeded_implementations_agree(self):
        """Every seeded implementation yields the same edge set on one fixture."""
        state, lh, systems, cutoff = _fixture()
        nl = AdaptiveNeighborList.from_state(state, cutoff)
        edge_sets = [_run(c.neighborlist, lh, systems) for c in nl.implementations]
        assert all(edges == edge_sets[0] for edges in edge_sets)
        assert edge_sets[0]  # non-empty
