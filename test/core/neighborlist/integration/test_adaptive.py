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
    systems, cutoffs = make_systems(cell, jnp.array([cutoff]))
    params = UniversalNeighborlistParameters(
        avg_edges=64, avg_candidates=64, avg_image_candidates=64, cells=64
    )
    state = EvalState(particles=lh, systems=systems, neighborlist_params=params)
    return state, lh, systems, cutoffs


def _run(
    nl: NeighborList[Literal[2]],
    lh: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
) -> set[tuple[int, int]]:
    result = jax.jit(as_result_function(nl))(keys=lh, systems=systems)
    result.raise_assertion()
    return valid_edge_set(result.value)


class TestAdaptiveNeighborListEndToEnd:
    def test_auto_matches_forced_dense(self):
        """The factory returns an adaptive object whose AUTO dispatch (dense for
        the small fixture) matches a forced ``DenseNearestNeighborList``."""
        state, lh, systems, cutoffs = _fixture()
        nl = AdaptiveNeighborList.from_state(state, cutoffs)
        assert isinstance(nl, AdaptiveNeighborList)

        ref = DenseNearestNeighborList(
            avg_candidates=FixedCapacity(900),
            avg_edges=FixedCapacity(900),
            avg_image_candidates=FixedCapacity(900),
            cutoffs=cutoffs,
        )
        assert _run(nl, lh, systems) == _run(ref, lh, systems)

    def test_seeded_implementations_agree(self):
        """Every seeded implementation yields the same edge set on one fixture."""
        state, lh, systems, cutoffs = _fixture()
        nl = AdaptiveNeighborList.from_state(state, cutoffs)
        edge_sets = [_run(c.neighborlist, lh, systems) for c in nl.implementations]
        assert all(edges == edge_sets[0] for edges in edge_sets)
        assert edge_sets[0]  # non-empty


# Cutoff 6: the 15 Å box is minimum image (1 image), the 5 Å box replicates
# 3 x 3 x 3 = 27 images and the slab (30 Å along z) 3 x 3 x 1 = 9.
_CUTOFF = 6.0
_BULK = (216, (15.0, 15.0, 15.0))
_SMALL = (27, (5.0, 5.0, 5.0))
_SLAB = (14, (5.0, 5.0, 30.0))


def _batch(boxes):
    """One orthorhombic system per ``(n, lengths)``, positions seeded by ``n``."""
    positions = [
        np.random.default_rng(n).uniform(0.0, 1.0, (n, 3)) * lengths
        for n, lengths in boxes
    ]
    system = np.concatenate([np.full(len(p), i) for i, p in enumerate(positions)])
    lh = make_lh(jnp.asarray(np.concatenate(positions)), jnp.asarray(system))
    cell = PeriodicCell(OrthogonalFrame(jnp.asarray([lengths for _, lengths in boxes])))
    systems, cutoffs = make_systems(cell, jnp.full(len(boxes), _CUTOFF))
    return lh, systems, cutoffs


def _estimated_edges(boxes, multiplier: float) -> tuple[set[tuple[int, int]], int]:
    """Edges of the adaptive list sized by ``estimate``, and the fixes it needed."""
    lh, systems, cutoffs = _batch(boxes)
    counts = Table(systems.keys, lh.data.system.counts.data)
    params = UniversalNeighborlistParameters.estimate(
        counts, systems, cutoffs, multiplier=multiplier
    )
    state = EvalState(particles=lh, systems=systems, neighborlist_params=params)

    @jax.jit
    @as_result_function
    def edges(state: EvalState):
        nl = AdaptiveNeighborList.from_state(state, cutoffs)
        return nl(state.particles, state.systems)

    fixes = 0
    while (result := edges(state)).failed_assertions:
        state = result.fix_or_raise(state)
        fixes += 1
    return valid_edge_set(result.value), fixes


def _reference_edges(boxes) -> set[tuple[int, int]]:
    lh, systems, cutoffs = _batch(boxes)
    ref = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(256),
        avg_edges=FixedCapacity(1024),
        avg_image_candidates=FixedCapacity(4096),
        cutoffs=cutoffs,
    )
    return _run(ref, lh, systems)


class TestEstimatedImageCapacity:
    """``estimate`` sizes the image candidates from the real candidates: padding
    candidates take no image rows, whatever the image count of the batch's last
    system, and no image is dropped once the candidate buffer has grown."""

    def test_replicated_slab_fits_estimate(self):
        # 14 candidates per particle in a 16-row buffer, 9 images: the real demand
        # 14 * 9 = 126 fits the estimate of 128, the padded 16 * 9 = 144 does not.
        edges, fixes = _estimated_edges([_SLAB], 1.0)
        assert fixes == 0
        assert edges == _reference_edges([_SLAB])

    def test_batch_order_does_not_change_capacity(self):
        # Last system replicated (27 images) or minimum image: neither grows.
        bulk_first, bulk_first_fixes = _estimated_edges([_BULK, _SMALL], 2.0)
        small_first, small_first_fixes = _estimated_edges([_SMALL, _BULK], 2.0)
        assert bulk_first_fixes == small_first_fixes == 0
        assert bulk_first == _reference_edges([_BULK, _SMALL])
        n_bulk, n_small = _BULK[0], _SMALL[0]

        def reorder(i: int) -> int:
            return i + n_small if i < n_bulk else i - n_bulk

        assert {(reorder(i), reorder(j)) for i, j in bulk_first} == small_first

    def test_grown_candidates_keep_slab_images(self):
        # The candidate buffer grows from 128 to 256 rows per particle, as many as
        # the image buffer; the slab's 9 images still need more rows than that.
        edges, _ = _estimated_edges([_BULK, _SLAB], 1.0)
        assert edges == _reference_edges([_BULK, _SLAB])
