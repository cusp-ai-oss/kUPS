# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end checks for the adaptive cutoff neighbor-list factory.

The factory picks a concrete implementation at construction time; these tests
run the chosen object to confirm it produces correct edges and that jitting it
compiles only the selected branch.
"""

import jax
import jax.numpy as jnp
import numpy as np

from kups.core.capacity import FixedCapacity
from kups.core.cell import OrthogonalFrame, PeriodicCell
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    adaptive_cutoff_neighborlist_from_state,
)
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.result import as_result_function

from .._builders import (
    EvalState,
    cutoff_table,
    make_adaptive_state,
    make_lh,
    make_systems,
    valid_edge_set,
)


class TestAdaptiveCutoffFactoryEndToEnd:
    def test_adaptive_matches_forced_dense_on_small_fixture(self):
        """When the policy picks dense, the adaptive object's edges equal a
        forced ``DenseNearestNeighborList`` on the same fixture."""
        rng = np.random.default_rng(0)
        L = 12.0
        positions = jnp.asarray(rng.uniform(1.0, L - 1.0, size=(30, 3)))
        cell = PeriodicCell(OrthogonalFrame(jnp.array([L, L, L])[None]))
        cutoff = 3.5
        lh = make_lh(positions, jnp.zeros(30, dtype=int))
        systems, cutoffs = make_systems(cell, jnp.array([cutoff]))

        params = UniversalNeighborlistParameters(
            avg_edges=64, avg_candidates=64, avg_image_candidates=64, cells=64
        )
        state = EvalState(particles=lh, systems=systems, neighborlist_params=params)
        nl = adaptive_cutoff_neighborlist_from_state(state, cutoffs)
        assert isinstance(nl, DenseNearestNeighborList)

        adaptive = jax.jit(as_result_function(nl))(lh=lh, systems=systems)
        adaptive.raise_assertion()

        ref_nl = DenseNearestNeighborList(
            avg_candidates=FixedCapacity(900),
            avg_edges=FixedCapacity(900),
            avg_image_candidates=FixedCapacity(900),
            cutoffs=cutoffs,
        )
        ref = jax.jit(as_result_function(ref_nl))(lh=lh, systems=systems)
        ref.raise_assertion()

        assert valid_edge_set(adaptive.value, 30) == valid_edge_set(ref.value, 30)

    def test_jit_traces_only_chosen_branch(self):
        """Jitting the adaptive NL must not require the other branch to compile."""
        state = make_adaptive_state(n_particles=64, n_systems=1)
        cutoffs = cutoff_table(jnp.array([2.0]))
        nl = adaptive_cutoff_neighborlist_from_state(state, cutoffs)
        # AUTO -> DENSE for 64 particles, so we can jit-call the dense path
        # directly; no cell-list machinery is referenced.
        assert isinstance(nl, DenseNearestNeighborList)
        positions = jnp.zeros((4, 3))
        lh = make_lh(positions, jnp.zeros(4, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])[None])),
            jnp.array([2.0]),
        )

        result = jax.jit(as_result_function(nl))(lh=lh, systems=systems)
        result.raise_assertion()
        assert result.value.indices.shape[-1] == 2
