# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the adaptive cutoff policy and construction-time factory."""

import jax.numpy as jnp
import numpy.testing as npt

from kups.core.neighborlist import (
    CellListNeighborList,
    CutoffNeighborListPolicy,
    CutoffNeighborListStrategy,
    DenseNearestNeighborList,
    adaptive_cutoff_neighborlist_from_state,
)

from ._builders import cutoff_table, make_adaptive_state


class TestCutoffNeighborListPolicy:
    """Pure-Python policy over the avg-particles-per-system threshold."""

    def test_small_avg_picks_dense(self):
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=100, num_systems=1)
            is CutoffNeighborListStrategy.DENSE
        )

    def test_sparse_multi_system_picks_dense(self):
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=10_000, num_systems=200)
            is CutoffNeighborListStrategy.DENSE
        )

    def test_large_avg_picks_cell_list(self):
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=20_000, num_systems=1)
            is CutoffNeighborListStrategy.CELL_LIST
        )

    def test_just_below_default_threshold_stays_dense(self):
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=9_999, num_systems=1)
            is CutoffNeighborListStrategy.DENSE
        )

    def test_custom_threshold(self):
        policy = CutoffNeighborListPolicy(
            min_avg_particles_per_system_for_cell_list=500
        )
        assert (
            policy.choose(num_particles=600, num_systems=1)
            is CutoffNeighborListStrategy.CELL_LIST
        )

    def test_zero_systems_falls_back_to_dense(self):
        policy = CutoffNeighborListPolicy()
        assert (
            policy.choose(num_particles=100_000, num_systems=0)
            is CutoffNeighborListStrategy.DENSE
        )


class TestAdaptiveCutoffFactory:
    """The factory returns the right concrete class for the chosen strategy."""

    def test_auto_picks_dense_for_small(self):
        state = make_adaptive_state(n_particles=64, n_systems=1)
        nl = adaptive_cutoff_neighborlist_from_state(
            state, cutoff_table(jnp.array([2.0]))
        )
        assert isinstance(nl, DenseNearestNeighborList)

    def test_auto_picks_cell_list_for_large(self):
        state = make_adaptive_state(n_particles=20_000, n_systems=1)
        nl = adaptive_cutoff_neighborlist_from_state(
            state, cutoff_table(jnp.array([2.0]))
        )
        assert isinstance(nl, CellListNeighborList)

    def test_forced_dense_bypasses_policy(self):
        state = make_adaptive_state(n_particles=20_000, n_systems=1)
        nl = adaptive_cutoff_neighborlist_from_state(
            state,
            cutoff_table(jnp.array([2.0])),
            policy=CutoffNeighborListPolicy(strategy=CutoffNeighborListStrategy.DENSE),
        )
        assert isinstance(nl, DenseNearestNeighborList)

    def test_forced_cell_list_bypasses_policy(self):
        state = make_adaptive_state(n_particles=64, n_systems=1)
        nl = adaptive_cutoff_neighborlist_from_state(
            state,
            cutoff_table(jnp.array([2.0])),
            policy=CutoffNeighborListPolicy(
                strategy=CutoffNeighborListStrategy.CELL_LIST
            ),
        )
        assert isinstance(nl, CellListNeighborList)

    def test_carries_cutoffs(self):
        state = make_adaptive_state(n_particles=64, n_systems=1)
        nl = adaptive_cutoff_neighborlist_from_state(
            state, cutoff_table(jnp.array([3.5]))
        )
        assert isinstance(nl, DenseNearestNeighborList | CellListNeighborList)
        npt.assert_array_equal(nl.cutoffs.data, jnp.array([3.5]))
