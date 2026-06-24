# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the adaptive cost functions and per-call dispatch."""

import math

import jax.numpy as jnp
import numpy.testing as npt

from kups.core.neighborlist import (
    AdaptiveNeighborList,
    AllDenseNearestNeighborList,
    CellListNeighborList,
    DenseNearestNeighborList,
    NeighborListCandidate,
    all_dense_cost,
    cell_list_cost,
    dense_cost,
)

from ._builders import cutoff_table, make_adaptive_state


def _always_cheapest(num_particles: int, num_systems: int) -> float:
    return -1.0


class TestDefaultCostFunctions:
    """Default cost guesses reproduce the dense/cell-list crossover."""

    def test_dense_cheaper_below_crossover(self):
        assert dense_cost(9_999, 1) < cell_list_cost(9_999, 1)

    def test_cell_list_cheaper_above_crossover(self):
        assert cell_list_cost(20_000, 1) < dense_cost(20_000, 1)

    def test_dense_divides_work_across_systems(self):
        assert dense_cost(10_000, 100) < dense_cost(10_000, 1)

    def test_all_dense_invalid_for_multiple_systems(self):
        assert all_dense_cost(100, 2) == math.inf
        assert math.isfinite(all_dense_cost(100, 1))

    def test_all_dense_ties_dense_for_single_system(self):
        # Tie -> the earlier (dense) entry wins in dispatch.
        assert all_dense_cost(500, 1) == dense_cost(500, 1)


class TestAdaptiveNeighborList:
    """The adaptive object dispatches to the cheapest implementation per call."""

    def _nl(self, n_particles: int = 64, n_systems: int = 1) -> AdaptiveNeighborList:
        state = make_adaptive_state(n_particles=n_particles, n_systems=n_systems)
        return AdaptiveNeighborList.from_state(
            state, cutoff_table(jnp.array([2.0] * n_systems))
        )

    def test_from_state_returns_adaptive(self):
        assert isinstance(self._nl(), AdaptiveNeighborList)

    def test_seeds_three_implementations(self):
        nl = self._nl()
        types = {type(c.neighborlist) for c in nl.implementations}
        assert types == {
            DenseNearestNeighborList,
            CellListNeighborList,
            AllDenseNearestNeighborList,
        }

    def test_chooses_dense_for_small(self):
        assert isinstance(self._nl()._choose(64, 1), DenseNearestNeighborList)

    def test_chooses_cell_list_for_large(self):
        assert isinstance(self._nl()._choose(20_000, 1), CellListNeighborList)

    def test_per_call_dispatch_by_counts(self):
        # One object routes differently depending on each call's counts.
        nl = self._nl()
        assert isinstance(nl._choose(64, 1), DenseNearestNeighborList)
        assert isinstance(nl._choose(20_000, 1), CellListNeighborList)

    def test_augmentation_overrides_choice(self):
        # Appending a cheaper pair makes it win, demonstrating easy augmentation.
        base = self._nl()
        all_dense = base.implementations[2].neighborlist
        augmented = AdaptiveNeighborList(
            base.implementations + (NeighborListCandidate(all_dense, _always_cheapest),)
        )
        assert augmented._choose(64, 1) is all_dense

    def test_implementations_carry_cutoffs(self):
        for candidate in self._nl().implementations:
            impl = candidate.neighborlist
            assert isinstance(
                impl,
                DenseNearestNeighborList
                | CellListNeighborList
                | AllDenseNearestNeighborList,
            )
            npt.assert_array_equal(impl.cutoffs.data, jnp.array([2.0]))
