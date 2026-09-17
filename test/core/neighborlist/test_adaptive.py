# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the adaptive cost functions and per-call dispatch."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.cell import OrthogonalFrame, PeriodicCell
from kups.core.data import Index
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    AllDenseNearestNeighborList,
    CellListNeighborList,
    DenseNearestNeighborList,
    NeighborListCandidate,
    SelectableNeighborList,
    all_dense_cost,
    cell_list_cost,
    dense_cost,
)
from kups.core.result import as_result_function

from ._builders import cutoff_table, make_adaptive_state, make_lh, make_systems


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

    @pytest.mark.parametrize("implementation", [0, 1, 2, 3])
    @pytest.mark.parametrize("inclusion", [False, True])
    @pytest.mark.parametrize("exclusion", [False, True])
    def test_pair_candidates_retain_group_pairs_and_self_images(
        self, implementation: int, inclusion: bool, exclusion: bool
    ):
        # Different inclusion groups, one shared exclusion group. Pair terms
        # must decide which interactions to include after geometric selection.
        state = make_adaptive_state(2, 1)
        state = replace(
            state,
            neighborlist_params=replace(
                state.neighborlist_params, avg_edges=32, avg_image_candidates=256
            ),
        )
        particles = make_lh(
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            jnp.zeros(2, dtype=int),
            jnp.zeros(2, dtype=int),
        ).map_data(lambda p: replace(p, inclusion=Index.integer(jnp.arange(2))))
        systems, cutoffs = make_systems(
            PeriodicCell(OrthogonalFrame(jnp.full((1, 3), 4.0))), jnp.array([4.4])
        )
        neighbors = AdaptiveNeighborList.from_state(state, cutoffs)
        candidate = neighbors.implementations[implementation % 3]
        if implementation == 3:
            base = candidate.neighborlist
            assert isinstance(base, SelectableNeighborList)

            # Structural delegation: this class inherits no built-in selector.
            class CustomNeighborList:
                cutoffs = base.cutoffs
                avg_edges = base.avg_edges
                selector = staticmethod(base.selector)
                __call__ = staticmethod(base)

            candidate = replace(candidate, neighborlist=CustomNeighborList())
        neighbors = replace(neighbors, implementations=(candidate,))
        result = jax.jit(
            as_result_function(
                lambda p, s: neighbors.pair_candidates(
                    p, s, inclusion=inclusion, exclusion=exclusion
                )
            )
        )(particles, systems)
        result.raise_assertion()
        batch = result.value
        edges = batch.edges.indices.indices
        valid = (edges < 2).all(-1)
        same_particle = edges[:, 0] == edges[:, 1]
        # Each atom has six self images; the zero-shift self edge is absent.
        assert int((valid & same_particle).sum()) == 12
        assert not bool((valid & same_particle & batch.is_minimum_image).any())
        # Group policies apply before compaction, preserving nonminimum images.
        assert int((valid & ~same_particle & batch.is_minimum_image).sum()) == (
            0 if inclusion or exclusion else 2
        )
        assert int((valid & ~same_particle & ~batch.is_minimum_image).sum()) == (
            0 if inclusion else 10
        )
        if inclusion and exclusion:
            normal = neighbors(particles, systems)
            npt.assert_array_equal(edges, normal.indices.indices)
            npt.assert_array_equal(batch.edges.shifts, normal.shifts)

    @pytest.mark.parametrize("implementation", [0, 1, 2])
    def test_pair_candidates_empty(self, implementation: int):
        particles = make_lh(jnp.zeros((1, 3)), jnp.zeros(1, dtype=int))
        particles = particles.subset(Index(particles.keys, jnp.empty(0, dtype=int)))
        systems, cutoffs = make_systems(
            PeriodicCell(OrthogonalFrame(jnp.full((1, 3), 4.0))), jnp.array([2.0])
        )
        neighbors = AdaptiveNeighborList.from_state(make_adaptive_state(1, 1), cutoffs)
        neighbors = replace(
            neighbors, implementations=(neighbors.implementations[implementation],)
        )
        result = jax.jit(
            as_result_function(
                lambda: neighbors.pair_candidates(
                    particles, systems, inclusion=True, exclusion=True
                )
            )
        )()
        result.raise_assertion()
        assert result.value.edges.indices.shape == (0, 2)
        assert result.value.is_minimum_image.shape == (0,)
