# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the adaptive cost functions and per-call dispatch."""

import math

import pytest

from kups.core.neighborlist import (
    AdaptiveNeighborList,
    AllDenseNearestNeighborList,
    CellListNeighborList,
    DenseNearestNeighborList,
    NeighborListCandidate,
    NvalchemiCellListNeighborList,
    NvalchemiNaiveNeighborList,
    all_dense_cost,
    cell_list_cost,
    dense_cost,
    nvalchemi_cell_list_cost,
    nvalchemi_naive_cost,
)
from kups.core.neighborlist import adaptive as adaptive_mod

from ._builders import make_adaptive_state


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

    def test_all_dense_cheaper_than_dense_for_single_system(self):
        # All-dense undercuts dense, so it is the single-system default.
        assert all_dense_cost(500, 1) < dense_cost(500, 1)


class TestNvalchemiCostFunctions:
    """Nvalchemi costs undercut their JAX equivalents and keep the crossover."""

    def test_naive_cheaper_than_dense(self):
        assert nvalchemi_naive_cost(500, 1) < dense_cost(500, 1)

    def test_cell_list_cheaper_than_jax_cell_list(self):
        assert nvalchemi_cell_list_cost(20_000, 1) < cell_list_cost(20_000, 1)

    def test_naive_wins_below_crossover(self):
        assert nvalchemi_naive_cost(9_999, 1) < nvalchemi_cell_list_cost(9_999, 1)

    def test_cell_list_wins_above_crossover(self):
        assert nvalchemi_cell_list_cost(20_000, 1) < nvalchemi_naive_cost(20_000, 1)

    def test_cell_list_finite_for_multiple_systems(self):
        assert math.isfinite(nvalchemi_cell_list_cost(20_000, 10))


class TestAdaptiveNeighborList:
    """The adaptive object dispatches to the cheapest implementation per call."""

    @pytest.fixture(autouse=True)
    def _no_nvalchemi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin the library seeding regardless of whether nvalchemiops is installed.
        monkeypatch.setattr(adaptive_mod, "_nvalchemi_installed", lambda: False)

    def _nl(self, n_particles: int = 64, n_systems: int = 1) -> AdaptiveNeighborList:
        state = make_adaptive_state(n_particles=n_particles, n_systems=n_systems)
        return AdaptiveNeighborList.from_state(state, 2.0)

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

    def test_chooses_all_dense_for_small(self):
        assert isinstance(self._nl()._choose(64, 1), AllDenseNearestNeighborList)

    def test_chooses_cell_list_for_large(self):
        assert isinstance(self._nl()._choose(20_000, 1), CellListNeighborList)

    def test_per_call_dispatch_by_counts(self):
        # One object routes differently depending on each call's counts.
        nl = self._nl()
        assert isinstance(nl._choose(64, 1), AllDenseNearestNeighborList)
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
            assert impl.cutoff == 2.0


class TestAdaptiveNeighborListWithNvalchemi:
    """When nvalchemiops is installed, the GPU kernels are seeded and preferred."""

    @pytest.fixture(autouse=True)
    def _with_nvalchemi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(adaptive_mod, "_nvalchemi_installed", lambda: True)

    def _nl(self, n_particles: int = 64, n_systems: int = 1) -> AdaptiveNeighborList:
        state = make_adaptive_state(n_particles=n_particles, n_systems=n_systems)
        return AdaptiveNeighborList.from_state(state, 2.0)

    def test_seeds_five_implementations(self):
        types = {type(c.neighborlist) for c in self._nl().implementations}
        assert types == {
            DenseNearestNeighborList,
            CellListNeighborList,
            AllDenseNearestNeighborList,
            NvalchemiNaiveNeighborList,
            NvalchemiCellListNeighborList,
        }

    def test_nvalchemi_candidates_reject_queries(self):
        for candidate in self._nl().implementations:
            nvalchemi = isinstance(
                candidate.neighborlist,
                NvalchemiNaiveNeighborList | NvalchemiCellListNeighborList,
            )
            assert candidate.supports_queries is not nvalchemi

    def test_prefers_nvalchemi_for_self_graph(self):
        nl = self._nl()
        assert isinstance(nl._choose(64, 1), NvalchemiNaiveNeighborList)
        assert isinstance(nl._choose(20_000, 1), NvalchemiCellListNeighborList)

    def test_prefers_nvalchemi_cell_list_for_multiple_systems(self):
        # Crossover is on per-system count: 10 systems * >10k particles each.
        nl = self._nl()
        assert isinstance(nl._choose(2_000_000, 10), NvalchemiCellListNeighborList)

    def test_bipartite_falls_back_to_library_impl(self):
        # Nvalchemi rejects queries, so bipartite calls skip it for a JAX impl.
        nl = self._nl()
        for counts in [(64, 1), (20_000, 1)]:
            chosen = nl._choose(*counts, bipartite=True)
            assert not isinstance(
                chosen, NvalchemiNaiveNeighborList | NvalchemiCellListNeighborList
            )
