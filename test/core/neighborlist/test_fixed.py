# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``EmptyNeighborList`` and ``FixedEdgesNeighborList``."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.cell import OrthogonalFrame, PeriodicCell
from kups.core.data.index import Index
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.fixed import EmptyNeighborList, FixedEdgesNeighborList
from kups.core.result import as_result_function
from kups.core.typing import ParticleId

from ._builders import make_edges, make_lh, make_systems


def _simple_lh(n: int):
    return make_lh(jnp.zeros((n, 3)), jnp.zeros(n, dtype=int))


def _simple_systems():
    cell = PeriodicCell(OrthogonalFrame(jnp.array([10.0, 10.0, 10.0])[None]))
    systems, _ = make_systems(cell, jnp.array([2.0]))
    return systems


class TestEmptyNeighborList:
    """``EmptyNeighborList`` returns zero-row ``Edges[D]`` regardless of input."""

    def test_degree_zero_returns_empty_pointcloud(self):
        nl = EmptyNeighborList[Literal[0]](degree=0)
        edges = nl(_simple_lh(4), _simple_systems())
        assert edges.indices.indices.shape == (0, 0)
        assert edges.shifts.shape == (0, 0, 3)

    def test_degree_two_returns_pair_shaped_empty(self):
        nl = EmptyNeighborList[Literal[2]](degree=2)
        edges = nl(_simple_lh(4), _simple_systems())
        assert edges.indices.indices.shape == (0, 2)
        assert edges.shifts.shape == (0, 1, 3)

    def test_degree_three_returns_triple_shaped_empty(self):
        nl = EmptyNeighborList[Literal[3]](degree=3)
        edges = nl(_simple_lh(4), _simple_systems())
        assert edges.indices.indices.shape == (0, 3)
        assert edges.shifts.shape == (0, 2, 3)

    def test_rh_or_for_indices_arguments_are_ignored(self):
        nl = EmptyNeighborList[Literal[2]](degree=2)
        lh = _simple_lh(4)
        systems = _simple_systems()
        rh = _simple_lh(2)
        for_indices = Index(lh.keys, jnp.array([0, 2]))
        assert nl(lh, systems, rh=rh).indices.indices.shape == (0, 2)
        assert nl(lh, systems, for_indices=for_indices).indices.indices.shape == (0, 2)

    def test_rh_and_for_indices_are_mutually_exclusive(self):
        nl = EmptyNeighborList[Literal[2]](degree=2)
        lh = _simple_lh(4)
        with pytest.raises(AssertionError, match="cannot combine rh with for_indices"):
            nl(
                lh,
                _simple_systems(),
                rh=_simple_lh(2),
                for_indices=Index(lh.keys, jnp.array([0, 2])),
            )

    def test_keys_come_from_lh(self):
        nl = EmptyNeighborList[Literal[2]](degree=2)
        lh = _simple_lh(5)
        edges = nl(lh, _simple_systems())
        assert edges.indices.keys == lh.keys

    def test_jit_compiles(self):
        nl = EmptyNeighborList[Literal[2]](degree=2)
        lh = _simple_lh(4)
        systems = _simple_systems()
        edges = jax.jit(lambda l, s: nl(l, s))(lh, systems)
        assert edges.indices.indices.shape == (0, 2)


class TestFixedEdgesNeighborList:
    """``FixedEdgesNeighborList`` returns full or patch-affected fixed edges."""

    def _edges(self):
        return make_edges(jnp.array([0, 1, 2]), jnp.array([1, 2, 3]), n_particles=4)

    def test_full_call_roundtrip_indices(self):
        stored = self._edges()
        nl = FixedEdgesNeighborList[Literal[2]](indices=stored.indices)
        out = nl(_simple_lh(4), _simple_systems())
        npt.assert_array_equal(out.indices.indices, stored.indices.indices)
        npt.assert_array_equal(out.shifts, jnp.zeros_like(stored.shifts))

    def test_full_call_computes_shifts_from_current_positions(self):
        indices = Index((ParticleId(0), ParticleId(1)), jnp.array([[0, 1]]))
        nl = FixedEdgesNeighborList[Literal[2]](indices=indices)
        lh = make_lh(
            jnp.array([[9.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), jnp.zeros(2, dtype=int)
        )
        systems = _simple_systems()
        out = nl(lh, systems)
        npt.assert_allclose(out.shifts, jnp.array([[[1.0, 0.0, 0.0]]]))
        npt.assert_allclose(
            out.difference_vectors(lh, systems), jnp.array([[[2.0, 0.0, 0.0]]])
        )

    def test_for_indices_filters_without_rh(self):
        stored = self._edges()
        nl = FixedEdgesNeighborList[Literal[2]](
            indices=stored.indices, avg_edges=FixedCapacity(2)
        )
        lh = _simple_lh(4)
        for_indices = Index(lh.keys, jnp.array([2]))
        out = nl(lh, _simple_systems(), for_indices=for_indices)
        npt.assert_array_equal(out.indices.indices, jnp.array([[1, 2], [2, 3]]))

    def test_for_indices_returns_touched_edges(self):
        stored = self._edges()
        nl = FixedEdgesNeighborList[Literal[2]](
            indices=stored.indices, avg_edges=FixedCapacity(2)
        )
        lh = _simple_lh(4)
        for_indices = Index(lh.keys, jnp.array([2]))
        out = nl(lh, _simple_systems(), for_indices=for_indices)
        npt.assert_array_equal(out.indices.indices, jnp.array([[1, 2], [2, 3]]))
        npt.assert_array_equal(out.shifts, jnp.zeros((2, 1, 3), dtype=int))

    def test_patch_call_uses_capacity_and_pads_with_oob_rows(self):
        stored = self._edges()
        nl = FixedEdgesNeighborList[Literal[2]](
            indices=stored.indices, avg_edges=FixedCapacity(3)
        )
        lh = _simple_lh(4)
        for_indices = Index(lh.keys, jnp.array([2]))
        out = nl(lh, _simple_systems(), for_indices=for_indices)
        npt.assert_array_equal(out.indices.indices, jnp.array([[1, 2], [2, 3], [4, 4]]))
        npt.assert_array_equal(out.shifts[-1], jnp.zeros((1, 3), dtype=int))

    def test_for_indices_multiplies_avg_edges_by_affected_count(self):
        stored = self._edges()
        nl = FixedEdgesNeighborList[Literal[2]](
            indices=stored.indices, avg_edges=FixedCapacity(2)
        )
        lh = _simple_lh(4)
        for_indices = Index(lh.keys, jnp.array([0, 3]))
        out = nl(lh, _simple_systems(), for_indices=for_indices)
        npt.assert_array_equal(
            out.indices.indices, jnp.array([[0, 1], [2, 3], [4, 4], [4, 4]])
        )

    def test_rh_call_is_rejected(self):
        stored = self._edges()
        nl = FixedEdgesNeighborList[Literal[2]](indices=stored.indices)
        with pytest.raises(AssertionError, match="only supports self-graph"):
            nl(_simple_lh(4), _simple_systems(), rh=_simple_lh(1))

    def test_patch_call_capacity_assertion(self):
        stored = self._edges()
        nl = FixedEdgesNeighborList[Literal[2]](
            indices=stored.indices, avg_edges=FixedCapacity(1)
        )
        lh = _simple_lh(4)
        for_indices = Index(lh.keys, jnp.array([2]))
        result = as_result_function(nl)(
            lh=lh, systems=_simple_systems(), for_indices=for_indices
        )
        with pytest.raises(CapacityError):
            result.raise_assertion()

    def test_patch_call_supports_higher_degree_fixed_edges(self):
        idx = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 4, 3]])
        shifts = jnp.zeros((4, 2, 3), dtype=int)
        edges = Edges(Index(tuple(ParticleId(i) for i in range(5)), idx), shifts)
        nl = FixedEdgesNeighborList[Literal[3]](
            indices=edges.indices, avg_edges=FixedCapacity(3)
        )
        lh = _simple_lh(5)
        for_indices = Index(lh.keys, jnp.array([0]))
        out = nl(lh, _simple_systems(), for_indices=for_indices)
        npt.assert_array_equal(
            out.indices.indices, jnp.array([[0, 1, 2], [0, 4, 3], [5, 5, 5]])
        )
        npt.assert_array_equal(out.shifts[-1], jnp.zeros((2, 3), dtype=int))
