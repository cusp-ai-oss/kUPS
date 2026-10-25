# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.types``: the carrier dataclasses and the
``NeighborList`` call contract."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    AllDenseNearestNeighborList,
    CellListNeighborList,
    CellScanNeighborList,
    DenseNearestNeighborList,
    Edges,
    EmptyNeighborList,
    FixedEdgesNeighborList,
    RefineCutoffNeighborList,
    RefineMaskNeighborList,
    all_connected_neighborlist,
)
from kups.core.neighborlist.types import NeighborList, PipelineContext
from kups.core.patch import IdPatch
from kups.core.typing import ParticleId, SystemId

from ._builders import (
    EvalState,
    after,
    make_adaptive_state,
    make_batch,
    make_edges,
    make_lh,
    make_pipeline_ctx,
    make_systems,
    systems_from_lvecs,
)


class TestCandidateBatch:
    def test_key_and_query_idx_default_to_edge_keys(self):
        keys = (ParticleId(0), ParticleId(1), ParticleId(2))
        batch = make_batch(keys, jnp.array([0, 1]), jnp.array([2, 0]))
        npt.assert_array_equal(np.asarray(batch.key_idx), [0, 1])
        npt.assert_array_equal(np.asarray(batch.query_idx), [2, 0])
        assert batch.key_idx.keys == keys
        assert batch.query_idx.keys == keys

    def test_bipartite_endpoints_resolve_each_table(self):
        lh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        rh = after(lh, make_lh(jnp.eye(3), jnp.zeros(3, dtype=int)))
        ctx = make_pipeline_ctx(lh, rh)
        assert ctx.edge_keys == lh.keys + rh.keys
        batch = make_batch(ctx.edge_keys, jnp.array([1]), jnp.array([4]))
        key, query = ctx.endpoints(batch)
        npt.assert_array_equal(key.positions, [[0.0, 0.0, 0.0]])
        npt.assert_array_equal(query.positions, [[0.0, 0.0, 1.0]])


class TestPipelineContext:
    def test_rejects_rh_and_queried_keys_together(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        rh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None])), jnp.array([1.0])
        )
        with pytest.raises(
            AssertionError, match="cannot combine queries with queried_keys"
        ):
            PipelineContext(
                keys=lh,
                queries=after(lh, rh),
                systems=systems,
                queried_keys=Index(lh.keys, jnp.array([0, 1])),
            )

    def test_allows_queried_keys_without_rh(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None])), jnp.array([1.0])
        )
        queried_keys = Index(lh.keys, jnp.array([0, 2]))
        ctx = PipelineContext(
            keys=lh, queries=None, systems=systems, queried_keys=queried_keys
        )
        assert ctx.queried_keys is queried_keys

    def test_rejects_overlapping_bipartite_ids(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        with pytest.raises(AssertionError, match="disjoint"):
            make_pipeline_ctx(lh, lh)


def _neighbor_lists(
    cutoffs: Table[SystemId, Array],
) -> dict[str, NeighborList[Literal[2]]]:
    pairs = make_edges(jnp.array([0, 1]), jnp.array([1, 0]))
    capacity, cells = FixedCapacity(8), FixedCapacity(64)
    return {
        "adaptive": AdaptiveNeighborList.from_state(
            make_adaptive_state(2, 1), identity_lens(EvalState), cutoffs
        ),
        "all_connected": all_connected_neighborlist,
        "all_dense": AllDenseNearestNeighborList(capacity, capacity, cutoffs),
        "cell_list": CellListNeighborList(capacity, capacity, cells, capacity, cutoffs),
        "cell_scan": CellScanNeighborList(capacity, cells, cutoffs),
        "dense": DenseNearestNeighborList(capacity, capacity, capacity, cutoffs),
        "empty": EmptyNeighborList[Literal[2]](degree=2),
        "fixed_edges": FixedEdgesNeighborList[Literal[2]](pairs.indices),
        "refine_cutoff": RefineCutoffNeighborList(pairs, capacity, cutoffs),
        "refine_mask": RefineMaskNeighborList(pairs),
    }


class TestNeighborList:
    @pytest.mark.parametrize(
        "name",
        [
            "adaptive",
            "all_connected",
            "all_dense",
            "cell_list",
            "cell_scan",
            "dense",
            "empty",
            "fixed_edges",
            "refine_cutoff",
            "refine_mask",
        ],
    )
    def test_library_lists_return_an_identity_patch(self, name):
        """Lists that refresh no cache on the state return their edges with an
        ``IdPatch``."""
        lh = make_lh(
            jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            jnp.zeros(2, dtype=int),
            inclusion_max_count=2,
        )
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 5.0, jnp.array([1.5]))
        neighborlist = _neighbor_lists(cutoffs)[name]
        result = jax.eval_shape(lambda: neighborlist(lh, systems))
        assert isinstance(result.data, Edges)
        assert isinstance(result.patch, IdPatch)
