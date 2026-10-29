# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.types`` carrier dataclasses."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.types import PipelineContext
from kups.core.typing import ParticleId
from test.units import A

from ._builders import after, make_batch, make_lh, make_pipeline_ctx, make_systems


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
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * A)),
            jnp.array([1.0]),
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
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * A)),
            jnp.array([1.0]),
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
