# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.types`` carrier dataclasses."""

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.types import CandidateBatch, PipelineContext
from kups.core.typing import ParticleId

from ._builders import make_batch, make_lh, make_systems


class TestCandidateBatch:
    def test_key_and_query_idx_default_to_edge_keys(self):
        keys = (ParticleId(0), ParticleId(1), ParticleId(2))
        batch = make_batch(keys, jnp.array([0, 1]), jnp.array([2, 0]))
        npt.assert_array_equal(batch.key_idx.indices, jnp.array([0, 1]))
        npt.assert_array_equal(batch.query_idx.indices, jnp.array([2, 0]))
        assert batch.key_idx.keys == keys
        assert batch.query_idx.keys == keys

    def test_query_keys_override_addresses_query_column(self):
        lh_keys = (ParticleId(0), ParticleId(1))
        rh_keys = (ParticleId(5), ParticleId(6), ParticleId(7))
        edges = Edges(Index(lh_keys, jnp.array([[0, 2]])), jnp.zeros((1, 1, 3)))
        batch = CandidateBatch(
            edges=edges, is_minimum_image=jnp.ones((1,), dtype=bool), query_keys=rh_keys
        )
        # key side uses edge keys; query side resolves against the override vocabulary.
        assert batch.key_idx.keys == lh_keys
        assert batch.query_idx.keys == rh_keys
        npt.assert_array_equal(batch.query_idx.indices, jnp.array([2]))


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
                keys=lh, queries=rh, systems=systems, queried_keys=jnp.array([0, 1])
            )

    def test_allows_queried_keys_without_rh(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None])), jnp.array([1.0])
        )
        ctx = PipelineContext(
            keys=lh, queries=None, systems=systems, queried_keys=jnp.array([0, 2])
        )
        npt.assert_array_equal(ctx.queried_keys, jnp.array([0, 2]))
