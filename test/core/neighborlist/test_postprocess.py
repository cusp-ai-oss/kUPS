# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``MirrorPairEdges`` postprocessor."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.core.capacity import FixedCapacity
from kups.core.neighborlist.compact import ReduceCompactor
from kups.core.neighborlist.postprocess import MirrorPairEdges

from ._builders import make_batch, make_edges, make_lh, make_pipeline_ctx


class TestMirrorPairEdges:
    def test_mirrors_each_edge_with_reverse_when_queried_keys_set(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh, queried_keys=jnp.array([1, 3]))
        keep = jnp.array([True])
        shifts = jnp.array([[[0.5, 0.0, 0.0]]])
        batch = make_batch(lh.keys, jnp.array([0]), jnp.array([1]), shifts=shifts)
        compacted = ReduceCompactor(avg_edges=FixedCapacity(1))(keep, batch, ctx)
        edges = MirrorPairEdges()(compacted, ctx)
        npt.assert_array_equal(
            np.asarray(edges.indices.indices), np.array([[0, 1], [1, 0]])
        )
        npt.assert_allclose(
            np.asarray(edges.shifts),
            np.array([[[0.5, 0.0, 0.0]], [[-0.5, -0.0, -0.0]]]),
        )

    def test_noop_without_queried_keys_by_default(self):
        lh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        ctx = make_pipeline_ctx(lh)
        edges = make_edges(
            jnp.array([0]),
            jnp.array([1]),
            n_particles=2,
            shifts=jnp.array([[0.25, 0.0, 0.0]]),
        )
        out = MirrorPairEdges()(edges, ctx)
        npt.assert_array_equal(np.asarray(out.indices.indices), np.array([[0, 1]]))
        npt.assert_allclose(np.asarray(out.shifts), np.array([[[0.25, 0.0, 0.0]]]))

    def test_only_when_queried_keys_false_mirrors_unconditionally(self):
        lh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        ctx = make_pipeline_ctx(lh)
        edges = make_edges(
            jnp.array([0]),
            jnp.array([1]),
            n_particles=2,
            shifts=jnp.array([[0.25, 0.0, 0.0]]),
        )
        out = MirrorPairEdges(only_when_queried_keys=False)(edges, ctx)
        npt.assert_array_equal(
            np.asarray(out.indices.indices), np.array([[0, 1], [1, 0]])
        )
