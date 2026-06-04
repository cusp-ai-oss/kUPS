# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the neighbor list compactors."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.neighborlist.compact import MaskOnlyCompactor, ReduceCompactor

from ._builders import make_batch, make_lh, make_pipeline_ctx


class TestReduceCompactor:
    def test_compacts_to_capacity_size(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh)
        keep = jnp.array([True, False, True, False, True])
        batch = make_batch(
            lh.keys, jnp.array([0, 1, 2, 3, 0]), jnp.array([1, 2, 3, 0, 2])
        )
        compactor = ReduceCompactor(avg_edges=FixedCapacity(6))
        edges = compactor(keep, batch, ctx)
        npt.assert_array_equal(
            np.asarray(edges.indices.indices[:, 0]), np.array([0, 2, 0, 4, 4, 4])
        )
        npt.assert_array_equal(
            np.asarray(edges.indices.indices[:, 1]), np.array([1, 3, 2, 4, 4, 4])
        )

    def test_compacts_rows_without_mirroring(self):
        """ReduceCompactor only compacts; graph symmetry is postprocessing."""
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh, queried_keys=jnp.array([1, 3]))
        keep = jnp.array([True])
        shifts = jnp.array([[[0.5, 0.0, 0.0]]])
        batch = make_batch(lh.keys, jnp.array([0]), jnp.array([1]), shifts=shifts)
        compactor = ReduceCompactor(avg_edges=FixedCapacity(1))
        edges = compactor(keep, batch, ctx)
        npt.assert_array_equal(np.asarray(edges.indices.indices), np.array([[0, 1]]))
        npt.assert_allclose(np.asarray(edges.shifts), np.array([[[0.5, 0.0, 0.0]]]))

    def test_capacity_overflow_emits_assertion(self):
        from kups.core.result import as_result_function

        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh)
        keep = jnp.array([True, True, True])
        batch = make_batch(lh.keys, jnp.array([0, 1, 2]), jnp.array([1, 2, 3]))

        compactor = ReduceCompactor(avg_edges=FixedCapacity(1))
        result = as_result_function(lambda: compactor(keep, batch, ctx))()
        with pytest.raises(CapacityError):
            result.raise_assertion()


class TestMaskOnlyCompactor:
    def test_stamps_oob_on_dropped_entries(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        ctx = make_pipeline_ctx(lh, queried_keys=jnp.array([1, 3]))
        keep = jnp.array([True, False, True])
        batch = make_batch(lh.keys, jnp.array([0, 1, 2]), jnp.array([1, 3, 3]))
        edges = MaskOnlyCompactor()(keep, batch, ctx)
        oob = lh.size
        npt.assert_array_equal(
            np.asarray(edges.indices.indices),
            np.array([[0, 1], [oob, oob], [2, 3]]),
        )

    def test_zeros_shifts_on_dropped_entries(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        ctx = make_pipeline_ctx(lh)
        keep = jnp.array([True, False])
        shifts = jnp.array([[[0.5, 0.0, 0.0]], [[0.7, 0.0, 0.0]]])
        batch = make_batch(lh.keys, jnp.array([0, 1]), jnp.array([1, 2]), shifts=shifts)
        edges = MaskOnlyCompactor()(keep, batch, ctx)
        npt.assert_allclose(np.asarray(edges.shifts[1]), np.zeros((1, 3)))
