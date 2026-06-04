# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``Pipeline`` runner and ``_prepare``."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.compact import MaskOnlyCompactor, ReduceCompactor
from kups.core.neighborlist.masks import DistanceCutoffMask
from kups.core.neighborlist.pipeline import Pipeline, _prepare
from kups.core.neighborlist.postprocess import MirrorPairEdges
from kups.core.neighborlist.refine import PrecomputedEdgesSelector

from ._builders import make_edges, make_lh, make_systems, systems_from_lvecs


class TestPrepare:
    def test_converts_positions_to_fractional(self):
        # In a 10 Å cell, real position 5.0 maps to fractional 0.5.
        lh = make_lh(jnp.array([[5.0, 0.0, 0.0]]), jnp.zeros(1, dtype=int))
        systems, _ = systems_from_lvecs(jnp.eye(3)[None] * 10.0, jnp.array([1.0]))
        ctx = _prepare(lh, None, systems, None)
        npt.assert_allclose(
            np.asarray(ctx.keys.data.positions), np.array([[0.5, 0.0, 0.0]]), atol=1e-6
        )

    def test_resolves_queried_keys_to_raw_array(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None])), jnp.array([1.0])
        )
        ctx = _prepare(lh, None, systems, Index(lh.keys, jnp.array([1, 3])))
        npt.assert_array_equal(np.asarray(ctx.queried_keys), np.array([1, 3]))

    def test_rejects_rh_and_queried_keys_together(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        rh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None])), jnp.array([1.0])
        )
        with pytest.raises(
            AssertionError, match="cannot combine queries with queried_keys"
        ):
            _prepare(lh, rh, systems, Index(lh.keys, jnp.array([0, 1])))


class TestPipelineComposition:
    def test_distance_only_pipeline(self):
        # Candidates (0,1) and (0,2); only the close pair survives cutoff 1.5.
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        lh = make_lh(positions, jnp.zeros(3, dtype=int))
        candidates = make_edges(jnp.array([0, 0]), jnp.array([1, 2]), n_particles=3)
        systems, cutoffs = systems_from_lvecs(jnp.eye(3)[None] * 10.0, jnp.array([1.5]))
        pipeline = Pipeline[Literal[2]](
            selector=PrecomputedEdgesSelector(candidates, recompute_mic_shifts=True),
            masks=(DistanceCutoffMask(cutoffs=cutoffs),),
            compactor=MaskOnlyCompactor(),
        )
        edges = pipeline(lh, systems)
        oob = lh.size
        npt.assert_array_equal(
            np.asarray(edges.indices.indices), np.array([[0, 1], [oob, oob]])
        )

    def test_runs_postprocessors_in_order(self):
        lh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None])), jnp.array([1.0])
        )
        candidates = make_edges(
            jnp.array([0]),
            jnp.array([1]),
            n_particles=2,
            shifts=jnp.array([[0.25, 0.0, 0.0]]),
        )
        pipeline = Pipeline[Literal[2]](
            selector=PrecomputedEdgesSelector(candidates),
            masks=(),
            compactor=ReduceCompactor(avg_edges=FixedCapacity(1)),
            postprocessors=(
                MirrorPairEdges(only_when_queried_keys=False),
                MirrorPairEdges(only_when_queried_keys=False),
            ),
        )
        edges = pipeline(lh, systems)
        npt.assert_array_equal(
            np.asarray(edges.indices.indices),
            np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
        )
        npt.assert_allclose(
            np.asarray(edges.shifts),
            np.array(
                [
                    [[0.25, 0.0, 0.0]],
                    [[-0.25, -0.0, -0.0]],
                    [[-0.25, -0.0, -0.0]],
                    [[0.25, 0.0, 0.0]],
                ]
            ),
        )
