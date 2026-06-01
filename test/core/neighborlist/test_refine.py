# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``PrecomputedEdgesSelector`` (shared by both refine variants).

The selector has two modes: reuse precomputed shifts (default, used by
``RefineMaskNeighborList``) or recompute minimum-image shifts on the current
positions (``RefineCutoffNeighborList``).
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.neighborlist.refine import PrecomputedEdgesSelector

from ._builders import make_edges, make_lh, make_pipeline_ctx


class TestPrecomputedEdgesSelectorModes:
    def test_default_mode_reuses_precomputed_shifts(self):
        lh = make_lh(
            jnp.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]), jnp.zeros(2, dtype=int)
        )
        # A non-MIC precomputed shift, to confirm it is preserved verbatim.
        custom_shift = jnp.array([[[7.0, 7.0, 7.0]]])
        candidates = make_edges(
            jnp.array([0]), jnp.array([1]), n_particles=2, shifts=custom_shift
        )
        ctx = make_pipeline_ctx(lh)
        selector = PrecomputedEdgesSelector(candidates)  # recompute_mic=False
        batch = selector(ctx)
        npt.assert_allclose(np.asarray(batch.edges.shifts), np.asarray(custom_shift))
        assert bool(batch.is_minimum_image.all())

    def test_recompute_mic_mode_overrides_precomputed_shifts(self):
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
        lh = make_lh(
            jnp.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]), jnp.zeros(2, dtype=int)
        )
        # Garbage precomputed shift, recomputed from positions instead.
        candidates = make_edges(
            jnp.array([0]),
            jnp.array([1]),
            n_particles=2,
            shifts=jnp.array([[99.0, 99.0, 99.0]]),
        )
        ctx = make_pipeline_ctx(lh, cell=cell)
        selector = PrecomputedEdgesSelector(candidates, recompute_mic_shifts=True)
        batch = selector(ctx)
        npt.assert_allclose(
            np.asarray(batch.edges.shifts), np.array([[[-1.0, 0.0, 0.0]]])
        )
