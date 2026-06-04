# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``kups.core.neighborlist.all_dense``."""

import logging

import jax.numpy as jnp
import numpy as np
import numpy.testing as nptest

from kups.core.capacity import FixedCapacity, LensCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.neighborlist.all_dense import (
    AllDenseNearestNeighborList,
    _all_subselect,
)
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.result import as_result_function

from ._builders import EvalState, cutoff_table, make_lh, make_systems


class TestAllSubselect:
    def test_emits_full_cartesian_product(self):
        lh = make_lh(jnp.zeros((3, 3)), jnp.zeros(3, dtype=int))
        rh = make_lh(jnp.zeros((2, 3)), jnp.zeros(2, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0)),
            jnp.array([1.0]),
        )
        candidates = _all_subselect(lh, rh, systems)
        nptest.assert_array_equal(
            np.asarray(candidates.key_idx.indices), np.array([0, 0, 1, 1, 2, 2])
        )
        nptest.assert_array_equal(
            np.asarray(candidates.query_idx.indices), np.array([0, 1, 0, 1, 0, 1])
        )


class TestMultiSystemWarning:
    def test_warns_on_multiple_inclusion_segments(self, caplog):
        lh = make_lh(jnp.zeros((4, 3)), jnp.array([0, 0, 1, 1]))
        systems, _ = make_systems(
            PeriodicCell(
                TriclinicFrame.from_matrix(jnp.eye(3)[None].repeat(2, 0) * 10.0)
            ),
            jnp.array([1.0, 1.0]),
        )
        nl = AllDenseNearestNeighborList(
            avg_edges=FixedCapacity(32),
            avg_image_candidates=FixedCapacity(32),
            cutoffs=cutoff_table(jnp.array([1.0, 1.0])),
        )
        with caplog.at_level(logging.WARNING):
            result = as_result_function(nl)(keys=lh, systems=systems)
            result.raise_assertion()
        assert "single-system" in caplog.text


class TestFromState:
    def test_from_state_wires_lens_capacities(self):
        lh = make_lh(jnp.zeros((4, 3)), jnp.zeros(4, dtype=int))
        systems, _ = make_systems(
            PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 10.0)),
            jnp.array([2.0]),
        )
        params = UniversalNeighborlistParameters(
            avg_edges=16, avg_candidates=32, avg_image_candidates=32, cells=64
        )
        state = EvalState(particles=lh, systems=systems, neighborlist_params=params)
        nl = AllDenseNearestNeighborList.from_state(
            state, cutoff_table(jnp.array([2.0]))
        )

        assert isinstance(nl.avg_edges, LensCapacity)
        assert isinstance(nl.avg_image_candidates, LensCapacity)
        assert int(nl.avg_edges.size) == 16
