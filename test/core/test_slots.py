# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.data import Index
from kups.core.data.slots import SlotLayout
from kups.core.typing import SystemId

LAYOUT = SlotLayout(n_slots=2, capacity=3)


class TestGeometry:
    def test_slot_of_row_is_contiguous(self):
        npt.assert_array_equal(LAYOUT.slot_of_row.indices, [0, 0, 0, 1, 1, 1])
        assert LAYOUT.slot_of_row.max_count == LAYOUT.capacity
        npt.assert_array_equal(LAYOUT.slot_of_row.counts.data, [3, 3])

    def test_keys_and_size(self):
        assert LAYOUT.keys == (SystemId(0), SystemId(1))
        assert LAYOUT.n_rows == 6

    def test_invalid(self):
        with pytest.raises(ValueError):
            SlotLayout(n_slots=0, capacity=3)


class TestReshape:
    def test_round_trip_with_index_leaf(self):
        tree = {
            "x": jnp.arange(12.0).reshape(6, 2),
            "sys": Index(LAYOUT.keys, jnp.array([0, 0, 2, 1, 2, 2]), _cls=SystemId),
        }
        slotted = LAYOUT.to_slots(tree)
        assert slotted["x"].shape == (2, 3, 2)
        assert slotted["sys"].indices.shape == (2, 3)
        assert slotted["sys"].keys == LAYOUT.keys
        back = LAYOUT.from_slots(slotted)
        npt.assert_array_equal(back["x"], tree["x"])
        npt.assert_array_equal(back["sys"].indices, tree["sys"].indices)

    def test_system_index(self):
        valid = jnp.array([[True, False, False], [True, True, False]])
        npt.assert_array_equal(LAYOUT.system_index(valid).indices, [0, 2, 2, 1, 1, 2])
        npt.assert_array_equal(
            LAYOUT.system_index(valid.reshape(-1)).indices, [0, 2, 2, 1, 1, 2]
        )
