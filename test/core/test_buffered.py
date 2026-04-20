# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.data.buffered import Buffered, add_buffers
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


def _make_status(occ: list[bool]) -> Index[int]:
    """Create an Index where True=valid(0), False=OOB(1)."""
    return Index((0,), jnp.array([0 if o else 1 for o in occ]))


@dataclass
class TD:
    """Test data with a viewed Index leaf."""

    values: jax.Array
    status: Index[int]


@dataclass
class D:
    """Test data with an additional Index leaf for sanitization tests."""

    idx: Index[str]
    status: Index[int]


def _view(d):
    return d.status


def _td(values: list[float], occ: list[bool]) -> TD:
    return TD(values=jnp.array(values), status=_make_status(occ))


def _td_from_array(arr: jax.Array, occ: list[bool]) -> TD:
    return TD(values=arr, status=_make_status(occ))


@dataclass
class TDSys:
    values: jax.Array
    status: Index[int]
    system: Index[SystemId]


@dataclass
class Pair:
    x: jax.Array
    y: jax.Array
    status: Index[int]


def _pair_view(d):
    return d.status


def _pair(x: list[float], y: list[float], occ: list[bool]) -> Pair:
    return Pair(x=jnp.array(x), y=jnp.array(y), status=_make_status(occ))


class TestConstruction:
    def test_construction(self):
        """Merged: basic, full_occupation, partial_occupation, wrong_shape."""
        # basic
        data = _td([0.0, 1.0, 2.0], [True, True, True])
        buf = Buffered(("a", "b", "c"), data, _view)
        assert buf.keys == ("a", "b", "c")
        npt.assert_array_equal(buf.occupation, [True, True, True])

        # full occupation
        data = _td([1.0, 1.0], [True, True])
        buf = Buffered((0, 1), data, _view)
        npt.assert_array_equal(buf.num_occupied, 2)

        # partial occupation
        data = _td([0.0, 1.0, 2.0], [True, False, True])
        buf = Buffered(("a", "b", "c"), data, _view)
        npt.assert_array_equal(buf.num_occupied, 2)

        # wrong shape raises
        bad = TD(values=jnp.arange(3.0), status=_make_status([True, True]))
        with pytest.raises(ValueError, match="Leaf has size"):
            Buffered(("a", "b"), bad, _view)


class TestSanitization:
    def test_sanitization(self):
        """Merged: array_leaves_zeroed, index_leaves_sentinel, pytree_mixed, multidim."""
        # array leaves zeroed
        data = _td([10.0, 20.0, 30.0], [True, False, True])
        buf = Buffered(("a", "b", "c"), data, _view)
        npt.assert_array_equal(buf.data.values, [10.0, 0.0, 30.0])

        # index leaves get OOB sentinel
        idx = Index.new(["x", "y", "z"])
        status = _make_status([True, False, True])
        buf = Buffered((0, 1, 2), D(idx=idx, status=status), lambda d: d.status)
        assert buf.data.idx.indices[1] == 3

        # pytree with mixed types
        data = _pair([1.0, 2.0], [3.0, 4.0], [False, True])
        buf = Buffered((0, 1), data, _pair_view)
        npt.assert_array_equal(buf.data.x, [0.0, 2.0])
        npt.assert_array_equal(buf.data.y, [0.0, 4.0])

        # multidim array zeroed
        data = TD(values=jnp.ones((3, 2)), status=_make_status([True, False, True]))
        buf = Buffered(("a", "b", "c"), data, _view)
        npt.assert_array_equal(buf.data.values[1], [0.0, 0.0])
        npt.assert_array_equal(buf.data.values[0], [1.0, 1.0])


class TestSelectFree:
    def test_select_free(self):
        """Merged: num_occupied count, returns_index, fill_value, usable_with_at."""
        # num_occupied count
        data = _td([0.0] * 5, [True, False, True, False, True])
        buf = Buffered(tuple(range(5)), data, _view)
        npt.assert_array_equal(buf.num_occupied, 3)

        # returns index
        data = _td([0.0] * 4, [True, False, True, False])
        buf = Buffered(tuple(range(4)), data, _view)
        free = buf.select_free(2)
        assert isinstance(free, Index)
        assert free.keys == (0, 1, 2, 3)
        npt.assert_array_equal(free.indices, [1, 3])

        # fill value when fewer free
        data = _td([0.0] * 3, [True, False, True])
        buf = Buffered(tuple(range(3)), data, _view)
        free = buf.select_free(3)
        npt.assert_array_equal(free.indices, [1, 3, 3])

        # usable with at
        data = _td([1.0, 0.0, 3.0], [True, False, True])
        buf = Buffered(("a", "b", "c"), data, _view)
        free = buf.select_free(1)
        updated = buf.at(free).set(_td([99.0], [True]))
        npt.assert_array_equal(updated.values, [1.0, 99.0, 3.0])


class TestArange:
    def test_arange(self):
        """Merged: basic, all_occupied, none_occupied, pytree_data."""
        # basic
        data = _td([1.0, 2.0, 3.0, 4.0], [True] * 4)
        buf = Buffered.arange(data, num_occupied=2, view=_view)
        assert buf.keys == (0, 1, 2, 3)
        npt.assert_array_equal(buf.occupation, [True, True, False, False])
        npt.assert_array_equal(buf.data.values, [1.0, 2.0, 0.0, 0.0])

        # all occupied
        data = _td([5.0, 6.0], [True, True])
        buf = Buffered.arange(data, num_occupied=2, view=_view)
        npt.assert_array_equal(buf.occupation, [True, True])
        npt.assert_array_equal(buf.data.values, [5.0, 6.0])

        # none occupied
        data = _td([1.0, 2.0, 3.0], [True, True, True])
        buf = Buffered.arange(data, num_occupied=0, view=_view)
        npt.assert_array_equal(buf.occupation, [False, False, False])
        npt.assert_array_equal(buf.data.values, [0.0, 0.0, 0.0])

        # pytree data
        data = _pair([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [True, True, True])
        buf = Buffered.arange(data, num_occupied=1, view=_pair_view)
        npt.assert_array_equal(buf.data.x, [1.0, 0.0, 0.0])
        npt.assert_array_equal(buf.data.y, [4.0, 0.0, 0.0])


class TestPad:
    def test_pad(self):
        """Merged: full, basic, zero_free, pytree_data."""
        # full
        data = _td([1.0, 2.0, 3.0], [True, True, True])
        indexed = Table(("a", "b", "c"), data)
        buf = Buffered.full(indexed, view=_view)
        npt.assert_array_equal(buf.occupation, [True, True, True])
        npt.assert_array_equal(buf.data.values, [1.0, 2.0, 3.0])
        assert buf.keys == ("a", "b", "c")

        # basic
        data = _td([1.0, 2.0], [True, True])
        indexed = Table((0, 1), data)
        buf = Buffered.pad(indexed, 2, view=_view)
        assert buf.keys == (0, 1, 2, 3)
        npt.assert_array_equal(buf.occupation, [True, True, False, False])
        npt.assert_array_equal(buf.data.values, [1.0, 2.0, 0.0, 0.0])

        # zero free
        data = _td([5.0, 6.0], [True, True])
        indexed = Table((0, 1), data)
        buf = Buffered.pad(indexed, 0, view=_view)
        assert len(buf.keys) == 2
        assert buf.occupation.all()

        # pytree data
        data = _pair([1.0, 2.0], [3.0, 4.0], [True, True])
        indexed = Table((0, 1), data)
        buf = Buffered.pad(indexed, 1, view=_pair_view)
        assert buf.keys == (0, 1, 2)
        npt.assert_array_equal(buf.data.x, [1.0, 2.0, 0.0])
        npt.assert_array_equal(buf.data.y, [3.0, 4.0, 0.0])
        npt.assert_array_equal(buf.occupation, [True, True, False])


class TestSubset:
    def test_subset(self):
        """Merged: basic + preserves_unoccupied."""
        # basic
        data = _td([10.0, 20.0, 30.0, 40.0], [True, True, False, True])
        buf = Buffered((0, 1, 2, 3), data, _view)
        idx = Index((0, 1, 2, 3), jnp.array([0, 3]))
        sub = buf.subset(idx)
        assert sub.keys == (0, 1)
        npt.assert_array_equal(sub.data.values, [10.0, 40.0])
        npt.assert_array_equal(sub.occupation, [True, True])

        # preserves unoccupied
        data = _td([1.0, 2.0, 3.0], [True, False, True])
        buf = Buffered((0, 1, 2), data, _view)
        idx = Index((0, 1, 2), jnp.array([1, 2]))
        sub = buf.subset(idx)
        assert sub.keys == (0, 1)
        npt.assert_array_equal(sub.occupation, [False, True])


class TestUpdateIf:
    def _make_buf(self, values, occ):
        n = len(values)
        sys_idx = Index.new([SystemId(i) for i in range(n)])
        data = TDSys(
            values=jnp.array(values),
            status=_make_status(occ),
            system=sys_idx,
        )
        return Buffered(tuple(range(n)), data, lambda d: d.status)

    def _make_new(self, values, occ, sys_ids):
        sys_idx = Index.new([SystemId(s) for s in sys_ids])
        data = TDSys(
            values=jnp.array(values),
            status=_make_status(occ),
            system=sys_idx,
        )
        return data

    def test_update_if(self):
        """Merged: accept_true, accept_false, mixed_accept."""
        # accept true writes
        buf = self._make_buf([1.0, 2.0, 3.0], [True, True, True])
        idx = Index(buf.keys, jnp.array([1]))
        new_data = self._make_new([99.0], [True], [1])
        accept = Table.arange(jnp.array([True, True, True]), label=SystemId)
        result = buf.update_if(accept, idx, new_data)
        npt.assert_array_equal(result.data.values, [1.0, 99.0, 3.0])

        # accept false keeps original
        buf = self._make_buf([1.0, 2.0, 3.0], [True, True, True])
        idx = Index(buf.keys, jnp.array([1]))
        new_data = self._make_new([99.0], [True], [1])
        accept = Table.arange(jnp.array([False, False, False]), label=SystemId)
        result = buf.update_if(accept, idx, new_data)
        npt.assert_array_equal(result.data.values, [1.0, 2.0, 3.0])

        # mixed accept
        buf = self._make_buf([1.0, 2.0, 3.0, 4.0], [True, True, True, True])
        idx = Index(buf.keys, jnp.array([0, 2]))
        new_data = self._make_new([90.0, 91.0], [True, False], [0, 2])
        accept = Table.arange(jnp.array([True, False, False, False]), label=SystemId)
        result = buf.update_if(accept, idx, new_data)
        npt.assert_array_equal(result.data.values[0], 90.0)
        npt.assert_array_equal(result.data.values[2], 3.0)
        assert result.occupation[0]
        assert result.occupation[2]


class TestInheritedBehavior:
    def test_getitem_and_at_set(self):
        """Merged: getitem + at_set."""
        # getitem
        data = _td([10.0, 20.0, 30.0], [True, True, True])
        buf = Buffered(("a", "b", "c"), data, _view)
        idx = Index.new(["c", "a"])
        result = buf[idx]
        npt.assert_array_equal(result.values, [30.0, 10.0])

        # at set
        data = _td([1.0, 2.0, 3.0], [True, True, True])
        buf = Buffered(("a", "b", "c"), data, _view)
        idx = Index.new(["b"])
        updated = buf.at(idx).set(_td([99.0], [True]))
        npt.assert_array_equal(updated.values, [1.0, 99.0, 3.0])


class TestUpdate:
    def test_update(self):
        """Merged: write_free_slots, preserves_existing, clear_slots, pytree_data."""
        # write into free slots
        data = _td([0.0] * 4, [False, False, False, False])
        buf = Buffered((0, 1, 2, 3), data, _view)
        src = Buffered((0, 1), _td([10.0, 20.0], [True, True]), _view)
        idx = Index((0, 1, 2, 3), jnp.array([1, 3]))
        result = buf.update(idx, src.data)
        npt.assert_array_equal(result.data.values, [0.0, 10.0, 0.0, 20.0])
        npt.assert_array_equal(result.occupation, [False, True, False, True])

        # preserves existing occupied
        data = _td([5.0, 6.0, 7.0], [True, True, False])
        buf = Buffered((0, 1, 2), data, _view)
        src = Buffered((0,), _td([99.0], [True]), _view)
        idx = Index((0, 1, 2), jnp.array([2]))
        result = buf.update(idx, src.data)
        npt.assert_array_equal(result.data.values, [5.0, 6.0, 99.0])
        npt.assert_array_equal(result.occupation, [True, True, True])

        # clear slots
        data = _td([1.0, 2.0, 3.0], [True, True, True])
        buf = Buffered((0, 1, 2), data, _view)
        src = Buffered((0,), _td([0.0], [False]), _view)
        idx = Index((0, 1, 2), jnp.array([1]))
        result = buf.update(idx, src.data)
        assert not result.occupation[1]
        npt.assert_array_equal(result.data.values[1], 0.0)
        npt.assert_array_equal(result.data.values[0], 1.0)
        npt.assert_array_equal(result.data.values[2], 3.0)

        # pytree data
        data = _pair([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [True, False, True])
        buf = Buffered((0, 1, 2), data, _pair_view)
        src = Buffered((0,), _pair([99.0], [88.0], [True]), _pair_view)
        idx = Index((0, 1, 2), jnp.array([1]))
        result = buf.update(idx, src.data)
        npt.assert_array_equal(result.data.x, [1.0, 99.0, 3.0])
        npt.assert_array_equal(result.data.y, [4.0, 88.0, 6.0])
        npt.assert_array_equal(result.occupation, [True, True, True])


class TestJitCompatibility:
    def test_jit_compatibility(self):
        """Merged: passes_through_jit, construction_inside_jit, sanitization_inside_jit."""
        # passes through jit
        data = _td([10.0, 20.0, 30.0], [True, False, True])
        buf = Buffered(("a", "b", "c"), data, _view)
        idx = Index.new(["c", "a"])

        @jax.jit
        def f(buf, idx):
            return buf[idx]

        result = f(buf, idx)
        npt.assert_array_equal(result.values, [30.0, 10.0])

        # construction inside jit
        @jax.jit
        def make_and_read(data):
            buf = Buffered(("a", "b", "c"), data, _view)
            return buf.data.values

        data = _td([10.0, 20.0, 30.0], [True, False, True])
        result = make_and_read(data)
        npt.assert_array_equal(result, [10.0, 0.0, 30.0])

        # sanitization inside jit
        @jax.jit
        def make_and_sum(data):
            buf = Buffered((0, 1, 2, 3), data, _view)
            return buf.data.values.sum()

        data = _td([1.0, 2.0, 3.0, 4.0], [True, False, False, True])
        npt.assert_array_equal(make_and_sum(data), 5.0)


@dataclass
class ParticleData:
    positions: jax.Array
    system: Index[SystemId]


@dataclass
class SystemData:
    energy: jax.Array
    self_id: Index[SystemId]


def _particle_view(d):
    return d.system


def _system_view(d):
    return d.self_id


class TestAddBuffers:
    def test_add_buffers(self):
        """Merged: disjoint_keys, leaf_index_remapped, data_preserved."""
        # disjoint keys
        p_data = ParticleData(
            positions=jnp.array([1.0, 2.0]),
            system=Index((SystemId(0),), jnp.array([0, 0])),
        )
        particles = Table((ParticleId(0), ParticleId(1)), p_data)
        s_data = SystemData(
            energy=jnp.array([10.0]),
            self_id=Index((SystemId(0),), jnp.array([0])),
        )
        systems = Table((SystemId(0),), s_data)

        bp, bs = add_buffers(
            (particles, 1, _particle_view),
            (systems, 2, _system_view),
        )
        assert len(bp.keys) == 3
        npt.assert_array_equal(bp.occupation, [True, True, False])
        assert len(bs.keys) == 3
        npt.assert_array_equal(bs.occupation, [True, False, False])

        # leaf index remapped
        p_data = ParticleData(
            positions=jnp.array([[0.0]]),
            system=Index((SystemId(0),), jnp.array([0])),
        )
        particles = Table((ParticleId(0),), p_data)
        s_data = SystemData(
            energy=jnp.array([5.0]),
            self_id=Index((SystemId(0),), jnp.array([0])),
        )
        systems = Table((SystemId(0),), s_data)

        bp, bs = add_buffers(
            (particles, 1, _particle_view),
            (systems, 1, _system_view),
        )
        assert SystemId(0) in bp.data.system.keys

        # data preserved
        p_data = ParticleData(
            positions=jnp.array([10.0, 20.0, 30.0]),
            system=Index((SystemId(0),), jnp.array([0, 0, 0])),
        )
        particles = Table((ParticleId(0), ParticleId(1), ParticleId(2)), p_data)
        s_data = SystemData(
            energy=jnp.array([99.0]),
            self_id=Index((SystemId(0),), jnp.array([0])),
        )
        systems = Table((SystemId(0),), s_data)

        bp, _ = add_buffers(
            (particles, 1, _particle_view),
            (systems, 1, _system_view),
        )
        npt.assert_array_equal(bp.data.positions[:3], [10.0, 20.0, 30.0])
        npt.assert_array_equal(bp.data.positions[3], 0.0)

    def test_buffer_larger_than_existing(self):
        """Buffer entries exceed existing entries (regression)."""
        p_data = ParticleData(
            positions=jnp.array([1.0, 2.0]),
            system=Index((SystemId(0),), jnp.array([0, 0])),
        )
        particles = Table((ParticleId(0), ParticleId(1)), p_data)
        s_data = SystemData(
            energy=jnp.array([10.0]),
            self_id=Index((SystemId(0),), jnp.array([0])),
        )
        systems = Table((SystemId(0),), s_data)

        bp, bs = add_buffers(
            (particles, 5, _particle_view),
            (systems, 3, _system_view),
        )
        assert len(bp.keys) == 7
        npt.assert_array_equal(bp.occupation, [True, True, *[False] * 5])
        npt.assert_array_equal(bp.data.positions[:2], [1.0, 2.0])
        npt.assert_array_equal(bp.data.positions[2:], [0.0] * 5)
        assert len(bs.keys) == 4
        npt.assert_array_equal(bs.occupation, [True, *[False] * 3])
