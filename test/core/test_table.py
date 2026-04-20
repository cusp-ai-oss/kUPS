# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import GroupId, Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass


@dataclass
class Pair:
    x: jax.Array
    y: jax.Array


@dataclass
class ParticleData:
    positions: jax.Array
    system: Index[SystemId]


@dataclass
class ParticleDataSelf:
    positions: jax.Array
    neighbor: Index[ParticleId]


@dataclass
class SystemData:
    energy: jax.Array


@dataclass
class GroupData:
    weight: jax.Array
    system: Index[SystemId]


class TestConstruction:
    def test_construction_and_arange(self):
        # Basic construction
        data = jnp.arange(3.0)
        indexed = Table(("a", "b", "c"), data)
        assert indexed.keys == ("a", "b", "c")
        npt.assert_array_equal(indexed.data, data)

        # Pytree data
        data_p = Pair(jnp.ones(4), jnp.zeros(4))
        indexed_p = Table(("w", "x", "y", "z"), data_p)
        assert len(indexed_p.keys) == 4

        # Mismatched leading dim raises
        with pytest.raises((AssertionError, ValueError)):
            Table(("a", "b", "c"), Pair(jnp.ones(3), jnp.ones(4)))

        # Wrong index length raises
        with pytest.raises((AssertionError, ValueError)):
            Table(("a", "b"), jnp.arange(3.0))

        # Broadcast scalar leaf
        data_bc = Pair(jnp.ones(3), jnp.ones(1))
        idx_bc = Table(("a", "b", "c"), data_bc)
        assert idx_bc.data.y.shape == (3,)

        # Arange with integer keys
        data_ar = jnp.arange(5.0)
        indexed_ar = Table.arange(data_ar)
        assert indexed_ar.keys == (0, 1, 2, 3, 4)
        npt.assert_array_equal(indexed_ar.data, data_ar)

        # Arange with pytree data
        data_ar2 = Pair(jnp.ones(3), jnp.zeros(3))
        assert Table.arange(data_ar2).keys == (0, 1, 2)


class TestUtilities:
    def test_utilities(self):
        indexed = Table(("a", "b", "c"), jnp.arange(3.0))
        assert len(indexed) == 3
        assert indexed.size == 3
        assert "a" in indexed
        assert "d" not in indexed

        # map_data
        data = Pair(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        mapped = Table(("x", "y"), data).map_data(lambda d: d.x * 2)
        assert mapped.keys == ("x", "y")
        npt.assert_array_equal(mapped.data, [2.0, 4.0])

        # set_data
        new = Table(("a", "b"), jnp.array([1.0, 2.0])).set_data(jnp.array([10.0, 20.0]))
        assert new.keys == ("a", "b")
        npt.assert_array_equal(new.data, [10.0, 20.0])


class TestAt:
    def test_get_and_set(self):
        data = jnp.array([1.0, 2.0, 3.0])
        indexed = Table(("a", "b", "c"), data)
        idx = Index.new(["b"])
        npt.assert_array_equal(indexed.at(idx).get(), [2.0])
        updated = indexed.at(idx).set(jnp.array([99.0]))
        npt.assert_array_equal(updated, [1.0, 99.0, 3.0])


class TestTableOperators:
    def test_table_table_ops(self):
        a = Table.arange(jnp.array([10.0, 20.0, 30.0]), label=SystemId)
        b = Table.arange(jnp.array([1.0, 2.0, 3.0]), label=SystemId)

        npt.assert_array_equal((a + b).data, [11.0, 22.0, 33.0])
        npt.assert_array_equal((a - b).data, [9.0, 18.0, 27.0])
        npt.assert_array_equal((a * b).data, [10.0, 40.0, 90.0])
        npt.assert_array_equal((a / b).data, [10.0, 10.0, 10.0])
        npt.assert_array_equal((a // b).data, [10.0, 10.0, 10.0])
        assert (a + b).keys == a.keys
        assert (a + b).cls is SystemId

    def test_table_scalar_and_errors(self):
        a = Table.arange(jnp.array([10.0, 20.0, 30.0]), label=SystemId)
        npt.assert_array_equal((a + 5.0).data, [15.0, 25.0, 35.0])
        npt.assert_array_equal((a * 2.0).data, [20.0, 40.0, 60.0])
        npt.assert_array_equal((a / 10.0).data, [1.0, 2.0, 3.0])

        c = Table.arange(jnp.array([1.0, 2.0]), label=SystemId)
        with pytest.raises(AssertionError):
            a + c


class TestGetitem:
    def test_getitem_basic(self):
        data = jnp.array([10.0, 20.0, 30.0])
        indexed = Table(("a", "b", "c"), data)
        npt.assert_array_equal(indexed[Index.new(["c", "a"])], [30.0, 10.0])
        npt.assert_array_equal(indexed[Index.new(["a", "b", "c"])], data)
        # Repeated keys
        indexed2 = Table(("x", "y"), jnp.array([10.0, 20.0]))
        npt.assert_array_equal(indexed2[Index.new(["y", "x", "y"])], [20.0, 10.0, 20.0])

    def test_getitem_pytree_and_integer(self):
        # Pytree data
        data = Pair(jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
        result = Table(("a", "b", "c"), data)[Index.new(["b"])]
        npt.assert_array_equal(result.x, [2.0])
        npt.assert_array_equal(result.y, [5.0])
        # Integer keys
        indexed = Table.arange(jnp.array([100.0, 200.0, 300.0]))
        npt.assert_array_equal(indexed[Index.new([2, 0])], [300.0, 100.0])


class TestSlice:
    def test_slice(self):
        indexed = Table(("a", "b", "c", "d"), jnp.array([1.0, 2.0, 3.0, 4.0]))
        result = indexed.slice(1, 3)
        assert result.keys == ("b", "c")
        npt.assert_array_equal(result.data, [2.0, 3.0])
        # With step
        result2 = indexed.slice(0, 4, 2)
        assert result2.keys == ("a", "c")
        npt.assert_array_equal(result2.data, [1.0, 3.0])
        # Defaults
        indexed2 = Table(("a", "b", "c"), jnp.array([1.0, 2.0, 3.0]))
        result3 = indexed2.slice()
        assert result3.keys == ("a", "b", "c")
        npt.assert_array_equal(result3.data, [1.0, 2.0, 3.0])


class TestUpdate:
    def test_update(self):
        indexed = Table(("a", "b", "c"), jnp.array([1.0, 2.0, 3.0]))
        result = indexed.update(Index.new(["b"]), jnp.array([99.0]))
        npt.assert_array_equal(result.data, [1.0, 99.0, 3.0])
        result2 = indexed.update(Index.new(["a", "c"]), jnp.array([10.0, 30.0]))
        npt.assert_array_equal(result2.data, [10.0, 2.0, 30.0])


class TestTableTransform:
    def test_single_and_two_args(self):
        a = Table.arange(jnp.array([1.0, 2.0, 3.0]), label=SystemId)
        fn1 = Table.transform(lambda x: x * 2)
        result1 = fn1(a)
        assert result1.keys == a.keys
        npt.assert_array_equal(result1.data, [2.0, 4.0, 6.0])

        b = Table.arange(jnp.array([3.0, 4.0]), label=SystemId)
        c = Table.arange(jnp.array([1.0, 2.0]), label=SystemId)
        result2 = Table.transform(lambda x, y: x + y)(b, c)
        npt.assert_array_equal(result2.data, [4.0, 6.0])

    def test_preserves_index_and_errors(self):
        idx = (SystemId(5), SystemId(10))
        a = Table(idx, jnp.array([1.0, 2.0]), _cls=SystemId)
        assert Table.transform(lambda x: x + 1)(a).keys == idx

        a2 = Table.arange(jnp.array([1.0, 2.0]), label=SystemId)
        b2 = Table((SystemId(5), SystemId(6)), jnp.array([3.0, 4.0]), _cls=SystemId)
        with pytest.raises(AssertionError):
            Table.transform(lambda x, y: x + y)(a2, b2)


class TestBroadcast:
    def test_broadcast_happy(self):
        a = Table.arange(jnp.array([1.0, 2.0]), label=SystemId)
        b = Table.arange(jnp.array([3.0, 4.0]), label=SystemId)
        ra, rb = Table.broadcast(a, b)
        npt.assert_array_equal(ra.data, [1.0, 2.0])
        npt.assert_array_equal(rb.data, [3.0, 4.0])

        # Single item
        (ra2,) = Table.broadcast(a)
        npt.assert_array_equal(ra2.data, [1.0, 2.0])

        # Broadcast size-one
        c = Table.arange(jnp.array([99.0]), label=SystemId)
        ra3, rc = Table.broadcast(
            Table.arange(jnp.array([1.0, 2.0, 3.0]), label=SystemId), c
        )
        assert len(rc) == 3
        npt.assert_array_equal(rc.data, [99.0, 99.0, 99.0])
        assert rc.keys == (SystemId(0), SystemId(1), SystemId(2))

        # Pytree broadcast
        ap = Table.arange(
            Pair(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), label=SystemId
        )
        bp = Table.arange(Pair(jnp.array([5.0]), jnp.array([6.0])), label=SystemId)
        _, rbp = Table.broadcast(ap, bp)
        npt.assert_array_equal(rbp.data.x, [5.0, 5.0])
        npt.assert_array_equal(rbp.data.y, [6.0, 6.0])

    def test_broadcast_errors(self):
        with pytest.raises(AssertionError, match="Key type mismatch"):
            Table.broadcast(
                Table.arange(jnp.array([1.0]), label=SystemId),
                Table.arange(jnp.array([2.0]), label=ParticleId),
            )
        with pytest.raises(AssertionError, match="Cannot broadcast"):
            Table.broadcast(
                Table.arange(jnp.array([1.0, 2.0, 3.0]), label=SystemId),
                Table.arange(jnp.array([4.0, 5.0]), label=SystemId),
            )
        with pytest.raises(AssertionError, match="int-based keys"):
            Table.broadcast(
                Table(("a",), jnp.array([1.0])),
                Table(("a", "b"), jnp.array([2.0, 3.0])),
            )


class TestMerge:
    def test_merge(self):
        a = Table(("x", "y"), jnp.array([1.0, 2.0]))
        b = Table(("x", "y"), jnp.array([3.0, 4.0]))
        result = Table.join(a, b)
        assert result.keys == ("x", "y")
        npt.assert_array_equal(result.data[0], [1.0, 2.0])
        npt.assert_array_equal(result.data[1], [3.0, 4.0])

        # Reorders to match self
        b2 = Table(("y", "x"), jnp.array([8.0, 6.0]))
        result2 = Table.join(a, b2)
        npt.assert_array_equal(result2.data[1], [6.0, 8.0])

        # Key mismatch raises
        with pytest.raises(ValueError, match="Key set mismatch"):
            Table.join(a, Table(("x", "z"), jnp.array([3.0, 4.0])))

        # Three indexed
        c = Table(("x", "y"), jnp.array([5.0, 6.0]))
        assert len(Table.join(a, b, c).data) == 3


class TestSubset:
    def test_subset(self):
        indexed = Table(("a", "b", "c"), jnp.array([10.0, 20.0, 30.0]))
        result = indexed.subset(Index.new(["c", "a"]))
        assert result.keys == ("0", "1")
        npt.assert_array_equal(result.data, [30.0, 10.0])

        indexed2 = Table.arange(jnp.array([1.0, 2.0, 3.0]), label=ParticleId)
        result2 = indexed2.subset(Index.new([ParticleId(2), ParticleId(0)]))
        assert result2.cls is ParticleId
        assert result2.keys == (ParticleId(0), ParticleId(1))
        npt.assert_array_equal(result2.data, [3.0, 1.0])


class TestUpdateIf:
    def test_update_if(self):
        sys_idx = Index.new([SystemId(0), SystemId(0), SystemId(1), SystemId(1)])
        data = ParticleData(positions=jnp.array([1.0, 2.0, 3.0, 4.0]), system=sys_idx)
        state = Table.arange(data, label=ParticleId)
        idx = Index.new([ParticleId(0), ParticleId(2)])
        new_sys = Index.new([SystemId(0), SystemId(1)])
        new_data = ParticleData(positions=jnp.array([10.0, 30.0]), system=new_sys)

        # Accept all
        result = state.update_if(
            Table.arange(jnp.array([True, True]), label=SystemId), idx, new_data
        )
        npt.assert_array_equal(result.data.positions, [10.0, 2.0, 30.0, 4.0])

        # Accept none
        result2 = state.update_if(
            Table.arange(jnp.array([False, False]), label=SystemId), idx, new_data
        )
        npt.assert_array_equal(result2.data.positions, [1.0, 2.0, 3.0, 4.0])

        # Mixed accept (system 0 only)
        result3 = state.update_if(
            Table.arange(jnp.array([True, False]), label=SystemId), idx, new_data
        )
        npt.assert_array_equal(result3.data.positions, [10.0, 2.0, 3.0, 4.0])


class TestJitCompatibility:
    def test_jit_getitem(self):
        data = jnp.array([10.0, 20.0, 30.0])
        indexed = Table(("a", "b", "c"), data)
        idx = Index.new(["c", "a"])

        @jax.jit
        def f(indexed, idx):
            return indexed[idx]

        result = f(indexed, idx)
        npt.assert_array_equal(result, [30.0, 10.0])


class TestIter:
    def test_basic(self):
        indexed = Table(("a", "b"), jnp.array([10.0, 20.0]))
        items = list(indexed)
        assert items[0][0] == "a"
        assert items[1][0] == "b"
        npt.assert_array_equal(items[0][1], 10.0)
        npt.assert_array_equal(items[1][1], 20.0)


class TestConcatenateTable:
    """Tests for union_tables - SQL UNION ALL with key remapping."""

    def test_union_single_and_two_groups(self):
        """Single group + two-group union with key shifting and remapping."""
        # Single group
        p0 = Table((ParticleId(0), ParticleId(1)), jnp.array([1.0, 2.0]))
        p1 = Table(
            (ParticleId(0), ParticleId(1), ParticleId(2)), jnp.array([3.0, 4.0, 5.0])
        )
        result = Table.union([p0, p1])
        assert result.keys == tuple(ParticleId(i) for i in range(5))
        npt.assert_array_equal(result.data, [1.0, 2.0, 3.0, 4.0, 5.0])

        # Two groups: particles + systems with foreign key remapping
        particles0 = Table(
            (ParticleId(0), ParticleId(1)),
            ParticleData(
                positions=jnp.array([[0.0], [1.0]]),
                system=Index((SystemId(0),), jnp.array([0, 0])),
            ),
        )
        particles1 = Table(
            (ParticleId(0),),
            ParticleData(
                positions=jnp.array([[2.0]]),
                system=Index((SystemId(0),), jnp.array([0])),
            ),
        )
        systems0 = Table((SystemId(0),), SystemData(energy=jnp.array([10.0])))
        systems1 = Table((SystemId(0),), SystemData(energy=jnp.array([20.0])))
        merged_p, merged_s = Table.union([particles0, particles1], [systems0, systems1])
        assert merged_p.keys == (ParticleId(0), ParticleId(1), ParticleId(2))
        assert merged_s.keys == (SystemId(0), SystemId(1))
        npt.assert_array_equal(merged_p.data.positions, [[0.0], [1.0], [2.0]])
        npt.assert_array_equal(merged_s.data.energy, [10.0, 20.0])
        npt.assert_array_equal(merged_p.data.system.indices, [0, 0, 1])
        assert merged_p.data.system.keys == (SystemId(0), SystemId(1))

    def test_union_three_groups_and_key_shifting(self):
        """Three groups with cross-references + integer vs string key shifting."""
        # Three groups
        particles0 = Table(
            (ParticleId(0),),
            ParticleData(
                positions=jnp.array([[0.0]]),
                system=Index((SystemId(0),), jnp.array([0])),
            ),
        )
        particles1 = Table(
            (ParticleId(0),),
            ParticleData(
                positions=jnp.array([[1.0]]),
                system=Index((SystemId(0),), jnp.array([0])),
            ),
        )
        systems0 = Table((SystemId(0),), SystemData(energy=jnp.array([5.0])))
        systems1 = Table((SystemId(0),), SystemData(energy=jnp.array([6.0])))
        groups0 = Table(
            (GroupId(0),),
            GroupData(
                weight=jnp.array([1.0]), system=Index((SystemId(0),), jnp.array([0]))
            ),
        )
        groups1 = Table(
            (GroupId(0),),
            GroupData(
                weight=jnp.array([2.0]), system=Index((SystemId(0),), jnp.array([0]))
            ),
        )
        mp, ms, mg = Table.union(
            [particles0, particles1], [systems0, systems1], [groups0, groups1]
        )
        assert mp.keys == (ParticleId(0), ParticleId(1))
        assert ms.keys == (SystemId(0), SystemId(1))
        assert mg.keys == (GroupId(0), GroupId(1))
        assert mp.data.system.keys == (SystemId(0), SystemId(1))
        assert mg.data.system.keys == (SystemId(0), SystemId(1))
        npt.assert_array_equal(mp.data.system.indices, [0, 1])
        npt.assert_array_equal(mg.data.system.indices, [0, 1])

        # Integer vs string key shifting
        lp0 = Table((ParticleId(0),), Index((Label("H"),), jnp.array([0])))
        lp1 = Table((ParticleId(0),), Index((Label("O"),), jnp.array([0])))
        lresult = Table.union([lp0, lp1])
        assert lresult.keys == (ParticleId(0), ParticleId(1))
        assert lresult.data.keys == (Label("H"), Label("O"))

    def test_index_remapping_and_max_count(self):
        """Index remapping inside data + max_count preservation."""
        p0 = Table(
            (ParticleId(0), ParticleId(1)),
            ParticleData(
                positions=jnp.array([[0.0], [1.0]]),
                system=Index((SystemId(0), SystemId(1)), jnp.array([0, 1])),
            ),
        )
        p1 = Table(
            (ParticleId(0),),
            ParticleData(
                positions=jnp.array([[2.0]]),
                system=Index((SystemId(0),), jnp.array([0])),
            ),
        )
        systems0 = Table(
            (SystemId(0), SystemId(1)), SystemData(energy=jnp.array([10.0, 20.0]))
        )
        systems1 = Table((SystemId(0),), SystemData(energy=jnp.array([30.0])))
        mp, ms = Table.union([p0, p1], [systems0, systems1])
        assert ms.keys == (SystemId(0), SystemId(1), SystemId(2))
        npt.assert_array_equal(ms.data.energy, [10.0, 20.0, 30.0])
        assert mp.data.system.keys == (SystemId(0), SystemId(1), SystemId(2))
        npt.assert_array_equal(mp.data.system.indices, [0, 1, 2])

        # max_count preserved
        mc0 = Table(
            (ParticleId(0),), Index((SystemId(0),), jnp.array([0]), max_count=3)
        )
        mc1 = Table(
            (ParticleId(0),), Index((SystemId(0),), jnp.array([0]), max_count=5)
        )
        assert Table.union([mc0, mc1]).data.max_count == 8

    def test_self_referential_and_empty(self):
        """Self-referential Index[ParticleId] + empty index handling."""
        p0 = Table(
            (ParticleId(0), ParticleId(1)),
            ParticleDataSelf(
                positions=jnp.array([[0.0], [1.0]]),
                neighbor=Index((ParticleId(0), ParticleId(1)), jnp.array([1, 0])),
            ),
        )
        p1 = Table(
            (ParticleId(0),),
            ParticleDataSelf(
                positions=jnp.array([[2.0]]),
                neighbor=Index((ParticleId(0),), jnp.array([0])),
            ),
        )
        result = Table.union([p0, p1])
        assert result.keys == (ParticleId(0), ParticleId(1), ParticleId(2))
        assert result.data.neighbor.keys == (
            ParticleId(0),
            ParticleId(1),
            ParticleId(2),
        )
        npt.assert_array_equal(result.data.neighbor.indices, [1, 0, 2])

        # Empty index
        e0 = Table((ParticleId(0),), Index((), jnp.zeros(0, dtype=jnp.int32), _cls=int))
        e1 = Table((ParticleId(0),), Index((), jnp.zeros(0, dtype=jnp.int32), _cls=int))
        eresult = Table.union([e0, e1])
        assert eresult.keys == (ParticleId(0), ParticleId(1))
        assert eresult.data.keys == ()

    def test_union_errors(self):
        """Duplicate key type and mismatched sequence lengths."""
        with pytest.raises(AssertionError, match="Duplicate key type"):
            Table.union(
                [Table((ParticleId(0),), jnp.array([1.0]))],
                [Table((ParticleId(0),), jnp.array([2.0]))],
            )
        with pytest.raises(AssertionError, match="same length"):
            Table.union(
                [
                    Table((ParticleId(0),), jnp.array([1.0])),
                    Table((ParticleId(0),), jnp.array([2.0])),
                ],
                [Table((SystemId(0),), jnp.array([3.0]))],
            )


class TestMatchIndices:
    def test_single_and_two_groups(self):
        """Single group unchanged + two groups align leaf Index."""
        particles_s = Table(
            (ParticleId(0), ParticleId(1)),
            Index((SystemId(0),), jnp.array([0, 0])),
        )
        (result,) = Table.match(particles_s)
        assert result.keys == particles_s.keys

        particles = Table(
            (ParticleId(0), ParticleId(1)),
            ParticleData(
                positions=jnp.array([[0.0], [1.0]]),
                system=Index((SystemId(0),), jnp.array([0, 0])),
            ),
        )
        systems = Table(
            (SystemId(0), SystemId(1)),
            SystemData(energy=jnp.array([10.0, 20.0])),
        )
        mp, ms = Table.match(particles, systems)
        assert mp.data.system.keys == (SystemId(0), SystemId(1))
        npt.assert_array_equal(mp.data.system.indices, [0, 0])
        assert ms.keys == systems.keys

    def test_three_groups_and_self_ref(self):
        """Three groups cross-reference + self-referencing leaf."""
        particles = Table(
            (ParticleId(0),),
            ParticleData(
                positions=jnp.array([[0.0]]),
                system=Index((SystemId(0),), jnp.array([0])),
            ),
        )
        systems = Table(
            (SystemId(0), SystemId(1)),
            SystemData(energy=jnp.array([5.0, 6.0])),
        )
        groups = Table(
            (GroupId(0),),
            GroupData(
                weight=jnp.array([1.0]),
                system=Index((SystemId(0),), jnp.array([0])),
            ),
        )
        mp, ms, mg = Table.match(particles, systems, groups)
        assert mp.data.system.keys == (SystemId(0), SystemId(1))
        assert mg.data.system.keys == (SystemId(0), SystemId(1))

        # Self-referencing leaf
        particles_sr = Table(
            (ParticleId(0), ParticleId(1), ParticleId(2)),
            ParticleDataSelf(
                positions=jnp.array([[0.0], [1.0], [2.0]]),
                neighbor=Index((ParticleId(0), ParticleId(1)), jnp.array([1, 0, 1])),
            ),
        )
        (result,) = Table.match(particles_sr)
        assert result.data.neighbor.keys == (
            ParticleId(0),
            ParticleId(1),
            ParticleId(2),
        )
        npt.assert_array_equal(result.data.neighbor.indices, [1, 0, 1])
