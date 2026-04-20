# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.data.index import (
    Index,
    unify_keys_by_cls,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


class TestNew:
    def test_new(self):
        sa = Index.new(["C", "H", "O", "H", "C"])
        assert sa.keys == ("C", "H", "O")
        npt.assert_array_equal(sa.indices, [0, 1, 2, 1, 0])
        # Single label
        sa2 = Index.new(["Ar", "Ar", "Ar"])
        assert sa2.keys == ("Ar",)
        npt.assert_array_equal(sa2.indices, [0, 0, 0])
        # 2D shape preserved
        sa3 = Index.new(np.array([["A", "B"], ["B", "A"]]))
        assert sa3.shape == (2, 2)
        npt.assert_array_equal(sa3.indices, [[0, 1], [1, 0]])
        # Keys sorted
        assert Index.new(["Z", "A", "M"]).keys == ("A", "M", "Z")


class TestProperties:
    def test_properties(self):
        sa = Index.new(["a", "b", "c"])
        assert sa.shape == (3,)
        assert len(sa) == 3
        assert sa.dtype == jnp.dtype(object)
        args = sa.scatter_args
        assert args.get("mode") == "fill" and args.get("fill_value") == 3

        sa2 = Index.new(np.array([["a", "b"], ["c", "d"]]))
        assert sa2.size == 4 and sa2.ndim == 2

        # counts
        counts = Index.new(["H", "O", "H", "H", "O"]).counts
        assert counts.keys == ("H", "O")
        npt.assert_array_equal(counts.data, [3, 2])
        counts2 = Index.new(["Ar", "Ar", "Ar"]).counts
        npt.assert_array_equal(counts2.data, [3])

        # iter
        items = list(Index.new(["X", "Y", "X"]))
        assert len(items) == 3 and all(isinstance(i, Index) for i in items)


class TestGetitem:
    def test_getitem(self):
        sa = Index.new(["A", "B", "C", "D"])
        result = sa[0]
        assert isinstance(result, Index) and result.keys == sa.keys
        # Slice
        result2 = sa[1:3]
        assert result2.shape == (2,)
        npt.assert_array_equal(result2.indices, sa.indices[1:3])
        # Fancy index
        sa2 = Index.new(["A", "B", "C"])
        npt.assert_array_equal(
            sa2[jnp.array([2, 0])].indices, sa2.indices[jnp.array([2, 0])]
        )


class TestReshape:
    def test_reshape_ravel_transpose(self):
        sa = Index.new(["a", "b", "c", "d"])
        result = sa.reshape(2, 2)
        assert result.shape == (2, 2) and result.keys == sa.keys
        sa2 = Index.new(np.array([["a", "b"], ["c", "d"]]))
        assert sa2.ravel().shape == (4,)
        assert sa2.transpose().shape == (2, 2)
        npt.assert_array_equal(sa2.transpose().indices, sa2.indices.T)
        npt.assert_array_equal(sa2.T.indices, sa2.transpose().indices)


class TestIndicesIn:
    def test_same_different_subset(self):
        sa = Index.new(["A", "B", "A"])
        npt.assert_array_equal(sa.indices_in(("A", "B")), sa.indices)
        # Different order
        sa2 = Index.new(["H", "O"])
        decoded = np.asarray(("H", "O", "Si"))[
            np.asarray(sa2.indices_in(("H", "O", "Si")))
        ]
        npt.assert_array_equal(decoded, ["H", "O"])
        # Subset
        idx3 = Index.new(["B"]).indices_in(("A", "B", "C"))
        assert np.asarray(("A", "B", "C"))[np.asarray(idx3).item()] == "B"

    def test_missing_key_and_oob(self):
        with pytest.raises(ValueError, match="Keys in self not found"):
            Index.new(["X", "Y"]).indices_in(("Y", "Z"))
        # OOB preserved
        sa = Index(("A", "B"), jnp.array([0, 2, 1]))
        npt.assert_array_equal(sa.indices_in(("B", "A")), [1, 2, 0])

    def test_allow_missing_maps_missing_to_zero(self):
        idx = Index.new(["A", "B", "C"])
        result = idx.indices_in(("A", "C"), allow_missing=True)
        # A->0, C->1, B is missing -> argmax fallback to 0
        npt.assert_array_equal(result, [0, 0, 1])

    def test_allow_missing_empty_tokens(self):
        idx = Index.new(["A", "B"])
        result = idx.indices_in((), allow_missing=True)
        # All mapped to 0 (fill value for empty tokens)
        npt.assert_array_equal(result, [0, 0])


class TestIsin:
    def test_isin(self):
        npt.assert_array_equal(
            Index.new(["H", "O", "H"]).isin(Index.new(["H", "O"])), [True, True, True]
        )
        npt.assert_array_equal(
            Index.new(["H", "O", "C"]).isin(Index.new(["H", "C"])), [True, False, True]
        )
        npt.assert_array_equal(
            Index.new(["H", "O"]).isin(Index.new(["C", "N"])), [False, False]
        )


class TestValidMask:
    def test_valid_mask(self):
        npt.assert_array_equal(
            Index.new(["A", "B", "C"]).valid_mask, [True, True, True]
        )
        npt.assert_array_equal(
            Index(("A", "B"), jnp.array([0, 2, 1])).valid_mask, [True, False, True]
        )
        npt.assert_array_equal(
            Index(("A",), jnp.array([1, 1, 1])).valid_mask, [False, False, False]
        )


class TestApplyMask:
    def test_apply_mask(self):
        idx = Index.new(["A", "B", "C"])
        result = idx.apply_mask(jnp.array([True, False, True]))
        assert (
            result.valid_mask[0] and not result.valid_mask[1] and result.valid_mask[2]
        )
        npt.assert_array_equal(result.indices[1], len(result.keys))
        # Preserves keys and max_count
        idx2 = Index(("X", "Y"), jnp.array([0, 1, 0]), max_count=5)
        result2 = idx2.apply_mask(jnp.array([False, True, True]))
        assert result2.keys == ("X", "Y") and result2.max_count == 5


class TestStr:
    def test_str(self):
        result = str(Index.new(["C", "H", "O", "H"]))
        assert "C" in result and "H" in result and "O" in result
        # Tracer fallback
        sa2 = Index.new(["A", "B"])
        output = {}

        def capture_str(data):
            output["s"] = str(Index(sa2.keys, data))
            return data

        jax.make_jaxpr(capture_str)(sa2.indices)
        assert "Index" in output["s"] and "keys" in output["s"]


class TestMatchIndices:
    def test_match_indices(self):
        # Same keys
        i, j = Index.match(Index.new(["H", "O", "H"]), Index.new(["H", "O"]))
        npt.assert_array_equal(i, Index.new(["H", "O", "H"]).indices)
        npt.assert_array_equal(j, Index.new(["H", "O"]).indices)
        # Disjoint
        i2, j2 = Index.match(Index.new(["H", "H"]), Index.new(["O"]))
        npt.assert_array_equal(i2, [0, 0])
        npt.assert_array_equal(j2, [1])
        # Three
        i3, j3, k3 = Index.match(Index.new(["A"]), Index.new(["B"]), Index.new(["C"]))
        npt.assert_array_equal(i3, [0])
        npt.assert_array_equal(j3, [1])
        npt.assert_array_equal(k3, [2])
        # Overlapping
        i4, j4 = Index.match(Index.new(["H", "O"]), Index.new(["O", "C"]))
        merged = ("C", "H", "O")
        assert merged[int(i4[0])] == "H" and merged[int(i4[1])] == "O"
        assert merged[int(j4[0])] == "O" and merged[int(j4[1])] == "C"


class TestConcatenate:
    def test_same_disjoint_overlapping(self):
        a = Index(("H", "O"), jnp.array([0, 1, 0]))
        b = Index(("H", "O"), jnp.array([1, 1]))
        result = Index.concatenate(a, b)
        assert set(result.keys) == {"H", "O"}
        assert [result.keys[int(i)] for i in result.indices] == [
            "H",
            "O",
            "H",
            "O",
            "O",
        ]
        # Disjoint
        result2 = Index.concatenate(
            Index(("H",), jnp.array([0, 0])), Index(("O",), jnp.array([0]))
        )
        assert [result2.keys[int(i)] for i in result2.indices] == ["H", "H", "O"]
        # Overlapping
        result3 = Index.concatenate(
            Index(("H", "O"), jnp.array([0, 1])), Index(("C", "O"), jnp.array([1, 0]))
        )
        assert [result3.keys[int(i)] for i in result3.indices] == ["H", "O", "O", "C"]
        # max_count + single input
        assert (
            Index.concatenate(
                Index(("X",), jnp.array([0]), max_count=3),
                Index(("X",), jnp.array([0]), max_count=7),
            ).max_count
            == 10
        )
        a3 = Index(("A", "B"), jnp.array([1, 0, 1]))
        result4 = Index.concatenate(a3)
        assert result4.keys == a3.keys
        npt.assert_array_equal(result4.indices, a3.indices)

    def test_offset_concatenate(self):
        # Basic
        result = Index.concatenate(
            Index((ParticleId(0), ParticleId(1)), jnp.array([0, 1])),
            Index((ParticleId(0),), jnp.array([0])),
            shift_keys=True,
        )
        assert result.keys == (ParticleId(0), ParticleId(1), ParticleId(2))
        npt.assert_array_equal(result.indices, [0, 1, 2])
        # Three items
        result2 = Index.concatenate(
            Index((SystemId(0),), jnp.array([0, 0])),
            Index((SystemId(0),), jnp.array([0])),
            Index((SystemId(0),), jnp.array([0, 0, 0])),
            shift_keys=True,
        )
        assert result2.keys == (SystemId(0), SystemId(1), SystemId(2))
        npt.assert_array_equal(result2.indices, [0, 0, 1, 2, 2, 2])
        # Keys strictly increasing
        result3 = Index.concatenate(
            Index((ParticleId(0), ParticleId(1)), jnp.array([0])),
            Index((ParticleId(0), ParticleId(1)), jnp.array([1])),
            shift_keys=True,
        )
        ints = [int(key) for key in result3.keys]
        assert ints == sorted(ints) and len(set(ints)) == len(ints)
        # max_count preserved
        assert (
            Index.concatenate(
                Index((ParticleId(0),), jnp.array([0]), max_count=5),
                Index((ParticleId(0),), jnp.array([0]), max_count=3),
                shift_keys=True,
            ).max_count
            == 5
        )
        # OOB remapped
        result5 = Index.concatenate(
            Index((ParticleId(0), ParticleId(1)), jnp.array([0, 2, 1])),
            Index((ParticleId(0),), jnp.array([0, 1])),
            shift_keys=True,
        )
        npt.assert_array_equal(result5.indices, [0, 3, 1, 2, 3])
        # Non-increasing input raises
        with pytest.raises(ValueError, match="Keys must be unique and sorted"):
            Index((ParticleId(1), ParticleId(0)), jnp.array([0, 1]))


class TestSelectPerLabel:
    def test_select_per_label(self):
        sa = Index.new(["H", "O", "H", "H", "O"])
        npt.assert_array_equal(sa.select_per_label(jnp.array([0, 1])), [0, 4])
        # Wraps with modulo
        sa2 = Index.new(["A", "A", "B"])
        npt.assert_array_equal(sa2.select_per_label(jnp.array([5, 0])), [1, 2])
        # Empty key returns OOB
        sa3 = Index(("A", "B"), jnp.array([0, 0, 0]))
        result = sa3.select_per_label(jnp.array([0, 0]))
        assert result[0] == 0 and result[1] == 3


class TestCombineIndices:
    def test_combine_indices(self):
        # Two indices
        idx1 = Index.new([ParticleId(0), ParticleId(1), ParticleId(0)])
        idx2 = Index.new([SystemId(0), SystemId(0), SystemId(1)])
        result = Index.combine(idx1, idx2)
        assert result.keys == (
            (ParticleId(0), SystemId(0)),
            (ParticleId(0), SystemId(1)),
            (ParticleId(1), SystemId(0)),
            (ParticleId(1), SystemId(1)),
        )
        npt.assert_array_equal(result.indices, [0, 2, 1])
        # Single
        result2 = Index.combine(Index.new([SystemId(1), SystemId(0)]))
        assert result2.keys == ((SystemId(0),), (SystemId(1),))
        npt.assert_array_equal(result2.indices, [1, 0])
        # Three
        result3 = Index.combine(
            Index.new([ParticleId(0), ParticleId(0)]),
            Index.new([SystemId(0), SystemId(1)]),
            Index.new([ParticleId(1), ParticleId(1)]),
        )
        assert len(result3.keys) == 2
        assert (ParticleId(0), SystemId(0), ParticleId(1)) in result3.keys
        # Shape mismatch
        with pytest.raises(AssertionError, match="shapes must match"):
            Index.combine(
                Index.new([ParticleId(0), ParticleId(1)]), Index.new([SystemId(0)])
            )
        # Unoccupied combinations
        result4 = Index.combine(
            Index.new([SystemId(0), SystemId(0), SystemId(1)]),
            Index.new([ParticleId(0), ParticleId(0), ParticleId(0)]),
        )
        assert result4.keys == (
            (SystemId(0), ParticleId(0)),
            (SystemId(1), ParticleId(0)),
        )
        npt.assert_array_equal(result4.indices, [0, 0, 1])


class TestSumOver:
    def test_sum_over(self):
        idx = Index.new([SystemId(0), SystemId(1), SystemId(0), SystemId(1)])
        result = idx.sum_over(jnp.array([1.0, 2.0, 3.0, 4.0]))
        assert result.keys == (SystemId(0), SystemId(1))
        npt.assert_array_equal(result.data, [4.0, 6.0])
        # Single group
        npt.assert_array_equal(
            Index.new([SystemId(0)] * 3).sum_over(jnp.array([1.0, 2.0, 3.0])).data,
            [6.0],
        )
        # Multidimensional
        idx3 = Index.new([SystemId(0), SystemId(1), SystemId(0)])
        npt.assert_array_equal(
            idx3.sum_over(jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).data,
            [[6.0, 8.0], [3.0, 4.0]],
        )


class TestToCls:
    def test_to_cls(self):
        idx = Index((0, 1, 2), jnp.array([2, 0, 1]))
        result = idx.to_cls(ParticleId)
        assert result.keys == (ParticleId(0), ParticleId(1), ParticleId(2))
        npt.assert_array_equal(result.indices, [2, 0, 1])
        assert result.cls is ParticleId
        # max_count
        assert (
            Index((0, 1), jnp.array([0, 1]), max_count=5).to_cls(SystemId).max_count
            == 5
        )
        # Callable
        result3 = Index((0, 1), jnp.array([0, 1, 0])).to_cls(
            lambda x: ParticleId(x * 10)
        )
        assert result3.keys == (ParticleId(0), ParticleId(10))
        # OOB preserved
        assert not Index((0, 1), jnp.array([0, 2, 1])).to_cls(ParticleId).valid_mask[1]


class TestWhereAndSubselect:
    def test_where_and_subselect(self):
        # where_rectangular
        sa = Index.new(["H", "O", "H", "O", "H"])
        result = sa.where_rectangular(Index.new(["H", "O"]), max_count=3)
        npt.assert_array_equal(result[0], [0, 2, 4])
        npt.assert_array_equal(result[1, :2], [1, 3])
        assert result[1, 2] == 5
        # Single label
        sa2 = Index.new(["X", "X", "X"])
        result2 = sa2.where_rectangular(Index.new(["X"]), max_count=4)
        npt.assert_array_equal(result2[0, :3], [0, 1, 2])
        assert result2[0, 3] == 3
        # Different key order
        sa3 = Index.new(["H", "O", "H"])
        result3 = sa3.where_rectangular(Index.new(["O"]), max_count=2)
        npt.assert_array_equal(result3[0, :1], [1])
        assert result3[0, 1] == 3

        # where_flat
        npt.assert_array_equal(
            sa.where_flat(Index.new(["H"]), capacity=FixedCapacity(3)), [0, 2, 4]
        )
        sa4 = Index.new(["A", "B", "C", "A", "B"])
        npt.assert_array_equal(
            sa4.where_flat(Index.new(["A", "C"]), capacity=FixedCapacity(3)), [0, 2, 3]
        )

        # subselect
        sub = sa.subselect(Index.new(["O"]), capacity=FixedCapacity(2))
        assert isinstance(sub.scatter, Index) and isinstance(sub.gather, Index)
        npt.assert_array_equal(sub.scatter.value, ["O", "O"])
        npt.assert_array_equal(sub.gather.value, ["O", "O"])
        # Multiple needles
        sub2 = sa4.subselect(Index.new(["A", "C"]), capacity=FixedCapacity(3))
        npt.assert_array_equal(sorted(sub2.gather.value), ["A", "A", "C"])
        # Key spaces
        sa5 = Index.new(["X", "Y", "X", "Z"])
        sub3 = sa5.subselect(Index.new(["Y"]), capacity=FixedCapacity(1))
        assert sub3.scatter.keys == ("Y",) and sub3.gather.keys == ("X", "Y", "Z")
        # OOB with overcapacity
        sub4 = Index.new(["A", "B", "A"]).subselect(
            Index.new(["B"]), capacity=FixedCapacity(3)
        )
        assert int((sub4.scatter.indices < len(sub4.scatter.keys)).sum()) == 1
        # Iter unpacking
        scatter, gather = Index.new(["A", "B", "A"]).subselect(
            Index.new(["A"]), capacity=FixedCapacity(2)
        )
        assert isinstance(scatter, Index) and isinstance(gather, Index)


class TestValueArrayAndJit:
    def test_value_array_and_jit(self):
        # Value property
        npt.assert_array_equal(
            Index.new(["C", "H", "O", "H"]).value, ["C", "H", "O", "H"]
        )
        npt.assert_array_equal(
            Index.new(np.array([["A", "B"], ["B", "A"]])).value,
            [["A", "B"], ["B", "A"]],
        )
        npt.assert_array_equal(np.asarray(Index.new(["X", "Y", "X"])), ["X", "Y", "X"])

        # JIT passthrough
        sa = Index.new(["A", "B", "A"])

        @jax.jit
        def f(x: Index) -> Index:
            return x[jnp.array([2, 0, 1])]

        result = f(sa)
        assert isinstance(result, Index) and result.keys == sa.keys
        npt.assert_array_equal(result.indices, sa.indices[jnp.array([2, 0, 1])])


class TestUnifyKeysByCls:
    def test_unify_keys_by_cls(self):
        # Same cls
        a, b = unify_keys_by_cls((Index.new(["H", "O"]), Index.new(["O", "N"])))
        assert a.keys == b.keys and set(a.keys) == {"H", "N", "O"}
        npt.assert_array_equal(a.indices, [0, 2])
        npt.assert_array_equal(b.indices, [2, 1])
        # Different cls independent
        p, s = unify_keys_by_cls(
            (
                Index((ParticleId(0),), jnp.array([0])),
                Index((SystemId(0), SystemId(1)), jnp.array([1])),
            )
        )
        assert p.keys == (ParticleId(0),) and s.keys == (SystemId(0), SystemId(1))
        # Three same cls
        r0, r1, r2 = unify_keys_by_cls(
            (Index.new(["A"]), Index.new(["B"]), Index.new(["C"]))
        )
        assert r0.keys == r1.keys == r2.keys == ("A", "B", "C")
        # Non-index untouched
        arr = jnp.array([1.0, 2.0])
        ri, ra = unify_keys_by_cls((Index.new(["H"]), arr))
        assert isinstance(ri, Index)
        npt.assert_array_equal(ra, arr)


class TestFactoryAndLabels:
    def test_factory_and_labels(self):
        # update_labels
        idx = Index.new(["A", "B"])
        result = idx.update_labels(("A", "B", "C"))
        assert result.keys == ("A", "B", "C")
        npt.assert_array_equal(result.value, idx.value)
        idx2 = Index.new(["H", "O"])
        result2 = idx2.update_labels(("H", "N", "O"))
        npt.assert_array_equal(result2.value, idx2.value)
        assert result2.keys == ("H", "N", "O")

        # zeros
        idx3 = Index.zeros(5)
        assert idx3.shape == (5,) and idx3.num_labels == 1
        npt.assert_array_equal(idx3.indices, [0, 0, 0, 0, 0])
        assert Index.zeros((2, 3)).shape == (2, 3)
        idx_s = Index.zeros(3, label=SystemId)
        assert idx_s.keys == (SystemId(0),) and idx_s.cls is SystemId

        # integer
        idx4 = Index.integer([0, 1, 0, 2], n=3)
        assert idx4.num_labels == 3
        npt.assert_array_equal(idx4.indices, [0, 1, 0, 2])
        assert Index.integer([0, 1, 2]).num_labels == 3
        idx5 = Index.integer([0, 0, 1], n=2, label=ParticleId)
        assert idx5.keys == (ParticleId(0), ParticleId(1)) and idx5.cls is ParticleId


@dataclass
class _NestedData:
    values: jax.Array
    system: Index[SystemId]


class TestFindIndex:
    def test_find_index(self):
        sys_idx = Index.new([SystemId(0), SystemId(1)])
        obj = _NestedData(values=jnp.array([1.0, 2.0]), system=sys_idx)
        assert Index.find(obj, SystemId) is sys_idx
        # No match
        with pytest.raises(AssertionError, match="No Index"):
            Index.find(
                _NestedData(values=jnp.array([1.0]), system=Index.new([SystemId(0)])),
                ParticleId,
            )
        # Multiple matches
        with pytest.raises(AssertionError, match="Multiple Index"):
            Index.find((sys_idx, sys_idx), SystemId)
