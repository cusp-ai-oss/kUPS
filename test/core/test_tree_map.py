# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.data.index import Index
from kups.core.utils.jax import dataclass, tree_map


@dataclass
class Pair:
    x: jax.Array
    y: jax.Array


@dataclass
class ParticleState:
    position: jax.Array
    species: Index[str]


class TestPlainArrays:
    """tree_map on plain JAX arrays behaves like jax.tree.map."""

    def test_single_array(self):
        a = jnp.array([1.0, 2.0, 3.0])
        result = tree_map(lambda x: x * 2, a)
        npt.assert_array_equal(result, a * 2)

    def test_two_arrays(self):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        result = tree_map(jnp.add, a, b)
        npt.assert_array_equal(result, [4.0, 6.0])

    def test_three_arrays(self):
        a = jnp.array([1.0])
        b = jnp.array([2.0])
        c = jnp.array([3.0])
        result = tree_map(lambda x, y, z: x + y + z, a, b, c)
        npt.assert_array_equal(result, [6.0])


class TestPytreeOfArrays:
    """tree_map on a dataclass containing arrays."""

    def test_single_dataclass(self):
        p = Pair(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        result = tree_map(lambda x: x + 10, p)
        assert isinstance(result, Pair)
        npt.assert_array_equal(result.x, [11.0, 12.0])
        npt.assert_array_equal(result.y, [13.0, 14.0])

    def test_two_dataclasses(self):
        a = Pair(jnp.array([1.0]), jnp.array([2.0]))
        b = Pair(jnp.array([3.0]), jnp.array([4.0]))
        result = tree_map(jnp.add, a, b)
        npt.assert_array_equal(result.x, [4.0])
        npt.assert_array_equal(result.y, [6.0])


class TestIsLeaf:
    """Custom is_leaf predicate stops traversal."""

    def test_is_leaf_treats_pair_as_leaf(self):
        """When is_leaf matches Pair, fn receives the whole Pair."""
        p = Pair(jnp.array([1.0]), jnp.array([2.0]))
        results: list = []

        def capture(x):
            results.append(x)
            return x

        tree_map(capture, p, is_leaf=lambda x: isinstance(x, Pair))
        assert len(results) == 1
        assert isinstance(results[0], Pair)

    def test_is_leaf_on_nested_list(self):
        """is_leaf can prevent descent into list elements."""
        tree = [jnp.array(1.0), jnp.array(2.0)]
        result = tree_map(lambda x: x * 3, tree, is_leaf=lambda x: isinstance(x, list))
        assert isinstance(result, list)


class TestTreeMatchAlignment:
    """TreeMatch alignment via Index.__tree_match__.

    Index is a TreeMatch *and* a pytree. tree_map first aligns keys via
    __tree_match__, then recurses into the Index children (the `indices`
    array). So fn receives plain arrays (the remapped integer indices).
    """

    def test_same_keys_no_remapping(self):
        """Indices with identical keys: integer values unchanged after alignment."""
        a = Index(("H", "O"), jnp.array([0, 1, 0]))
        b = Index(("H", "O"), jnp.array([1, 0, 1]))
        result = tree_map(jnp.add, a, b)
        assert isinstance(result, Index)
        npt.assert_array_equal(result.indices, [1, 1, 1])
        assert result.keys == ("H", "O")

    def test_different_key_order(self):
        """Indices with different key orderings are aligned before fn.

        a: keys=("A","B"), indices=[0,1] -> values A, B
        b: keys=("B","C"), indices=[0,1] -> values B, C
        After alignment, merged keys = ("A", "B", "C"):
          a remapped: A->0, B->1 => [0, 1]
          b remapped: B->1, C->2 => [1, 2]
        """
        a = Index(("A", "B"), jnp.array([0, 1]))
        b = Index(("B", "C"), jnp.array([0, 1]))
        result = tree_map(jnp.add, a, b)
        assert isinstance(result, Index)
        assert result.keys == ("A", "B", "C")
        npt.assert_array_equal(result.indices, [0 + 1, 1 + 2])

    def test_disjoint_keys_merged(self):
        """Indices with disjoint key sets are merged into a union."""
        a = Index(("X",), jnp.array([0, 0]))
        b = Index(("Y",), jnp.array([0, 0]))
        result = tree_map(jnp.add, a, b)
        assert isinstance(result, Index)
        assert set(result.keys) == {"X", "Y"}
        # a: X->0, b: Y->1 => sum [0+1, 0+1] = [1, 1]
        npt.assert_array_equal(result.indices, [1, 1])

    def test_alignment_preserves_semantic_values(self):
        """After alignment, both indices decode to the same label vocabulary."""
        a = Index.new(["O", "H", "O"])  # keys=("H","O"), indices=[1,0,1]
        b = Index.new(["H", "H", "O"])  # keys=("H","O"), indices=[0,0,1]
        # Same keys, so no remapping needed. Sum of indices:
        result = tree_map(jnp.add, a, b)
        npt.assert_array_equal(result.indices, [1 + 0, 0 + 0, 1 + 1])


class TestTreeMatchInsidePytree:
    """Dataclass containing an Index field: tree_map aligns Index fields."""

    def test_index_inside_dataclass(self):
        """Index fields inside a dataclass are aligned across trees."""
        a = ParticleState(
            position=jnp.array([1.0, 2.0]),
            species=Index(("H", "O"), jnp.array([0, 1])),
        )
        b = ParticleState(
            position=jnp.array([3.0, 4.0]),
            species=Index(("O", "Si"), jnp.array([0, 1])),
        )
        result = tree_map(jnp.add, a, b)
        assert isinstance(result, ParticleState)
        npt.assert_array_equal(result.position, [4.0, 6.0])
        # species aligned to ("H", "O", "Si"):
        #   a: H->0, O->1 => [0, 1]
        #   b: O->1, Si->2 => [1, 2]
        assert result.species.keys == ("H", "O", "Si")
        npt.assert_array_equal(result.species.indices, [0 + 1, 1 + 2])

    def test_index_keys_merged_inside_dataclass(self):
        """Verify that Index keys are merged when inside a dataclass."""
        a = ParticleState(
            position=jnp.array([0.0]),
            species=Index(("A",), jnp.array([0])),
        )
        b = ParticleState(
            position=jnp.array([0.0]),
            species=Index(("B",), jnp.array([0])),
        )
        result = tree_map(jnp.add, a, b)
        assert result.species.keys == ("A", "B")


class TestMultipleTrees:
    """tree_map with 2+ input trees."""

    def test_three_plain_trees(self):
        a = {"v": jnp.array([1.0])}
        b = {"v": jnp.array([2.0])}
        c = {"v": jnp.array([3.0])}
        result = tree_map(lambda x, y, z: x + y + z, a, b, c)
        npt.assert_array_equal(result["v"], [6.0])

    def test_three_dataclasses(self):
        a = Pair(jnp.array([1.0]), jnp.array([10.0]))
        b = Pair(jnp.array([2.0]), jnp.array([20.0]))
        c = Pair(jnp.array([3.0]), jnp.array([30.0]))
        result = tree_map(lambda x, y, z: x + y + z, a, b, c)
        npt.assert_array_equal(result.x, [6.0])
        npt.assert_array_equal(result.y, [60.0])

    def test_three_indices_aligned(self):
        """Three Index objects with partially overlapping keys are all aligned."""
        a = Index(("A", "B"), jnp.array([0, 1]))
        b = Index(("B", "C"), jnp.array([0, 1]))
        c = Index(("A", "C"), jnp.array([0, 1]))
        # Merged keys: ("A", "B", "C")
        #   a: A->0, B->1 => [0, 1]
        #   b: B->1, C->2 => [1, 2]
        #   c: A->0, C->2 => [0, 2]
        result = tree_map(lambda x, y, z: x + y + z, a, b, c)
        assert isinstance(result, Index)
        assert result.keys == ("A", "B", "C")
        npt.assert_array_equal(result.indices, [0 + 1 + 0, 1 + 2 + 2])
