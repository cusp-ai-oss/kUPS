# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Batched mixin."""

import jax
import jax.numpy as jnp
import pytest

from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Batched
from kups.core.utils.jax import dataclass


@dataclass
class _Pair(Batched):
    a: jax.Array
    b: jax.Array


class TestBatched:
    def test_consistent_leading_dim(self):
        pair = _Pair(jnp.zeros((4, 3)), jnp.ones((4,)))
        assert len(pair) == 4 and pair.size == 4

    def test_inconsistent_leading_dim(self):
        with pytest.raises(ValueError, match="Inconsistent shapes"):
            _Pair(jnp.zeros((4, 3)), jnp.ones((2,)))

    def test_missing_leading_dim(self):
        with pytest.raises(ValueError, match="no leading dimension"):
            _Pair(jnp.zeros(()), jnp.zeros(()))

    @pytest.mark.parametrize(
        "obj",
        [
            _Pair(jnp.zeros((4, 3)), jnp.ones((4,))),
            PeriodicCell(frame=TriclinicFrame.from_matrix(jnp.eye(3)[None])),
        ],
        ids=["pair", "cell"],
    )
    def test_unflatten_with_none_leaves(self, obj: Batched):
        """Rebuilding with ``None`` leaves must not fail.

        ``flax.nnx`` replaces every non-graph-node leaf with ``None`` and
        unflattens, so ``__post_init__`` runs on a leafless placeholder.
        """
        leaves, treedef = jax.tree.flatten(obj)
        rebuilt = jax.tree.unflatten(treedef, [None] * len(leaves))
        assert type(rebuilt) is type(obj)
