# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system L2-norm clip transform."""

import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.core.data.index import Index
from kups.core.typing import SystemId
from kups.relaxation.transforms.clip_by_global_norm import (
    ClipByGlobalNorm,
    ClipByGlobalNormState,
)

from ...clear_cache import clear_cache  # noqa: F401


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


class TestClipByGlobalNormGlobalFallback:
    """index_prefix=None reduces to a single tree-global L2 clip."""

    def test_no_clip_when_below(self):
        opt = ClipByGlobalNorm(max_norm=10.0)
        state = opt.init(None)
        updates = jnp.array([[0.1, 0.2, 0.0], [0.3, 0.0, 0.4]])
        new_updates, _ = opt.update(updates, state)
        npt.assert_allclose(new_updates, updates, atol=1e-6)

    def test_clips_when_above(self):
        opt = ClipByGlobalNorm(max_norm=1.0)
        state = opt.init(None)
        updates = jnp.array([[3.0, 4.0, 0.0]])  # global L2 norm = 5.0
        new_updates, _ = opt.update(updates, state)
        npt.assert_allclose(new_updates, updates / 5.0, atol=1e-6)

    def test_matches_optax_clip_by_global_norm(self):
        """Bit-identity check vs ``optax.clip_by_global_norm`` for a single system."""
        new = ClipByGlobalNorm(max_norm=2.0)
        ref = optax.clip_by_global_norm(max_norm=2.0)
        updates = {
            "a": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "b": jnp.array([[0.5, -0.5]]),
        }
        new_state = new.init(None)
        ref_state = ref.init(updates)  # type: ignore[arg-type]
        new_out, _ = new.update(updates, new_state)
        ref_out_a, ref_out_b = ref.update(updates, ref_state)[0].values()  # type: ignore[union-attr]
        npt.assert_allclose(new_out["a"], jnp.asarray(ref_out_a), atol=1e-6)
        npt.assert_allclose(new_out["b"], jnp.asarray(ref_out_b), atol=1e-6)


class TestClipByGlobalNormPerSystem:
    def test_independent_clip_per_system(self):
        idx = _system_index([0, 0, 1, 1], 2)
        # System 0: total L2 = sqrt(0.3² + 0.4² + 0² + 0²) = 0.5 — well below.
        # System 1: total L2 = sqrt(3² + 0² + 4² + 0²) = 5.0 — must clip to 1.0.
        updates = jnp.array([[0.3, 0.0], [0.4, 0.0], [3.0, 0.0], [4.0, 0.0]])
        opt = ClipByGlobalNorm(max_norm=1.0)
        state = opt.init(updates, index_prefix=idx)
        new_updates, _ = opt.update(updates, state)

        npt.assert_allclose(new_updates[0:2], updates[0:2], atol=1e-6)
        # System 1's per-element values scaled by 1/5.
        npt.assert_allclose(new_updates[2:4], updates[2:4] / 5.0, atol=1e-6)

    def test_batched_matches_separate(self):
        """Batched run = concatenation of independent per-system runs."""

        def run_alone(arr: jnp.ndarray) -> jnp.ndarray:
            opt = ClipByGlobalNorm(max_norm=1.0)
            state = opt.init(None)
            new_updates, _ = opt.update(arr, state)
            return jnp.asarray(new_updates)

        sys_a = jnp.array([[0.5, 0.0], [0.4, 0.0]])
        sys_b = jnp.array([[3.0, 0.0], [4.0, 0.0]])
        sep = jnp.concatenate([run_alone(sys_a), run_alone(sys_b)], axis=0)

        opt = ClipByGlobalNorm(max_norm=1.0)
        batched = jnp.concatenate([sys_a, sys_b], axis=0)
        idx = _system_index([0, 0, 1, 1], 2)
        state = opt.init(batched, index_prefix=idx)
        new_updates, _ = opt.update(batched, state)
        npt.assert_allclose(new_updates, sep, atol=1e-6)

    def test_pytree_with_shared_systems(self):
        """L2 norm is taken across leaves *within* a system."""
        positions_idx = _system_index([0, 0, 1], 2)
        cell_idx = _system_index([0, 1], 2)

        # System 0: positions sum-of-squares = 0² + 0² = 0; cell row = (3,0,0) → 9.
        # Total = sqrt(9) = 3 → clip to 1.0 → scale = 1/3.
        # System 1: positions = 1² = 1; cell row = (0,0.5,0) → 0.25. Total = sqrt(1.25) ≈ 1.118 → scale = 1/1.118.
        positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = jnp.array(
            [
                [[3.0, 0.0, 0.0]],  # system 0
                [[0.0, 0.5, 0.0]],  # system 1
            ]
        )

        opt = ClipByGlobalNorm(max_norm=1.0)
        params = (positions, cells)
        index_prefix = (positions_idx, cell_idx)
        state = opt.init(params, index_prefix=index_prefix)
        new_pos, new_cells = opt.update(params, state)[0]

        # System 0 scale = 1/3 (only the cell row contributes).
        npt.assert_allclose(new_pos[0], jnp.zeros(3), atol=1e-6)
        npt.assert_allclose(new_cells[0], jnp.array([[1.0, 0.0, 0.0]]), atol=1e-6)
        # System 1 scale = 1/sqrt(1.25), applied to both leaves.
        scale_b = 1.0 / float(jnp.sqrt(jnp.array(1.25)))
        npt.assert_allclose(new_pos[2], jnp.array([scale_b, 0.0, 0.0]), atol=1e-6)
        npt.assert_allclose(
            new_cells[1], jnp.array([[0.0, 0.5 * scale_b, 0.0]]), atol=1e-6
        )

    def test_state_passes_through_unchanged(self):
        idx = _system_index([0, 0, 1, 1], 2)
        updates = jnp.array([[3.0, 4.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
        opt = ClipByGlobalNorm(max_norm=1.0)
        state = opt.init(updates, index_prefix=idx)
        _, new_state = opt.update(updates, state)
        assert isinstance(state, ClipByGlobalNormState)
        assert isinstance(new_state, ClipByGlobalNormState)
