# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-system max_step_size transform."""

import jax.numpy as jnp
import numpy.testing as npt

from kups.core.data.index import Index
from kups.core.typing import ParticleId, SystemId
from kups.relaxation.transforms.max_step_size import MaxStepSize


def _system_index(system_ids: list[int], num_systems: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(num_systems))
    return Index(keys, jnp.array(system_ids), _cls=SystemId)


class TestMaxStepSizeGlobalFallback:
    """When index_prefix is None, every update lives in a single system."""

    def test_global_clip_when_index_prefix_is_none(self):
        # Per-row L2 norms: [0.5, 4.0]; max = 4.0; scale = 1/4.
        updates = jnp.array([[0.5, 0.0, 0.0], [4.0, 0.0, 0.0]])
        transform = MaxStepSize(max_step_size=1.0)
        state = transform.init(updates)
        new_updates, _ = transform.update(updates, state)
        expected = jnp.array([[0.125, 0.0, 0.0], [1.0, 0.0, 0.0]])
        npt.assert_allclose(new_updates, expected, atol=1e-6)

    def test_pytree_global_fallback(self):
        transform = MaxStepSize(max_step_size=1.0)
        state = transform.init(None)
        updates = {
            "a": jnp.array([[0.5, 0.0, 0.0]]),
            "b": jnp.array([[2.0, 0.0, 0.0]]),
        }
        new_updates, _ = transform.update(updates, state)
        npt.assert_allclose(new_updates["a"], jnp.array([[0.25, 0.0, 0.0]]), atol=1e-6)
        npt.assert_allclose(new_updates["b"], jnp.array([[1.0, 0.0, 0.0]]), atol=1e-6)


class TestMaxStepSizePerSystem:
    def test_independent_systems_get_independent_scales(self):
        """Two systems should be clipped independently — system A unchanged, system B scaled."""
        # 4 particles: first two in system 0, last two in system 1
        idx = _system_index([0, 0, 1, 1], 2)
        updates = jnp.array(
            [
                [0.3, 0.0, 0.0],  # system 0: norm 0.3
                [0.4, 0.0, 0.0],  # system 0: norm 0.4 (max)
                [4.0, 0.0, 0.0],  # system 1: norm 4.0 (max)
                [1.0, 0.0, 0.0],  # system 1: norm 1.0
            ]
        )
        transform = MaxStepSize(max_step_size=1.0)
        state = transform.init(updates, index_prefix=idx)
        new_updates, _ = transform.update(updates, state)

        # System 0: max norm 0.4 < 1.0, no scaling.
        # System 1: max norm 4.0, scale by 1.0/4.0 = 0.25.
        expected = jnp.array(
            [
                [0.3, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ]
        )
        npt.assert_allclose(new_updates, expected, atol=1e-6)

    def test_matches_running_systems_separately(self):
        """Batched run must equal concatenation of per-system runs."""
        sys_a = jnp.array([[0.5, 0.0, 0.0], [0.4, 0.0, 0.0]])
        sys_b = jnp.array([[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        transform = MaxStepSize(max_step_size=1.0)

        def run_alone(arr: jnp.ndarray) -> jnp.ndarray:
            idx = _system_index([0] * arr.shape[0], 1)
            state = transform.init(arr, index_prefix=idx)
            new_updates, _ = transform.update(arr, state)
            return new_updates

        per_system = jnp.concatenate([run_alone(sys_a), run_alone(sys_b)], axis=0)

        batched = jnp.concatenate([sys_a, sys_b], axis=0)
        idx_batched = _system_index([0, 0, 1, 1], 2)
        state = transform.init(batched, index_prefix=idx_batched)
        batched_out, _ = transform.update(batched, state)

        npt.assert_allclose(batched_out, per_system, atol=1e-6)

    def test_pytree_with_shared_systems(self):
        """Per-system scale uses the worst-case across all leaves of the same system."""
        # System 0: 2 particles. System 1: 1 particle.
        positions_idx = _system_index([0, 0, 1], 2)
        # System 0: 1 cell. System 1: 1 cell.
        cell_idx = _system_index([0, 1], 2)

        positions = jnp.array(
            [
                [0.5, 0.0, 0.0],  # system 0
                [0.6, 0.0, 0.0],  # system 0 (max norm 0.6)
                [0.5, 0.0, 0.0],  # system 1
            ]
        )
        # cell row norms: system 0 has max 2.0 → dominates positions; system 1 max 0.3.
        cells = jnp.array(
            [
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # system 0
                [[0.3, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],  # system 1
            ]
        )

        transform = MaxStepSize(max_step_size=1.0)
        index_prefix = (positions_idx, cell_idx)
        params = (positions, cells)
        state = transform.init(params, index_prefix=index_prefix)
        new_pos, new_cells = transform.update(params, state)[0]

        # System 0 max = max(0.6, 2.0) = 2.0 → scale 0.5.
        # System 1 max = max(0.5, 0.3) = 0.5 → no scaling.
        npt.assert_allclose(new_pos[0], jnp.array([0.25, 0.0, 0.0]), atol=1e-6)
        npt.assert_allclose(new_pos[1], jnp.array([0.30, 0.0, 0.0]), atol=1e-6)
        npt.assert_allclose(new_pos[2], jnp.array([0.5, 0.0, 0.0]), atol=1e-6)
        npt.assert_allclose(new_cells[0], cells[0] * 0.5, atol=1e-6)
        npt.assert_allclose(new_cells[1], cells[1], atol=1e-6)

    def test_index_prefix_as_tree_prefix(self):
        """A single Index can act as a prefix shared by a whole subtree."""
        # Both leaves use the same particle->system mapping.
        idx = _system_index([0, 0, 1], 2)
        updates = {
            "x": jnp.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            "y": jnp.array([[0.0, 0.05, 0.0], [0.0, 0.05, 0.0], [0.0, 0.5, 0.0]]),
        }
        transform = MaxStepSize(max_step_size=1.0)
        state = transform.init(updates, index_prefix=idx)
        new_updates, _ = transform.update(updates, state)

        # System 0: max across leaves = max(0.1, 0.2, 0.05) = 0.2 → no scale.
        # System 1: max across leaves = max(3.0, 0.5) = 3.0 → scale 1/3.
        npt.assert_allclose(new_updates["x"][0], jnp.array([0.1, 0.0, 0.0]), atol=1e-6)
        npt.assert_allclose(new_updates["x"][2], jnp.array([1.0, 0.0, 0.0]), atol=1e-6)
        npt.assert_allclose(
            new_updates["y"][2], jnp.array([0.0, 0.5 / 3.0, 0.0]), atol=1e-6
        )

    def test_unused_keys_in_index(self):
        """Systems with no elements assigned should not produce NaNs/affect scaling."""
        # Vocabulary has 3 systems but only systems 0 and 2 have elements.
        keys = tuple(SystemId(i) for i in range(3))
        idx = Index(keys, jnp.array([0, 2]), _cls=SystemId)
        updates = jnp.array([[0.2, 0.0, 0.0], [4.0, 0.0, 0.0]])

        transform = MaxStepSize(max_step_size=1.0)
        state = transform.init(updates, index_prefix=idx)
        new_updates, _ = transform.update(updates, state)
        npt.assert_allclose(
            new_updates, jnp.array([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-6
        )

    def test_state_passes_through_unchanged(self):
        idx = _system_index([0, 0, 1, 1], 2)
        updates = jnp.array(
            [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        transform = MaxStepSize(max_step_size=1.0)
        state = transform.init(updates, index_prefix=idx)
        _, new_state = transform.update(updates, state)
        assert isinstance(state.index_prefix, Index)
        assert isinstance(new_state.index_prefix, Index)
        npt.assert_allclose(state.index_prefix.indices, new_state.index_prefix.indices)


class TestParticleIdIndex:
    """Sanity check that per-system clipping works with non-int Index keys."""

    def test_with_particle_id_keys(self):
        keys = tuple(ParticleId(i) for i in range(2))
        idx = Index(keys, jnp.array([0, 0, 1]), _cls=ParticleId)
        updates = jnp.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [4.0, 0.0, 0.0]])
        transform = MaxStepSize(max_step_size=1.0)
        state = transform.init(updates, index_prefix=idx)
        new_updates, _ = transform.update(updates, state)
        npt.assert_allclose(
            new_updates,
            jnp.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            atol=1e-6,
        )
