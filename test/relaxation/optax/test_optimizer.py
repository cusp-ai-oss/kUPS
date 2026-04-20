# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optimizer builder utilities."""

import jax.numpy as jnp
import optax
import pytest

from kups.relaxation.optax.optimizer import (
    get_transform,
    get_transformations,
    make_optimizer,
)

from ...clear_cache import clear_cache  # noqa: F401


class TestGetTransform:
    """Tests for get_transform."""

    def test_string_optax_transform(self):
        t = get_transform("identity")
        assert isinstance(t, optax.GradientTransformation)

    def test_dict_optax_transform_with_kwargs(self):
        t = get_transform({"transform": "sgd", "learning_rate": 0.01})
        assert isinstance(t, optax.GradientTransformation)

    def test_custom_scale_by_fire(self):
        t = get_transform("scale_by_fire")
        assert isinstance(t, optax.GradientTransformation)

    def test_custom_max_step_size(self):
        t = get_transform({"transform": "max_step_size", "max_step_size": 0.1})
        assert isinstance(t, optax.GradientTransformation)

    def test_custom_scale_by_ase_lbfgs(self):
        t = get_transform("scale_by_ase_lbfgs")
        assert isinstance(t, optax.GradientTransformation)

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transformation"):
            get_transform("this_does_not_exist")

    def test_dict_does_not_mutate_input(self):
        config = {"transform": "sgd", "learning_rate": 0.01}
        original = config.copy()
        get_transform(config)
        assert config == original

    def test_dict_kwargs_forwarded(self):
        """Kwargs from dict config should be forwarded to the constructor."""
        t = get_transform({"transform": "scale_by_fire", "dt_start": 0.05})
        params = jnp.array([1.0])
        state = t.init(params)
        # dt_start=0.05 should have been forwarded
        assert float(state.dt) == pytest.approx(0.05)


class TestGetTransformations:
    """Tests for get_transformations."""

    def test_empty_list(self):
        assert get_transformations([]) == []

    def test_multiple_transforms(self):
        result = get_transformations(["identity", {"transform": "scale_by_fire"}])
        assert len(result) == 2
        assert all(isinstance(t, optax.GradientTransformation) for t in result)


class TestMakeOptimizer:
    """Tests for make_optimizer."""

    def test_returns_gradient_transformation(self):
        opt = make_optimizer(["identity"])
        assert isinstance(opt, optax.GradientTransformation)

    def test_chained_optimizer(self):
        opt = make_optimizer(
            [
                {"transform": "clip_by_global_norm", "max_norm": 1.0},
                {"transform": "scale_by_fire"},
            ]
        )
        assert isinstance(opt, optax.GradientTransformation)

    def test_chained_optimizer_produces_updates(self):
        """Chained optimizer should produce finite updates."""
        opt = make_optimizer(
            [
                {"transform": "clip_by_global_norm", "max_norm": 1.0},
                {"transform": "scale_by_fire", "dt_start": 0.1},
            ]
        )
        params = jnp.array([1.0, 2.0])
        state = opt.init(params)
        gradient = jnp.array([-0.5, -1.0])
        updates, _ = opt.update(gradient, state, params)
        assert updates.shape == (2,)
        assert jnp.all(jnp.isfinite(updates))
