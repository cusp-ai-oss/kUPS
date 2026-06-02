# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relaxation optimizer config-spec builder utilities."""

import jax.numpy as jnp
import optax
import pytest

from kups.relaxation.config import (
    get_transform,
    get_transformations,
    make_optimizer,
)
from kups.relaxation.optimizer import ChainOptimizer
from kups.relaxation.transforms import (
    ClipByGlobalNorm,
    MaxStepSize,
    ScaleByAseLbfgs,
    ScaleByFire,
)


class TestGetTransform:
    def test_string_optax_transform(self):
        t = get_transform("identity")
        assert isinstance(t, optax.GradientTransformation)

    def test_dict_optax_transform_with_kwargs(self):
        t = get_transform({"transform": "sgd", "learning_rate": 0.01})
        assert isinstance(t, optax.GradientTransformation)

    def test_custom_scale_by_fire(self):
        t = get_transform("scale_by_fire")
        assert isinstance(t, ScaleByFire)

    def test_custom_max_step_size(self):
        t = get_transform({"transform": "max_step_size", "max_step_size": 0.1})
        assert isinstance(t, MaxStepSize)

    def test_custom_scale_by_ase_lbfgs(self):
        t = get_transform("scale_by_ase_lbfgs")
        assert isinstance(t, ScaleByAseLbfgs)

    def test_clip_by_global_norm_resolves_to_kups(self):
        """The kups per-system clip overrides optax's tree-global one."""
        t = get_transform({"transform": "clip_by_global_norm", "max_norm": 1.0})
        assert isinstance(t, ClipByGlobalNorm)

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transformation"):
            get_transform("this_does_not_exist")

    def test_dict_does_not_mutate_input(self):
        config = {"transform": "sgd", "learning_rate": 0.01}
        original = config.copy()
        get_transform(config)
        assert config == original

    def test_dict_kwargs_forwarded_to_custom(self):
        """Kwargs from dict config land on the dataclass fields."""
        t = get_transform({"transform": "scale_by_fire", "dt_start": 0.05})
        assert isinstance(t, ScaleByFire)
        assert t.dt_start == 0.05


class TestGetTransformations:
    def test_empty_list(self):
        assert get_transformations([]) == []

    def test_multiple_transforms(self):
        result = get_transformations(
            [
                "identity",
                {"transform": "scale_by_fire"},
                {"transform": "clip_by_global_norm", "max_norm": 1.0},
            ]
        )
        assert len(result) == 3
        assert isinstance(result[0], optax.GradientTransformation)
        assert isinstance(result[1], ScaleByFire)
        assert isinstance(result[2], ClipByGlobalNorm)


class TestMakeOptimizer:
    def test_returns_chain_optimizer(self):
        opt = make_optimizer(["identity"])
        assert isinstance(opt, ChainOptimizer)

    def test_chained_with_only_custom_transforms(self):
        opt = make_optimizer(
            [
                {"transform": "clip_by_global_norm", "max_norm": 1.0},
                {"transform": "scale_by_fire"},
            ]
        )
        assert isinstance(opt, ChainOptimizer)

    def test_chained_optimizer_produces_finite_updates(self):
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

    def test_chain_mixes_optax_and_custom(self):
        """An optax transform and a kups Optimizer can be chained together."""
        opt = make_optimizer(
            [
                "identity",
                {"transform": "scale_by_fire", "dt_start": 0.1},
            ]
        )
        params = jnp.array([1.0, 2.0])
        state = opt.init(params)
        gradient = jnp.array([-0.5, -1.0])
        updates, _ = opt.update(gradient, state, params)
        assert updates.shape == (2,)
        assert jnp.all(jnp.isfinite(updates))
