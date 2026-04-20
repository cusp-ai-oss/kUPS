# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Optax transform that clips per-particle update norms."""

import functools

import jax
import jax.numpy as jnp
import optax


def max_step_size(max_step_size: float) -> optax.GradientTransformation:
    """Clip updates so no single particle moves more than ``max_step_size``.

    Computes per-particle displacement norms (last axis) across all leaves,
    then uniformly scales the entire update tree so the largest per-particle
    norm does not exceed the limit.

    Args:
        max_step_size: Maximum allowed per-particle displacement norm.

    Returns:
        Stateless Optax GradientTransformation.
    """

    def update_fn[P](updates: P, state: None, *_, **__) -> tuple[P, None]:
        per_particle_size = jax.tree.map(
            functools.partial(jnp.linalg.norm, axis=-1), updates
        )
        max_sizes = jax.tree.map(jnp.max, per_particle_size)
        max_size = jax.tree.reduce(jnp.maximum, max_sizes)
        scale = jnp.minimum(1.0, max_step_size / (max_size + 1e-12))
        updates = jax.tree.map(lambda g: g * scale, updates)
        return updates, state

    return optax.GradientTransformation(init=lambda params: None, update=update_fn)  # type: ignore
