# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""FIRE optimizer as composable Optax transform."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array


class ScaleByFireState(NamedTuple):
    """State for scale_by_fire transform.

    Attributes:
        velocity: Velocity estimate (PyTree matching params).
        dt: Current adaptive timestep.
        alpha: Current velocity mixing parameter.
        n_pos: Count of consecutive positive power steps.
    """

    velocity: optax.Params
    dt: Array
    alpha: Array
    n_pos: Array


def scale_by_fire(
    dt_start: float = 0.1,
    dt_max: float | None = None,
    dt_min: float | None = None,
    max_step: float | None = 0.2,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    n_min: int = 5,
) -> optax.GradientTransformation:
    """FIRE (Fast Inertial Relaxation Engine) optimizer.

    Composable Optax transform implementing the FIRE algorithm for
    structure relaxation. Can be chained with other transforms.

    Args:
        dt_start: Initial timestep.
        dt_max: Maximum timestep. Defaults to 10 * dt_start.
        dt_min: Minimum timestep. Defaults to dt_start * 1e-4.
        max_step: Maximum step size (clips position updates). Defaults to 0.2 Å.
            Set to None to disable clipping.
        f_inc: Factor to increase dt when making progress.
        f_dec: Factor to decrease dt on bad step.
        alpha_start: Initial velocity mixing parameter.
        f_alpha: Factor to decay alpha when making progress.
        n_min: Minimum positive power steps before increasing dt.

    Returns:
        Optax GradientTransformation implementing FIRE.

    Reference:
        Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).
    """
    if dt_max is None:
        dt_max = 10.0 * dt_start
    if dt_min is None:
        dt_min = dt_start * 1e-4

    def init_fn(params: optax.Params) -> ScaleByFireState:
        return ScaleByFireState(
            velocity=jax.tree.map(jnp.zeros_like, params),
            dt=jnp.array(dt_start),
            alpha=jnp.array(alpha_start),
            n_pos=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: optax.Updates,
        state: ScaleByFireState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByFireState]:
        del params

        # F = -gradient (FIRE uses forces, pointing downhill)
        forces = jax.tree.map(lambda g: -g, updates)

        # Update velocity: v = v + dt * F
        velocity = jax.tree.map(lambda v, f: v + state.dt * f, state.velocity, forces)

        # Compute power: P = F · v (positive when moving downhill)
        power = optax.tree_utils.tree_vdot(forces, velocity)
        positive_power = power > 0.0  # type: ignore

        # Velocity mixing: v = (1-α)v + α|v|F̂
        v_norm = optax.tree_utils.tree_norm(velocity)
        f_norm = optax.tree_utils.tree_norm(forces)
        safe_f_norm = jnp.maximum(f_norm, 1e-10)

        mixed_velocity = jax.tree.map(
            lambda v, f: (1 - state.alpha) * v + state.alpha * v_norm * f / safe_f_norm,
            velocity,
            forces,
        )

        # Adaptive timestep and mixing parameter
        should_increase = jnp.logical_and(positive_power, state.n_pos >= n_min)

        new_dt = jnp.where(
            positive_power,
            jnp.where(should_increase, jnp.minimum(state.dt * f_inc, dt_max), state.dt),
            jnp.maximum(state.dt * f_dec, dt_min),
        )
        new_alpha = jnp.where(
            positive_power,
            jnp.where(should_increase, state.alpha * f_alpha, state.alpha),
            alpha_start,
        )
        new_n_pos = jnp.where(positive_power, state.n_pos + 1, 0)

        # If P > 0: use mixed velocity for next step and position update
        # If P <= 0: reset velocity to zero, no position update
        final_velocity = jax.tree.map(
            lambda v: jnp.where(positive_power, v, jnp.zeros_like(v)),
            mixed_velocity,
        )

        # Position update: step only when making progress (P > 0)
        position_updates = jax.tree.map(
            lambda v: jnp.where(positive_power, state.dt * v, jnp.zeros_like(v)),
            mixed_velocity,
        )

        # Clip position updates to max_step (prevents runaway steps)
        if max_step is not None:
            update_norm = optax.tree_utils.tree_norm(position_updates)
            scale = jnp.minimum(1.0, max_step / jnp.maximum(update_norm, 1e-10))
            position_updates = jax.tree.map(lambda u: u * scale, position_updates)

        return position_updates, ScaleByFireState(
            velocity=final_velocity, dt=new_dt, alpha=new_alpha, n_pos=new_n_pos
        )

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore[arg-type]
