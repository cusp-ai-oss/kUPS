# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""L-BFGS optimizer with ASE-style initial Hessian scaling."""

import jax
import jax.numpy as jnp
import optax
from optax import ScaleByLBFGSState
from optax._src.transform import _precondition_by_lbfgs


def scale_by_ase_lbfgs(
    memory_size: int = 100, alpha: float = 70.0
) -> optax.GradientTransformation:
    """L-BFGS preconditioner with ASE-style initial inverse Hessian.

    Equivalent to ``optax.scale_by_lbfgs`` except the initial Hessian
    approximation is ``(1/alpha) * I`` (following the ASE convention)
    rather than the curvature-based initialization used by default in Optax.

    Args:
        memory_size: Number of past (param, gradient) differences to store.
            Must be >= 1.
        alpha: Initial inverse Hessian is ``(1/alpha) * I``.  In the ASE
            convention this controls the initial step size.

    Returns:
        Optax GradientTransformation applying L-BFGS preconditioning.

    Raises:
        ValueError: If ``memory_size < 1``.

    Example:
        >>> optimizer = optax.chain(
        ...     scale_by_ase_lbfgs(memory_size=10, alpha=70.0),
        ...     optax.scale(-1.0),
        ... )
    """
    if memory_size < 1:
        raise ValueError("memory_size must be >= 1")

    def init_fn(params) -> ScaleByLBFGSState:
        stacked_zero_params = jax.tree.map(
            lambda leaf: jnp.zeros((memory_size,) + leaf.shape, dtype=leaf.dtype),
            params,
        )
        return ScaleByLBFGSState(
            count=jnp.asarray(0, dtype=jnp.int32),
            params=optax.tree.zeros_like(params),
            updates=optax.tree.zeros_like(params),
            diff_params_memory=stacked_zero_params,
            diff_updates_memory=optax.tree.zeros_like(stacked_zero_params),
            weights_memory=jnp.zeros(memory_size),
        )

    def update_fn[P](
        updates: P, state: ScaleByLBFGSState, params: P
    ) -> tuple[P, ScaleByLBFGSState]:
        memory_idx = state.count % memory_size  # type: ignore[arg-type] - optax typing
        prev_memory_idx = (state.count - 1) % memory_size  # type: ignore

        # Update the memory buffers with fresh params and gradients
        diff_params = optax.tree.sub(params, state.params)
        diff_updates = optax.tree.sub(updates, state.updates)
        vdot_diff_params_updates = optax.tree.real(
            optax.tree.vdot(diff_updates, diff_params)
        )
        weight = jnp.where(
            vdot_diff_params_updates == 0.0, 0.0, 1.0 / vdot_diff_params_updates
        )
        # Differences are undefined at first iteration; keep at zero
        diff_params, diff_updates, weight = jax.tree.map(
            lambda x: jnp.where(state.count > 0, x, jnp.zeros_like(x)),  # type: ignore
            (diff_params, diff_updates, weight),
        )
        diff_params_memory, diff_updates_memory, weights_memory = jax.tree.map(
            lambda x, y: x.at[prev_memory_idx].set(y),
            (
                state.diff_params_memory,
                state.diff_updates_memory,
                state.weights_memory,
            ),
            (diff_params, diff_updates, weight),
        )
        identity_scale = 1.0 / alpha

        # Compute the L-BFGS preconditioned update
        precond_updates = _precondition_by_lbfgs(
            updates,  # type: ignore[arg-type] - optax typing
            diff_params_memory,
            diff_updates_memory,
            weights_memory,
            identity_scale,
            memory_idx,  # type: ignore[arg-type] - optax typing
        )
        return precond_updates, ScaleByLBFGSState(  # type: ignore[arg-type] - optax typing
            count=state.count + 1,
            params=params,  # type: ignore[arg-type] - optax typing
            updates=updates,  # type: ignore[arg-type] - optax typing
            diff_params_memory=diff_params_memory,
            diff_updates_memory=diff_updates_memory,
            weights_memory=weights_memory,
        )

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore[arg-type] - optax typing
