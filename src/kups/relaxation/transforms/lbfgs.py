# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system L-BFGS preconditioner with ASE-style initial Hessian.

Unlike :func:`kups.relaxation.optax.scale_by_ase_lbfgs`, this version takes
an ``index_prefix`` pytree at init time mapping each parameter element to a
system. Every reduction in the L-BFGS two-loop recursion (the
``s · q`` and ``y · r`` inner products) is taken per-system, the per-slot
weights ``ρᵢ = 1/(yᵢ · sᵢ)`` become per-system scalars stored in a
``Table[K, Array]`` of shape ``(n_systems, memory_size)``, and the
resulting inverse-Hessian approximation is therefore block-diagonal across
systems. Running batched independent systems through this transform is
bit-identical to running them one at a time.
"""

from __future__ import annotations

from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data.index import Index, SupportsSorting
from kups.core.data.table import Table
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass, field, tree_copy
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms._segmented_tree import (
    tree_scale_per_row,
    tree_vdot,
)


@dataclass
class ScaleByAseLbfgsState:
    """State for the per-system ASE-flavor L-BFGS preconditioner.

    Attributes:
        count: Total update steps taken so far (scalar int32).
        params: Last seen parameters, pytree matching ``parameters``.
        updates: Last seen gradients/updates.
        diff_params_memory: Stacked past parameter differences, shape
            ``(memory_size, *leaf_shape)`` per leaf.
        diff_updates_memory: Stacked past update differences, same shape.
        weights_memory: Per-system per-slot ``ρᵢ = 1/(yᵢ · sᵢ)`` weights as
            ``Table[K, Array]`` with data shape ``(n_systems, memory_size)``.
        index_prefix: Tree prefix of the parameter pytree whose leaves are
            ``Index[K]`` objects, captured at init time.
    """

    count: Array
    params: PyTree
    updates: PyTree
    diff_params_memory: PyTree
    diff_updates_memory: PyTree
    weights_memory: Table[SupportsSorting, Array]
    index_prefix: PyTree


@dataclass
class ScaleByAseLbfgs[Params](Optimizer[Params, ScaleByAseLbfgsState]):
    """L-BFGS preconditioner with per-system block-diagonal Hessian.

    With a trivial ``index_prefix`` (one system) this reduces to the same
    algorithm as :func:`kups.relaxation.optax.scale_by_ase_lbfgs`:
    the initial inverse Hessian is ``(1/alpha) * I`` (ASE convention) and
    the recursion buffers ``memory_size`` past ``(diff_params, diff_updates)``
    pairs. With multiple systems, every system maintains its own
    independent inverse-Hessian approximation and its own ``ρᵢ`` weights.

    Attributes:
        memory_size: Number of past difference pairs to store. ``>= 1``.
        alpha: Fixed initial inverse Hessian is ``(1/alpha) * I``. Used as the
            initial scale and as the fallback when ``adaptive_scale`` cannot use
            the curvature pair.
        adaptive_scale: If ``True``, scale the initial inverse Hessian per system
            by ``γ = (s·y)/(y·y)`` (Nocedal & Wright eq. 7.20) from the freshest
            difference pair, falling back to ``1/alpha`` on the first step or when
            the curvature pair is non-positive.
    """

    memory_size: int = field(static=True, default=100)
    alpha: float = field(static=True, default=70.0)
    adaptive_scale: bool = field(static=True, default=False)

    def __post_init__(self) -> None:
        if self.memory_size < 1:
            raise ValueError("memory_size must be >= 1")

    @override
    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> ScaleByAseLbfgsState:
        if index_prefix is None:
            index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), parameters)
        idx_leaves = jax.tree.leaves(
            index_prefix, is_leaf=lambda x: isinstance(x, Index)
        )
        first = next(x for x in idx_leaves if isinstance(x, Index))
        keys = first.keys
        n_systems = len(keys)

        stacked_zero = jax.tree.map(
            lambda leaf: jnp.zeros((self.memory_size,) + leaf.shape, dtype=leaf.dtype),
            parameters,
        )
        return ScaleByAseLbfgsState(
            count=jnp.asarray(0, dtype=jnp.int32),
            params=jax.tree.map(jnp.zeros_like, parameters),
            updates=jax.tree.map(jnp.zeros_like, parameters),
            diff_params_memory=stacked_zero,
            diff_updates_memory=jax.tree.map(jnp.zeros_like, stacked_zero),
            weights_memory=Table(keys, jnp.zeros((n_systems, self.memory_size))),
            index_prefix=tree_copy(index_prefix),
        )

    @override
    def update(
        self,
        updates: Params,
        state: ScaleByAseLbfgsState,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, ScaleByAseLbfgsState]:
        del kwargs
        if params is None:
            raise ValueError("ScaleByASELBFGS.update requires params")
        idx = state.index_prefix
        keys = state.weights_memory.keys
        memory_idx = state.count % self.memory_size
        prev_memory_idx = (state.count - 1) % self.memory_size
        inv_alpha = 1.0 / self.alpha

        # Compute fresh (s, y) differences and corresponding ρ = 1/(y·s).
        diff_params = jax.tree.map(jnp.subtract, params, state.params)
        diff_updates = jax.tree.map(jnp.subtract, updates, state.updates)
        sy = tree_vdot(diff_updates, diff_params, idx).data  # (s·y) per system
        weight = jnp.where(sy == 0.0, 0.0, 1.0 / sy)

        is_first = state.count == 0

        # Per-system initial inverse-Hessian scale γ.
        if self.adaptive_scale:
            yy = tree_vdot(diff_updates, diff_updates, idx).data  # (y·y) per system
            valid = jnp.logical_and(jnp.logical_not(is_first), (yy > 0) & (sy > 0))
            gamma_data = jnp.where(valid, sy / jnp.where(yy > 0, yy, 1.0), inv_alpha)
        else:
            gamma_data = jnp.broadcast_to(
                jnp.asarray(inv_alpha, dtype=sy.dtype), sy.shape
            )
        gamma = Table(keys, gamma_data)

        # Differences are undefined at the very first iteration; stay zero.
        diff_params = jax.tree.map(
            lambda x: jnp.where(is_first, jnp.zeros_like(x), x), diff_params
        )
        diff_updates = jax.tree.map(
            lambda x: jnp.where(is_first, jnp.zeros_like(x), x), diff_updates
        )
        weight = jnp.where(is_first, jnp.zeros_like(weight), weight)

        diff_params_memory = jax.tree.map(
            lambda mem, x: mem.at[prev_memory_idx].set(x),
            state.diff_params_memory,
            diff_params,
        )
        diff_updates_memory = jax.tree.map(
            lambda mem, x: mem.at[prev_memory_idx].set(x),
            state.diff_updates_memory,
            diff_updates,
        )
        weights_data = state.weights_memory.data.at[:, prev_memory_idx].set(weight)

        precond = _precondition_by_lbfgs_segmented(
            updates,
            diff_params_memory,
            diff_updates_memory,
            weights_data,
            gamma=gamma,
            memory_idx=memory_idx,
            index_prefix=idx,
            keys=keys,
        )
        return precond, ScaleByAseLbfgsState(
            count=state.count + 1,
            params=params,
            updates=updates,
            diff_params_memory=diff_params_memory,
            diff_updates_memory=diff_updates_memory,
            weights_memory=state.weights_memory.set_data(weights_data),
            index_prefix=idx,
        )


def _precondition_by_lbfgs_segmented[P](
    updates: P,
    diff_params_memory: PyTree,
    diff_updates_memory: PyTree,
    weights_data: Array,
    gamma: Table[SupportsSorting, Array],
    memory_idx: Array,
    index_prefix: PyTree,
    keys: tuple[SupportsSorting, ...],
) -> P:
    """Per-system version of ``optax._src.transform._precondition_by_lbfgs``.

    Runs Nocedal's two-loop recursion (Algorithm 7.4) with all inner
    products replaced by their per-system equivalents — ``α_i`` and ``β_i``
    are arrays of shape ``(n_systems,)``, and the initial inverse-Hessian
    ``γ_i I`` is applied per system via its own ``gamma`` entry. The
    block-diagonal structure of the resulting approximation across systems
    is what makes the batched run bit-identical to running each system alone.
    """
    memory_size = weights_data.shape[1]
    indices = (memory_idx + jnp.arange(memory_size)) % memory_size

    def right_product(q: P, mem_idx: Array) -> tuple[P, Array]:
        s_i = jax.tree.map(lambda x: x[mem_idx], diff_params_memory)
        y_i = jax.tree.map(lambda x: x[mem_idx], diff_updates_memory)
        rho_i = weights_data[:, mem_idx]
        sq = tree_vdot(s_i, q, index_prefix).data
        alpha = rho_i * sq
        scaled_y = tree_scale_per_row(y_i, Table(keys, alpha), index_prefix)
        new_q = jax.tree.map(jnp.subtract, q, scaled_y)
        return new_q, alpha

    q, alphas = jax.lax.scan(right_product, updates, indices, reverse=True)
    q = tree_scale_per_row(q, gamma, index_prefix)

    def left_product(q: P, args: tuple[Array, Array]) -> tuple[P, Array]:
        mem_idx, alpha = args
        s_i = jax.tree.map(lambda x: x[mem_idx], diff_params_memory)
        y_i = jax.tree.map(lambda x: x[mem_idx], diff_updates_memory)
        rho_i = weights_data[:, mem_idx]
        yq = tree_vdot(y_i, q, index_prefix).data
        beta = rho_i * yq
        coeff = alpha - beta
        scaled_s = tree_scale_per_row(s_i, Table(keys, coeff), index_prefix)
        new_q = jax.tree.map(jnp.add, q, scaled_s)
        return new_q, beta

    q, _ = jax.lax.scan(left_product, q, (indices, alphas))
    return q
