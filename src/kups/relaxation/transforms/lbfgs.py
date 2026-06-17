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
from kups.core.utils.jax import (
    PyTreeDef,
    dataclass,
    field,
    tree_copy,
    tree_structure,
)
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms._segmented_tree import (
    _layout_and_leaves,
    tree_scale_per_row,
    tree_vdot,
)


@dataclass
class ScaleByAseLbfgsState[Params]:
    """State for the per-system ASE-flavor L-BFGS preconditioner.

    All parameter-shaped fields are stored as flat lists of raw array leaves
    (aligned with ``index_leaves``), never as ``Table``-bearing pytrees, so the
    stacked ``memory_size`` leading axis on the history buffers never re-runs
    ``Table`` validation. ``treedef`` reconstructs the parameter pytree on exit.

    Attributes:
        count: Total update steps taken so far (scalar int32).
        params: Last seen parameter leaves (flat list of arrays).
        updates: Last seen gradient/update leaves (flat list of arrays).
        diff_params_memory: Per leaf, stacked past parameter differences of
            shape ``(memory_size, *leaf_shape)``.
        diff_updates_memory: Per leaf, stacked past update differences.
        weights_memory: Per-system per-slot ``ρᵢ = 1/(yᵢ · sᵢ)`` weights as
            ``Table[K, Array]`` with data shape ``(n_systems, memory_size)``.
        index_prefix: Tree prefix of the parameter pytree whose leaves are
            ``Index[K]`` objects, captured at init. Stored as the prefix (each
            ``Index`` once) rather than the per-leaf expansion so a shared
            ``Index`` buffer is not donated multiple times across a jitted step.
        treedef: Static pytree structure of the parameters, used to unflatten
            the preconditioned update back into the parameter pytree.
    """

    count: Array
    params: list[Array]
    updates: list[Array]
    diff_params_memory: list[Array]
    diff_updates_memory: list[Array]
    weights_memory: Table[SupportsSorting, Array]
    index_prefix: PyTree
    treedef: PyTreeDef[Params] = field(static=True)


@dataclass
class ScaleByAseLbfgs[Params](Optimizer[Params, ScaleByAseLbfgsState[Params]]):
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
    ) -> ScaleByAseLbfgsState[Params]:
        if index_prefix is None:
            index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), parameters)
        # Flatten params to raw leaves with an Index aligned to each (DFS order),
        # matching ``jax.tree.leaves`` so ``update`` can flatten the same way.
        index_leaves, (param_leaves,) = _layout_and_leaves(index_prefix, parameters)
        keys = index_leaves[0].keys
        n_systems = len(keys)

        zeros = [jnp.zeros_like(x) for x in param_leaves]
        stacked = [
            jnp.zeros((self.memory_size,) + x.shape, dtype=x.dtype)
            for x in param_leaves
        ]
        return ScaleByAseLbfgsState(
            count=jnp.asarray(0, dtype=jnp.int32),
            params=zeros,
            updates=[jnp.zeros_like(x) for x in param_leaves],
            diff_params_memory=stacked,
            diff_updates_memory=[jnp.zeros_like(x) for x in stacked],
            weights_memory=Table(keys, jnp.zeros((n_systems, self.memory_size))),
            index_prefix=tree_copy(index_prefix),
            treedef=tree_structure(parameters),
        )

    @override
    def update(
        self,
        updates: Params,
        state: ScaleByAseLbfgsState[Params],
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, ScaleByAseLbfgsState[Params]]:
        del kwargs
        if params is None:
            raise ValueError("ScaleByASELBFGS.update requires params")
        keys = state.weights_memory.keys
        memory_idx = state.count % self.memory_size
        prev_memory_idx = (state.count - 1) % self.memory_size
        inv_alpha = 1.0 / self.alpha

        # Flatten on entry; raw leaves carry no ``Table`` validation. Expanding the
        # stored prefix yields one Index per leaf, aligned with the flattened params.
        idx, (param_leaves, update_leaves) = _layout_and_leaves(
            state.index_prefix, params, updates
        )

        # Compute fresh (s, y) differences and corresponding ρ = 1/(y·s).
        diff_params = jax.tree.map(jnp.subtract, param_leaves, state.params)
        diff_updates = jax.tree.map(jnp.subtract, update_leaves, state.updates)
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
        diff_params = [jnp.where(is_first, jnp.zeros_like(x), x) for x in diff_params]
        diff_updates = [jnp.where(is_first, jnp.zeros_like(x), x) for x in diff_updates]
        weight = jnp.where(is_first, jnp.zeros_like(weight), weight)

        diff_params_memory = [
            mem.at[prev_memory_idx].set(x)
            for mem, x in zip(state.diff_params_memory, diff_params, strict=True)
        ]
        diff_updates_memory = [
            mem.at[prev_memory_idx].set(x)
            for mem, x in zip(state.diff_updates_memory, diff_updates, strict=True)
        ]
        weights_data = state.weights_memory.data.at[:, prev_memory_idx].set(weight)

        precond_leaves = _precondition_by_lbfgs_segmented(
            update_leaves,
            diff_params_memory,
            diff_updates_memory,
            weights_data,
            gamma=gamma,
            memory_idx=memory_idx,
            index_leaves=idx,
            keys=keys,
        )
        # Unflatten on exit, back into the parameter pytree.
        precond = state.treedef.unflatten(precond_leaves)
        return precond, ScaleByAseLbfgsState(
            count=state.count + 1,
            params=param_leaves,
            updates=update_leaves,
            diff_params_memory=diff_params_memory,
            diff_updates_memory=diff_updates_memory,
            weights_memory=state.weights_memory.set_data(weights_data),
            index_prefix=state.index_prefix,
            treedef=state.treedef,
        )


def _precondition_by_lbfgs_segmented(
    update_leaves: list[Array],
    diff_params_memory: list[Array],
    diff_updates_memory: list[Array],
    weights_data: Array,
    gamma: Table[SupportsSorting, Array],
    memory_idx: Array,
    index_leaves: list[Index[SupportsSorting]],
    keys: tuple[SupportsSorting, ...],
) -> list[Array]:
    """Per-system version of ``optax._src.transform._precondition_by_lbfgs``.

    Operates on flat lists of raw array leaves (aligned with ``index_leaves``).
    Runs Nocedal's two-loop recursion (Algorithm 7.4) with all inner products
    replaced by their per-system equivalents — ``α_i`` and ``β_i`` are arrays of
    shape ``(n_systems,)``, and the initial inverse-Hessian ``γ_i I`` is applied
    per system via its own ``gamma`` entry. The block-diagonal structure across
    systems is what makes the batched run bit-identical to running each alone.
    """
    memory_size = weights_data.shape[1]
    indices = (memory_idx + jnp.arange(memory_size)) % memory_size

    def right_product(q: list[Array], mem_idx: Array) -> tuple[list[Array], Array]:
        s_i = [x[mem_idx] for x in diff_params_memory]
        y_i = [x[mem_idx] for x in diff_updates_memory]
        rho_i = weights_data[:, mem_idx]
        sq = tree_vdot(s_i, q, index_leaves).data
        alpha = rho_i * sq
        scaled_y = tree_scale_per_row(y_i, Table(keys, alpha), index_leaves)
        new_q = jax.tree.map(jnp.subtract, q, scaled_y)
        return new_q, alpha

    q, alphas = jax.lax.scan(right_product, update_leaves, indices, reverse=True)
    q = tree_scale_per_row(q, gamma, index_leaves)

    def left_product(
        q: list[Array], args: tuple[Array, Array]
    ) -> tuple[list[Array], Array]:
        mem_idx, alpha = args
        s_i = [x[mem_idx] for x in diff_params_memory]
        y_i = [x[mem_idx] for x in diff_updates_memory]
        rho_i = weights_data[:, mem_idx]
        yq = tree_vdot(y_i, q, index_leaves).data
        beta = rho_i * yq
        coeff = alpha - beta
        scaled_s = tree_scale_per_row(s_i, Table(keys, coeff), index_leaves)
        new_q = jax.tree.map(jnp.add, q, scaled_s)
        return new_q, beta

    q, _ = jax.lax.scan(left_product, q, (indices, alphas))
    return q
