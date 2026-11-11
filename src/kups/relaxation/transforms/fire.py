# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system FIRE optimizer transform.

Unlike :func:`kups.relaxation.optax.scale_by_fire`, this version takes an
``index_prefix`` pytree at init time mapping each parameter element to a
system. Every reduction that the FIRE algorithm uses (``F·v`` power,
``||v||``, ``||F||``, position-update norm) is taken per-system, and the
adaptive ``dt`` / ``alpha`` / ``n_pos`` state is stored as a
``Table[K, Array]`` — one entry per system. Running batched independent
systems through this transform is bit-identical to running them one at a
time.

API convention
--------------
Following the optax composability pattern, the ``updates`` argument to
:meth:`ScaleByFire.update` is the *descent direction* (force
``F = -∇L``), not the raw gradient. The transform integrates ``updates``
directly as a force and emits a position step ``Δx`` such that
``apply_updates(x, Δx) = x + Δx`` descends.

If your upstream produces a raw gradient ``∇L``, prepend a sign-flip in
the chain. Any per-system clipping (force cap or per-particle step cap)
composes the same way:

.. code-block:: python

    from kups.relaxation.optimizer import chain
    from kups.relaxation.transforms import (
        ClipByGlobalNorm, MaxStepSize, ScaleByFire,
    )
    import optax

    optimizer = chain(
        optax.scale(-1.0),                 # ∇L  →  F = -∇L
        ScaleByFire(dt_start=0.1),
        ClipByGlobalNorm(max_norm=10.0),   # per-system L2 force cap
        MaxStepSize(max_step_size=0.1),    # per-particle Δx cap
    )
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
    tree_segment_norm,
    tree_vdot,
)


@dataclass
class ScaleByFireState:
    """Optimizer state for the per-system FIRE transform.

    Attributes:
        velocity: Velocity estimate (PyTree matching params).
        dt: Per-system adaptive timestep.
        alpha: Per-system velocity-mixing parameter.
        n_pos: Per-system count of consecutive positive-power steps.
        index_prefix: Tree prefix of the parameter pytree whose leaves are
            ``Index[K]`` objects, captured at init time.
    """

    velocity: PyTree
    dt: Table[SupportsSorting, Array]
    alpha: Table[SupportsSorting, Array]
    n_pos: Table[SupportsSorting, Array]
    index_prefix: PyTree


@dataclass
class ScaleByFire[Params](Optimizer[Params, ScaleByFireState]):
    """FIRE (Fast Inertial Relaxation Engine) optimizer with per-system state.

    Implements Bitzek et al. *Phys. Rev. Lett.* **97**, 170201 (2006) with
    every reduction taken per system over ``index_prefix``, so each system
    adapts its own ``dt``, ``alpha`` and ``n_pos``. Following optax,
    ``updates`` is the force ``F = -∇L``; sign conversion and clipping belong
    in :func:`kups.relaxation.optimizer.chain`, e.g.
    :class:`~kups.relaxation.transforms.ClipByGlobalNorm` before FIRE or
    :class:`~kups.relaxation.transforms.MaxStepSize` after it.

    :class:`~kups.relaxation.transforms.ScaleByFire2` usually needs 1.5–3×
    fewer force calls (Guénolé et al. 2020); this class serves as a baseline.

    Attributes:
        dt_start: Initial timestep.
        dt_max: Maximum timestep. Defaults to ``10 * dt_start``.
        dt_min: Minimum timestep. Defaults to ``dt_start * 1e-4``.
        f_inc: Factor to increase dt when making progress.
        f_dec: Factor to decrease dt on a bad step.
        alpha_start: Initial velocity-mixing parameter.
        f_alpha: Factor to decay alpha when making progress.
        n_min: Minimum positive-power steps before dt is allowed to grow.
    """

    dt_start: float = field(static=True, default=0.1)
    dt_max: float | None = field(static=True, default=None)
    dt_min: float | None = field(static=True, default=None)
    f_inc: float = field(static=True, default=1.1)
    f_dec: float = field(static=True, default=0.5)
    alpha_start: float = field(static=True, default=0.1)
    f_alpha: float = field(static=True, default=0.99)
    n_min: int = field(static=True, default=5)

    @property
    def _dt_max(self) -> float:
        return self.dt_max if self.dt_max is not None else 10.0 * self.dt_start

    @property
    def _dt_min(self) -> float:
        return self.dt_min if self.dt_min is not None else self.dt_start * 1e-4

    @override
    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> ScaleByFireState:
        if index_prefix is None:
            index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), parameters)
        idx_leaves = jax.tree.leaves(
            index_prefix, is_leaf=lambda x: isinstance(x, Index)
        )
        first = next(x for x in idx_leaves if isinstance(x, Index))
        keys = first.keys
        n = len(keys)
        return ScaleByFireState(
            velocity=jax.tree.map(jnp.zeros_like, parameters),
            dt=Table(keys, jnp.full((n,), self.dt_start)),
            alpha=Table(keys, jnp.full((n,), self.alpha_start)),
            n_pos=Table(keys, jnp.zeros((n,), dtype=jnp.int32)),
            index_prefix=tree_copy(index_prefix),
        )

    @override
    def update(
        self,
        updates: Params,
        state: ScaleByFireState,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, ScaleByFireState]:
        del params, kwargs
        idx = state.index_prefix

        # ``updates`` IS the force F = -∇L (optax convention); see module
        # docstring. v <- v + dt[s] · F (per-system dt broadcast per particle).
        velocity = jax.tree.map(
            lambda v, sf: v + sf,
            state.velocity,
            tree_scale_per_row(updates, state.dt, idx),
        )

        # Per-system power P = F · v and its sign.
        power = tree_vdot(updates, velocity, idx)
        positive = power.data > 0.0

        # Per-system L2 norms; safe denominator for ||F||.
        v_norm = tree_segment_norm(velocity, idx)
        f_norm = tree_segment_norm(updates, idx)
        safe_f_norm = jnp.maximum(f_norm.data, 1e-10)

        # Mixed velocity per particle: v' = (1-α)·v + α·||v||/||F|| · F.
        v_scale = state.alpha.set_data(1.0 - state.alpha.data)
        f_scale = state.alpha.set_data(state.alpha.data * v_norm.data / safe_f_norm)
        mixed_velocity = jax.tree.map(
            lambda a, b: a + b,
            tree_scale_per_row(velocity, v_scale, idx),
            tree_scale_per_row(updates, f_scale, idx),
        )

        # Adaptive dt / alpha / n_pos updates per system.
        should_increase = positive & (state.n_pos.data >= self.n_min)
        new_dt_data = jnp.where(
            positive,
            jnp.where(
                should_increase,
                jnp.minimum(state.dt.data * self.f_inc, self._dt_max),
                state.dt.data,
            ),
            jnp.maximum(state.dt.data * self.f_dec, self._dt_min),
        )
        new_alpha_data = jnp.where(
            positive,
            jnp.where(
                should_increase,
                state.alpha.data * self.f_alpha,
                state.alpha.data,
            ),
            jnp.full_like(state.alpha.data, self.alpha_start),
        )
        new_n_pos_data = jnp.where(
            positive, state.n_pos.data + 1, jnp.zeros_like(state.n_pos.data)
        )

        # Per-system gating: zero velocity / position update where P <= 0.
        gate = state.dt.set_data(positive.astype(state.dt.data.dtype))
        final_velocity = tree_scale_per_row(mixed_velocity, gate, idx)
        position_updates = tree_scale_per_row(
            mixed_velocity,
            state.dt.set_data(jnp.where(positive, state.dt.data, 0.0)),
            idx,
        )

        return position_updates, ScaleByFireState(
            velocity=final_velocity,
            dt=state.dt.set_data(new_dt_data),
            alpha=state.alpha.set_data(new_alpha_data),
            n_pos=state.n_pos.set_data(new_n_pos_data),
            index_prefix=idx,
        )
