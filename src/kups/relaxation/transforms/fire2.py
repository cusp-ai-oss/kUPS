# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system FIRE 2.0 / ABC-FIRE optimizer transform.

Per-system port of the LAMMPS-style FIRE 2.0 algorithm
(Guénolé et al. 2020) with optional ABC-FIRE bias correction
(Echeverri Restrepo & Andric 2023). Every tree reduction (``F·v``,
``v·v``, ``F·F`` and the ``dmax`` ∞-norm) is taken per-system, and the
adaptive ``dt`` / ``alpha`` / ``n_pos`` state is stored as a
``Table[K, Array]`` — one entry per system. Running batched independent
systems through this transform is bit-identical to running them one at a
time.

API convention
--------------
Following the optax composability pattern, the ``updates`` argument to
:meth:`ScaleByFire2.update` is the *descent direction* (force
``F = -∇L``), not the raw gradient. The transform integrates ``updates``
as a force and emits a position step ``Δx`` such that
``apply_updates(x, Δx) = x + Δx`` descends.

Convert a raw gradient with a sign-flip transform at the head of the
chain. Per-system L2 clipping on the input force and per-particle Δx
clipping on the output both compose around FIRE 2.0 — the built-in
LAMMPS ``dmax`` (``max_step``) is independent of these:

.. code-block:: python

    from kups.relaxation.optimizer import chain
    from kups.relaxation.transforms import (
        ClipByGlobalNorm, MaxStepSize, ScaleByFire2,
    )
    import optax

    optimizer = chain(
        optax.scale(-1.0),                  # ∇L  →  F = -∇L
        ScaleByFire2(dt_start=0.1),         # built-in dmax also active
        ClipByGlobalNorm(max_norm=10.0),    # per-system L2 force cap
        MaxStepSize(max_step_size=0.05),    # extra per-particle Δx cap
    )
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass, field, tree_copy
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms._segmented_tree import (
    tree_clip_per_row,
    tree_scale_per_row,
    tree_segment_max,
    tree_vdot,
    tree_where_per_row,
)


@dataclass
class ScaleByFire2State:
    """Optimizer state for the per-system FIRE 2.0 transform.

    Attributes:
        velocity: Velocity estimate, pytree matching the parameters.
        dt: Per-system adaptive timestep.
        alpha: Per-system velocity-mixing parameter.
        n_pos: Per-system count of consecutive positive-power steps
            (LAMMPS ``ntimestep - last_negative``; also the ABC-FIRE bias
            exponent).
        n_total: Scalar — total update steps taken so far (drives
            ``delaystep_start``).
        index_prefix: Tree prefix of the parameter pytree whose leaves are
            ``Index[K]`` objects, captured at init time.
    """

    velocity: PyTree
    dt: Table[Any, Array]
    alpha: Table[Any, Array]
    n_pos: Table[Any, Array]
    n_total: Array
    index_prefix: PyTree


@dataclass
class ScaleByFire2[Params](Optimizer[Params, ScaleByFire2State]):
    """FIRE 2.0 (with optional ABC-FIRE) with per-system block-diagonal state.

    Per-system port of the LAMMPS-style FIRE 2.0 integrator described in
    Guénolé et al. 2020, with the ABC-FIRE bias correction
    (``use_abc=True``) of Echeverri Restrepo & Andric 2023. With a single
    system this reduces to the algorithm from
    ``kups.relaxation.optax.scale_by_fire2``; with multiple systems each
    system independently adapts its own ``dt`` / ``alpha`` / ``n_pos``
    and sees its own per-system power, norms and ``dmax``.

    The transform follows the optax convention: ``updates`` passed to
    :meth:`update` is interpreted as the force ``F = -∇L`` (the descent
    direction). Sign conversion from a raw gradient and any external
    clipping live in the surrounding
    :func:`kups.relaxation.optimizer.chain` — see the module docstring
    for a worked example. The LAMMPS-style ``dmax`` clip configured via
    :attr:`max_step` is internal to FIRE 2.0 and applies on top of any
    composed clipping.

    Attributes:
        dt_start: Initial timestep.
        dt_max: Maximum timestep (LAMMPS ``dtmax``).
        dt_min: Minimum timestep (LAMMPS ``dtmin``).
        max_step: Per-step displacement bound ``dmax``. ``use_abc=False``
            applies it as a one-shot ∞-norm timestep rescale
            (``max_i |Δx_i| ≤ max_step``); ``use_abc=True`` applies it as
            a per-component velocity clip that persists into the next
            step. ``None`` disables it. The clip is per-system: each
            system's ∞-norm or component limit is computed independently.
        f_inc: Factor to grow ``dt`` (LAMMPS ``dtgrow``).
        f_dec: Factor to shrink ``dt`` (LAMMPS ``dtshrink``).
        alpha_start: Initial velocity-mixing parameter (LAMMPS ``alpha0``).
        f_alpha: Factor to decay ``alpha`` (LAMMPS ``alphashrink``).
        n_min: Minimum positive-power steps before ``dt`` is allowed to
            grow (LAMMPS ``delaystep``).
        use_abc: If True, apply ABC-FIRE bias correction to the mixing.
        halfstepback: If True, apply ``x -= 0.5·new_dt·v_old`` on the
            non-positive-power branch.
        delaystep_start: If True, suppress ``dt`` shrink and ``alpha``
            reset while ``n_total < n_min``.

    References:
        * Guénolé et al., *Comput. Mater. Sci.* **175**, 109584 (2020).
        * Echeverri Restrepo & Andric, *Comput. Mater. Sci.* **218**,
          111978 (2023).
        * LAMMPS ``src/min_fire.cpp`` (develop branch).
    """

    dt_start: float = field(static=True, default=0.1)
    dt_max: float = field(static=True, default=1.0)
    dt_min: float = field(static=True, default=2e-3)
    max_step: float | None = field(static=True, default=0.1)
    f_inc: float = field(static=True, default=1.1)
    f_dec: float = field(static=True, default=0.5)
    alpha_start: float = field(static=True, default=0.25)
    f_alpha: float = field(static=True, default=0.99)
    n_min: int = field(static=True, default=20)
    use_abc: bool = field(static=True, default=False)
    halfstepback: bool = field(static=True, default=True)
    delaystep_start: bool = field(static=True, default=True)

    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> ScaleByFire2State:
        if index_prefix is None:
            index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), parameters)
        idx_leaves = jax.tree.leaves(
            index_prefix, is_leaf=lambda x: isinstance(x, Index)
        )
        first = next(x for x in idx_leaves if isinstance(x, Index))
        keys = first.keys
        n = len(keys)
        return ScaleByFire2State(
            velocity=jax.tree.map(jnp.zeros_like, parameters),
            dt=Table(keys, jnp.full((n,), self.dt_start)),
            alpha=Table(keys, jnp.full((n,), self.alpha_start)),
            n_pos=Table(keys, jnp.zeros((n,), dtype=jnp.int32)),
            n_total=jnp.asarray(0, dtype=jnp.int32),
            index_prefix=tree_copy(index_prefix),
        )

    def update(
        self,
        updates: Params,
        state: ScaleByFire2State,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, ScaleByFire2State]:
        del params, kwargs
        idx = state.index_prefix
        keys = state.dt.keys
        dt_data = state.dt.data
        alpha_data = state.alpha.data
        float_dtype = dt_data.dtype
        n_total = state.n_total + 1

        # ``updates`` IS the force F = -∇L (optax convention); see module
        # docstring. P = v_old · F per system (LAMMPS: vdotfall).
        power = tree_vdot(updates, state.velocity, idx).data
        positive = power > 0.0

        # ----- n_pos (LAMMPS: ntimestep - last_negative) ------------------
        new_n_pos = jnp.where(positive, state.n_pos.data + 1, 0)
        should_increase = positive & (new_n_pos > self.n_min)

        # ----- dt adaptation per system -----------------------------------
        dt_increased = jnp.minimum(dt_data * self.f_inc, self.dt_max)
        dt_decreased = jnp.where(
            dt_data * self.f_dec >= self.dt_min,
            dt_data * self.f_dec,
            dt_data,
        )
        new_dt = jnp.where(
            positive,
            jnp.where(should_increase, dt_increased, dt_data),
            dt_decreased,
        )

        # ----- alpha adaptation per system --------------------------------
        alpha_for_mixing = (
            jnp.maximum(alpha_data, 1e-10) if self.use_abc else alpha_data
        )
        new_alpha = jnp.where(
            positive,
            jnp.where(
                should_increase,
                alpha_for_mixing * self.f_alpha,
                alpha_for_mixing,
            ),
            jnp.full_like(alpha_data, self.alpha_start),
        )

        # ----- delaystep_start: suppress shrink during startup ------------
        if self.delaystep_start:
            in_startup = (~positive) & (n_total < self.n_min)
            new_dt = jnp.where(in_startup, dt_data, new_dt)
            new_alpha = jnp.where(in_startup, alpha_data, new_alpha)

        # ----- Mixing scales (use OLD velocity, per system) --------------
        v_old_sq = tree_vdot(state.velocity, state.velocity, idx).data
        f_sq = tree_vdot(updates, updates, idx).data

        if self.use_abc:
            abc = jnp.where(
                positive,
                1.0 - jnp.power(1.0 - alpha_for_mixing, new_n_pos.astype(float_dtype)),
                1.0,
            )
            safe_abc = jnp.maximum(abc, 1e-30)
            scale1 = jnp.where(positive, (1.0 - alpha_for_mixing) / safe_abc, 1.0)
            scale2_raw = jnp.where(
                f_sq <= 1e-20,  # type: ignore[operator]
                0.0,
                (alpha_for_mixing * jnp.sqrt(v_old_sq / jnp.maximum(f_sq, 1e-20)))
                / safe_abc,
            )
            scale2 = jnp.where(positive, scale2_raw, 0.0)
        else:
            scale1 = 1.0 - alpha_data
            scale2 = jnp.where(
                f_sq <= 1e-20,  # type: ignore[operator]
                0.0,
                alpha_data * jnp.sqrt(v_old_sq / jnp.maximum(f_sq, 1e-20)),
            )

        # ----- dmax: compute dtv (non-ABC only) per system ----------------
        if self.max_step is not None and not self.use_abc:
            abs_v = jax.tree.map(jnp.abs, state.velocity)
            abs_f = jax.tree.map(jnp.abs, updates)
            vmax_pos = tree_segment_max(abs_v, idx).data
            vmax_neg = new_dt * tree_segment_max(abs_f, idx).data
            vmax = jnp.where(positive, vmax_pos, vmax_neg)
            dtv = jnp.where(
                new_dt * vmax > self.max_step,
                self.max_step / jnp.maximum(vmax, 1e-30),
                new_dt,
            )
        else:
            dtv = new_dt

        # ----- Half-step backtrack: -0.5·new_dt·v_old per particle --------
        if self.halfstepback:
            backtrack = tree_scale_per_row(
                state.velocity, Table(keys, -0.5 * new_dt), idx
            )
        else:
            backtrack = jax.tree.map(jnp.zeros_like, state.velocity)

        # ----- v_pre: zero on P<=0, keep on P>0 ---------------------------
        gate = Table(keys, positive.astype(float_dtype))
        v_pre = tree_scale_per_row(state.velocity, gate, idx)

        # ----- Euler-implicit kick: v += dtv · F --------------------------
        scaled_f = tree_scale_per_row(updates, Table(keys, dtv), idx)
        v_int = jax.tree.map(jnp.add, v_pre, scaled_f)

        # ----- Mixing (applied only when P > 0): v = s1·v + s2·F ----------
        v_mixed = jax.tree.map(
            jnp.add,
            tree_scale_per_row(v_int, Table(keys, scale1), idx),
            tree_scale_per_row(updates, Table(keys, scale2), idx),
        )
        new_velocity = tree_where_per_row(Table(keys, positive), v_mixed, v_int, idx)

        # ----- ABC per-component dmax clip (P>0 only) ---------------------
        if self.max_step is not None and self.use_abc:
            effective_limit = jnp.where(
                positive,
                self.max_step / jnp.maximum(dtv, 1e-30),
                jnp.inf,
            )
            new_velocity = tree_clip_per_row(
                new_velocity, Table(keys, effective_limit), idx
            )

        # ----- Position update: dtv · v + (~positive) · backtrack ---------
        main = tree_scale_per_row(new_velocity, Table(keys, dtv), idx)
        not_positive = Table(keys, (1.0 - positive.astype(float_dtype)))
        gated_backtrack = tree_scale_per_row(backtrack, not_positive, idx)
        position_updates = jax.tree.map(jnp.add, main, gated_backtrack)

        return position_updates, ScaleByFire2State(
            velocity=new_velocity,
            dt=state.dt.set_data(new_dt),
            alpha=state.alpha.set_data(new_alpha),
            n_pos=state.n_pos.set_data(new_n_pos),
            n_total=n_total,
            index_prefix=idx,
        )
