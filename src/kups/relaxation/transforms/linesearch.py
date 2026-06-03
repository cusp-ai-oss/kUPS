# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system Armijo and strong-Wolfe line searches for relaxation.

Two transforms that rescale an incoming descent direction ``d`` by a
per-system step length ``t``. Like every other transform in this package,
the search is taken per system: ``φ(t)``, ``φ'(t)`` and the accept /
backtrack decision are arrays of shape ``(n_systems,)`` keyed on the
system ids, so a system that is already satisfied never throttles the step
of one that is not, and a batched run is bit-identical to running each
system on its own.

* :class:`ScaleByBacktrackingLinesearch` shrinks ``t`` from ``t_init`` until
  the Armijo sufficient-decrease condition holds.
* :class:`ScaleByZoomLinesearch` brackets then bisects to satisfy the strong
  Wolfe conditions (Nocedal & Wright Algorithm 3.5/3.6) and pairs naturally
  with :class:`ScaleByAseLbfgs`.

API convention
--------------
Following the optax composability pattern, the ``updates`` passed to
:meth:`update` is the *descent direction* ``d`` produced by the preceding
transforms (e.g. ``-H⁻¹∇L`` once the L-BFGS preconditioner is sign-flipped),
and the raw gradient ``∇L`` arrives as the ``grad`` keyword. The search emits
``t · d`` per system, so it belongs at the tail of a chain:

.. code-block:: python

    from kups.relaxation.optimizer import chain
    from kups.relaxation.transforms import ScaleByAseLbfgs, ScaleByZoomLinesearch
    import optax

    optimizer = chain(
        ScaleByAseLbfgs(memory_size=10),   # H⁻¹∇L
        optax.scale(-1.0),                 # descent direction d = -H⁻¹∇L
        ScaleByZoomLinesearch(),           # t · d
    )

Each step :class:`kups.relaxation.propagator.RelaxationPropagator` supplies
the current per-system energies (``energies``), the raw gradient (``grad``)
and a ``value_and_grad_fn`` that returns the per-system energies and gradient
at a trial point. A system whose direction is not a descent direction
(``∇L · d ≥ 0``) is left unmoved (``t = 0``).
"""

from __future__ import annotations

from typing import Any, Callable, override

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data.index import Index, SupportsSorting
from kups.core.data.table import Table
from kups.core.typing import PyTree
from kups.core.utils.jax import dataclass, field, tree_copy
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms._segmented_tree import tree_scale_per_row, tree_vdot

type ValueAndGradFn = Callable[[PyTree], tuple[Table[SupportsSorting, Array], PyTree]]
"""Maps trial params to ``(per-system energies, gradient pytree)``."""


@dataclass
class LineSearchState:
    """Line-search state: just the ``index_prefix`` captured at init.

    The search is otherwise stateless — every step restarts from ``t_init``.
    """

    index_prefix: PyTree


def _init_state(parameters: PyTree, index_prefix: PyTree | None) -> LineSearchState:
    if index_prefix is None:
        index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), parameters)
    return LineSearchState(index_prefix=tree_copy(index_prefix))


def _setup(
    direction: PyTree,
    state: LineSearchState,
    params: PyTree | None,
    kwargs: dict[str, Any],
) -> tuple[PyTree, tuple[SupportsSorting, ...], ValueAndGradFn, Array, Array]:
    """Validate inputs and return ``(idx, keys, value_and_grad_fn, φ0, φ'(0))``."""
    grad = kwargs.get("grad")
    energies = kwargs.get("energies")
    value_and_grad_fn = kwargs.get("value_and_grad_fn")
    if params is None or grad is None or energies is None or value_and_grad_fn is None:
        raise ValueError(
            "line search needs params and the `grad`, `energies` and "
            "`value_and_grad_fn` keywords that RelaxationPropagator supplies."
        )
    idx = state.index_prefix
    leaves = jax.tree.leaves(idx, is_leaf=lambda x: isinstance(x, Index))
    keys = next(leaf for leaf in leaves if isinstance(leaf, Index)).keys
    if tuple(energies.keys) != tuple(keys):
        raise ValueError(
            f"total_energies keys {energies.keys} do not match index_prefix "
            f"keys {keys}; init the optimizer with a matching index_prefix."
        )
    return (
        idx,
        keys,
        value_and_grad_fn,
        energies.data,
        tree_vdot(grad, direction, idx).data,
    )


def _trial(
    params: PyTree,
    direction: PyTree,
    t: Array,
    keys: tuple[SupportsSorting, ...],
    idx: PyTree,
) -> PyTree:
    """Trial params ``params + t · d``, with ``t`` applied per system."""
    return jax.tree.map(
        jnp.add, params, tree_scale_per_row(direction, Table(keys, t), idx)
    )


def _backtracking(
    params: PyTree,
    direction: PyTree,
    idx: PyTree,
    keys: tuple[SupportsSorting, ...],
    phi0: Array,
    dphi0: Array,
    value_and_grad_fn: ValueAndGradFn,
    *,
    c1: float,
    decrease_factor: float,
    max_steps: int,
    t_init: float,
) -> Array:
    """Per-system Armijo backtracking; returns the step ``t`` per system.

    Starts every descent system at ``t_init`` and multiplies by
    ``decrease_factor`` until ``φ(t) ≤ φ(0) + c1·t·φ'(0)`` holds, freezing the
    step the moment it does. Non-descent systems (``φ'(0) ≥ 0``) stay at
    ``t = 0``; systems that exhaust ``max_steps`` keep their final shrunk step.
    """
    descent = dphi0 < 0
    t0 = jnp.where(descent, jnp.asarray(t_init, phi0.dtype), 0.0)

    def cond(carry: tuple[Array, Array, Array]) -> Array:
        i, _, done = carry
        return (i < max_steps) & ~jnp.all(done)

    def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        i, t, done = carry
        phi = value_and_grad_fn(_trial(params, direction, t, keys, idx))[0].data
        done = done | (phi <= phi0 + c1 * t * dphi0)
        return i + 1, jnp.where(done, t, t * decrease_factor), done

    _, t, _ = jax.lax.while_loop(cond, body, (jnp.asarray(0, jnp.int32), t0, ~descent))
    return t


def _zoom(
    params: PyTree,
    direction: PyTree,
    idx: PyTree,
    keys: tuple[SupportsSorting, ...],
    phi0: Array,
    dphi0: Array,
    value_and_grad_fn: ValueAndGradFn,
    *,
    c1: float,
    c2: float,
    max_steps: int,
    t_init: float,
    t_max: float,
    expand_factor: float,
) -> Array:
    """Per-system strong-Wolfe search (bracket then bisection zoom).

    Returns the per-system step ``t`` satisfying, where reached within
    ``max_steps``, Armijo (``φ(t) ≤ φ0 + c1·t·φ'(0)``) and curvature
    (``|φ'(t)| ≤ -c2·φ'(0)``). Systems that exhaust the budget fall back to the
    last step that met sufficient decrease (``t = 0`` if none ever did, e.g. a
    non-descent direction). Every quantity carries a leading ``(n_systems,)``
    axis so each system brackets and zooms independently.
    """
    shape, dtype = phi0.shape, phi0.dtype

    def evald(t: Array) -> tuple[Array, Array]:
        energies, grad = value_and_grad_fn(_trial(params, direction, t, keys, idx))
        return energies.data, tree_vdot(grad, direction, idx).data  # φ(t), φ'(t)

    init: dict[str, Array] = {
        "i": jnp.asarray(0, jnp.int32),
        "done": jnp.zeros(shape, bool),
        "t_acc": jnp.zeros(shape, dtype),  # last sufficient-decrease step (fallback)
        "zooming": jnp.zeros(shape, bool),
        "t": jnp.full(shape, t_init, dtype),
        "t_prev": jnp.zeros(shape, dtype),
        "phi_prev": phi0,  # φ at t_prev (=0 ⇒ φ0)
        "lo": jnp.zeros(shape, dtype),
        "hi": jnp.zeros(shape, dtype),
        "phi_lo": phi0,
    }

    def cond(c: dict[str, Array]) -> Array:
        return (c["i"] < max_steps) & ~jnp.all(c["done"])

    def body(c: dict[str, Array]) -> dict[str, Array]:
        t = c["t"]
        phi, dphi = evald(t)
        active = ~c["done"]
        armijo = phi <= phi0 + c1 * t * dphi0
        wolfe = jnp.abs(dphi) <= -c2 * dphi0

        bracketing = active & ~c["zooming"]
        in_zoom = active & c["zooming"]

        # Bracketing transitions (mutually exclusive, in precedence order).
        b_zoom1 = bracketing & ((~armijo) | ((phi >= c["phi_prev"]) & (c["i"] > 0)))
        b_accept = bracketing & (~b_zoom1) & wolfe
        b_zoom2 = bracketing & (~b_zoom1) & (~wolfe) & (dphi >= 0)
        b_expand = bracketing & (~b_zoom1) & (~wolfe) & (dphi < 0)

        # Zoom transitions (``t`` is the current bracket midpoint).
        z_shrink = in_zoom & ((~armijo) | (phi >= c["phi_lo"]))
        z_accept = in_zoom & (~z_shrink) & wolfe
        z_advance = in_zoom & (~z_shrink) & (~wolfe)

        lo, hi, phi_lo = c["lo"], c["hi"], c["phi_lo"]
        # Enter zoom from bracketing.
        lo = jnp.where(b_zoom1, c["t_prev"], jnp.where(b_zoom2, t, lo))
        hi = jnp.where(b_zoom1, t, jnp.where(b_zoom2, c["t_prev"], hi))
        phi_lo = jnp.where(b_zoom1, c["phi_prev"], jnp.where(b_zoom2, phi, phi_lo))
        # Zoom: shrink the upper bound.
        hi = jnp.where(z_shrink, t, hi)
        # Zoom: advance the lower bound (flip hi←lo when the slope points outward).
        flip = z_advance & (dphi * (hi - lo) >= 0)
        hi = jnp.where(flip, lo, hi)
        lo = jnp.where(z_advance, t, lo)
        phi_lo = jnp.where(z_advance, phi, phi_lo)

        accept = b_accept | z_accept
        t_acc = jnp.where(accept, t, jnp.where(active & armijo, t, c["t_acc"]))
        return {
            "i": c["i"] + 1,
            "done": c["done"] | accept,
            "t_acc": t_acc,
            "zooming": c["zooming"] | b_zoom1 | b_zoom2,
            # Next trial: expand while bracketing, else the new bracket midpoint.
            "t": jnp.where(
                b_expand, jnp.minimum(expand_factor * t, t_max), 0.5 * (lo + hi)
            ),
            "t_prev": jnp.where(b_expand, t, c["t_prev"]),
            "phi_prev": jnp.where(b_expand, phi, c["phi_prev"]),
            "lo": lo,
            "hi": hi,
            "phi_lo": phi_lo,
        }

    return jax.lax.while_loop(cond, body, init)["t_acc"]


@dataclass
class ScaleByBacktrackingLinesearch[Params](Optimizer[Params, LineSearchState]):
    """Per-system Armijo backtracking line search.

    Rescales the incoming descent direction by the largest ``t ∈ (0, t_init]``
    of the form ``t_init · decrease_factor**k`` whose step meets the Armijo
    sufficient-decrease condition, deciding per system. See the module
    docstring for the chain convention.

    Attributes:
        c1: Armijo sufficient-decrease constant (``0 < c1 < 1``).
        decrease_factor: Per-iteration shrink factor for ``t`` (``0 < ρ < 1``).
        max_steps: Maximum backtracking iterations.
        t_init: Initial trial step. ``1.0`` suits Newton-scaled directions.
    """

    c1: float = field(static=True, default=1e-4)
    decrease_factor: float = field(static=True, default=0.5)
    max_steps: int = field(static=True, default=20)
    t_init: float = field(static=True, default=1.0)

    @override
    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> LineSearchState:
        return _init_state(parameters, index_prefix)

    @override
    def update(
        self,
        updates: Params,
        state: LineSearchState,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, LineSearchState]:
        idx, keys, value_and_grad_fn, phi0, dphi0 = _setup(
            updates, state, params, kwargs
        )
        t = _backtracking(
            params,
            updates,
            idx,
            keys,
            phi0,
            dphi0,
            value_and_grad_fn,
            c1=self.c1,
            decrease_factor=self.decrease_factor,
            max_steps=self.max_steps,
            t_init=self.t_init,
        )
        return tree_scale_per_row(updates, Table(keys, t), idx), state


@dataclass
class ScaleByZoomLinesearch[Params](Optimizer[Params, LineSearchState]):
    """Per-system strong-Wolfe (zoom) line search.

    Brackets an interval then bisects it to meet the strong Wolfe conditions
    (Nocedal & Wright Algorithm 3.5/3.6), deciding per system. Pairs naturally
    with :class:`ScaleByAseLbfgs`, whose secant updates rely on the curvature
    condition. See the module docstring for the chain convention.

    Attributes:
        c1: Armijo sufficient-decrease constant (``0 < c1 < c2 < 1``).
        c2: Curvature constant. ``0.9`` is the usual quasi-Newton choice.
        max_steps: Maximum combined bracketing + zoom iterations.
        t_init: Initial trial step.
        t_max: Upper bound on ``t`` during the bracketing expansion.
        expand_factor: Growth factor for ``t`` while bracketing.
    """

    c1: float = field(static=True, default=1e-4)
    c2: float = field(static=True, default=0.9)
    max_steps: int = field(static=True, default=20)
    t_init: float = field(static=True, default=1.0)
    t_max: float = field(static=True, default=1e3)
    expand_factor: float = field(static=True, default=2.0)

    @override
    def init(
        self, parameters: Params, index_prefix: PyTree | None = None
    ) -> LineSearchState:
        return _init_state(parameters, index_prefix)

    @override
    def update(
        self,
        updates: Params,
        state: LineSearchState,
        params: Params | None = None,
        **kwargs: Any,
    ) -> tuple[Params, LineSearchState]:
        idx, keys, value_and_grad_fn, phi0, dphi0 = _setup(
            updates, state, params, kwargs
        )
        t = _zoom(
            params,
            updates,
            idx,
            keys,
            phi0,
            dphi0,
            value_and_grad_fn,
            c1=self.c1,
            c2=self.c2,
            max_steps=self.max_steps,
            t_init=self.t_init,
            t_max=self.t_max,
            expand_factor=self.expand_factor,
        )
        return tree_scale_per_row(updates, Table(keys, t), idx), state
