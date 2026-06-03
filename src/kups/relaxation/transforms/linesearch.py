# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Per-system line searches that reproduce ASE's implementations.

Two transforms that rescale an incoming descent direction ``d`` by a
per-system step length ``t``. Like every other transform in this package,
the search is taken per system: ``φ(t)``, ``φ'(t)`` and the accept /
backtrack decision are arrays of shape ``(n_systems,)`` keyed on the
system ids, so a system that is already satisfied never throttles the step
of one that is not, and a batched run is bit-identical to running each
system on its own.

The two searches reproduce ASE's two line searches step-for-step (a single
system matches ASE to floating-point tolerance):

* :class:`ScaleByBacktrackingLinesearch` is :class:`ase.utils.linesearcharmijo`
  ``LineSearchArmijo``: quadratic-interpolation backtracking enforcing the
  Armijo sufficient-decrease condition, with the ``t ← max(t_quad, t/10)``
  floor.
* :class:`ScaleByMoreThuenteLinesearch` is :class:`ase.utils.linesearch` ``LineSearch``:
  the MINPACK More–Thuente search (``dcsrch``/``dcstep``) targeting the strong
  Wolfe conditions, which ASE's ``LBFGS(use_line_search=True)`` uses and which
  pairs naturally with :class:`ScaleByAseLbfgs`.

ASE bakes a per-atom ``maxstep`` clamp into the search (via ``determine_step``);
here that clamp is left to the separate :class:`MaxStepSize` transform, so
``determine_step`` is the identity and the searches reproduce the pure
algorithms. ASE also fixes the initial trial step at ``1.0``; ``t_init = 1.0``
(the default) reproduces ASE exactly.

API convention
--------------
Following the optax composability pattern, the ``updates`` passed to
:meth:`update` is the *descent direction* ``d`` produced by the preceding
transforms (e.g. ``-H⁻¹∇L`` once the L-BFGS preconditioner is sign-flipped),
and the raw gradient ``∇L`` arrives as the ``grad`` keyword. The search emits
``t · d`` per system, so it belongs at the tail of a chain:

.. code-block:: python

    from kups.relaxation.optimizer import chain
    from kups.relaxation.transforms import ScaleByAseLbfgs, ScaleByMoreThuenteLinesearch
    import optax

    optimizer = chain(
        ScaleByAseLbfgs(memory_size=10),   # H⁻¹∇L
        optax.scale(-1.0),                 # descent direction d = -H⁻¹∇L
        ScaleByMoreThuenteLinesearch(),           # t · d
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
    """Line-search state.

    Attributes:
        index_prefix: Tree prefix whose leaves are ``Index`` objects, captured at
            init and used to take every reduction per system.
        prev_phi0: Previous step's per-system energies, ``NaN`` until a step
            records them. :class:`ScaleByBacktrackingLinesearch` uses them for its
            Nocedal & Wright eq. 3.60 initial-step estimate;
            :class:`ScaleByMoreThuenteLinesearch` leaves them untouched.
    """

    index_prefix: PyTree
    prev_phi0: Array


def _init_state(parameters: PyTree, index_prefix: PyTree | None) -> LineSearchState:
    if index_prefix is None:
        index_prefix = jax.tree.map(lambda x: Index.new((0,) * len(x)), parameters)
    leaves = jax.tree.leaves(index_prefix, is_leaf=lambda x: isinstance(x, Index))
    keys = next(leaf for leaf in leaves if isinstance(leaf, Index)).keys
    return LineSearchState(
        index_prefix=tree_copy(index_prefix), prev_phi0=jnp.full(len(keys), jnp.nan)
    )


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


def _armijo(
    params: PyTree,
    direction: PyTree,
    idx: PyTree,
    keys: tuple[SupportsSorting, ...],
    phi0: Array,
    dphi0: Array,
    value_and_grad_fn: ValueAndGradFn,
    func_old: Array,
    *,
    c1: float,
    a_min: float,
    a_max: float,
    max_steps: int,
    t_init: float,
) -> Array:
    """Per-system Armijo backtracking — ASE ``LineSearchArmijo`` (no maxstep).

    Each step evaluates ``φ(t)``; while Armijo (``φ(t) ≤ φ(0) + c1·t·φ'(0)``)
    fails it replaces ``t`` by the minimiser of the quadratic through
    ``φ(0), φ'(0), φ(t)`` floored at ``t/10`` (ASE: ``t = max(t_quad, t/10)``).
    The first trial is the Nocedal & Wright eq. 3.60 estimate
    ``2·(φ(0) − φ_old)/φ'(0)`` when ``φ_old`` (the previous step's energy) is
    available and lands in ``[a_min, a_max]``, else ``t_init``; either way it is
    rounded to ``1.0`` when within ``0.5`` (ASE's rule). Non-descent systems
    (``φ'(0) ≥ 0``) stay at ``t = 0``. ASE raises once ``t < a_min``; per system
    that path instead freezes the step.
    """
    n = phi0.shape[0]
    dt = phi0.dtype
    descent = dphi0 < 0.0

    def phi(t: Array) -> Array:
        energies, _ = value_and_grad_fn(_trial(params, direction, t, keys, idx))
        return energies.data.astype(dt)

    estimate = 2.0 * (phi0 - func_old.astype(dt)) / dphi0  # NaN on the first step
    a1_init = jnp.where(
        (estimate >= a_min) & (estimate <= a_max), estimate, jnp.asarray(t_init, dt)
    )
    # Cast back to φ's dtype: φ' may carry higher precision, which must not leak
    # into the while_loop carry (see _more_thuente for the same guard).
    a1_init = jnp.where(
        jnp.abs(a1_init - 1.0) <= 0.5, jnp.asarray(1.0, dt), a1_init
    ).astype(dt)
    init = {
        "i": jnp.asarray(0, jnp.int32),
        "done": ~descent,
        "t": jnp.zeros(n, dt),  # accepted step; 0 for non-descent
        "a": jnp.where(descent, a1_init, jnp.zeros(n, dt)),  # current trial
    }

    def cond(c: dict[str, Array]) -> Array:
        return (c["i"] < max_steps) & ~jnp.all(c["done"])

    def body(c: dict[str, Array]) -> dict[str, Array]:
        a = c["a"]
        ph = phi(a)
        suff = ph <= phi0 + c1 * a * dphi0
        below = a < a_min  # underflowed the step floor
        accept = (~c["done"]) & (~below) & suff
        t = jnp.where(accept, a, c["t"])
        a_quad = -(dphi0 * a) / (2.0 * ((ph - phi0) / a - dphi0))
        a_next = jnp.maximum(a_quad, a / 10.0)
        proceed = (~c["done"]) & (~below) & (~suff)
        return {
            "i": c["i"] + 1,
            "done": c["done"] | below | suff,
            "t": t,
            "a": jnp.where(proceed, a_next, a).astype(dt),
        }

    return jax.lax.while_loop(cond, body, init)["t"]


def _dcstep(
    stx: Array,
    fx: Array,
    gx: Array,
    sty: Array,
    fy: Array,
    gy: Array,
    stp: Array,
    fp: Array,
    gp: Array,
    brackt: Array,
    lo: Array,
    hi: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Per-system MINPACK ``dcstep`` (ASE ``LineSearch.update``).

    Computes a safeguarded trial step from the best step so far (``stx``), the
    other endpoint (``sty``) and the latest trial (``stp``), selecting among four
    cubic/secant interpolation cases. ``lo``/``hi`` are the current bracket bounds
    (``stmin``/``stmax``), not the global step limits. Returns the updated
    ``(stx, sty, stp, gx, fx, gy, fy, brackt)``.
    """
    sgnd = gp * jnp.sign(gx)

    def cubic(theta: Array, ga: Array, gb: Array) -> Array:
        s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(ga)), jnp.abs(gb))
        return s * jnp.sqrt((theta / s) ** 2 - (ga / s) * (gb / s))

    # case 1: higher function value -> minimum bracketed.
    theta1 = 3.0 * (fx - fp) / (stp - stx) + gx + gp
    g1 = jnp.where(stp < stx, -cubic(theta1, gx, gp), cubic(theta1, gx, gp))
    r1 = ((g1 - gx) + theta1) / (((g1 - gx) + g1) + gp)
    stpc1 = stx + r1 * (stp - stx)
    stpq1 = stx + ((gx / ((fx - fp) / (stp - stx) + gx)) / 2.0) * (stp - stx)
    stpf1 = jnp.where(
        jnp.abs(stpc1 - stx) < jnp.abs(stpq1 - stx),
        stpc1,
        stpc1 + (stpq1 - stpc1) / 2.0,
    )

    # case 2: lower value, opposite-sign derivatives -> minimum bracketed.
    theta2 = 3.0 * (fx - fp) / (stp - stx) + gx + gp
    g2 = jnp.where(stp > stx, -cubic(theta2, gx, gp), cubic(theta2, gx, gp))
    r2 = ((g2 - gp) + theta2) / (((g2 - gp) + g2) + gx)
    stpc2 = stp + r2 * (stx - stp)
    stpq2 = stp + (gp / (gp - gx)) * (stx - stp)
    stpf2 = jnp.where(jnp.abs(stpc2 - stp) > jnp.abs(stpq2 - stp), stpc2, stpq2)

    # case 3: lower value, same-sign derivatives, magnitude decreasing.
    theta3 = 3.0 * (fx - fp) / (stp - stx) + gx + gp
    s3 = jnp.maximum(jnp.maximum(jnp.abs(theta3), jnp.abs(gx)), jnp.abs(gp))
    g3 = s3 * jnp.sqrt(jnp.maximum(0.0, (theta3 / s3) ** 2 - (gx / s3) * (gp / s3)))
    g3 = jnp.where(stp > stx, -g3, g3)
    r3 = ((g3 - gp) + theta3) / ((g3 + (gx - gp)) + g3)
    stpc3 = jnp.where(
        (r3 < 0.0) & (g3 != 0.0),
        stp + r3 * (stx - stp),
        jnp.where(stp > stx, hi, lo),
    )
    stpq3 = stp + (gp / (gp - gx)) * (stx - stp)
    closer = jnp.abs(stpc3 - stp) < jnp.abs(stpq3 - stp)
    stpf3_br = jnp.where(closer, stpc3, stpq3)
    stpf3_br = jnp.where(
        stp > stx,
        jnp.minimum(stp + 0.66 * (sty - stp), stpf3_br),
        jnp.maximum(stp + 0.66 * (sty - stp), stpf3_br),
    )
    stpf3_nb = jnp.clip(jnp.where(~closer, stpc3, stpq3), lo, hi)
    stpf3 = jnp.where(brackt, stpf3_br, stpf3_nb)

    # case 4: lower value, same-sign derivatives, magnitude not decreasing.
    theta4 = 3.0 * (fp - fy) / (sty - stp) + gy + gp
    g4 = jnp.where(stp > sty, -cubic(theta4, gy, gp), cubic(theta4, gy, gp))
    r4 = ((g4 - gp) + theta4) / (((g4 - gp) + g4) + gy)
    stpf4 = jnp.where(brackt, stp + r4 * (sty - stp), jnp.where(stp > stx, hi, lo))

    m1 = fp > fx
    m2 = (~m1) & (sgnd < 0.0)
    m3 = (~m1) & (~(sgnd < 0.0)) & (jnp.abs(gp) < jnp.abs(gx))
    stpf = jnp.where(m1, stpf1, jnp.where(m2, stpf2, jnp.where(m3, stpf3, stpf4)))
    brackt = brackt | m1 | m2

    # Update the interval bounds (using the pre-update best point stx/fx/gx).
    nstx = jnp.where(m1, stx, stp)
    nfx = jnp.where(m1, fx, fp)
    ngx = jnp.where(m1, gx, gp)
    nsty = jnp.where(m1, stp, jnp.where(sgnd < 0.0, stx, sty))
    nfy = jnp.where(m1, fp, jnp.where(sgnd < 0.0, fx, fy))
    ngy = jnp.where(m1, gp, jnp.where(sgnd < 0.0, gx, gy))
    return nstx, nsty, stpf, ngx, nfx, ngy, nfy, brackt


def _more_thuente(
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
    t_init: float,
    stpmin: float,
    stpmax: float,
    xtol: float,
    xtrapl: float,
    xtrapu: float,
    max_steps: int,
) -> Array:
    """Per-system MINPACK More–Thuente search — ASE ``LineSearch`` (no maxstep).

    Drives ``dcsrch``: it brackets a step satisfying the strong Wolfe conditions
    (``φ(t) ≤ φ0 + c1·t·φ'(0)`` and ``|φ'(t)| ≤ -c2·φ'(0)``) and refines it with
    :func:`_dcstep`. Every quantity carries a leading ``(n_systems,)`` axis so
    each system brackets independently. Non-descent systems (``φ'(0) ≥ 0``) stay
    at ``t = 0``; a system that exhausts ``max_steps`` falls back to its best
    bracket endpoint.
    """
    n = phi0.shape[0]
    dt = phi0.dtype
    descent = dphi0 < 0.0
    # Run in φ's dtype so the while_loop carry stays single-dtype even when
    # energies and gradient projections differ in precision.
    ginit, finit, gtest = dphi0.astype(dt), phi0, c1 * dphi0.astype(dt)

    def evald(t: Array) -> tuple[Array, Array]:
        energies, grad = value_and_grad_fn(_trial(params, direction, t, keys, idx))
        return energies.data.astype(dt), tree_vdot(grad, direction, idx).data.astype(dt)

    init = {
        "i": jnp.asarray(0, jnp.int32),
        "done": ~descent,
        "t": jnp.zeros(n, dt),  # accepted step (strong-Wolfe point)
        "pending": jnp.zeros(n, bool),  # flagged to stop after one more evaluation
        "stp": jnp.where(descent, jnp.asarray(t_init, dt), jnp.zeros(n, dt)),
        "stx": jnp.zeros(n, dt),
        "fx": finit,
        "gx": ginit,
        "sty": jnp.zeros(n, dt),
        "fy": finit,
        "gy": ginit,
        "stmin": jnp.zeros(n, dt),
        "stmax": jnp.full(n, t_init + xtrapu * t_init, dt),
        "brackt": jnp.zeros(n, bool),
        "width": jnp.full(n, stpmax - stpmin, dt),
        "width1": jnp.full(n, (stpmax - stpmin) / 0.5, dt),
    }

    def cond(c: dict[str, Array]) -> Array:
        return (c["i"] < max_steps) & ~jnp.all(c["done"])

    def body(c: dict[str, Array]) -> dict[str, Array]:
        stp = c["stp"]
        f, g = evald(stp)
        ftest = finit + stp * gtest
        warn = (
            (c["brackt"] & ((stp <= c["stmin"]) | (stp >= c["stmax"])))
            | (c["brackt"] & ((c["stmax"] - c["stmin"]) <= xtol * c["stmax"]))
            | ((stp == stpmax) & (f <= ftest) & (g <= gtest))
            | ((stp == stpmin) & ((f > ftest) | (g >= gtest)))
        )
        conv = (f <= ftest) & (jnp.abs(g) <= c2 * (-ginit))
        terminate = warn | conv | c["pending"]
        t = jnp.where((~c["done"]) & terminate, stp, c["t"])

        ustx, usty, ustp, ugx, ufx, ugy, ufy, ubr = _dcstep(
            c["stx"],
            c["fx"],
            c["gx"],
            c["sty"],
            c["fy"],
            c["gy"],
            stp,
            f,
            g,
            c["brackt"],
            c["stmin"],
            c["stmax"],
        )
        do_bisect = ubr & (jnp.abs(usty - ustx) >= 0.66 * c["width1"])
        ustp = jnp.where(do_bisect, ustx + 0.5 * (usty - ustx), ustp)
        nwidth1 = jnp.where(ubr, c["width"], c["width1"])
        nwidth = jnp.where(ubr, jnp.abs(usty - ustx), c["width"])
        nstmin = jnp.where(ubr, jnp.minimum(ustx, usty), ustp + xtrapl * (ustp - ustx))
        nstmax = jnp.where(ubr, jnp.maximum(ustx, usty), ustp + xtrapu * (ustp - ustx))
        ustp = jnp.clip(ustp, stpmin, stpmax)
        no_upd = (ustx == ustp) & (ustp == stpmax) & (nstmin > stpmax)
        fallback = (
            (ubr & (ustp < nstmin))
            | (ustp >= nstmax)
            | (ubr & ((nstmax - nstmin) < xtol * nstmax))
        )
        ustp = jnp.where(fallback, ustx, ustp)

        proceed = (~c["done"]) & (~terminate)

        def sel(new: Array, old: Array) -> Array:
            return jnp.where(proceed, new, old)

        return {
            "i": c["i"] + 1,
            "done": c["done"] | terminate,
            "t": t,
            "pending": proceed & no_upd,
            "stp": jnp.where(proceed, ustp, stp).astype(dt),
            "stx": sel(ustx, c["stx"]),
            "fx": sel(ufx, c["fx"]),
            "gx": sel(ugx, c["gx"]),
            "sty": sel(usty, c["sty"]),
            "fy": sel(ufy, c["fy"]),
            "gy": sel(ugy, c["gy"]),
            "stmin": sel(nstmin, c["stmin"]),
            "stmax": sel(nstmax, c["stmax"]),
            "brackt": jnp.where(proceed, ubr, c["brackt"]),
            "width": sel(nwidth, c["width"]),
            "width1": sel(nwidth1, c["width1"]),
        }

    fin = jax.lax.while_loop(cond, body, init)
    return jnp.where(fin["done"] | ~descent, fin["t"], fin["stx"])


@dataclass
class ScaleByBacktrackingLinesearch[Params](Optimizer[Params, LineSearchState]):
    """Per-system Armijo backtracking line search — ASE ``LineSearchArmijo``.

    Rescales the incoming descent direction by a per-system step ``t``: it shrinks
    ``t`` by quadratic interpolation (floored at ``t/10``) until the step meets the
    Armijo sufficient-decrease condition, deciding per system. The first trial uses
    a Nocedal & Wright eq. 3.60 estimate from the previous step's energy (clamped
    to ``[a_min, a_max]``, else ``t_init``). See the module docstring for the chain
    convention.

    Attributes:
        c1: Armijo sufficient-decrease constant (``0 < c1 < 0.5``).
        a_min: Smallest allowed step; the search freezes a system that reaches it.
            Also the lower clamp on the eq. 3.60 initial-step estimate.
        a_max: Upper clamp on the eq. 3.60 initial-step estimate; above it the
            first trial falls back to ``t_init``.
        max_steps: Maximum backtracking iterations.
        t_init: Initial trial step when no estimate applies, rounded to ``1.0``
            when within ``0.5``. ``1.0`` suits Newton-scaled directions.
    """

    c1: float = field(static=True, default=0.1)
    a_min: float = field(static=True, default=1e-10)
    a_max: float = field(static=True, default=2.0)
    max_steps: int = field(static=True, default=50)
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
        t = _armijo(
            params,
            updates,
            idx,
            keys,
            phi0,
            dphi0,
            value_and_grad_fn,
            state.prev_phi0,
            c1=self.c1,
            a_min=self.a_min,
            a_max=self.a_max,
            max_steps=self.max_steps,
            t_init=self.t_init,
        )
        new_state = LineSearchState(index_prefix=state.index_prefix, prev_phi0=phi0)
        return tree_scale_per_row(updates, Table(keys, t), idx), new_state


@dataclass
class ScaleByMoreThuenteLinesearch[Params](Optimizer[Params, LineSearchState]):
    """Per-system More–Thuente line search — ASE ``LineSearch``.

    Brackets and refines a step meeting the strong Wolfe conditions via the
    MINPACK ``dcsrch``/``dcstep`` algorithm, deciding per system. This is the
    search ASE's ``LBFGS(use_line_search=True)`` uses; it pairs naturally with
    :class:`ScaleByAseLbfgs`, whose secant updates rely on the curvature
    condition. See the module docstring for the chain convention.

    Attributes:
        c1: Armijo sufficient-decrease constant (``0 < c1 < c2 < 1``).
        c2: Curvature constant (``c1 < c2 < 1``); smaller demands a flatter slope.
        t_init: Initial trial step.
        stpmin: Smallest step the search will return.
        stpmax: Largest step the search will return.
        xtol: Relative bracket-width tolerance ending the search.
        xtrapl: Lower extrapolation factor while bracketing.
        xtrapu: Upper extrapolation factor while bracketing.
        max_steps: Maximum combined bracketing + zoom iterations.
    """

    c1: float = field(static=True, default=0.23)
    c2: float = field(static=True, default=0.46)
    t_init: float = field(static=True, default=1.0)
    stpmin: float = field(static=True, default=1e-8)
    stpmax: float = field(static=True, default=50.0)
    xtol: float = field(static=True, default=1e-14)
    xtrapl: float = field(static=True, default=1.1)
    xtrapu: float = field(static=True, default=4.0)
    max_steps: int = field(static=True, default=50)

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
        t = _more_thuente(
            params,
            updates,
            idx,
            keys,
            phi0,
            dphi0,
            value_and_grad_fn,
            c1=self.c1,
            c2=self.c2,
            t_init=self.t_init,
            stpmin=self.stpmin,
            stpmax=self.stpmax,
            xtol=self.xtol,
            xtrapl=self.xtrapl,
            xtrapu=self.xtrapu,
            max_steps=self.max_steps,
        )
        return tree_scale_per_row(updates, Table(keys, t), idx), state
