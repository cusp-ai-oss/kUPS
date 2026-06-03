# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The per-system line searches reproduce ASE's implementations exactly.

Each system is given one degree of freedom so the per-system search reduces to a
1-D problem, run through the full transform interface and compared element-by-
element against ASE's ``LineSearch`` (More-Thuente) and ``LineSearchArmijo``.
ASE bakes a per-atom ``maxstep`` clamp into the search; we disable it (the clamp
is left to ``MaxStepSize``) by passing ``maxstep=1e30``, which makes ASE's
``determine_step`` the identity. Runs in float64 to match ASE's double precision.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from ase.utils.linesearch import LineSearch
from ase.utils.linesearcharmijo import LineSearchArmijo
from jax import Array

from kups.core.data.index import Index
from kups.core.typing import SystemId
from kups.relaxation.optimizer import Optimizer
from kups.relaxation.transforms.linesearch import (
    LineSearchState,
    ScaleByBacktrackingLinesearch,
    ScaleByMoreThuenteLinesearch,
)

Scalar = Callable[[float], float]


@pytest.fixture(autouse=True)
def _x64():
    """Run these comparisons in float64 to match ASE; isolate the cache."""
    prev = bool(jax.config.read("jax_enable_x64"))
    jax.clear_caches()
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)
        jax.clear_caches()


def _ase_more_thuente(phi: Scalar, dphi: Scalar, c1: float, c2: float) -> float:
    # Embed the scalar problem on the first of three coords (ASE's gradient norm
    # reshapes to (-1, 3)); maxstep=1e30 makes determine_step the identity, so the
    # per-atom clamp (left to MaxStepSize) never fires. Portable across ASE versions.
    ls = LineSearch()
    pk = np.array([1.0, 0.0, 0.0])
    alpha, *_ = ls._line_search(
        lambda x, *a: phi(x[0]),
        lambda x, *a: np.array([dphi(x[0]), 0.0, 0.0]),
        np.zeros(3),
        pk,
        np.array([dphi(0.0), 0.0, 0.0]),
        phi(0.0),
        None,
        c1=c1,
        c2=c2,
        stpmax=50.0,
        maxstep=1e30,
    )
    assert alpha is not None  # ASE returns None only on failure; our cases succeed
    return float(alpha)


def _ase_armijo(
    phi: Scalar, dphi: Scalar, c1: float, func_old: float | None = None
) -> float:
    # dirn reshaped to (-1, 3) by ASE, so embed the scalar on the first coord.
    ls = LineSearchArmijo(lambda x: phi(x[0]), c1=c1)
    alpha, *_ = ls.run(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        a1=None,
        func_start=phi(0.0),
        func_old=func_old,
        func_prime_start=np.array([dphi(0.0), 0.0, 0.0]),
        maxstep=1e30,
    )
    return float(alpha)


def _index(n: int) -> Index[SystemId]:
    keys = tuple(SystemId(i) for i in range(n))
    return Index(keys, jnp.arange(n), _cls=SystemId)


def _step(
    opt: Optimizer[Array, LineSearchState],
    x0: Array,
    direction: Array,
    vag: Callable[[Array], tuple[Any, Array]],
    idx: Index[SystemId],
) -> np.ndarray:
    state = opt.init(x0, index_prefix=idx)

    @jax.jit
    def run(x0: Array, direction: Array) -> Array:
        energies, grad = vag(x0)
        step, _ = opt.update(
            direction, state, x0, grad=grad, energies=energies, value_and_grad_fn=vag
        )
        return step

    return np.array(run(x0, direction))


# Per-system diagonal quadratics with steepest-descent direction. The line
# minimiser is t* = 1/hess, spanning overshoot (hess > 1), exact (hess = 1) and
# undershoot/expansion (hess < 1).
HESS = jnp.array([1.0, 100.0, 0.2, 2.0, 0.5, 0.05], dtype=jnp.float64)
X0 = jnp.array([1.0, 1.0, -2.0, 3.0, -1.0, 0.7], dtype=jnp.float64)

# Per-system quartics evaluated from x0 = 0 along direction 1, so phi(t) is the
# quartic itself; a1 < 0 keeps them descent, a4 > 0 keeps them bounded below.
QUARTIC = jnp.array(
    [[1.0, -0.5, 0.3, -1.0], [0.5, 0.2, -0.4, -0.8], [2.0, -1.0, 0.5, -0.3]],
    dtype=jnp.float64,
)


def _quad_setup() -> tuple[Array, Array, Index[SystemId], Callable[[Array], Any]]:
    idx = _index(len(HESS))
    direction = -HESS * X0

    def vag(p: Array) -> tuple[Any, Array]:
        return idx.sum_over(0.5 * HESS * p**2), HESS * p

    return X0, direction, idx, vag


def _quad_scalars(i: int) -> tuple[Scalar, Scalar]:
    h, x, d = float(HESS[i]), float(X0[i]), float(-HESS[i] * X0[i])
    return (lambda t: 0.5 * h * (x + t * d) ** 2, lambda t: h * (x + t * d) * d)


def _quartic_setup() -> tuple[Array, Array, Index[SystemId], Callable[[Array], Any]]:
    n = QUARTIC.shape[0]
    idx = _index(n)
    a4, a3, a2, a1 = (QUARTIC[:, k] for k in range(4))

    def vag(p: Array) -> tuple[Any, Array]:
        e = a4 * p**4 + a3 * p**3 + a2 * p**2 + a1 * p
        g = 4 * a4 * p**3 + 3 * a3 * p**2 + 2 * a2 * p + a1
        return idx.sum_over(e), g

    return jnp.zeros(n, jnp.float64), jnp.ones(n, jnp.float64), idx, vag


def _quartic_scalars(i: int) -> tuple[Scalar, Scalar]:
    c = np.array(QUARTIC[i])
    return (
        lambda t: c[0] * t**4 + c[1] * t**3 + c[2] * t**2 + c[3] * t,
        lambda t: 4 * c[0] * t**3 + 3 * c[1] * t**2 + 2 * c[2] * t + c[3],
    )


def _shifted_quartic_scalars(ci: float, xi: float, di: float) -> tuple[Scalar, Scalar]:
    """φ(s) = 0.5·ci·(xi + s·di)² + 0.25·(xi + s·di)⁴ and its derivative in s."""
    return (
        lambda s: 0.5 * ci * (xi + s * di) ** 2 + 0.25 * (xi + s * di) ** 4,
        lambda s: (ci * (xi + s * di) + (xi + s * di) ** 3) * di,
    )


class TestMoreThuenteMatchesAse:
    def test_quadratic(self):
        x0, direction, idx, vag = _quad_setup()
        t = _step(ScaleByMoreThuenteLinesearch(), x0, direction, vag, idx) / np.array(
            direction
        )
        ase = [
            _ase_more_thuente(*_quad_scalars(i), 0.23, 0.46) for i in range(len(HESS))
        ]
        npt.assert_allclose(t, ase, rtol=1e-7, atol=1e-9)

    def test_quartic(self):
        x0, direction, idx, vag = _quartic_setup()
        t = _step(
            ScaleByMoreThuenteLinesearch(), x0, direction, vag, idx
        )  # direction = 1
        ase = [
            _ase_more_thuente(*_quartic_scalars(i), 0.23, 0.46)
            for i in range(QUARTIC.shape[0])
        ]
        npt.assert_allclose(t, ase, rtol=1e-7, atol=1e-9)

    def test_custom_c1_c2(self):
        x0, direction, idx, vag = _quartic_setup()
        opt = ScaleByMoreThuenteLinesearch(c1=1e-4, c2=0.9)
        t = _step(opt, x0, direction, vag, idx)
        ase = [
            _ase_more_thuente(*_quartic_scalars(i), 1e-4, 0.9)
            for i in range(QUARTIC.shape[0])
        ]
        npt.assert_allclose(t, ase, rtol=1e-7, atol=1e-9)


class TestBacktrackingMatchesAse:
    def test_quadratic(self):
        x0, direction, idx, vag = _quad_setup()
        t = _step(ScaleByBacktrackingLinesearch(), x0, direction, vag, idx) / np.array(
            direction
        )
        ase = [_ase_armijo(*_quad_scalars(i), 0.1) for i in range(len(HESS))]
        npt.assert_allclose(t, ase, rtol=1e-7, atol=1e-9)

    def test_quartic(self):
        x0, direction, idx, vag = _quartic_setup()
        t = _step(ScaleByBacktrackingLinesearch(), x0, direction, vag, idx)
        ase = [_ase_armijo(*_quartic_scalars(i), 0.1) for i in range(QUARTIC.shape[0])]
        npt.assert_allclose(t, ase, rtol=1e-7, atol=1e-9)

    def test_custom_c1(self):
        x0, direction, idx, vag = _quartic_setup()
        t = _step(ScaleByBacktrackingLinesearch(c1=0.4), x0, direction, vag, idx)
        ase = [_ase_armijo(*_quartic_scalars(i), 0.4) for i in range(QUARTIC.shape[0])]
        npt.assert_allclose(t, ase, rtol=1e-7, atol=1e-9)

    def test_initial_step_estimate_matches_ase(self):
        """Step 2 uses the N&W eq. 3.60 estimate from step 1's energy, like ASE.

        One dof per system on convex quartics ``0.5·c·x² + 0.25·x⁴`` (non-quadratic
        so step 1 does not land on the minimum and step 2 is a genuine search).
        """
        c = jnp.array([1.0, 4.0, 0.5], dtype=jnp.float64)
        x0 = jnp.array([1.5, 1.2, 2.0], dtype=jnp.float64)
        idx = _index(len(c))

        def vag(p: Array):
            return idx.sum_over(0.5 * c * p**2 + 0.25 * p**4), c * p + p**3

        opt = ScaleByBacktrackingLinesearch()

        @jax.jit
        def update(state: LineSearchState, x: Array):
            e, g = vag(x)
            step, state = opt.update(
                -g, state, x, grad=g, energies=e, value_and_grad_fn=vag
            )
            return step, state

        state = opt.init(x0, index_prefix=idx)
        e0, _ = vag(x0)
        s0, state = update(state, x0)
        x1 = x0 + s0
        _, g1 = vag(x1)
        s1, _ = update(state, x1)
        t2 = np.array(s1) / np.array(-g1)

        cn, x1n, d1n = np.array(c), np.array(x1), np.array(-g1)
        e0n = np.array(e0.data)  # func_old for step 2
        ase = [
            _ase_armijo(
                *_shifted_quartic_scalars(float(cn[i]), float(x1n[i]), float(d1n[i])),
                0.1,
                func_old=float(e0n[i]),
            )
            for i in range(len(c))
        ]
        npt.assert_allclose(t2, ase, rtol=1e-7, atol=1e-9)
