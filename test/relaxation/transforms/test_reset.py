# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""``Optimizer.reset``: a reset system continues bit-identically to a fresh run."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

from kups.core.data import Index, Table
from kups.core.typing import SystemId
from kups.relaxation.optimizer import ChainOptimizer, Optimizer, chain
from kups.relaxation.transforms import (
    ClipByGlobalNorm,
    MaxStepSize,
    ScaleByAseLbfgs,
    ScaleByBacktrackingLinesearch,
    ScaleByFire,
    ScaleByFire2,
    ScaleByMoreThuenteLinesearch,
)

# Two systems: rows 0-2 belong to system 0, rows 3-4 to system 1.
PREFIX = Index.integer(jnp.array([0, 0, 0, 1, 1]), n=2, label=SystemId)
SINGLE = Index.integer(jnp.array([0, 0]), n=1, label=SystemId)
X0 = jnp.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
)
STIFFNESS = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0])[:, None]


def _objective(prefix: Index[SystemId], x0: jax.Array, k: jax.Array):
    """Per-system harmonic wells ``E_s = sum_i k_i |x_i - x0_i|^2 / 2``."""

    def value_and_grad(x: jax.Array):
        grad = k * (x - x0)
        energies = prefix.sum_over(0.5 * jnp.sum(grad * (x - x0), axis=-1))
        return energies, grad

    return value_and_grad


def _run[OptState](
    opt: Optimizer[jax.Array, OptState],
    params: jax.Array,
    state: OptState,
    prefix: Index[SystemId],
    x0: jax.Array,
    k: jax.Array,
    steps: int,
) -> tuple[jax.Array, OptState]:
    vg = _objective(prefix, x0, k)
    for _ in range(steps):
        energies, grad = vg(params)
        updates, state = opt.update(
            grad, state, params, grad=grad, energies=energies, value_and_grad_fn=vg
        )
        params = params + updates
    return params, state


OPTIMIZERS: dict[str, Callable[[], ChainOptimizer[jax.Array]]] = {
    "fire": lambda: chain(optax.scale(-1.0), ScaleByFire[jax.Array](dt_start=0.1)),
    "fire2": lambda: chain(
        optax.scale(-1.0), ScaleByFire2[jax.Array](dt_start=0.1, n_min=3)
    ),
    "fire2_abc": lambda: chain(
        optax.scale(-1.0), ScaleByFire2[jax.Array](dt_start=0.1, use_abc=True)
    ),
    "lbfgs": lambda: chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3), optax.scale(-1.0)
    ),
    "lbfgs_adaptive": lambda: chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3, adaptive_scale=True),
        optax.scale(-1.0),
    ),
    "lbfgs_more_thuente": lambda: chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3),
        optax.scale(-1.0),
        ScaleByMoreThuenteLinesearch[jax.Array](),
    ),
    "lbfgs_backtracking": lambda: chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3),
        optax.scale(-1.0),
        ScaleByBacktrackingLinesearch[jax.Array](),
    ),
    "fire_clipped": lambda: chain(
        optax.scale(-1.0),
        ScaleByFire[jax.Array](dt_start=0.1),
        ClipByGlobalNorm[jax.Array](max_norm=0.5),
        MaxStepSize[jax.Array](max_step_size=0.2),
    ),
}


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
@pytest.mark.parametrize("k_before", [1, 5])
def test_reset_system_matches_fresh_run(name: str, k_before: int):
    opt = OPTIMIZERS[name]()
    start = X0 + 0.7
    state = opt.init(start, PREFIX)
    params, state = _run(opt, start, state, PREFIX, X0, STIFFNESS, k_before)

    # System 1 receives a new structure; system 0 keeps relaxing.
    new_rows = jnp.array([[3.0, 0.0, 1.0], [0.5, 2.0, 0.0]])
    params_reset = params.at[3:].set(new_rows)
    mask = Table((SystemId(0), SystemId(1)), jnp.array([False, True]))
    state_reset = opt.reset(state, params_reset, PREFIX, mask)
    after, _ = _run(opt, params_reset, state_reset, PREFIX, X0, STIFFNESS, 4)

    # Fresh single-system run of the new structure.
    fresh_state = opt.init(new_rows, SINGLE)
    fresh, _ = _run(opt, new_rows, fresh_state, SINGLE, X0[3:], STIFFNESS[3:], 4)
    npt.assert_array_equal(after[3:], fresh)

    # System 0 is untouched by the reset.
    continued, _ = _run(opt, params, state, PREFIX, X0, STIFFNESS, 4)
    npt.assert_array_equal(after[:3], continued[:3])


def test_fire2_n_total_is_per_system():
    opt = ScaleByFire2[jax.Array](dt_start=0.1, n_min=3)
    state = opt.init(X0, PREFIX)
    for _ in range(4):
        _, state = opt.update(-STIFFNESS * (X0 + 0.5 - X0), state, X0 + 0.5)
    npt.assert_array_equal(state.n_total.data, [4, 4])
    mask = Table((SystemId(0), SystemId(1)), jnp.array([False, True]))
    reset = opt.reset(state, X0, PREFIX, mask)
    npt.assert_array_equal(reset.n_total.data, [4, 0])
    npt.assert_array_equal(reset.n_pos.data[0], state.n_pos.data[0])
    npt.assert_array_equal(reset.velocity[3:], 0.0)
    npt.assert_array_equal(reset.velocity[:3], state.velocity[:3])


def test_lbfgs_reset_blanks_history_rows_only():
    opt = ScaleByAseLbfgs[jax.Array](memory_size=3)
    state = opt.init(X0, PREFIX)
    params = X0 + 0.5
    params, state = _run(
        chain(opt, optax.scale(-1.0)),
        params,
        (state, optax.EmptyState()),
        PREFIX,
        X0,
        STIFFNESS,
        4,
    )
    lbfgs_state = state[0]
    mask = Table((SystemId(0), SystemId(1)), jnp.array([False, True]))
    reset = opt.reset(lbfgs_state, params, PREFIX, mask)
    assert int(reset.count) == int(lbfgs_state.count)
    npt.assert_array_equal(reset.steps.data, [4, 0])
    for mem in reset.diff_params_memory + reset.diff_updates_memory:
        npt.assert_array_equal(mem[:, 3:], 0.0)
    npt.assert_array_equal(
        reset.diff_params_memory[0][:, :3], lbfgs_state.diff_params_memory[0][:, :3]
    )
    npt.assert_array_equal(reset.weights_memory.data[1], 0.0)
    npt.assert_array_equal(
        reset.weights_memory.data[0], lbfgs_state.weights_memory.data[0]
    )
    npt.assert_array_equal(reset.params[0][3:], params[3:])
    npt.assert_array_equal(reset.updates[0][3:], 0.0)


def test_stateful_optax_requires_explicit_reset_adapter() -> None:
    opt = chain(optax.adam(0.1))
    state = opt.init(X0, PREFIX)
    mask = Table((SystemId(0), SystemId(1)), jnp.array([True, False]))
    with pytest.raises(ValueError, match="explicit reset adapter"):
        opt.reset(state, X0, PREFIX, mask)


def test_reset_is_jittable():
    opt = OPTIMIZERS["lbfgs_more_thuente"]()
    state = opt.init(X0, PREFIX)
    mask = Table((SystemId(0), SystemId(1)), jnp.array([False, True]))
    reset = jax.jit(lambda s, p, m: opt.reset(s, p, PREFIX, m))(state, X0, mask)
    assert jax.tree.structure(reset) == jax.tree.structure(state)
