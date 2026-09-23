# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Indexed state lenses: a reset system continues identically to a fresh run."""

from __future__ import annotations

from typing import Callable, assert_type

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

from kups.core.data import Index, Table
from kups.core.data.index import SupportsSorting
from kups.core.lens import lens
from kups.core.patch import IndexLensPatch
from kups.core.typing import SystemId
from kups.relaxation.optimizer import (
    ChainOptimizer,
    ChainOptState,
    Optimizer,
    ResetLayout,
    chain,
)
from kups.relaxation.transforms import (
    ClipByGlobalNorm,
    MaxStepSize,
    ScaleByAseLbfgs,
    ScaleByBacktrackingLinesearch,
    ScaleByFire,
    ScaleByFire2,
    ScaleByMoreThuenteLinesearch,
)
from kups.relaxation.transforms.fire import FireReset, fire_reset_layout
from kups.relaxation.transforms.fire2 import fire2_reset_layout
from kups.relaxation.transforms.lbfgs import LbfgsResetIndices, lbfgs_reset_layout
from kups.relaxation.transforms.linesearch import linesearch_reset_layout

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


type ResetIndices = (
    FireReset[Index[SupportsSorting], Index[SupportsSorting]]
    | LbfgsResetIndices
    | tuple[LbfgsResetIndices, Index[SupportsSorting]]
)


def _layout(name: str) -> ResetLayout[ChainOptState, object, ResetIndices]:
    if name.startswith("fire"):
        fire = fire2_reset_layout() if name.startswith("fire2") else fire_reset_layout()
        return ResetLayout(
            fields=lens(lambda s: s[1]).nest(fire.fields),
            system_index=lambda s: fire.system_index(s[1]),
        )
    lbfgs = lbfgs_reset_layout()
    outer = lens(lambda s: s[0], cls=tuple).nest(lbfgs.fields)
    if name in ("lbfgs_more_thuente", "lbfgs_backtracking"):
        search = linesearch_reset_layout()
        return ResetLayout(
            fields=outer.merge(lens(lambda s: s[2]).nest(search.fields)),
            system_index=lambda s: (
                lbfgs.system_index(s[0]),
                search.system_index(s[2]),
            ),
        )
    return ResetLayout(fields=outer, system_index=lambda s: lbfgs.system_index(s[0]))


def _replace[State, Data, Indices](
    state: State,
    fresh: State,
    layout: ResetLayout[State, Data, Indices],
    mask: Table[SystemId, jax.Array],
) -> State:
    return IndexLensPatch(
        layout.fields.get(fresh), layout.system_index(state), layout.fields
    )(state, mask)


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
    state_reset = _replace(state, opt.init(params_reset, PREFIX), _layout(name), mask)
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
    reset = _replace(state, opt.init(X0, PREFIX), fire2_reset_layout(), mask)
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
    reset = _replace(lbfgs_state, opt.init(params, PREFIX), lbfgs_reset_layout(), mask)
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
    npt.assert_array_equal(reset.params[0][3:], 0.0)
    npt.assert_array_equal(reset.updates[0][3:], 0.0)


def test_ordinary_chain_does_not_require_reset_support() -> None:
    opt = chain(optax.adam(0.1))
    state = opt.init(X0, PREFIX)
    updates, _ = opt.update(X0, state, X0)
    assert updates.shape == X0.shape


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_reset_is_jittable(name: str) -> None:
    opt = OPTIMIZERS[name]()
    state = opt.init(X0, PREFIX)
    _, state = _run(opt, X0 + 0.5, state, PREFIX, X0, STIFFNESS, 4)
    mask = Table((SystemId(0), SystemId(1)), jnp.array([False, True]))
    layout = _layout(name)

    def replace(s: ChainOptState) -> ChainOptState:
        return _replace(s, opt.init(X0, PREFIX), layout, mask)

    reset = jax.jit(replace)(state)
    assert jax.tree.structure(reset) == jax.tree.structure(state)
    for eager, compiled in zip(
        jax.tree.leaves(replace(state)), jax.tree.leaves(reset), strict=True
    ):
        npt.assert_array_equal(eager, compiled)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_reset_lens_round_trip(name: str) -> None:
    opt = OPTIMIZERS[name]()
    state = opt.init(X0, PREFIX)
    _, state = _run(opt, X0 + 0.5, state, PREFIX, X0, STIFFNESS, 4)
    fields = _layout(name).fields
    restored = fields.set(state, fields.get(state))
    for before, after in zip(
        jax.tree.leaves(state), jax.tree.leaves(restored), strict=True
    ):
        npt.assert_array_equal(before, after)


def test_reset_projection_field_types() -> None:
    fire_state = ScaleByFire[jax.Array]().init(X0, PREFIX)
    fire = fire_reset_layout()
    assert_type(fire.fields.get(fire_state).dt, Table[SupportsSorting, jax.Array])
    assert_type(fire.system_index(fire_state).dt, Index[SupportsSorting])

    fire2_state = ScaleByFire2[jax.Array]().init(X0, PREFIX)
    fire2 = fire2_reset_layout()
    assert_type(
        fire2.fields.get(fire2_state).n_total, Table[SupportsSorting, jax.Array]
    )
    assert_type(fire2.system_index(fire2_state).n_total, Index[SupportsSorting])

    lbfgs_state = ScaleByAseLbfgs[jax.Array](memory_size=3).init(X0, PREFIX)
    lbfgs = lbfgs_reset_layout()
    assert_type(lbfgs.fields.get(lbfgs_state).diff_params_memory, list[jax.Array])
    assert_type(lbfgs.system_index(lbfgs_state), LbfgsResetIndices)
    assert_type(
        lbfgs.system_index(lbfgs_state).diff_params_memory, list[Index[SupportsSorting]]
    )


def test_reset_layout_preserves_concrete_index_type() -> None:
    layout: ResetLayout[
        tuple[Index[SystemId], jax.Array], jax.Array, Index[SystemId]
    ] = ResetLayout(
        fields=lens(lambda s: s[1], cls=tuple[Index[SystemId], jax.Array]),
        system_index=lambda s: s[0],
    )
    indices = layout.system_index((PREFIX, X0))
    assert_type(indices, Index[SystemId])
    npt.assert_array_equal(indices.indices, PREFIX.indices)
