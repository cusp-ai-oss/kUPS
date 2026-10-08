# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Indexed state lenses: a reset system continues identically to a fresh run."""

from __future__ import annotations

from typing import Any, Callable, assert_type, override

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

from kups.core.data import Index, Table
from kups.core.data.index import SupportsSorting
from kups.core.lens import lens
from kups.core.typing import PyTree, SystemId
from kups.core.utils.jax import dataclass
from kups.relaxation.optimizer import (
    ChainOptimizer,
    ChainOptState,
    Optimizer,
    ResetLayout,
    Resettable,
    ResettableChain,
    Stateless,
    chain,
    resettable_chain,
)
from kups.relaxation.transforms import (
    ClipByGlobalNorm,
    ClipByGlobalNormState,
    LineSearchState,
    MaxStepSize,
    MaxStepSizeState,
    ScaleByAseLbfgs,
    ScaleByAseLbfgsState,
    ScaleByBacktrackingLinesearch,
    ScaleByFire,
    ScaleByFire2,
    ScaleByFire2State,
    ScaleByMoreThuenteLinesearch,
)
from kups.relaxation.transforms.fire import (
    FireReset,
    FireResetData,
    FireResetIndices,
    ScaleByFireState,
    fire_reset_layout,
)
from kups.relaxation.transforms.fire2 import (
    Fire2ResetData,
    Fire2ResetIndices,
    fire2_reset_layout,
)
from kups.relaxation.transforms.lbfgs import (
    LbfgsResetData,
    LbfgsResetIndices,
    lbfgs_reset_layout,
)
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
MASK = Table((SystemId(0), SystemId(1)), jnp.array([False, True]))


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


NEGATE = Stateless(optax.scale(-1.0))

# The OPTIMIZERS chains, with each layout declared by its member.
RESETTABLE: dict[str, Callable[[], ResettableChain[jax.Array]]] = {
    "fire": lambda: resettable_chain(NEGATE, ScaleByFire[jax.Array](dt_start=0.1)),
    "fire2": lambda: resettable_chain(
        NEGATE, ScaleByFire2[jax.Array](dt_start=0.1, n_min=3)
    ),
    "fire2_abc": lambda: resettable_chain(
        NEGATE, ScaleByFire2[jax.Array](dt_start=0.1, use_abc=True)
    ),
    "lbfgs": lambda: resettable_chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3), NEGATE
    ),
    "lbfgs_adaptive": lambda: resettable_chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3, adaptive_scale=True), NEGATE
    ),
    "lbfgs_more_thuente": lambda: resettable_chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3),
        NEGATE,
        ScaleByMoreThuenteLinesearch[jax.Array](),
    ),
    "lbfgs_backtracking": lambda: resettable_chain(
        ScaleByAseLbfgs[jax.Array](memory_size=3),
        NEGATE,
        ScaleByBacktrackingLinesearch[jax.Array](),
    ),
    "fire_clipped": lambda: resettable_chain(
        NEGATE,
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
        return fire.within(lens(lambda s: s[1], cls=tuple))
    lbfgs = lbfgs_reset_layout().within(lens(lambda s: s[0], cls=tuple))
    if name in ("lbfgs_more_thuente", "lbfgs_backtracking"):
        search = linesearch_reset_layout().within(lens(lambda s: s[2], cls=tuple))
        return lbfgs.merge(search)
    return lbfgs


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
# The first post-reset L-BFGS write lands in ring slot (k_before - 1) % 3: 0, 1, 2.
@pytest.mark.parametrize("k_before", [1, 5, 6])
def test_reset_system_matches_fresh_run(name: str, k_before: int):
    opt = OPTIMIZERS[name]()
    start = X0 + 0.7
    state = opt.init(start, PREFIX)
    params, state = _run(opt, start, state, PREFIX, X0, STIFFNESS, k_before)

    # System 1 receives a new structure; system 0 keeps relaxing.
    new_rows = jnp.array([[3.0, 0.0, 1.0], [0.5, 2.0, 0.0]])
    params_reset = params.at[3:].set(new_rows)
    state_reset = _layout(name).reset(state, opt.init(params_reset, PREFIX), MASK)
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
    reset = fire2_reset_layout().reset(state, opt.init(X0, PREFIX), MASK)
    npt.assert_array_equal(reset.n_total.data, [4, 0])
    npt.assert_array_equal(reset.n_pos.data[0], state.n_pos.data[0])
    npt.assert_array_equal(reset.velocity[3:], 0.0)
    npt.assert_array_equal(reset.velocity[:3], state.velocity[:3])

    # A non-positive-power step (force opposing velocity) shrinks dt for system 0,
    # past its n_min warm-up, but not for the reset system 1, which is back inside it.
    _, after = opt.update(-reset.velocity, reset, X0 + 0.5)
    assert float(after.dt.data[0]) < float(reset.dt.data[0])
    npt.assert_array_equal(after.dt.data[1], reset.dt.data[1])


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
    reset = lbfgs_reset_layout().reset(lbfgs_state, opt.init(params, PREFIX), MASK)
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


def test_chain_supports_stateful_optax_transforms() -> None:
    opt = chain(optax.adam(0.1))
    state = opt.init(X0, PREFIX)
    updates, _ = opt.update(X0, state, X0)
    assert updates.shape == X0.shape


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_reset_is_jittable(name: str) -> None:
    opt = OPTIMIZERS[name]()
    state = opt.init(X0, PREFIX)
    _, state = _run(opt, X0 + 0.5, state, PREFIX, X0, STIFFNESS, 4)
    layout = _layout(name)

    def replace(s: ChainOptState) -> ChainOptState:
        return layout.reset(s, opt.init(X0, PREFIX), MASK)

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


@pytest.mark.parametrize(
    ("opt", "layout"),
    [
        (ScaleByFire[jax.Array](dt_start=0.1, n_min=1), fire_reset_layout()),
        (
            ScaleByFire2[jax.Array](dt_start=0.1, n_min=1, delaystep_start=False),
            fire2_reset_layout(),
        ),
    ],
    ids=["fire", "fire2"],
)
def test_fire_reset_restores_per_system_adaptation(
    opt: Optimizer[jax.Array, Any],
    layout: ResetLayout[Any, FireReset[Any, Table[SupportsSorting, jax.Array]], Any],
) -> None:
    state = opt.init(X0, PREFIX)
    for _ in range(4):
        state = opt.update(-STIFFNESS * 0.5 * jnp.ones_like(X0), state, X0)[1]
    fresh = opt.init(X0, PREFIX)
    reset = layout.reset(state, fresh, MASK)
    for name in ("dt", "alpha", "n_pos"):
        before, init, after = (getattr(s, name).data for s in (state, fresh, reset))
        assert (before != init).all(), name  # the reset must be observable
        npt.assert_array_equal(after, [before[0], init[1]], err_msg=name)


def test_linesearch_reset_clears_previous_energy() -> None:
    opt = OPTIMIZERS["lbfgs_backtracking"]()
    state = opt.init(X0 + 0.5, PREFIX)
    _, state = _run(opt, X0 + 0.5, state, PREFIX, X0, STIFFNESS, 2)
    search = state[2]
    assert not jnp.isnan(search.prev_phi0.data).any()
    reset = linesearch_reset_layout().reset(search, opt.init(X0, PREFIX)[2], MASK)
    assert jnp.isnan(reset.prev_phi0.data[1])
    npt.assert_array_equal(reset.prev_phi0.data[0], search.prev_phi0.data[0])


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_resettable_chain_matches_explicit_layout(name: str) -> None:
    plain, opt = OPTIMIZERS[name](), RESETTABLE[name]()
    _, expected = _run(
        plain, X0 + 0.5, plain.init(X0 + 0.5, PREFIX), PREFIX, X0, STIFFNESS, 4
    )
    _, state = _run(opt, X0 + 0.5, opt.init(X0 + 0.5, PREFIX), PREFIX, X0, STIFFNESS, 4)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(expected), strict=True):
        npt.assert_array_equal(a, b)

    fresh = opt.init(X0, PREFIX)
    explicit = _layout(name).reset(state, fresh, MASK)
    composed = opt.reset_layout.reset(state, fresh, MASK)
    compiled = jax.jit(lambda s: opt.reset_layout.reset(s, fresh, MASK))(state)
    assert jax.tree.structure(composed) == jax.tree.structure(explicit)
    for a, b, c in zip(
        jax.tree.leaves(explicit),
        jax.tree.leaves(composed),
        jax.tree.leaves(compiled),
        strict=True,
    ):
        npt.assert_array_equal(b, a)
        npt.assert_array_equal(c, a)


def test_resettable_chains_nest() -> None:
    inner = resettable_chain(NEGATE, ScaleByFire[jax.Array](dt_start=0.1, n_min=1))
    opt = resettable_chain(inner, MaxStepSize[jax.Array](max_step_size=0.2))
    _, state = _run(opt, X0 + 0.5, opt.init(X0 + 0.5, PREFIX), PREFIX, X0, STIFFNESS, 4)
    fresh = opt.init(X0, PREFIX)
    reset = opt.reset_layout.reset(state, fresh, MASK)
    fire, new_fire = state[0][1], reset[0][1]
    npt.assert_array_equal(new_fire.dt.data, [fire.dt.data[0], fresh[0][1].dt.data[1]])
    npt.assert_array_equal(new_fire.velocity[:3], fire.velocity[:3])
    npt.assert_array_equal(new_fire.velocity[3:], 0.0)


def test_stateless_rejects_stateful_optax_transforms() -> None:
    with pytest.raises(ValueError, match="without state"):
        Stateless(optax.adam(0.1)).init(X0, PREFIX)


@pytest.mark.parametrize(
    "make",
    [
        lambda: optax.sgd(0.1),
        lambda: optax.chain(optax.clip(1.0), optax.scale(-1.0)),
        lambda: optax.masked(
            optax.scale(-1.0), lambda p: jax.tree.map(lambda _: True, p)
        ),
    ],
    ids=["sgd", "chain", "masked"],
)
def test_stateless_passes_composite_states_through(
    make: Callable[[], optax.GradientTransformation],
) -> None:
    # Composite stateless transforms have structured, leafless states.
    plain = chain(make(), ScaleByFire[jax.Array](dt_start=0.1))
    opt = resettable_chain(Stateless(make()), ScaleByFire[jax.Array](dt_start=0.1))
    _, expected = _run(
        plain, X0 + 0.5, plain.init(X0 + 0.5, PREFIX), PREFIX, X0, STIFFNESS, 3
    )
    _, state = _run(opt, X0 + 0.5, opt.init(X0 + 0.5, PREFIX), PREFIX, X0, STIFFNESS, 3)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(expected), strict=True):
        npt.assert_array_equal(a, b)
    reset = opt.reset_layout.reset(state, opt.init(X0, PREFIX), MASK)
    assert jax.tree.structure(reset) == jax.tree.structure(state)
    npt.assert_array_equal(reset[1].velocity[3:], 0.0)


@dataclass
class _Forgetful(Resettable[jax.Array, ScaleByFireState]):
    """Declares itself resettable but never says what to reset."""

    @override
    def init(
        self, parameters: jax.Array, index_prefix: PyTree | None = None
    ) -> ScaleByFireState:
        return ScaleByFire[jax.Array]().init(parameters, index_prefix)

    @override
    def update(
        self,
        updates: jax.Array,
        state: ScaleByFireState,
        params: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, ScaleByFireState]:
        return ScaleByFire[jax.Array]().update(updates, state, params, **kwargs)


def test_resettable_without_layout_cannot_be_built() -> None:
    with pytest.raises(TypeError, match="reset_layout"):
        _Forgetful()


def test_native_transforms_declare_their_layouts() -> None:
    assert_type(
        ScaleByFire[jax.Array]().reset_layout,
        ResetLayout[ScaleByFireState, FireResetData, FireResetIndices],
    )
    assert_type(
        ScaleByFire2[jax.Array]().reset_layout,
        ResetLayout[ScaleByFire2State, Fire2ResetData, Fire2ResetIndices],
    )
    assert_type(
        ScaleByAseLbfgs[jax.Array]().reset_layout,
        ResetLayout[ScaleByAseLbfgsState[jax.Array], LbfgsResetData, LbfgsResetIndices],
    )
    search = ResetLayout[
        LineSearchState, Table[SupportsSorting, jax.Array], Index[SupportsSorting]
    ]
    assert_type(ScaleByBacktrackingLinesearch[jax.Array]().reset_layout, search)
    assert_type(ScaleByMoreThuenteLinesearch[jax.Array]().reset_layout, search)
    assert_type(
        MaxStepSize[jax.Array](max_step_size=0.2).reset_layout,
        ResetLayout[MaxStepSizeState, tuple[()], tuple[()]],
    )
    assert_type(
        ClipByGlobalNorm[jax.Array](max_norm=0.5).reset_layout,
        ResetLayout[ClipByGlobalNormState, tuple[()], tuple[()]],
    )
    assert_type(
        RESETTABLE["fire"]().reset_layout,
        ResetLayout[ChainOptState, tuple[Any, ...], tuple[Any, ...]],
    )
    # Params comes from the native member, whichever position it takes.
    assert_type(
        resettable_chain(NEGATE, ScaleByFire[jax.Array]()), ResettableChain[jax.Array]
    )


@dataclass
class _StructuralOnly(Optimizer[jax.Array, ScaleByFireState]):
    """Has a reset layout but does not subclass Resettable."""

    @override
    def init(
        self, parameters: jax.Array, index_prefix: PyTree | None = None
    ) -> ScaleByFireState:
        return ScaleByFire[jax.Array]().init(parameters, index_prefix)

    @override
    def update(
        self,
        updates: jax.Array,
        state: ScaleByFireState,
        params: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, ScaleByFireState]:
        return ScaleByFire[jax.Array]().update(updates, state, params, **kwargs)

    @property
    def reset_layout(
        self,
    ) -> ResetLayout[ScaleByFireState, FireResetData, FireResetIndices]:
        return fire_reset_layout()


def _rejected_by_the_type_checker() -> None:
    """Never called: pyrefly in pre-commit fails if any call here type-checks.

    A bare Optax transform and a plain chain have no reset layout, and a
    layout alone does not make an optimizer resettable: membership is nominal,
    so only subclasses of Resettable may join a resettable chain.
    """
    resettable_chain(optax.adam(0.1))  # pyrefly: ignore[bad-argument-type]
    # Bound first: inline, chain(...) would be inferred against the unsolved
    # Params and fail on that instead of on the missing layout.
    plain = chain(ScaleByFire[jax.Array]())
    resettable_chain(plain)  # pyrefly: ignore[bad-argument-type]
    resettable_chain(_StructuralOnly())  # pyrefly: ignore[bad-argument-type]
