# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Relaxation with the Verlet-skin cache refreshed before every step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.potential.verlet import (
    make_skin_refresh_from_state,
    verlet_neighborlist_factory,
)
from kups.application.relaxation.data import (
    RelaxRunConfig,
    RelaxState,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import make_relax_propagator
from kups.application.utils.propagate import make_cycle_function
from kups.core.lens import Lens, identity_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    UniversalNeighborlistParameters,
    VerletSkinState,
)
from kups.core.propagator import (
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.utils.jax import tree_copy
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.common.geometry import Geometry, PositionsAndCell
from kups.relaxation.config import TransformationConfig, make_optimizer

from ._builders import LBFGS_OPTIMIZER, ar_cif, tmp_h5

# Ar fcc 2x2x2: 32 atoms in a 10.6 A box, so cutoff + skin stays below half the
# box. The 0.2 A step clamp keeps one step inside the skin; the rattle makes the
# skin run out within a few steps.
CUTOFF, SKIN, NUM_STEPS = 4.0, 0.6, 8
TINY = UniversalNeighborlistParameters(1, 1, 1, 1)
LJ = LennardJonesParameters.from_dict(
    cutoff=CUTOFF, parameters={"Ar": (3.405, 0.010326)}, mixing_rule="lorentz_berthelot"
)
LINE_SEARCH: TransformationConfig = [
    {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
    {"transform": "scale", "step_size": -1},
    {"transform": "scale_by_more_thuente_linesearch"},
]


def _build(
    skin: float,
    cached: bool,
    optimizer: TransformationConfig = LBFGS_OPTIMIZER,
    gradient: Lens[Geometry, PositionsAndCell] = POSITIONS_ONLY,
    *,
    neighborlist_params: UniversalNeighborlistParameters | None = None,
    skin_params: UniversalNeighborlistParameters | None = None,
) -> tuple[Propagator[RelaxState], RelaxState]:
    """The app composition; ``cached=False`` keeps the refresh but ignores the cache."""
    state_lens = identity_lens(RelaxState)
    cache = state_lens.focus(lambda s: s.verlet_skin)
    potential = make_lennard_jones_from_state(
        state_lens,
        parameters=LJ,
        gradient=gradient,
        neighborlist_factory=(
            verlet_neighborlist_factory(cache) if cached else AdaptiveNeighborList.new
        ),
    )
    relax, opt_init = make_relax_propagator(
        state_lens, potential, make_optimizer(optimizer), gradient
    )
    refresh = make_skin_refresh_from_state(state_lens, LJ.cutoff, skin)
    propagator = ResetOnErrorPropagator(SequentialPropagator((refresh, relax)))

    particles, systems = relax_state_from_ase(ar_cif(rattle=0.15, cubic=True))
    counts = particles.data.system.counts
    estimate = UniversalNeighborlistParameters.estimate
    state = RelaxState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params or estimate(counts, systems, LJ.cutoff),
        opt_state=opt_init(particles, systems),
        step=jnp.array([0]),
        verlet_skin=VerletSkinState.new(
            particles,
            systems,
            skin_params
            or estimate(counts, systems, LJ.cutoff.map_data(lambda c: c + skin)),
        ),
    )
    return propagator, state


def _run(propagator: Propagator[RelaxState], state: RelaxState, steps: int):
    cycle = make_cycle_function(propagator)
    for i in range(steps):
        state = propagate_and_fix(cycle, jax.random.key(i), state)
    return state


def _assert_same_geometry(a: RelaxState, b: RelaxState) -> None:
    np.testing.assert_allclose(
        a.particles.data.positions, b.particles.data.positions, rtol=0, atol=1e-8
    )
    for x, y in zip(
        jax.tree.leaves(a.systems.data.cell), jax.tree.leaves(b.systems.data.cell)
    ):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "gradient", [POSITIONS_ONLY, FRECHET_FILTER], ids=["positions", "cell"]
)
def test_cached_relaxation_matches_fresh_builds(gradient) -> None:
    plain = _run(*_build(SKIN, False, gradient=gradient), NUM_STEPS)
    propagator, initial = _build(SKIN, True, gradient=gradient)
    cached = _run(propagator, initial, NUM_STEPS)
    rebuilt_at = cached.verlet_skin.reference.particles.data.positions
    assert not np.array_equal(rebuilt_at, initial.particles.data.positions)
    assert int(cached.step[0]) == NUM_STEPS
    _assert_same_geometry(plain, cached)


@pytest.mark.parametrize("skin", [1e-4, SKIN])
def test_line_search_trials_match_fresh_builds(skin: float) -> None:
    plain = _run(*_build(skin, False, LINE_SEARCH), 3)
    cached = _run(*_build(skin, True, LINE_SEARCH), 3)
    _assert_same_geometry(plain, cached)


def test_trial_evaluations_never_write_the_cache() -> None:
    propagator, state = _build(SKIN, True, LINE_SEARCH)
    stepped = _run(propagator, tree_copy(state), 1)  # the cycle donates its input
    refresh = make_skin_refresh_from_state(identity_lens(RelaxState), LJ.cutoff, SKIN)
    refreshed = _run(ResetOnErrorPropagator(refresh), state, 1)
    for a, b in zip(
        jax.tree.leaves(stepped.verlet_skin), jax.tree.leaves(refreshed.verlet_skin)
    ):
        np.testing.assert_array_equal(a, b)


def test_undersized_capacities_are_repaired() -> None:
    plain = _run(*_build(SKIN, False), 3)
    propagator, state = _build(SKIN, True, neighborlist_params=TINY, skin_params=TINY)
    cached = _run(propagator, state, 3)
    assert cached.neighborlist_params.avg_edges > TINY.avg_edges
    assert cached.verlet_skin.params.avg_edges > TINY.avg_edges
    _assert_same_geometry(plain, cached)


def test_run_entry_point() -> None:
    from kups.application.relaxation.analysis import analyze_relax_file
    from kups.application.simulations.potentials import LjPotentialConfig
    from kups.application.simulations.relax import Config, run

    out_file = tmp_h5()
    config = Config(
        run=RelaxRunConfig(
            out_file=out_file,
            max_steps=NUM_STEPS,
            seed=42,
            force_tolerance=1e-6,
            optimizer=LBFGS_OPTIMIZER,
            optimize_cell=False,
        ),
        potential=LjPotentialConfig(
            cutoff=CUTOFF,
            parameters={"Ar": (3.405, 0.010326)},
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(ar_cif(rattle=0.15, cubic=True),),
    )
    run(config)
    result = next(iter(analyze_relax_file(out_file).values()))
    assert jnp.isfinite(jnp.asarray(result.final_energy)).item()
    assert jnp.isfinite(jnp.asarray(result.final_max_force)).item()
    assert result.n_steps >= 1
