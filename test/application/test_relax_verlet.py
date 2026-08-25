# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Relaxation integration tests for the Verlet-skin neighbor list.

Mirrors ``test/application/md/test_verlet.py`` for the relaxation driver: the
rebuild schedule is driven by the real on-device trigger and the resulting
optimisation trajectory is compared against the every-step dense path. Pure
trigger/refine unit tests live in ``test/core/neighborlist/test_verlet.py``.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_ONLY
from kups.application.relaxation.data import (
    RelaxRunConfig,
    RelaxState,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import make_relax_propagator
from kups.application.utils.propagate import make_cycle_function
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    UniversalNeighborlistParameters,
    VerletSkinState,
    skin_neighborlist,
)
from kups.core.propagator import propagate_and_fix
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import make_optimizer

from ._builders import LBFGS_OPTIMIZER, ar_cif, tmp_h5

# Ar fcc 2x2x2 -> 32 atoms, 10.6 A box; (cutoff + skin) / box < 0.5 (single
# image). The 0.2 A max step size keeps one step's margin consumption (at most
# 0.4 A) inside the 0.6 A budget, while the rattle is large enough that the
# displacement trigger fires within a few optimisation steps.
CUTOFF, SKIN, NUM_STEPS = 4.0, 0.6, 8

_LJ_PARAMS = LennardJonesParameters.from_dict(
    cutoff=CUTOFF, parameters={"Ar": (3.405, 0.010326)}, mixing_rule="lorentz_berthelot"
)


def _build(skin: float, rattle: float = 0.15):
    """A relaxation propagator and its initial state, with or without a skin."""
    state_lens = identity_lens(RelaxState)
    optimizer = make_optimizer(LBFGS_OPTIMIZER)
    potential = make_lennard_jones_from_state(
        state_lens,
        parameters=_LJ_PARAMS,
        gradient=POSITIONS_ONLY,
        neighborlist_factory=skin_neighborlist
        if skin > 0
        else AdaptiveNeighborList.from_state,
    )
    particles, systems = relax_state_from_ase(ar_cif(rattle, cubic=True))
    cutoffs = _LJ_PARAMS.cutoff
    propagator, opt_init = make_relax_propagator(
        state_lens,
        potential,
        optimizer,
        POSITIONS_ONLY,
        verlet_skin=skin,
        cutoffs=cutoffs,
    )
    state = RelaxState(
        particles=particles,
        systems=systems,
        neighborlist_params=UniversalNeighborlistParameters.estimate(
            particles.data.system.counts, systems, cutoffs
        ),
        opt_state=opt_init(particles, systems),
        step=jnp.array([0]),
        verlet_skin=VerletSkinState.seed(particles, systems, cutoffs, skin)
        if skin > 0
        else None,
    )
    return propagator, state


def test_verlet_relaxation_matches_dense() -> None:
    """The on-device trigger reproduces the every-step dense optimisation
    trajectory, and it actually fires at least once."""
    dense_prop, dense_state = _build(0.0)
    verlet_prop, verlet_state = _build(SKIN)

    dense = make_cycle_function(dense_prop)
    verlet = make_cycle_function(verlet_prop)
    flags = []
    for i in range(NUM_STEPS):
        key = jax.random.key(i)  # relaxation consumes no randomness
        dense_state = propagate_and_fix(dense, key, dense_state)
        verlet_state = propagate_and_fix(verlet, key, verlet_state)
        assert verlet_state.verlet_skin is not None
        flags.append(bool(jax.device_get(verlet_state.verlet_skin.should_rebuild)))

    assert any(flags), "the rebuild trigger never fired; weaken the skin"
    max_diff = float(
        jnp.max(
            jnp.abs(
                dense_state.particles.data.positions
                - verlet_state.particles.data.positions
            )
        )
    )
    assert max_diff < 1e-8, f"Verlet diverged from dense: {max_diff:.2e}"
    assert int(verlet_state.step[0]) == NUM_STEPS


def test_unabsorbable_skin_escalates_with_config_hint() -> None:
    """A skin far below one optimisation step's motion cannot be repaired by
    any rebuild schedule; the backstop must fail with the configuration hint
    instead of silently missing pairs."""
    tiny = 1e-4
    prop, state = _build(tiny)
    cycle = make_cycle_function(prop)
    with pytest.raises(ValueError, match="cannot absorb"):
        propagate_and_fix(cycle, jax.random.key(0), state)


def test_run_entry_point_with_verlet_skin() -> None:
    """The unified relaxation entry point wires the skin path end to end
    (seeding, factory injection, HDF5 output readable by the analyzer)."""
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
            verlet_skin=SKIN,
        ),
        potential=LjPotentialConfig(
            cutoff=CUTOFF,
            parameters={"Ar": (3.405, 0.010326)},
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(ar_cif(rattle=0.15, cubic=True),),
    )
    run(config)
    results = analyze_relax_file(out_file)
    result = next(iter(results.values()))
    assert jnp.isfinite(jnp.asarray(result.final_energy)).item()
    assert jnp.isfinite(jnp.asarray(result.final_max_force)).item()
    assert result.n_steps >= 1


def test_verlet_group_is_seedable_with_undersized_params() -> None:
    """Deliberately undersized skin capacities are grown by the fix loop on the
    first traced steps instead of truncating the list (mirrors the MD test)."""
    prop, state = _build(SKIN)
    assert state.verlet_skin is not None
    sabotaged = dataclasses.replace(state.verlet_skin.neighborlist_params, avg_edges=1)
    group = VerletSkinState.seed(
        state.particles,
        state.systems,
        _LJ_PARAMS.cutoff,
        SKIN,
        params=sabotaged,
    )
    state = dataclasses.replace(state, verlet_skin=group)

    cycle = make_cycle_function(prop)
    for i in range(3):
        state = propagate_and_fix(cycle, jax.random.key(i), state)

    assert state.verlet_skin is not None
    assert state.verlet_skin.neighborlist_params.avg_edges > 1
    assert int(state.step[0]) == 3
