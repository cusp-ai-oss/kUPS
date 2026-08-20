# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD integration tests for the Verlet-skin neighbor list.

The rebuild schedule is driven by the real on-device trigger — no hardcoded
rebuild schedule — and the resulting trajectory is compared against the
every-step dense path with the same PRNG keys, both per-step and fused into
blocks. Pure trigger/refine unit tests live in
``test/core/neighborlist/test_verlet.py``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import ase.build
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from kups.application.md.data import (
    MdParameters,
    MdRunConfig,
    MdState,
    md_state_from_ase,
)
from kups.application.md.simulation import make_md_propagator
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.utils.propagate import make_cycle_function
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    UniversalNeighborlistParameters,
    VerletSkinState,
    skin_neighborlist,
)
from kups.core.propagator import LoopPropagator, propagate_and_fix
from kups.core.typing import SystemId
from kups.core.utils.jax import key_chain
from kups.potential.classical.lennard_jones import LennardJonesParameters

from .._builders import ar_cif, tmp_h5

# Ar fcc 3x3x3 -> 108 atoms, ~15.8 A box; (cutoff+skin)/box ~ 0.41 < 0.5 (single
# image). 200 K and a thin 0.4 A skin make the displacement trigger fire within
# a few tens of 2 fs steps.
CUTOFF, SKIN, NUM_STEPS = 6.0, 0.4, 25

_LJ_PARAMS = LennardJonesParameters.from_dict(
    cutoff=CUTOFF, parameters={"Ar": (3.4, 0.0103)}, mixing_rule="lorentz_berthelot"
)


def _cutoff_table(systems) -> Table[SystemId, Array]:
    return Table(systems.keys, jnp.full((len(systems.keys),), CUTOFF))


def _make_state(
    skin: float,
    integrator: str,
    skin_params: UniversalNeighborlistParameters | None = None,
) -> MdState:
    atoms = ase.build.bulk("Ar", "fcc", a=5.26, cubic=True) * (3, 3, 3)
    md = MdParameters(
        temperature=200.0,
        time_step=2.0,
        friction_coefficient=0.01,
        thermostat_time_constant=100.0,
        target_pressure=1e5,
        pressure_coupling_time=1000.0,
        compressibility=5e-10,
        minimum_scale_factor=0.9,
        integrator=cast(Any, integrator),
        initialize_momenta=True,
        verlet_skin=skin,
    )
    particles, systems = md_state_from_ase(atoms, md, key=jax.random.key(0))
    cutoffs = _cutoff_table(systems)
    state = MdState(
        particles=particles,
        systems=systems,
        neighborlist_params=UniversalNeighborlistParameters.estimate(
            particles.data.system.counts, systems, cutoffs
        ),
        step=jnp.array([0]),
    )
    if skin > 0:
        group = VerletSkinState.seed(
            particles, systems, cutoffs, skin, params=skin_params
        )
        state = dataclasses.replace(state, verlet_skin=group)
    return state


def _make_propagators(integrator: str, cutoffs: Table[SystemId, Array]):
    """(dense, verlet) propagators sharing the LJ parameters."""
    state_lens = identity_lens(MdState)
    dense_potential = make_lennard_jones_from_state(
        state_lens, parameters=_LJ_PARAMS, gradient=POSITIONS_AND_CELL
    )
    verlet_potential = make_lennard_jones_from_state(
        state_lens,
        parameters=_LJ_PARAMS,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=skin_neighborlist,
    )
    dense = make_md_propagator(state_lens, cast(Any, integrator), dense_potential)
    verlet = make_md_propagator(
        state_lens,
        cast(Any, integrator),
        verlet_potential,
        verlet_skin=SKIN,
        cutoffs=cutoffs,
    )
    return dense, verlet


def _positions_close(a: MdState, b: MdState, what: str) -> None:
    max_diff = float(
        jnp.max(jnp.abs(a.particles.data.positions - b.particles.data.positions))
    )
    assert max_diff < 1e-8, f"Verlet diverged from dense ({what}): {max_diff:.2e}"


@pytest.mark.parametrize("integrator", ["csvr", "csvr_npt"])
def test_verlet_trajectory_matches_dense(integrator: str) -> None:
    """The on-device trigger reproduces the every-step dense trajectory (same
    keys), and it actually fires at least once."""
    dense_state = _make_state(0.0, integrator)
    verlet_state = _make_state(SKIN, integrator)
    dense_prop, verlet_prop = _make_propagators(
        integrator, _cutoff_table(dense_state.systems)
    )

    dense = make_cycle_function(dense_prop)
    verlet = make_cycle_function(verlet_prop)
    dense_chain = key_chain(jax.random.key(1))
    verlet_chain = key_chain(jax.random.key(1))
    flags = []
    for _ in range(NUM_STEPS):
        dense_state = propagate_and_fix(dense, next(dense_chain), dense_state)
        verlet_state = propagate_and_fix(verlet, next(verlet_chain), verlet_state)
        assert verlet_state.verlet_skin is not None
        flags.append(bool(jax.device_get(verlet_state.verlet_skin.should_rebuild)))

    assert any(flags), "the rebuild trigger never fired; weaken the skin"
    _positions_close(dense_state, verlet_state, integrator)


def test_verlet_fused_blocks_match_dense() -> None:
    """The skin path composes with blocked stepping: rebuilds happen on device
    inside the fused loop and the trajectory still matches the dense one."""
    block_size = 5
    dense_state = _make_state(0.0, "csvr")
    verlet_state = _make_state(SKIN, "csvr")
    dense_prop, verlet_prop = _make_propagators(
        "csvr", _cutoff_table(dense_state.systems)
    )

    dense = make_cycle_function(LoopPropagator(dense_prop, block_size))
    verlet = make_cycle_function(LoopPropagator(verlet_prop, block_size))
    dense_chain = key_chain(jax.random.key(1))
    verlet_chain = key_chain(jax.random.key(1))
    for _ in range(NUM_STEPS // block_size):
        dense_state = propagate_and_fix(dense, next(dense_chain), dense_state)
        verlet_state = propagate_and_fix(verlet, next(verlet_chain), verlet_state)

    # No backstop replay happened (the extrapolating trigger kept ahead), so the
    # two trajectories are at the same step and directly comparable.
    assert int(verlet_state.step[0]) == NUM_STEPS
    assert int(dense_state.step[0]) == NUM_STEPS
    _positions_close(dense_state, verlet_state, "fused blocks")


@pytest.mark.parametrize(
    "sabotage",
    [
        # avg_edges only: the eager seed build OVERGROWS its local capacity (its
        # assertions cannot surface), which desynchronizes the stored edges from
        # the static params unless the seed refits them.
        pytest.param({"avg_edges": 1}, id="avg_edges"),
        pytest.param(
            {"avg_edges": 1, "avg_candidates": 1, "avg_image_candidates": 1},
            id="all",
        ),
    ],
)
def test_undersized_seed_capacities_are_repaired(sabotage: dict) -> None:
    """The eager seed build cannot surface capacity assertions, so the first
    traced step rebuilds under assertion coverage: deliberately undersized skin
    capacities must be grown by the fix loop, not silently truncate the list or
    crash the first trace with mismatched cond branch shapes."""
    healthy = _make_state(SKIN, "csvr")
    _, verlet_prop = _make_propagators("csvr", _cutoff_table(healthy.systems))
    assert healthy.verlet_skin is not None
    sabotaged = dataclasses.replace(healthy.verlet_skin.neighborlist_params, **sabotage)
    state = _make_state(SKIN, "csvr", skin_params=sabotaged)

    verlet = make_cycle_function(verlet_prop)
    chain = key_chain(jax.random.key(1))
    for _ in range(3):
        state = propagate_and_fix(verlet, next(chain), state)

    assert state.verlet_skin is not None
    assert state.verlet_skin.neighborlist_params.avg_edges > 1
    assert int(state.step[0]) == 3


def test_unabsorbable_skin_escalates_with_config_hint() -> None:
    """A skin far below one step's motion cannot be repaired by any rebuild
    schedule; the backstop must fail with the configuration hint instead of
    silently missing pairs."""
    state_lens = identity_lens(MdState)
    potential = make_lennard_jones_from_state(
        state_lens,
        parameters=_LJ_PARAMS,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=skin_neighborlist,
    )
    tiny = 1e-4
    state = _make_state(tiny, "csvr")
    prop = make_md_propagator(
        state_lens,
        cast(Any, "csvr"),
        potential,
        verlet_skin=tiny,
        cutoffs=_cutoff_table(state.systems),
    )
    cycle = make_cycle_function(prop)
    with pytest.raises(ValueError, match="cannot absorb"):
        propagate_and_fix(cycle, jax.random.key(1), state)


@pytest.mark.parametrize("block_size", [1, 3])
def test_run_entry_point_with_verlet_skin(block_size: int) -> None:
    """The unified MD entry point wires the skin path end to end (seeding,
    factory injection, fused blocks, HDF5 output readable by the analyzer)."""
    from kups.application.md.analysis import analyze_md_file
    from kups.application.simulations.md import Config, run
    from kups.application.simulations.potentials import LjPotentialConfig

    # The cubic 10.6 A cell (32 atoms) keeps (cutoff + skin) / box = 5.0 / 10.6
    # inside the single-image limit.
    inp_file = ar_cif(cubic=True)
    out_file = tmp_h5()
    config = Config(
        run=MdRunConfig(
            out_file=out_file,
            num_steps=6,
            num_warmup_steps=2,
            block_size=block_size,
            seed=42,
        ),
        md=MdParameters(
            temperature=100.0,
            time_step=2.0,
            friction_coefficient=1.0,
            thermostat_time_constant=100.0,
            target_pressure=1.0,
            pressure_coupling_time=1.0e10,
            compressibility=4.5e-5,
            minimum_scale_factor=1.0,
            integrator="baoab_langevin",
            initialize_momenta=True,
            verlet_skin=1.0,
        ),
        potential=LjPotentialConfig(
            cutoff=4.0,
            parameters={"Ar": (3.405, 0.010326)},
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(inp_file,),
    )
    run(config)
    results = analyze_md_file(out_file, n_blocks=2)
    result = next(iter(results.values()))
    assert jnp.isfinite(result.total_energy.mean).all().item()
    assert jnp.isfinite(result.temperature.mean).all().item()
