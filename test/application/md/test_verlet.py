# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD with the Verlet-skin cache refreshed before every step."""

from __future__ import annotations

import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from kups.application.potential.verlet import (
    make_skin_refresh_from_state,
    verlet_neighborlist_factory,
)
from kups.application.utils.propagate import make_cycle_function
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    UniversalNeighborlistParameters,
    VerletSkinState,
)
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    propagate_and_fix,
)
from kups.core.utils.jax import key_chain
from kups.md.integrators import Integrator
from kups.potential.classical.lennard_jones import LennardJonesParameters

from .._builders import ar_cif, tmp_h5

# Ar fcc 3x3x3: 108 atoms in a ~15.8 A box, so cutoff + skin stays below half
# the box. 200 K and a 0.4 A skin exhaust the skin within a few tens of steps.
CUTOFF, SKIN, NUM_STEPS = 6.0, 0.4, 25
TINY = UniversalNeighborlistParameters(1, 1, 1, 1)
LJ = LennardJonesParameters.from_dict(
    cutoff=CUTOFF, parameters={"Ar": (3.4, 0.0103)}, mixing_rule="lorentz_berthelot"
)


def _state(
    skin: float,
    integrator: Integrator,
    *,
    neighborlist_params: UniversalNeighborlistParameters | None = None,
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
        integrator=integrator,
        initialize_momenta=True,
        verlet_skin=skin,
    )
    particles, systems = md_state_from_ase(atoms, md, key=jax.random.key(0))
    counts = particles.data.system.counts
    cutoffs = Table(systems.keys, jnp.full(systems.size, CUTOFF))
    estimate = UniversalNeighborlistParameters.estimate
    return MdState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params or estimate(counts, systems, cutoffs),
        step=jnp.array([0]),
        verlet_skin=VerletSkinState.new(
            particles,
            systems,
            skin_params
            or estimate(counts, systems, cutoffs.map_data(lambda c: c + skin)),
        ),
    )


def _propagator(
    integrator: Integrator, skin: float, cached: bool
) -> Propagator[MdState]:
    """The app composition; ``cached=False`` keeps the refresh but ignores the cache."""
    state_lens = identity_lens(MdState)
    cache = state_lens.focus(lambda s: s.verlet_skin)
    potential = make_lennard_jones_from_state(
        state_lens,
        parameters=LJ,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=(
            verlet_neighborlist_factory(cache) if cached else AdaptiveNeighborList.new
        ),
    )
    refresh = make_skin_refresh_from_state(state_lens, LJ.cutoff, skin)
    step = make_md_propagator(state_lens, integrator, potential)
    return ResetOnErrorPropagator(SequentialPropagator((refresh, step)))


def _run(propagator: Propagator[MdState], state: MdState, steps: int) -> MdState:
    cycle = make_cycle_function(propagator)
    chain = key_chain(jax.random.key(1))
    for _ in range(steps):
        state = propagate_and_fix(cycle, next(chain), state)
    return state


def _assert_same_positions(a: MdState, b: MdState) -> None:
    np.testing.assert_allclose(
        a.particles.data.positions, b.particles.data.positions, rtol=0, atol=1e-8
    )


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count executed ``AdaptiveNeighborList`` builds, per call site at run time."""
    executed: list[int] = []
    call = AdaptiveNeighborList.__call__

    def counted(self, keys, systems, *, queries=None, queried_keys=None):
        jax.debug.callback(lambda: executed.append(1))
        if queries is not None:
            return call(self, keys, systems, queries=queries)
        return call(self, keys, systems, queried_keys=queried_keys)

    monkeypatch.setattr(AdaptiveNeighborList, "__call__", counted)
    return executed


@pytest.mark.parametrize("integrator", ["csvr", "csvr_npt"])
def test_cached_trajectory_matches_fresh_builds(integrator: Integrator) -> None:
    plain = _run(
        _propagator(integrator, SKIN, False), _state(SKIN, integrator), NUM_STEPS
    )
    cached = _run(
        _propagator(integrator, SKIN, True), _state(SKIN, integrator), NUM_STEPS
    )
    initial = _state(SKIN, integrator).particles.data.positions
    rebuilt_at = cached.verlet_skin.reference.particles.data.positions
    assert not np.array_equal(rebuilt_at, initial), "the skin never ran out"
    _assert_same_positions(plain, cached)


def test_cache_saves_most_builds(builds: list[int]) -> None:
    _run(_propagator("csvr", SKIN, False), _state(SKIN, "csvr"), NUM_STEPS)
    jax.effects_barrier()
    plain = len(builds)
    builds.clear()
    _run(_propagator("csvr", SKIN, True), _state(SKIN, "csvr"), NUM_STEPS)
    jax.effects_barrier()
    assert plain >= NUM_STEPS
    assert 2 <= len(builds) <= NUM_STEPS // 3


def test_fused_blocks_match_fresh_builds() -> None:
    block = 5

    def run(cached: bool) -> MdState:
        loop = LoopPropagator(_propagator("csvr", SKIN, cached), block)
        return _run(loop, _state(SKIN, "csvr"), NUM_STEPS // block)

    plain, cached = run(False), run(True)
    assert int(cached.step[0]) == NUM_STEPS
    _assert_same_positions(plain, cached)


def test_undersized_capacities_are_repaired() -> None:
    plain = _run(_propagator("csvr", SKIN, False), _state(SKIN, "csvr"), 3)
    state = _state(SKIN, "csvr", neighborlist_params=TINY, skin_params=TINY)
    cached = _run(_propagator("csvr", SKIN, True), state, 3)
    assert cached.neighborlist_params.avg_edges > TINY.avg_edges
    assert cached.verlet_skin.params.avg_edges > TINY.avg_edges
    _assert_same_positions(plain, cached)


def test_zero_skin_builds_every_step_without_a_cache() -> None:
    plain = _run(_propagator("csvr", 0.0, False), _state(0.0, "csvr"), 3)
    cached = _run(_propagator("csvr", 0.0, True), _state(0.0, "csvr"), 3)
    np.testing.assert_array_equal(cached.verlet_skin.radii.data, 0.0)
    _assert_same_positions(plain, cached)


@pytest.mark.parametrize("block_size", [1, 3])
def test_run_entry_point(block_size: int) -> None:
    from kups.application.md.analysis import analyze_md_file
    from kups.application.simulations.md import Config, run
    from kups.application.simulations.potentials import LjPotentialConfig

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
        ),
        potential=LjPotentialConfig(
            cutoff=4.0,
            parameters={"Ar": (3.405, 0.010326)},
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(ar_cif(cubic=True),),
    )
    run(config)
    result = next(iter(analyze_md_file(out_file, n_blocks=2).values()))
    assert jnp.isfinite(result.total_energy.mean).all().item()
    assert jnp.isfinite(result.temperature.mean).all().item()
