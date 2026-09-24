# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet-skin refresh and neighbor-list factory bound to conventional fields."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.potential.verlet import (
    make_skin_refresh_from_state,
    verlet_neighborlist_factory,
)
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.lens import Lens, bind, identity_lens, lens
from kups.core.neighborlist import (
    UniversalNeighborlistParameters,
    VerletSkinState,
    skin_covers,
)
from kups.core.propagator import propagator_with_assertions
from kups.core.result import as_result_function
from kups.core.typing import ExclusionId, InclusionId, Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import LennardJonesParameters

N_PARTICLES, BOX, CUTOFF, SKIN = 48, 9.0, 3.5, 0.8
TINY = UniversalNeighborlistParameters(1, 1, 1, 1)
LJ = LennardJonesParameters.from_dict(
    cutoff=CUTOFF, parameters={"Ar": (1.0, 0.5)}, mixing_rule="lorentz_berthelot"
)
CUTOFFS = Table.arange(jnp.array([CUTOFF]), label=SystemId)


@dataclass
class Particles:
    positions: Array
    labels: Index[Label]
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class Systems:
    cell: PeriodicCell


@dataclass
class State:
    particles: Table[ParticleId, Particles]
    systems: Table[SystemId, Systems]
    neighborlist_params: UniversalNeighborlistParameters
    verlet_skin: VerletSkinState


@dataclass
class Outer:
    inner: State


def _state() -> State:
    particles = Table.arange(
        Particles(
            positions=jax.random.uniform(jax.random.key(0), (N_PARTICLES, 3)) * BOX,
            labels=Index.new([Label("Ar")] * N_PARTICLES),
            system=Index.new([SystemId(0)] * N_PARTICLES),
            inclusion=Index.new([InclusionId(0)] * N_PARTICLES),
            exclusion=Index.new([ExclusionId(i) for i in range(N_PARTICLES)]),
        ),
        label=ParticleId,
    )
    cell = PeriodicCell(TriclinicFrame.from_matrix(BOX * jnp.eye(3)[None]))
    systems = Table.arange(Systems(cell), label=SystemId)
    cache = VerletSkinState.new(particles, systems, TINY)
    return State(particles, systems, TINY, cache)


def _refreshed[S](state_lens: Lens[S, State], state: S) -> S:
    step = jax.jit(
        propagator_with_assertions(
            make_skin_refresh_from_state(state_lens, CUTOFFS, SKIN)
        )
    )
    for _ in range(10):
        result = step(jax.random.key(0), state)
        if result.all_assertions_pass:
            return result.value
        state = result.fix_or_raise(state)
    raise AssertionError("capacity repair did not converge")


def _repaired[S](potential, state: S) -> tuple[object, S]:
    evaluate = jax.jit(as_result_function(lambda s: potential(s).data))
    for _ in range(10):
        result = evaluate(state)
        if result.all_assertions_pass:
            return result.value, state
        state = result.fix_or_raise(state)
    raise AssertionError("capacity repair did not converge")


def test_refresh_commits_a_covering_cache_through_a_nested_lens() -> None:
    focus = lens(lambda s: s.inner, cls=Outer)
    outer = _refreshed(focus, Outer(_state()))
    inner = outer.inner
    assert inner.verlet_skin.params.avg_edges > TINY.avg_edges
    np.testing.assert_allclose(inner.verlet_skin.radii.data, [CUTOFF + SKIN])
    assert bool(skin_covers(inner.verlet_skin, inner.particles, inner.systems, CUTOFFS))


def test_refresh_keeps_a_covering_cache() -> None:
    state = _refreshed(identity_lens(State), _state())
    moved = (
        bind(state)
        .focus(lambda s: s.particles.data.positions)
        .apply(lambda x: x.at[0, 0].add(0.2 * SKIN))
    )
    again = _refreshed(identity_lens(State), moved)
    np.testing.assert_array_equal(
        again.verlet_skin.reference.particles.data.positions,
        state.particles.data.positions,
    )


def test_factory_reads_the_cache_and_matches_a_plain_potential() -> None:
    focus = lens(lambda s: s.inner, cls=Outer)
    outer = _refreshed(focus, Outer(_state()))
    moved = (
        bind(outer)
        .focus(lambda s: s.inner.particles.data.positions)
        .apply(lambda x: x.at[0, 0].add(0.2 * SKIN))
    )
    inner = moved.inner
    assert bool(skin_covers(inner.verlet_skin, inner.particles, inner.systems, CUTOFFS))
    cached = make_lennard_jones_from_state(
        focus,
        parameters=LJ,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=verlet_neighborlist_factory(
            focus.focus(lambda s: s.verlet_skin)
        ),
    )
    plain = make_lennard_jones_from_state(
        focus, parameters=LJ, gradient=POSITIONS_AND_CELL
    )
    actual, repaired = _repaired(cached, moved)
    expected, _ = _repaired(plain, moved)
    assert repaired.inner.neighborlist_params.avg_edges > TINY.avg_edges
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-12)
