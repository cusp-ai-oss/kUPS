# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Neighbor-list capacity repair for potentials bound to a nested sub-state."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.application.potential.classical.blocking import (
    make_blocking_spheres_from_state,
)
from kups.application.potential.classical.coulomb import make_coulomb_vacuum_from_state
from kups.application.potential.classical.ewald import make_ewald_from_state
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.core.cell import AnyPeriodicity, Cell, PeriodicCell, TriclinicFrame
from kups.core.cell import VacuumCell
from kups.core.data import Index, Table
from kups.core.lens import Lens, identity_lens, lens
from kups.core.neighborlist import CellListNeighborList, UniversalNeighborlistParameters
from kups.core.result import as_result_function
from kups.core.typing import (
    ExclusionId,
    GroupId,
    InclusionId,
    Label,
    MotifId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass
from kups.potential.classical.blocking import BlockingSpheresParameters
from kups.potential.classical.ewald import EwaldParameters
from kups.potential.classical.lennard_jones import LennardJonesParameters

N_PARTICLES, BOX, CUTOFF = 48, 9.0, 3.5
TINY = UniversalNeighborlistParameters(1, 1, 1, 1)


@dataclass
class Particles:
    positions: Array
    labels: Index[Label]
    charges: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]
    group: Index[GroupId]


@dataclass
class Systems[P: AnyPeriodicity]:
    cell: Cell[P]


@dataclass
class Groups:
    motif: Index[MotifId]


@dataclass
class State[P: AnyPeriodicity]:
    particles: Table[ParticleId, Particles]
    systems: Table[SystemId, Systems[P]]
    groups: Table[GroupId, Groups]
    neighborlist_params: UniversalNeighborlistParameters


@dataclass
class Outer[P: AnyPeriodicity]:
    inner: State[P]


def _state[P: AnyPeriodicity](cell: Cell[P]) -> State[P]:
    particles = Table.arange(
        Particles(
            positions=jax.random.uniform(jax.random.key(0), (N_PARTICLES, 3)) * BOX,
            labels=Index.new([Label("Ar")] * N_PARTICLES),
            charges=jnp.where(jnp.arange(N_PARTICLES) % 2 == 0, 1.0, -1.0),
            system=Index.new([SystemId(0)] * N_PARTICLES),
            inclusion=Index.new([InclusionId(0)] * N_PARTICLES),
            exclusion=Index.new([ExclusionId(i) for i in range(N_PARTICLES)]),
            group=Index.new([GroupId(i) for i in range(N_PARTICLES)]),
        ),
        label=ParticleId,
    )
    return State(
        particles,
        Table.arange(Systems(cell), label=SystemId),
        Table.arange(Groups(Index.new([MotifId(0)] * N_PARTICLES)), label=GroupId),
        TINY,
    )


FRAME = TriclinicFrame.from_matrix(BOX * jnp.eye(3)[None])
PERIODIC, VACUUM = PeriodicCell(FRAME), VacuumCell(FRAME)
LJ = LennardJonesParameters.from_dict(
    cutoff=CUTOFF, parameters={"Ar": (1.0, 0.5)}, mixing_rule="lorentz_berthelot"
)
SPHERES = BlockingSpheresParameters(
    radii=jnp.array([2.0, 2.5]),
    positions=jnp.array([[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]]),
    system=Index.new([SystemId(0), SystemId(0)]),
    motif=Index.new([MotifId(0), MotifId(0)]),
)
_EWALD_STATE = _state(PERIODIC)
EWALD = EwaldParameters.make(
    _EWALD_STATE.particles, _EWALD_STATE.systems, real_cutoff=CUTOFF
)

type Binding = Callable[[Lens[Any, Any]], Any]

BINDINGS: dict[str, tuple[Binding, Cell[Any]]] = {
    "lennard_jones": (
        lambda s: make_lennard_jones_from_state(
            s, parameters=LJ, gradient=POSITIONS_AND_CELL
        ),
        PERIODIC,
    ),
    "coulomb_vacuum": (
        lambda s: make_coulomb_vacuum_from_state(
            s, parameters=Table.arange(jnp.array([CUTOFF]), label=SystemId)
        ),
        VACUUM,
    ),
    "ewald": (lambda s: make_ewald_from_state(s, parameters=EWALD), PERIODIC),
    "blocking_spheres": (
        lambda s: make_blocking_spheres_from_state(s, parameters=SPHERES),
        PERIODIC,
    ),
}


def _repaired[S](potential: Any, state: S) -> tuple[Any, S]:
    evaluate = jax.jit(as_result_function(lambda s: potential(s).data))
    for _ in range(12):
        result = evaluate(state)
        if result.all_assertions_pass:
            return result.value, state
        state = result.fix_or_raise(state)
    raise AssertionError("capacity repair did not converge")


def _assert_nested_repair_matches[P: AnyPeriodicity](
    bind: Binding, cell: Cell[P]
) -> None:
    expected, _ = _repaired(bind(identity_lens(State)), _state(cell))
    actual, repaired = _repaired(
        bind(lens(lambda s: s.inner, cls=Outer)), Outer(_state(cell))
    )
    assert repaired.inner.neighborlist_params.avg_edges > TINY.avg_edges
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", BINDINGS)
def test_default_factory_repairs_through_a_nested_state_lens(name: str) -> None:
    _assert_nested_repair_matches(*BINDINGS[name])


def test_injected_factory_repairs_through_a_nested_state_lens() -> None:
    _assert_nested_repair_matches(
        lambda s: make_lennard_jones_from_state(
            s,
            parameters=LJ,
            gradient=POSITIONS_AND_CELL,
            neighborlist_factory=CellListNeighborList.new,
        ),
        PERIODIC,
    )
