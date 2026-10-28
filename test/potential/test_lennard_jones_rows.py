# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""LJ over a skin list's rows equals the graph LJ through wraps, rebuilds and strain."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Table
from kups.core.lens import bind, const_lens, lens, view
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    SkinList,
    wraps_since_build,
)
from kups.core.patch import IdPatch
from kups.core.potential import EMPTY_LENS, PotentialAsPropagator
from kups.core.propagator import propagate_and_fix
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    make_lennard_jones_potential,
    make_lennard_jones_row_potential,
)
from kups.potential.common.graph import GRAPH_GEOMETRY
from kups.potential.common.rows import ROW_GEOMETRY

from ._rows import (
    COUNTS,
    LATTICES,
    SKIN,
    Particles,
    Systems,
    lj_systems,
)


@dataclass
class _State:
    particles: Table[ParticleId, Particles]
    systems: Table[SystemId, Systems]
    skin_list: SkinList


def _setup() -> tuple[_State, LennardJonesParameters]:
    particles, systems, params = lj_systems()
    skin = Table.arange(jnp.array([SKIN]), label=SystemId)
    skin_list = SkinList.create(particles, systems, params.cutoff, skin)
    return _State(particles, systems, skin_list), params


def _potentials(params: LennardJonesParameters):
    n = sum(COUNTS)
    dense = DenseNearestNeighborList(
        FixedCapacity(n), FixedCapacity(128), FixedCapacity(32 * n), params.cutoff
    )
    rows = make_lennard_jones_row_potential(
        view(lambda s: s.particles),
        view(lambda s: s.systems),
        lens(lambda s: s.skin_list, cls=_State),
        const_lens(params),
        ROW_GEOMETRY.nest(POSITIONS_AND_CELL),
    )
    graph = make_lennard_jones_potential(
        view(lambda s: s.particles),
        view(lambda s: s.systems),
        lambda s: dense,
        const_lens(params),
        None,
        GRAPH_GEOMETRY.nest(POSITIONS_AND_CELL),
        EMPTY_LENS,
        EMPTY_LENS,
    )
    return rows, graph


def _assert_match(state: _State, params: LennardJonesParameters):
    rows, graph = _potentials(params)
    a, b = jax.jit(lambda s: (rows(s).data, graph(s).data))(state)
    np.testing.assert_allclose(
        a.total_energies.data, b.total_energies.data, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        a.gradients.positions.data, b.gradients.positions.data, atol=1e-10
    )
    np.testing.assert_allclose(
        a.gradients.cell.data.vectors, b.gradients.cell.data.vectors, atol=1e-10
    )


def _step(params: LennardJonesParameters):
    """Evaluate the row potential and apply its patch, which stores the skin list
    it read."""
    rows, _ = _potentials(params)
    return jax.jit(as_result_function(PotentialAsPropagator(rows)))


def _moved(state: _State, positions: Array, lattices: Array | None = None) -> _State:
    if lattices is not None:
        cell = PeriodicCell(TriclinicFrame.from_matrix(lattices))
        state = bind(state, lambda s: s.systems).set(
            state.systems.set_data(Systems(cell))
        )
    # integrators wrap positions into the cell
    cell = state.systems[state.particles.data.system].cell
    return bind(state, lambda s: s.particles.data.positions).set(cell.wrap(positions))


def _wraps(state: _State) -> Array:
    system = state.particles.data.system
    frames = state.systems[system].cell.frame
    fractional = frames.to_fractional(state.particles.data.positions)
    return wraps_since_build(
        state.skin_list.reference, fractional, system, state.systems
    )


def test_random_walk_with_wraps_and_rebuilds():
    state, params = _setup()
    step = _step(params)
    _assert_match(state, params)
    rng = np.random.default_rng(1)
    rebuilds, wrapped = 0, False
    for i in range(16):
        noise = rng.normal(0, 0.08, (state.particles.size, 3))
        moved = _moved(state, state.particles.data.positions + noise)
        state = propagate_and_fix(step, jax.random.key(i), moved)
        rebuilds += not np.array_equal(
            moved.skin_list.reference.positions, state.skin_list.reference.positions
        )
        wrapped |= bool(jnp.any(_wraps(state) != 0))
        _assert_match(state, params)
    assert 0 < rebuilds < 16 and wrapped


def test_strain_and_shear():
    state, params = _setup()
    step = _step(params)
    strain = jnp.eye(3) + 0.01 * jnp.array(
        [[-1.0, 0.3, 0.0], [0.0, -0.5, 0.2], [0.0, 0.0, 0.8]]
    )
    lattices = jnp.asarray(LATTICES)
    for i in range(8):
        frames = state.systems[state.particles.data.system].cell.frame
        fractional = frames.to_fractional(state.particles.data.positions)
        lattices = lattices @ strain
        system = state.particles.data.system.raw_indices
        positions = jnp.einsum("ni,nij->nj", fractional, lattices[system])
        state = propagate_and_fix(
            step, jax.random.key(i), _moved(state, positions, lattices)
        )
        _assert_match(state, params)


def test_evaluation_rebuilds_a_stale_list():
    state, params = _setup()
    # The stored list lacks pairs of another draw, so the read must rebuild it.
    other, _, _ = lj_systems(seed=1)
    moved = _moved(state, other.data.positions)
    assert not moved.skin_list.is_complete(moved.particles, moved.systems)
    _assert_match(moved, params)


def test_proposals_raise():
    state, params = _setup()
    rows, _ = _potentials(params)
    with pytest.raises(ValueError, match="whole states"):
        rows(state, IdPatch())
