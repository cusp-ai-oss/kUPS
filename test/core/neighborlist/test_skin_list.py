# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""A stored skin list returns the pairs of a fresh search while its skin lasts."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.cell import Cell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    Edges,
    SkinList,
    SkinListUpdate,
    SkinNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.propagator import propagate_and_fix
from kups.core.result import as_result_function
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from test.units import A

from ._builders import SamplePoints, SampleSystems


@contextmanager
def _x64(enabled: bool) -> Iterator[None]:
    """Scoped x64 toggle; `jax.enable_x64` only exists from jax 0.8."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


CUTOFF, SKIN = 3.0, 1.0


@dataclass
class _State:
    particles: Table[ParticleId, SamplePoints]
    systems: Table[SystemId, SampleSystems]
    neighborlist_params: UniversalNeighborlistParameters
    skin_list: SkinList


LENS = identity_lens(_State)
UPDATE = jax.jit(as_result_function(SkinListUpdate(LENS)))


def _state(
    positions: Array,
    lattice: Array,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    n_sys: int = 1,
    compacted: bool = False,
) -> _State:
    """``n_sys`` copies of one cell, each with the same share of the particles.

    Reads mask the stored edges in place, or compact them if ``compacted``.
    """
    n = len(positions)
    system = Index(tuple(SystemId(i) for i in range(n_sys)), jnp.arange(n) * n_sys // n)
    points = SamplePoints(
        positions * A,
        system,
        system.to_cls(InclusionId),
        Index.integer(jnp.arange(n), n=n, label=ExclusionId),
    )
    particles = Table(tuple(ParticleId(i) for i in range(n)), points)
    vectors = jnp.repeat(jnp.asarray(lattice, positions.dtype)[None], n_sys, axis=0)
    cell = Cell.from_pbc(TriclinicFrame.from_matrix(vectors * A), periodic)
    systems = Table(system.keys, SampleSystems(cell))
    cutoffs = systems.set_data(jnp.full(n_sys, CUTOFF, positions.dtype) * A)
    skin = jnp.asarray(SKIN, positions.dtype) * A
    skin_list = SkinList.create(particles, systems, cutoffs, skin)
    capacities = skin_list.parameters
    if compacted:
        capacities = UniversalNeighborlistParameters.estimate(
            system.counts, systems, cutoffs
        )
        assert capacities.avg_edges < skin_list.parameters.avg_edges
    return _State(particles, systems, capacities, skin_list)


def _moved(state: _State, positions: Array) -> _State:
    data = state.particles.data
    points = SamplePoints(positions * A, data.system, data.inclusion, data.exclusion)
    return dataclasses.replace(state, particles=state.particles.set_data(points))


def _pairs(state: _State, edges: Edges) -> set[tuple[int, int, int, int, int]]:
    """Each pair with its vector, which does not depend on how positions wrap."""
    raw = np.asarray(edges.indices.raw_indices)
    vectors = edges.difference_vectors(state.particles, state.systems).to(A)
    vectors = np.round(np.asarray(vectors)[:, 0], 3)
    keep = raw[:, 0] < state.particles.size
    return {
        (int(i), int(j), *map(float, v))
        for (i, j), v in zip(raw[keep], vectors[keep], strict=True)
    }


@jax.jit
@as_result_function
def _edges(state: _State, queried_keys: Index[ParticleId] | None) -> Edges:
    reader = SkinNeighborList.from_state(state, state.skin_list.cutoffs)
    return reader(state.particles, state.systems, queried_keys=queried_keys)


def _read(
    state: _State, queried_keys: Index[ParticleId] | None = None
) -> set[tuple[int, int, int, int, int]]:
    while (result := _edges(state, queried_keys)).failed_assertions:
        state = result.fix_or_raise(state)
    size = state.particles.size if queried_keys is None else queried_keys.size
    assert len(result.value) <= state.neighborlist_params.avg_edges * size
    return _pairs(state, result.value)


def _searched(state: _State) -> set[tuple[int, int, int, int, int]]:
    """Pairs of a search at the cutoff: a list created without a skin."""
    stored = state.skin_list
    fresh = SkinList.create(
        state.particles, state.systems, stored.cutoffs, 0.0 * stored.skin
    )
    return _pairs(state, fresh.edges)


def _fluid(n: int, box: float, dtype: type, seed: int = 0) -> Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0, box, (n, 3)), dtype)


CELLS = {
    "cubic": 9.0 * np.eye(3),
    "triclinic": np.array([[9.0, 0.0, 0.0], [2.0, 8.5, 0.0], [1.0, -1.5, 9.5]]),
    # narrower than twice the cutoff: a particle pairs with several images
    "narrow": 4.5 * np.eye(3),
}


PERIODIC = [(True, True, True), (True, True, False), (False, False, False)]


@pytest.mark.parametrize(
    ("cell", "periodic", "dtype", "n_sys", "compacted"),
    [
        (cell, periodic, dtype, n_sys, compacted)
        for i, cell in enumerate(CELLS)
        for j, periodic in enumerate(PERIODIC)
        for dtype, n_sys, compacted in [
            (jnp.float32, 1 + (i + j) % 2, bool(j % 2)),
            (jnp.float64, 2 - (i + j) % 2, not j % 2),
        ]
    ],
)
def test_reads_the_pairs_of_a_search_after_motion_and_wraps(
    cell: str,
    periodic: tuple[bool, bool, bool],
    dtype: type,
    n_sys: int,
    compacted: bool,
):
    with _x64(dtype == jnp.float64):
        lattice = CELLS[cell]
        n = 20 if cell == "narrow" else 80
        positions = _fluid(n, float(lattice[0, 0]), dtype)
        state = _state(positions, lattice, periodic, n_sys, compacted)
        assert _read(state) == _searched(state)
        # Less than half the skin of motion, then whole cells of displacement.
        rng = np.random.default_rng(1)
        step = rng.normal(size=positions.shape)
        step *= 0.3 * SKIN / np.linalg.norm(step, axis=-1, keepdims=True)
        images = rng.integers(-1, 2, positions.shape) * np.asarray(periodic)
        moved = positions + jnp.asarray(step + images @ lattice, dtype)
        state = _moved(state, moved)
        # A cell narrower than twice the build radius has no skin to spend.
        complete = state.skin_list.is_complete(state.particles, state.systems)
        assert bool(complete) is (cell != "narrow" or not any(periodic))
        state = propagate_and_fix(UPDATE, jax.random.key(0), state)
        assert _read(state) == _searched(state)


def test_update_rebuilds_only_when_the_skin_is_used_up():
    positions = _fluid(60, 9.0, jnp.float32)
    state = _state(positions, CELLS["cubic"])
    near = _moved(state, positions.at[0, 0].add(0.49 * SKIN))
    out = UPDATE(jax.random.key(0), near)
    out.raise_assertion()
    npt.assert_array_equal(
        out.value.skin_list.reference.positions.to(A), positions, strict=False
    )
    far = _moved(state, positions.at[0, 0].add(0.51 * SKIN))
    assert not bool(far.skin_list.is_complete(far.particles, far.systems))
    out = UPDATE(jax.random.key(0), far)
    out.raise_assertion()
    npt.assert_array_equal(
        out.value.skin_list.reference.positions.to(A),
        far.particles.data.positions.to(A),
    )
    assert _read(out.value) == _searched(far)


def test_update_rebuilds_after_the_cell_shrank():
    positions = _fluid(60, 9.0, jnp.float32)
    state = _state(positions, CELLS["cubic"])
    scale = 0.7
    cell = Cell.from_pbc(
        TriclinicFrame.from_matrix(
            jnp.asarray(scale * CELLS["cubic"], positions.dtype)[None] * A
        ),
        (True, True, True),
    )
    shrunk = dataclasses.replace(
        _moved(state, scale * positions),
        systems=state.systems.set_data(SampleSystems(cell)),
    )
    assert not bool(shrunk.skin_list.is_complete(shrunk.particles, shrunk.systems))
    updated = propagate_and_fix(UPDATE, jax.random.key(0), shrunk)
    assert _read(updated) == _searched(shrunk)


def test_a_rebuild_that_outgrows_its_capacities_keeps_the_list_and_is_fixed():
    positions = _fluid(60, 9.0, jnp.float32)
    state = _state(positions, CELLS["cubic"])
    stored = state.skin_list
    # All particles in one corner: more pairs than the stored edges hold.
    dense = _moved(state, positions / 3)
    out = UPDATE(jax.random.key(0), dense)
    assert out.failed_assertions
    npt.assert_array_equal(
        out.value.skin_list.edges.indices.raw_indices, stored.edges.indices.raw_indices
    )
    npt.assert_array_equal(
        out.value.skin_list.reference.positions.to(A), stored.reference.positions.to(A)
    )
    fixed = propagate_and_fix(UPDATE, jax.random.key(0), dense)
    assert fixed.skin_list.parameters.avg_edges > stored.parameters.avg_edges
    assert len(fixed.skin_list.edges) == (
        fixed.skin_list.parameters.avg_edges * dense.particles.size
    )
    assert _read(fixed) == _searched(dense)


def test_reader_refuses_cutoffs_beyond_the_list():
    state = _state(_fluid(20, 9.0, jnp.float32), CELLS["cubic"])
    cutoffs = state.skin_list.cutoffs
    reader = SkinNeighborList.from_state(state, cutoffs.set_data(cutoffs.data * 1.1))
    result = jax.jit(as_result_function(reader))(state.particles, state.systems)
    with pytest.raises(AssertionError, match="exceed the skin list"):
        result.raise_assertion()


@pytest.mark.parametrize("compacted", [False, True])
def test_reader_returns_the_pairs_of_queried_keys(compacted: bool):
    state = _state(_fluid(40, 9.0, jnp.float32), CELLS["cubic"], compacted=compacted)
    queried = Index(state.particles.keys, jnp.array([3, 7]))
    touching = {p for p in _searched(state) if p[0] in (3, 7) or p[1] in (3, 7)}
    assert _read(state, queried) == touching
