# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""``RowPairEnergy``: Lennard-Jones over a skin list's rows against the graph path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.neighborlist import DenseNearestNeighborList, Edges
from kups.core.result import as_result_function
from kups.core.typing import SystemId
from kups.potential.classical.lennard_jones import (
    lennard_jones_energy,
    lennard_jones_row_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph, PointCloud
from kups.potential.common.rows import RowPotentialInput

from ._rows import SKIN, lj_systems, row_energies, rows_skin_list, value_and_grad


def _dense_edges(particles: Table, systems: Table, cutoffs: Table) -> Edges:
    n = particles.size
    nl = DenseNearestNeighborList(
        FixedCapacity(n), FixedCapacity(128), FixedCapacity(32 * n), cutoffs
    )
    return nl(particles, systems).data


def _graph(params, edges: Edges):
    def energy(particles: Table, systems: Table) -> Array:
        graph = HyperGraph(particles, systems, edges)
        return lennard_jones_energy(GraphPotentialInput(params, graph)).data.data

    return energy


def _assert_match(particles: Table, systems: Table, params, skin_list):
    valid = np.asarray(particles.data.system.valid_mask)
    edges = _dense_edges(particles, systems, params.cutoff)
    e_rows, g_rows, c_rows = value_and_grad(
        row_energies(lennard_jones_row_energy, params, skin_list), particles, systems
    )
    e_graph, g_graph, c_graph = value_and_grad(
        _graph(params, edges), particles, systems
    )
    np.testing.assert_allclose(e_rows, e_graph, rtol=1e-12, atol=1e-12)
    assert np.all(np.abs(e_graph) > 1e-3)
    np.testing.assert_allclose(g_rows[valid], g_graph[valid], atol=1e-10)
    np.testing.assert_array_equal(g_rows[~valid], 0.0)
    for a, b in zip(jax.tree.leaves(c_rows), jax.tree.leaves(c_graph), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("groups", [False, True])
@pytest.mark.parametrize("periodic", [(True, True, True), (True, True, False)])
def test_matches_the_graph_path(periodic, groups):
    particles, systems, params = lj_systems(periodic, groups=groups)
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    rows = skin_list.rows
    filled = int(rows.offsets[-1])
    self_images = rows.neighbors[:filled] == rows.centers[:filled]
    assert np.any(np.asarray(self_images))
    _assert_match(particles, systems, params, skin_list)


def test_displacements_and_wraps_since_the_build():
    particles, systems, params = lj_systems()
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    rng = np.random.default_rng(1)
    step = rng.normal(size=(particles.size, 3))
    step *= SKIN / 2 / np.linalg.norm(step, axis=-1, keepdims=True)
    cell = systems[particles.data.system].cell
    moved = bind(particles, lambda p: p.data.positions).apply(
        lambda x: cell.wrap(x + step)
    )
    # integrators wrap positions into the cell: some particles crossed a face
    shift = np.abs(np.asarray(moved.data.positions - particles.data.positions))
    assert np.any(shift > 1)
    _assert_match(moved, systems, params, skin_list)


def test_one_cutoff_for_all_systems():
    particles, systems, params = lj_systems()
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    one = Table((SystemId(0),), jnp.array([2.5]))
    _assert_match(
        particles, systems, bind(params, lambda p: p.cutoff).set(one), skin_list
    )


def test_inactive_particles():
    particles, systems, params = lj_systems()
    raw = np.asarray(particles.data.system.raw_indices).copy()
    raw[:3] = systems.size  # out of bounds: inactive
    inactive = Index(particles.data.system.keys, jnp.asarray(raw))
    particles = bind(particles, lambda p: p.data.system).set(inactive)
    particles = bind(particles, lambda p: p.data.positions).apply(
        lambda x: x.at[:3].set(jnp.nan)
    )
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    _assert_match(particles, systems, params, skin_list)


def test_tables_are_not_differentiable():
    particles, systems, params = lj_systems()
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    cloud = PointCloud(particles, systems)

    def energy(epsilon: Array) -> Array:
        p = bind(params, lambda q: q.epsilon).set(epsilon)
        inp = RowPotentialInput(p, cloud, skin_list)
        return lennard_jones_row_energy(inp).data.data.sum()

    with pytest.raises(NotImplementedError):
        jax.grad(energy)(params.epsilon)


@pytest.mark.parametrize(
    "change,message",
    [("asymmetric", "symmetric"), ("cutoff", "exceed the skin list")],
)
def test_invalid_inputs_fail(change, message):
    particles, systems, params = lj_systems()
    skin_list = rows_skin_list(particles, systems, params.cutoff)
    if change == "asymmetric":
        params = bind(params, lambda p: p.sigma).apply(lambda s: s.at[0, 1].add(0.1))
    else:
        # above the list's cutoff, below its build radius
        params = bind(params, lambda p: p.cutoff.data).apply(lambda c: c + SKIN / 2)
    inp = RowPotentialInput(params, PointCloud(particles, systems), skin_list)
    result = jax.jit(as_result_function(lennard_jones_row_energy))(inp)
    with pytest.raises(AssertionError, match=message):
        result.raise_assertion()
