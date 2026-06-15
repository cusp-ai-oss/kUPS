# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fused real-space nonbonded potential (``kups.potential.classical.nonbonded``).

The fused energy (LJ + short-range Ewald on one shared graph, one vjp) must reproduce the
sum of the separate components — ``lennard_jones_energy`` + ``ewald_short_range_energy`` —
for energy AND forces (``dE/dr``) AND the NPT stress (``dE/dcell``), for both the Tier-A
(maximal-reuse) and Tier-B (single-pass edge) bodies, single- and multi-system. The trusted
oracles are the existing component energy functions.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist import Edges
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.ewald import EwaldParameters, ewald_short_range_energy
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    lennard_jones_energy,
)
from kups.potential.classical.nonbonded import (
    NonbondedParameters,
    fused_nonbonded_edge_energy,
    nonbonded_energy,
)
from kups.potential.common.graph import GraphPotentialInput, HyperGraph

from ..clear_cache import clear_cache  # noqa: F401

_LABELS = ("Na", "Cl")
_FUSED = {"TierA": nonbonded_energy, "TierB": fused_nonbonded_edge_energy}


@dataclass
class _NBPointData:
    positions: jax.Array
    labels: Index[str]
    charges: jax.Array
    system: Index[SystemId]


@dataclass
class _SysData:
    cell: Cell
    cutoff: jax.Array


def _params(
    n_sys: int, cutoff: float = 50.0, alpha: float = 0.3
) -> NonbondedParameters:
    keys = tuple(SystemId(i) for i in range(n_sys))
    lj = LennardJonesParameters(
        labels=_LABELS,
        sigma=jnp.array([[2.4, 3.4], [3.4, 4.4]]),
        epsilon=jnp.array([[0.0015, 0.003], [0.003, 0.005]]),
        cutoff=Table(keys, jnp.full((n_sys,), cutoff)),
    )
    ew = EwaldParameters(
        alpha=Table(keys, jnp.full((n_sys,), alpha)),
        cutoff=Table(keys, jnp.full((n_sys,), cutoff)),
        reciprocal_lattice_shifts=Table(keys, jnp.zeros((n_sys, 1, 3), dtype=int)),
    )
    return NonbondedParameters(lj=lj, ewald=ew)


def _graph(positions, species, charges, system_ids, cells) -> HyperGraph:
    particles = Table.arange(
        _NBPointData(positions, Index.new(species), charges, Index.new(system_ids)),
        label=ParticleId,
    )
    cell = PeriodicCell(TriclinicFrame.from_matrix(cells))
    systems = Table.arange(
        _SysData(cell, jnp.full((cells.shape[0],), 50.0)), label=SystemId
    )
    n = positions.shape[0]
    sysarr = jnp.asarray(system_ids)
    # all in-system pairs; large box => no periodic image crossing (zero shifts)
    pairs = jnp.array(
        [[i, j] for i in range(n) for j in range(n) if i < j and sysarr[i] == sysarr[j]]
    )
    edges = Edges(
        indices=Index(particles.keys, pairs),
        shifts=jnp.zeros((pairs.shape[0], 1, 3)),
    )
    return HyperGraph(particles, systems, edges)


def _reference_energy(params, graph):
    lj = lennard_jones_energy(GraphPotentialInput(params.lj, graph)).data.data
    sr = ewald_short_range_energy(GraphPotentialInput(params.ewald, graph)).data.data
    return lj + sr


def _single():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [0.0, 2.8, 0.0],
            [2.8, 2.8, 0.0],
            [1.4, 1.4, 2.0],
            [1.4, 1.4, -2.0],
        ]
    )
    return dict(
        positions=positions,
        species=["Na", "Cl", "Cl", "Na", "Na", "Cl"],
        charges=jnp.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]),
        system_ids=[0, 0, 0, 0, 0, 0],
        cells=jnp.eye(3)[None] * 100.0,
    )


def _two():
    base = jnp.array(
        [[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [0.0, 2.8, 0.0], [1.4, 1.4, 2.0]]
    )
    return dict(
        positions=jnp.concatenate([base, base + 0.3]),
        species=["Na", "Cl", "Cl", "Na"] * 2,
        charges=jnp.array([1.0, -1.0, -1.0, 1.0] * 2),
        system_ids=[0, 0, 0, 0, 1, 1, 1, 1],
        cells=jnp.broadcast_to(jnp.eye(3) * 100.0, (2, 3, 3)),
    )


def test_fused_energy_matches_components_single():
    cfg, params = _single(), _params(1)
    graph = _graph(**cfg)
    ref = _reference_energy(params, graph)
    for name, fn in _FUSED.items():
        got = fn(GraphPotentialInput(params, graph)).data.data
        npt.assert_allclose(got, ref, rtol=1e-5, err_msg=f"{name} energy mismatch")


def test_fused_energy_matches_components_batched():
    cfg, params = _two(), _params(2)
    graph = _graph(**cfg)
    ref = _reference_energy(params, graph)
    assert ref.shape == (2,)
    for name, fn in _FUSED.items():
        got = fn(GraphPotentialInput(params, graph)).data.data
        npt.assert_allclose(
            got, ref, rtol=1e-5, err_msg=f"{name} batched energy mismatch"
        )


def _grad_fns(energy_or_array_fn, params, cfg):
    def total(pos, cellm):
        g = _graph(pos, cfg["species"], cfg["charges"], cfg["system_ids"], cellm)
        return energy_or_array_fn(params, g).sum()

    return jax.grad(total, argnums=(0, 1))(cfg["positions"], cfg["cells"])


def test_fused_forces_and_stress_match_components():
    cfg, params = _single(), _params(1)
    dpos_ref, dcell_ref = _grad_fns(_reference_energy, params, cfg)
    for name, fn in _FUSED.items():
        arr_fn = lambda p, g, fn=fn: fn(GraphPotentialInput(p, g)).data.data  # noqa: E731
        dpos, dcell = _grad_fns(arr_fn, params, cfg)
        npt.assert_allclose(
            dpos,
            dpos_ref,
            rtol=1e-5,
            atol=1e-10,
            err_msg=f"{name} forces (dE/dr) mismatch",
        )
        npt.assert_allclose(
            dcell,
            dcell_ref,
            rtol=1e-5,
            atol=1e-10,
            err_msg=f"{name} stress (dE/dcell) mismatch",
        )
