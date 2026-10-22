# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared systems of the neighbor-row potential tests."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.cell import Cell, DeformedFrame, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.lens import bind
from kups.core.neighborlist import DenseNearestNeighborList, SkinList
from kups.core.typing import Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.common.graph import PointCloud
from kups.potential.common.rows import RowPotentialInput

SKIN = 0.5
LATTICES = np.array(
    [
        [[9.0, 0.0, 0.0], [3.6, 8.1, 0.0], [-2.7, 1.8, 9.0]],
        [[12.0, 0.0, 0.0], [4.8, 10.8, 0.0], [-3.6, 2.4, 12.0]],
        [[4.6, 0.0, 0.0], [0.8, 4.4, 0.0], [0.5, -0.7, 4.8]],  # < 2 x cutoff
    ]
)
COUNTS = (30, 60, 9)
CUTOFFS = (3.0, 3.5, 5.0)  # the small cell's particles see their own images


@dataclass
class Particles:
    positions: Array
    labels: Index[Label]
    system: Index[SystemId]
    inclusion: Index[SystemId]
    exclusion: Index


@dataclass
class Systems:
    cell: Cell


def _positions(rng: np.random.Generator, lattice: np.ndarray, count: int) -> list:
    """Uniform positions at least 0.8 Å apart over all periodic images."""
    images = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ lattice
    accepted: list[np.ndarray] = []
    while len(accepted) < count:
        x = rng.uniform(0, 1, 3) @ lattice
        if all(np.linalg.norm(x - y + images, axis=-1).min() > 0.8 for y in accepted):
            accepted.append(x)
    return accepted


def lj_systems(
    periodic=(True, True, True), seed: int = 0, groups: bool = False
) -> tuple[Table, Table, LennardJonesParameters]:
    """Three systems, two species; with ``groups``, exclusion groups of three and
    two inclusion segments in the first system."""
    rng = np.random.default_rng(seed)
    system = np.repeat(np.arange(len(COUNTS)), COUNTS)
    positions = np.concatenate(
        [_positions(rng, lattice, n) for lattice, n in zip(LATTICES, COUNTS)]
    )
    n = system.size
    inclusion = system
    exclusion = np.arange(n)
    if groups:
        inclusion = np.where(system == 0, np.arange(n) % 2, system + 1)
        exclusion = np.arange(n) // 3
    particles = Table(
        tuple(ParticleId(i) for i in range(n)),
        Particles(
            jnp.asarray(positions),
            Index.new([("A", "B")[i] for i in rng.integers(0, 2, n)]),
            Index.integer(system, label=SystemId),
            Index.integer(inclusion, label=SystemId),
            Index.integer(exclusion),
        ),
    )
    frame = TriclinicFrame.from_matrix(jnp.asarray(LATTICES))
    if not all(periodic):
        frame = DeformedFrame.from_frame(frame)
    systems = Table.arange(Systems(Cell.from_pbc(frame, periodic)), label=SystemId)
    params = LennardJonesParameters.from_lorentz_berthelot_mixing(
        ("A", "B"),
        jnp.array([1.0, 1.2]),
        jnp.array([0.5, 1.5]),
        Table(systems.keys, jnp.asarray(CUTOFFS)),
    )
    return particles, systems, params


def rows_skin_list(particles: Table, systems: Table, cutoffs: Table) -> SkinList:
    """A skin list, built densely."""
    skin = Table.arange(jnp.array([SKIN]), label=SystemId)
    return SkinList.create(
        particles, systems, cutoffs, skin, builder=DenseNearestNeighborList.new
    )


def row_energies(evaluator, params, skin_list: SkinList):
    """Energies of a row evaluator as a function of the particle and system tables."""

    def energy(particles: Table, systems: Table) -> Array:
        inp = RowPotentialInput(params, PointCloud(particles, systems), skin_list)
        return evaluator(inp).data.data

    return energy


def value_and_grad(energy, particles: Table, systems: Table):
    """Energies, position gradients and cell gradients of ``energy``."""

    def total(positions: Array, cell: Cell) -> tuple[Array, Array]:
        p = bind(particles, lambda q: q.data.positions).set(positions)
        s = bind(systems, lambda q: q.data.cell).set(cell)
        e = energy(p, s)
        return e.sum(), e

    grad = jax.grad(total, argnums=(0, 1), has_aux=True)
    (positions, cell), energies = grad(particles.data.positions, systems.data.cell)
    return np.asarray(energies), np.asarray(positions), cell
