# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Helpers turning the shared reference systems into kUPS graphs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.cell import Cell, TriclinicFrame, to_lower_triangular
from kups.core.data import Index, Table
from kups.core.neighborlist import AllDenseNearestNeighborList
from kups.core.typing import ExclusionId, InclusionId, ParticleId, SystemId
from kups.core.utils.jax import dataclass
from kups.potential.common.graph import GraphPotentialInput, HyperGraph
from test.potential.dispersion._systems import System


@dataclass
class D3Points:
    """Minimal particle payload satisfying the D3 and neighbor-list protocols."""

    positions: Array
    atomic_numbers: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


@dataclass
class D3Systems:
    """Minimal system payload: just the cell."""

    cell: Cell


def _cell_for(system: System) -> tuple[Cell, Array]:
    """Lower-triangular cell plus the rotated positions, or a vacuum cell."""
    if system.cell is None:
        frame = TriclinicFrame.from_matrix(jnp.eye(3)[None] * 1000.0)
        return Cell.from_pbc(frame, (False, False, False)), jnp.asarray(
            system.positions
        )
    lower, rotate = to_lower_triangular(jnp.asarray(system.cell))
    frame = TriclinicFrame.from_matrix(lower[None])
    return Cell.from_pbc(frame, system.pbc), rotate(jnp.asarray(system.positions))


def build_graph(
    systems: list[System], cutoff: float, *, edge_capacity: int | None = None
) -> HyperGraph:
    """Build one batched graph over ``systems`` using a dense neighbor list."""
    positions, numbers, system_ids = [], [], []
    cells = []
    for index, system in enumerate(systems):
        cell, pos = _cell_for(system)
        cells.append(cell)
        positions.append(np.asarray(pos))
        numbers.append(np.asarray(system.numbers))
        system_ids.append(np.full(len(system.numbers), index))

    # Cell.periodic is a static field shared by the whole systems table, so a
    # batch cannot mix periodic and non-periodic systems. Fail loudly here rather
    # than silently adopting the first system's flags.
    periodicities = {system.pbc for system in systems}
    assert len(periodicities) == 1, (
        f"cannot batch systems with differing periodicity: {sorted(periodicities)}"
    )

    n_total = sum(len(n) for n in numbers)
    system_index = Index.integer(
        jnp.asarray(np.concatenate(system_ids)), n=len(systems), label=SystemId
    )
    particles = Table.arange(
        D3Points(
            positions=jnp.asarray(np.concatenate(positions)),
            atomic_numbers=jnp.asarray(np.concatenate(numbers)),
            system=system_index,
            inclusion=Index(
                tuple(map(InclusionId, range(len(systems)))), system_index.indices
            ),
            exclusion=Index.integer(jnp.arange(n_total), n=n_total, label=ExclusionId),
        ),
        label=ParticleId,
    )
    stacked = Cell.from_pbc(
        TriclinicFrame(jnp.concatenate([c.frame.tril for c in cells])),
        cells[0].periodic,
    )
    system_table = Table.arange(D3Systems(stacked), label=SystemId)

    cutoffs = Table.arange(jnp.full(len(systems), float(cutoff)), label=SystemId)
    capacity = FixedCapacity(edge_capacity or max(4096, n_total * n_total))
    neighborlist = AllDenseNearestNeighborList(
        avg_edges=capacity, avg_image_candidates=capacity, cutoffs=cutoffs
    )
    edges = neighborlist(particles, system_table, queried_keys=None)
    return HyperGraph(particles, system_table, edges)


def build_input(systems: list[System], parameters, cutoff: float, **kwargs):
    """Graph input ready for the D3 energy functions."""
    return GraphPotentialInput(parameters, build_graph(systems, cutoff, **kwargs))


@dataclass
class VirialParticles:
    """Particle payload accepted by ``stress_via_virial_theorem``."""

    positions: Array
    position_gradients: Array
    system: Index[SystemId]


@dataclass
class VirialSystems:
    """System payload accepted by ``stress_via_virial_theorem``."""

    cell: Cell
    cell_gradients: Cell


def virial_tables(graph: HyperGraph, gradient) -> tuple[Table, Table]:
    """Write a ``PositionsAndCell`` gradient back into stress-shaped tables."""
    particles = Table.arange(
        VirialParticles(
            positions=graph.particles.data.positions,
            position_gradients=gradient.positions.data,
            system=graph.particles.data.system,
        ),
        label=ParticleId,
    )
    systems = Table.arange(
        VirialSystems(cell=graph.systems.data.cell, cell_gradients=gradient.cell.data),
        label=SystemId,
    )
    return particles, systems
