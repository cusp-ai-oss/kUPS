# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Capacity-hint parameters shared by neighbor list implementations.

[`UniversalNeighborlistParameters`][kups.core.neighborlist.parameters.UniversalNeighborlistParameters]
is the concrete dataclass every application state holds and threads into
``from_state`` for the neighbor list classes. The
[`estimate`][kups.core.neighborlist.parameters.UniversalNeighborlistParameters.estimate]
classmethod derives conservative power-of-two capacities from system geometry
so callers don't have to guess.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.neighborlist.common import candidate_image_counts, num_cells
from kups.core.neighborlist.types import NeighborListSystems
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field, no_jax_tracing, no_post_init
from kups.core.utils.math import next_higher_power


@no_jax_tracing
def _estimate_avg_num_edges(
    num_particles: int | Array,
    volume: float | Array,
    cutoff: float | Array,
    base: float = 2.0,
    multiplier: float = 1.0,
) -> int:
    """Estimate average number of neighbors per particle for neighbor list allocation.

    Calculates expected neighbors within cutoff radius based on particle density,
    with tolerance factor for small systems. Result is rounded up to next power of base.

    Args:
        num_particles: Total number of particles in the system.
        volume: Total volume of the simulation box.
        cutoff: Cutoff radius for neighbor interactions.
        base: Base for power rounding (default 2.0).
        multiplier: Multiplied with the estimate to create a buffer (default 1.0).

    Returns:
        Conservative estimate rounded to next power of base for array allocation.
    """
    # avg_edges ≈ (N/V) * (4π/3 * r³), i.e. uniform-density sphere of radius cutoff
    avg_particle_density = num_particles / volume
    cutoff_volume = 4 / 3 * jnp.pi * cutoff**3
    avg_particles_in_cutoff = cutoff_volume * avg_particle_density
    estimate = multiplier * avg_particles_in_cutoff
    return int(next_higher_power(jnp.array(estimate), base=base))


@dataclass
class UniversalNeighborlistParameters:
    """Concrete parameter dataclass satisfying ``IsUniversalNeighborlistParams``.

    Holds the capacity hints needed by every neighbor list implementation.
    Use the ``estimate()`` classmethod to compute reasonable initial values
    from system geometry rather than guessing manually.

    Attributes:
        avg_edges: Average number of edges per particle (for edge capacity).
        avg_candidates: Average number of candidate pairs per particle.
        avg_image_candidates: Average number of candidate pairs per particle after
            periodic-image replication (equals ``avg_candidates`` when every cutoff
            stays within the minimum-image regime).
        cells: Maximum number of spatial hash cells across all systems.
        max_neighbors: Maximum neighbors per atom (per-row width of the nvalchemi
            toolkit's neighbor matrix, counting each periodic image separately).
        max_shifts: Maximum per-system periodic-shift count (nvalchemi naive
            kernel launch dimension).
        max_total_cells: Maximum cell-grid buffer across systems (nvalchemi
            cell-list kernel).
    """

    avg_edges: int = field(static=True)
    avg_candidates: int = field(static=True)
    avg_image_candidates: int = field(static=True)
    cells: int = field(static=True)
    max_neighbors: int = field(static=True, default=0)
    max_shifts: int = field(static=True, default=0)
    max_total_cells: int = field(static=True, default=0)

    @classmethod
    @no_jax_tracing
    def estimate(
        cls,
        particles_per_system: Table[SystemId, Array],
        systems: Table[SystemId, NeighborListSystems],
        cutoff: float,
        *,
        base: float = 2,
        multiplier: float = 1.0,
    ) -> UniversalNeighborlistParameters:
        """Estimate parameters for all neighbor list types from system geometry.

        Computes conservative initial capacities based on particle density
        and cutoff radius. The estimates are rounded up to the next power of
        ``base`` to amortize future resizing.

        Args:
            particles_per_system: Number of particles per system.
            systems: System data with cell information.
            cutoff: Cutoff radius [Å].
            base: Base for power-of rounding (default 2).
            multiplier: Safety factor applied to the estimate (default 1.0).

        Returns:
            A ``UniversalNeighborlistParameters`` instance with estimated values.
        """

        def _next_power(total: float | Array) -> int:
            return int(next_higher_power(jnp.array(total * multiplier), base=base))

        cutoff_arr = jnp.asarray(cutoff)
        sys = Table.join(systems, particles_per_system)
        total_candidates = total_image_candidates = total_edges = max_cells = 0
        max_neighbors = max_shifts = 0
        with no_post_init():
            for _, (s, n_p) in sys:
                n_bins = num_cells(s, cutoff_arr).prod()
                candidates = min(n_p / n_bins * (3**3), n_p)
                # A cutoff reaching past perp/2 replicates each candidate once per
                # periodic image (product of per-axis image counts). Summing per
                # system keeps the estimate tight across heterogeneous box
                # geometries instead of assuming every system replicates at the
                # maximum rate.
                images = candidate_image_counts(s.cell, cutoff_arr).prod()
                image_candidates = _next_power(candidates) * images
                total_candidates += candidates
                total_image_candidates += image_candidates
                edges = _estimate_avg_num_edges(
                    n_p, s.cell.volume, cutoff, base, multiplier
                )
                total_edges += edges
                max_cells = max(n_bins, max_cells)
                max_neighbors = max(edges, max_neighbors)
                # nvalchemi naive shift count: prod(2 * ceil(cutoff/perp) * pbc + 1).
                periodic = jnp.asarray(s.cell.periodic)
                shift_range = jnp.where(
                    periodic,
                    jnp.ceil(cutoff_arr[..., None] / s.cell.perpendicular_lengths),
                    0,
                ).astype(int)
                max_shifts = max(int(jnp.prod(2 * shift_range + 1)), max_shifts)

        return UniversalNeighborlistParameters(
            avg_edges=int(total_edges // sys.size),
            avg_candidates=_next_power(total_candidates / sys.size),
            avg_image_candidates=_next_power(total_image_candidates / sys.size),
            cells=int(max_cells),
            max_neighbors=int(max_neighbors),
            max_shifts=max_shifts,
            # Multi-system cell_list buffers prod(cells) per system across systems.
            max_total_cells=int(max_cells) * sys.size,
        )
