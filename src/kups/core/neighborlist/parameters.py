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
from kups.core.utils.jax import dataclass, field, no_jax_tracing
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
    """

    avg_edges: int = field(static=True)
    avg_candidates: int = field(static=True)
    avg_image_candidates: int = field(static=True)
    cells: int = field(static=True)

    @classmethod
    @no_jax_tracing
    def estimate(
        cls,
        particles_per_system: Table[SystemId, Array],
        systems: Table[SystemId, NeighborListSystems],
        cutoffs: Table[SystemId, Array],
        *,
        base: float = 2,
        multiplier: float = 1.0,
    ) -> UniversalNeighborlistParameters:
        """Estimate parameters for all neighbor list types from system geometry.

        Computes conservative initial capacities based on particle density
        and cutoff radii. The estimates are rounded up to the next power of
        ``base`` to amortize future resizing.

        Args:
            particles_per_system: Number of particles per system.
            systems: System data with cell information.
            cutoffs: Cutoff distance per system.
            base: Base for power-of rounding (default 2).
            multiplier: Safety factor applied to the estimate (default 1.0).

        Returns:
            A ``UniversalNeighborlistParameters`` instance with estimated values.
        """

        def _next_power(total: float | Array) -> int:
            return int(next_higher_power(jnp.array(total * multiplier), base=base))

        sys = Table.join(systems, particles_per_system, cutoffs)
        total_candidates = total_image_candidates = total_edges = max_cells = 0
        for _, (s, n_p, c) in sys:
            n_bins = num_cells(s, c).prod()
            candidates = min(n_p / n_bins * (3**3), n_p)
            # A cutoff reaching past perp/2 replicates each candidate once per
            # periodic image (product of per-axis image counts). Summing per
            # system keeps the estimate tight for heterogeneous cutoffs instead
            # of assuming every system replicates at the maximum rate.
            images = candidate_image_counts(s.cell, c).prod()
            total_candidates += candidates
            total_image_candidates += _next_power(candidates) * images
            total_edges += _estimate_avg_num_edges(
                n_p, s.cell.volume, c, base, multiplier
            )
            max_cells = max(n_bins, max_cells)

        return UniversalNeighborlistParameters(
            avg_edges=int(total_edges // sys.size),
            avg_candidates=_next_power(total_candidates / sys.size),
            avg_image_candidates=_next_power(total_image_candidates / sys.size),
            cells=int(max_cells),
        )
