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
from kups.core.utils.jax import dataclass, field, jit, no_jax_tracing
from kups.core.utils.math import next_higher_power


@jit
def _capacity_estimates(
    systems: NeighborListSystems,
    num_particles: Array,
    cutoffs: Array,
    base: float,
    multiplier: float,
) -> tuple[Array, Array, Array, Array]:
    """Per-particle candidate, image-candidate and edge capacities, and max cells.

    Edges assume a uniform density within each system's cutoff sphere and are
    rounded up to the next power of ``base`` per system before averaging.
    """
    n_bins = num_cells(systems, cutoffs).prod(axis=-1)
    candidates = jnp.minimum(num_particles / n_bins * (3**3), num_particles)
    # A cutoff reaching past perp/2 replicates each candidate once per
    # periodic image (product of per-axis image counts). Summing per
    # system keeps the estimate tight for heterogeneous cutoffs instead
    # of assuming every system replicates at the maximum rate.
    images = candidate_image_counts(systems.cell, cutoffs).prod(axis=-1)
    in_cutoff = 4 / 3 * jnp.pi * cutoffs**3 * num_particles / systems.cell.volume
    edges = next_higher_power(multiplier * in_cutoff, base=base)
    n = num_particles.shape[0]

    def next_power(total: Array) -> Array:
        return next_higher_power(total / n * multiplier, base=base)

    return (
        next_power(candidates.sum()),
        # Apply headroom and rounding once, just as for candidates. In the
        # minimum-image regime both capacities must stay equal; spare image
        # capacity would select the replication pipeline.
        next_power((candidates * images).sum()),
        edges.sum() // n,
        n_bins.max(),
    )


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
        sys = Table.join(systems, particles_per_system, cutoffs)
        candidates, image_candidates, edges, cells = map(
            int, _capacity_estimates(*sys.data, base, multiplier)
        )
        return UniversalNeighborlistParameters(
            avg_edges=edges,
            avg_candidates=candidates,
            avg_image_candidates=image_candidates,
            cells=cells,
        )
