# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Edge representations for molecular systems.

An [`Edges`][kups.core.neighborlist.edges.Edges] value encodes the
connectivity produced by a neighbor list (or built explicitly for bonded
terms). It is generic in its ``Degree`` so the same dataclass represents
pairs (``Degree=2``), angles (``Degree=3``), dihedrals (``Degree=4``), etc.
"""

from __future__ import annotations

from typing import override

import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Sliceable, Table
from kups.core.typing import (
    HasCell,
    HasPositionsAndSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass
from kups.core.utils.math import triangular_3x3_matmul


@dataclass
class Edges[Degree: int](Sliceable):
    """Represents edges (connections) between particles in a molecular system.

    An edge connects `Degree` particles, where degree=2 represents pairwise
    interactions (bonds), degree=3 represents three-body interactions (angles), etc.

    For periodic systems, edges include shift vectors that indicate how many
    cells to traverse when computing distances between connected particles.

    Type Parameters:
        Degree: Number of particles connected by each edge (static type check)

    Attributes:
        indices: Particle indices for each edge, shape `(n_edges, Degree)`
        shifts: Periodic shift vectors, shape `(n_edges, Degree-1, 3)`.
            Shift vectors for the 2nd through Degree-th particle relative to the first.

    Example:
        ```python
        # Pairwise edges (bonds) between particles
        edges = Edges(
            indices=jnp.array([[0, 1], [1, 2], [0, 2]]),  # 3 edges
            shifts=jnp.array([[[0, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]])  # 3rd edge crosses boundary
        )
        ```
    """

    # The degree is purely for type checking and does not affect runtime behavior
    indices: Index[ParticleId]  # (n_edges, Degree)
    shifts: Array  # (n_edges, Degree - 1, 3)

    def __post_init__(self) -> None:
        # Resolve the underlying array for validation
        raw = self.indices.indices if isinstance(self.indices, Index) else self.indices
        if not isinstance(raw, Array):
            return
        assert jnp.issubdtype(raw.dtype, jnp.integer), (
            f"Indices must be of integer type, got {raw.dtype}"
        )
        target_shape = (
            *self.indices.shape[:-1],
            self.indices.shape[-1] - 1 if self.indices.shape[-1] > 1 else 0,
            3,
        )
        assert self.shifts.shape == target_shape, (
            f"Shifts must have shape {target_shape}, got {self.shifts.shape}"
        )

    def difference_vectors(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, HasCell[AnyPeriodicity]],
    ) -> Array:
        """Compute difference vectors between connected particles.

        For each edge, computes the vector from the first particle to each
        subsequent particle, accounting for periodic boundary conditions.

        Args:
            particles: Particle positions with system index information.
            systems: System data with cell for periodic boundary conditions.

        Returns:
            Array of shape `(n_edges, Degree-1, 3)` containing difference vectors.
        """

        shifts = self.absolute_shifts(particles, systems)
        pos = particles[self.indices].positions
        return pos[:, 1:] - pos[:, :1] + shifts

    def absolute_shifts(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, HasCell[AnyPeriodicity]],
    ) -> Array:
        """Compute absolute shift vectors for all particles in each edge.

        Converts relative shifts to absolute Cartesian shift vectors.

        Args:
            particles: Particle data with system index information.
            systems: System data with cell for periodic boundary conditions.

        Returns:
            Array of shape `(n_edges, Degree-1, 3)` containing absolute shift vectors.
        """
        lattice = systems.map_data(lambda x: x.cell.vectors)
        vecs = lattice[particles[self.indices[:, 0]].system]
        return triangular_3x3_matmul(vecs[:, None], self.shifts)

    @property
    def degree(self) -> int:
        return self.indices.shape[-1]

    @override
    def __len__(self) -> int:
        return self.indices.shape[0]
