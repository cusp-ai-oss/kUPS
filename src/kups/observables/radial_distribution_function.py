# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Radial distribution function (pair correlation function) calculations.

This module provides tools for computing the radial distribution function $g(r)$,
which measures the probability of finding a particle at distance $r$ from another
particle, normalized by the bulk density.

Key components:

- **[radial_distribution_function][kups.observables.radial_distribution_function.radial_distribution_function]**: Core RDF computation using neighbor lists
- **[RadialDistributionFunction][kups.observables.radial_distribution_function.RadialDistributionFunction]**: StateProperty wrapper for on-the-fly RDF calculation
- **[offline_radial_distribution_function][kups.observables.radial_distribution_function.offline_radial_distribution_function]**: Batch processing for trajectory analysis

## Radial Distribution Function

The RDF $g(r)$ characterizes local structure and correlations in particle systems:

$$ g(r) = \\frac{1}{4\\pi r^2 \\rho N} \\left\\langle \\sum_{i \\neq j} \\delta(|\\mathbf{r}_i - \\mathbf{r}_j| - r) \\right\\rangle $$

where $\\rho$ is the number density and $N$ is the number of particles. The integral
$\\int_0^{r_c} 4\\pi r^2 \\rho g(r) dr$ gives the average number of neighbors
within distance $r_c$ (coordination number).

Physical interpretation:

- $g(r) = 0$: Excluded volume (particles cannot overlap)
- $g(r) < 1$: Depletion (fewer neighbors than expected from random distribution)
- $g(r) = 1$: Random distribution (ideal gas limit at large $r$)
- $g(r) > 1$: Enrichment (structural peaks indicating shells/ordering)

The RDF reveals phase transitions (liquid vs. solid), molecular clustering,
and solvation structure.
"""

from collections import namedtuple
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.data import Index, Table
from kups.core.lens import View, lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    NearestNeighborList,
    NeighborListSystems,
)
from kups.core.propagator import StateProperty
from kups.core.result import as_result_function
from kups.core.typing import (
    ExclusionId,
    HasPositionsAndSystemIndex,
    InclusionId,
    ParticleId,
    SystemId,
)
from kups.core.unitcell import UnitCell
from kups.core.utils.jax import dataclass, field, jit, no_jax_tracing


@dataclass
class _NNListPointsImpl:
    """Wraps HasPositionsAndSystemIndex to satisfy NeighborListPoints."""

    positions: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


def _to_nnlist_points(
    positions: Table[ParticleId, HasPositionsAndSystemIndex],
) -> Table[ParticleId, _NNListPointsImpl]:
    """Wrap Indexed particle data to satisfy the NeighborListPoints protocol."""
    data = positions.data
    n = len(positions)
    inclusion = Index.integer(data.system.indices, label=InclusionId)
    exclusion = Index.integer(jnp.arange(n, dtype=int), label=ExclusionId)
    return Table(
        positions.keys,
        _NNListPointsImpl(data.positions, data.system, inclusion, exclusion),
    )


def radial_distribution_function(
    positions: Table[ParticleId, HasPositionsAndSystemIndex],
    systems: Table[SystemId, NeighborListSystems],
    rmax: float,
    bins: int,
    neighborlist: NearestNeighborList,
    cutoffs: Table[SystemId, Array] | None = None,
) -> Array:
    """Compute radial distribution function $g(r)$ from particle positions.

    Calculates the pair correlation function by:

    1. Finding all particle pairs within rmax using neighbor lists
    2. Computing pairwise distances
    3. Binning distances into histogram
    4. Normalizing by ideal gas distribution: $4\\pi r^2 \\rho dr$

    The normalization includes a finite bin width correction:

    $$g(r) = \\frac{n(r)}{4\\pi r^2 \\rho N \\Delta r} \\cdot \\frac{1}{1 + (\\Delta r)^2/(12r^2)}$$

    where $n(r)$ is the pair count in bin at $r$, $\\rho$ is number density, $N$ is particle
    count, and $\\Delta r$ is bin width.

    Args:
        positions: Indexed particle positions with system assignments.
        systems: Indexed system data (unit cells, cutoffs).
        rmax: Maximum distance to consider (cutoff radius).
        bins: Number of histogram bins for $g(r)$.
        neighborlist: Neighbor list algorithm for finding pairs.
        cutoffs: Per-system cutoff distances. Defaults to rmax for all systems.

    Returns:
        RDF array, shape `(n_systems, bins)`. The $r$-values for each bin are
        centered at $r_i = (i + 0.5) \times dr$ where $dr = r_{\text{max}} / \text{bins}$.

    Note:
        - Excludes self-pairs automatically via neighbor list construction
        - Uses periodic boundary conditions from unitcell
        - Assumes uniform density within each system
    """
    nnl_positions = _to_nnlist_points(positions)
    if cutoffs is None:
        cutoffs = Table(
            systems.keys,
            jnp.full(systems.size, rmax),
        )
    edge_result = neighborlist(nnl_positions, None, systems, cutoffs)
    diffs = edge_result.difference_vectors(positions, systems)
    dists = jnp.linalg.norm(diffs, axis=(-2, -1))
    dr = rmax / bins
    bin_indices = jnp.ceil(dists / dr).astype(int)
    system_index = positions.data.system
    n_sys = system_index.num_labels
    batch_id = system_index.indices[edge_result.indices.indices[:, 0]]
    rdf = jnp.zeros((n_sys, bins + 1), dtype=dists.dtype)
    rdf = rdf.at[batch_id, bin_indices].add(1, mode="drop")

    rr = jnp.linspace(dr / 2, rmax - dr / 2, bins)
    segment_sizes = system_index.counts.data
    phi = segment_sizes / systems.data.unitcell.volume
    norm = 4.0 * jnp.pi * dr * phi * segment_sizes
    rdf = rdf[:, 1:] / (norm[:, None] * (rr * rr + (dr * dr / 12)))
    return rdf


@dataclass
class RadialDistributionFunction[State](StateProperty[State, Array]):
    """Compute radial distribution function as a state property.

    Wraps [radial_distribution_function][kups.observables.radial_distribution_function.radial_distribution_function] in the
    [StateProperty][kups.core.propagator.StateProperty] interface for on-the-fly computation during simulations.
    Useful for tracking structural evolution or as input to other analyses.

    Type Parameters:
        State: Simulation state type

    Attributes:
        positions: View extracting indexed particle positions from state.
        systems: View extracting indexed system data from state.
        rmax: View extracting maximum distance (or constant float).
        bins: View extracting number of bins (or constant int).
        neighborlist: Neighbor list algorithm for pair finding.

    Note:
        For offline trajectory analysis of large datasets, use
        [offline_radial_distribution_function][kups.observables.radial_distribution_function.offline_radial_distribution_function]
        instead for better memory efficiency.
    """

    positions: View[State, Table[ParticleId, HasPositionsAndSystemIndex]] = field(
        static=True
    )
    systems: View[State, Table[SystemId, NeighborListSystems]] = field(static=True)
    rmax: View[State, float] = field(static=True)
    bins: View[State, int] = field(static=True)
    neighborlist: NearestNeighborList = field(static=True)

    def __call__(self, key: Array, state: State) -> Array:
        """Compute RDF from current state.

        Args:
            key: JAX PRNG key (unused)
            state: Current simulation state

        Returns:
            RDF array, shape `(n_systems, bins)`
        """
        return radial_distribution_function(
            self.positions(state),
            self.systems(state),
            self.rmax(state),
            self.bins(state),
            self.neighborlist,
        )


@no_jax_tracing
def offline_radial_distribution_function(
    positions: Array,
    unitcell: UnitCell,
    rmax: float,
    bins: int,
    *,
    batch_size: int | None = None,
) -> Array:
    """Compute RDF from trajectory data with automatic capacity management.

    Processes large trajectory datasets efficiently by:
    1. Automatically determining neighbor list capacity via trial runs
    2. Processing frames in batches using `jax.lax.map`
    3. Handling assertion failures and retrying with increased capacity

    This function is **not JIT-compilable** and intended for **offline analysis**
    of stored trajectories (e.g., from HDF5 files). For on-the-fly computation,
    use [RadialDistributionFunction][kups.observables.radial_distribution_function.RadialDistributionFunction].

    Args:
        positions: Particle positions, shape `(..., n_frames, n_particles, 3)`.
            Batch dimensions are preserved in output.
        unitcell: Unit cell for periodic boundary conditions (shared across frames)
        rmax: Maximum distance for RDF calculation (cutoff radius)
        bins: Number of histogram bins
        batch_size: Number of frames to process simultaneously in jax.lax.map.
            Lower values reduce memory usage. Defaults to None (process all at once).

    Returns:
        RDF array, shape `(..., n_frames, bins)`. Preserves all batch dimensions
        from input positions.

    Example:
        ```python
        import h5py
        from kups.observables.radial_distribution_function import offline_radial_distribution_function

        # Load trajectory from file
        with h5py.File("trajectory.h5", "r") as f:
            positions = f["positions"][:]  # Shape: (n_frames, n_particles, 3)
            unitcell = TriclinicUnitCell.from_matrix(f["unitcell"][()])

        # Compute time-averaged RDF
        g_r = offline_radial_distribution_function(
            positions, unitcell, rmax=10.0, bins=200, batch_size=100
        )  # Shape: (n_frames, 200)

        # Average over trajectory
        g_r_avg = g_r.mean(axis=0)  # Shape: (200,)

        # Plot
        import matplotlib.pyplot as plt
        r = jnp.linspace(0.025, 9.975, 200)
        plt.plot(r, g_r_avg)
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        ```

    Note:
        - Automatically retries with increased capacity if neighbor list is too small
        - Memory usage scales with `n_frames * batch_size * n_particles²`
        - For very large trajectories, use smaller batch_size to avoid OOM errors
    """

    @jit
    @as_result_function
    def rdf_with_capacity(nnlist: DenseNearestNeighborList) -> Array:
        def rdf(pos: Array) -> Array:
            _RDFParticles = namedtuple("_RDFParticles", ["positions", "system"])
            n = pos.shape[0]
            system = Index((SystemId(0),), jnp.zeros(n, dtype=int))
            particles = Table.arange(
                _RDFParticles(jnp.asarray(pos), system),
                label=ParticleId,
            )
            _SystemData = namedtuple("_SystemData", ["unitcell", "cutoff"])
            sys_data = Table.arange(
                _SystemData(unitcell, jnp.array([rmax])),
                label=SystemId,
            )
            return radial_distribution_function(
                particles,
                sys_data,
                rmax=rmax,
                bins=bins,
                neighborlist=nnlist,
            )

        return jax.lax.map(
            rdf, positions.reshape(-1, *positions.shape[-2:]), batch_size=batch_size
        )

    capacity = DenseNearestNeighborList.make(lens(lambda x: x))  # type: ignore[reportArgumentType]  # identity lens causes recursive type
    while (out := rdf_with_capacity(capacity)).failed_assertions:
        capacity = out.fix_or_raise(capacity)
    result = out.value
    result = result.reshape(*positions.shape[:-2], *result.shape[-2:])
    return result


if TYPE_CHECKING:

    def _[State](a: RadialDistributionFunction[State]) -> None:
        _: StateProperty[State, Array] = a
