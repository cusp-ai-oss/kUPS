# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

r"""Two-body Grimme D3 dispersion with Becke-Johnson damping.

$$
E = -\frac{1}{2}\sum_{i}\sum_{j,\mathbf{T}}\left[
    \frac{s_6\,C^{ij}_6}{r^6 + (R^{ij}_0)^6}
  + \frac{s_8\,C^{ij}_8}{r^8 + (R^{ij}_0)^8}\right],
$$

over neighbours $j$ and lattice translations $\mathbf{T}$, with
$C^{ij}_8 = 3\,Q_i Q_j\,C^{ij}_6$ and $R^{ij}_0 = a_1\sqrt{3\,Q_i Q_j} + a_2$,
where $Q_i$ is the stored ``r4r2``. The coefficients follow the chemical
environment through a fractional coordination number

$$
\mathrm{CN}_i = \sum_{j,\mathbf{T}}
    \left[1 + e^{-k_1\left((R^\text{cov}_i + R^\text{cov}_j)/r - 1\right)}\right]^{-1},
$$

which Gaussian-weights tabulated reference environments. The joint pair weight
factorizes, so the weights are formed and normalized per atom:

$$
C^{ij}_6 = \sum_{a,b} \hat g^a_i\, C^{ij,ab}_{6,\text{ref}}\, \hat g^b_j,
\qquad
\hat g^a_i \propto e^{-k_3\left(\mathrm{CN}_i - \mathrm{CN}^a_i\right)^2}.
$$

[d3_energy][kups.potential.dispersion.d3.d3_energy] is the sole boundary between
the kUPS graph representation and the mathematics below it: it owns edge
geometry, the atom-to-edge gathers, the per-system cutoffs and the reduction.
Edges are directed, so each pair appears twice and the sum is halved. A pair
contributes only while ``r < cutoff``, truncated abruptly as in
``simple-dftd3``. Forces and stress follow by differentiating the energy. The
Axilrod-Teller-Muto three-body term and the zero-damping variant are not
implemented.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.patch import IdPatch, WithPatch
from kups.core.potential import Energy
from kups.core.typing import (
    HasAtomicNumbers,
    HasCell,
    HasPositionsAndSystemIndex,
    SystemId,
)
from kups.core.utils.jax import dataclass, jit
from kups.potential.common.graph import GraphPotentialInput

_COORDINATION_STEEPNESS = 16.0  # k1
_REFERENCE_WEIGHTING_FACTOR = 4.0  # k3


@dataclass
class D3BJParameters:
    """Fitted Becke-Johnson damping parameters, in kUPS units."""

    s6: Array  # () dimensionless
    s8: Array  # () dimensionless
    a1: Array  # () dimensionless
    a2: Array  # () [Å], published in Bohr and converted by the caller


@dataclass
class D3ReferenceData:
    """Element and pair reference tables, indexed directly by atomic number.

    Row ``0`` is a finite neutral placeholder used by padded/unoccupied
    entries. Its numeric data must be finite (normally zero), and
    ``reference_mask[0]`` must be all ``False``. The preparation layer is
    responsible for enforcing this contract.
    """

    covalent_radii: Array  # (n_elements,) [Å], Grimme's 4/3-scaled radii
    r4r2: Array  # (n_elements,) [Å], sqrt(0.5 * sqrt(Z) * <r^4>/<r^2>)
    reference_cn: Array  # (n_elements, n_references)
    reference_c6: Array  # (n_elements, n_elements, n_refs, n_refs) [eV Å^6]
    reference_mask: Array  # (n_elements, n_references), environments defined


@dataclass
class D3Parameters:
    """Damping, reference tables and cutoffs, grouped into one parameter value.

    ``cn_cutoff`` must not exceed ``cutoff``, so that one neighbor list built at
    ``cutoff`` serves both sums.
    """

    damping: D3BJParameters
    reference: D3ReferenceData
    cutoff: Table[SystemId, Array]  # (n_systems,) [Å]
    cn_cutoff: Table[SystemId, Array]  # (n_systems,) [Å]


@runtime_checkable
class IsD3Particles(HasPositionsAndSystemIndex, HasAtomicNumbers, Protocol):
    """Particle properties D3 reads: positions, system assignment, elements."""


type D3Input = GraphPotentialInput[
    D3Parameters, IsD3Particles, HasCell[AnyPeriodicity], Literal[2]
]


def _masked_edge_distances(
    edge_vectors: Array,  # (n_edges, 3)
    valid: Array,  # (n_edges,)
) -> Array:  # (n_edges,)
    """Edge distances, with a finite stand-in on invalid/padded edges.

    Padded edges would otherwise give ``r == 0``; masking the energy afterwards
    is not enough, since a zero cotangent times an infinite partial derivative
    still yields a NaN gradient.
    """
    squared_distances = jnp.sum(edge_vectors * edge_vectors, axis=-1)
    return jnp.sqrt(jnp.where(valid, squared_distances, 1.0))


def _reference_weights(
    coordination_numbers: Array,  # (n_particles,)
    reference_cn: Array,  # (n_particles, n_references)
    reference_mask: Array,  # (n_particles, n_references)
) -> Array:  # (n_particles, n_references)
    """Normalized Gaussian weights over each atom's reference environments.

    Subtracting the largest exponent is exact for a normalized weight and keeps
    the sum from underflowing far from every reference.
    """
    exponent = (
        -_REFERENCE_WEIGHTING_FACTOR
        * (coordination_numbers[..., None] - reference_cn) ** 2
    )
    largest = jnp.max(
        jnp.where(reference_mask, exponent, -jnp.inf), axis=-1, keepdims=True
    )
    largest = jnp.where(jnp.isfinite(largest), largest, 0.0)  # fully masked row
    shifted = jnp.where(reference_mask, exponent - largest, 0.0)  # no overflow
    weights = jnp.where(reference_mask, jnp.exp(shifted), 0.0)
    normalizer = jnp.sum(weights, axis=-1, keepdims=True)
    return weights / jnp.where(normalizer > 0.0, normalizer, 1.0)


def d3_coordination_numbers(
    distances: Array,  # (n_edges,) [Å], safe to divide by
    covalent_radii_pairs: Array,  # (n_edges, 2) [Å]
    central_atom_indices: Array,  # (n_edges,)
    num_particles: int,
    edge_mask: Array,  # (n_edges,) within cn_cutoff and not padding
) -> Array:  # (num_particles,)
    """Fractional D3 coordination numbers, accumulated over directed edges.

    Self-image pairs contribute; the ``i == j``, ``T == 0`` pair does not,
    because a neighbor list never emits it.
    """
    covalent_distance = jnp.sum(covalent_radii_pairs, axis=-1)
    contribution = jax.nn.sigmoid(
        _COORDINATION_STEEPNESS * (covalent_distance / distances - 1.0)
    )
    contribution = jnp.where(edge_mask, contribution, 0.0)
    return jax.ops.segment_sum(
        contribution,
        central_atom_indices,
        num_segments=num_particles,
        mode="drop",
    )


def d3_c6_coefficients(
    edge_reference_weights: Array,  # (n_edges, 2, n_references)
    edge_reference_c6: Array,  # (n_edges, n_references, n_references) [eV Å^6]
) -> Array:  # (n_edges,) [eV Å^6]
    """Contract each endpoint's reference weights with the pair's ``C6`` matrix."""
    return jnp.einsum(
        "ea,eab,eb->e",
        edge_reference_weights[:, 0],
        edge_reference_c6,
        edge_reference_weights[:, 1],
    )


def d3_c8_coefficients(
    c6: Array,  # (n_edges,) [eV Å^6]
    r4r2_pairs: Array,  # (n_edges, 2) [Å]
) -> Array:  # (n_edges,) [eV Å^8]
    """Promote ``C6`` to ``C8`` through ``C8 = 3 Q_i Q_j C6``."""
    return 3.0 * r4r2_pairs[:, 0] * r4r2_pairs[:, 1] * c6


def d3_bj_edge_energy(
    distances: Array,  # (n_edges,) [Å]
    c6: Array,  # (n_edges,) [eV Å^6]
    c8: Array,  # (n_edges,) [eV Å^8]
    r4r2_pairs: Array,  # (n_edges, 2) [Å], fixes the damping radius
    damping: D3BJParameters,
) -> Array:  # (n_edges,) [eV]
    """Two-body D3(BJ) energy of each directed edge.

    Becke-Johnson damping adds the pair radius ``R0`` to the denominators, so no
    separate damping function appears. The caller applies the cutoff, drops
    padded edges and halves the directed sum.
    """
    c8_c6_ratio = 3.0 * r4r2_pairs[:, 0] * r4r2_pairs[:, 1]
    damping_length = damping.a1 * jnp.sqrt(c8_c6_ratio) + damping.a2
    r2 = distances**2
    r6 = r2**3
    r8 = r6 * r2
    return -(
        damping.s6 * c6 / (r6 + damping_length**6)
        + damping.s8 * c8 / (r8 + damping_length**8)
    )


@jit
def d3_energy(inp: D3Input) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Total two-body D3(BJ) dispersion energy per system [eV].

    The graph must carry directed pair edges out to ``cutoff``, which bounds
    ``cn_cutoff`` and so covers both sums. That ordering is a contract on
    [D3Parameters][kups.potential.dispersion.d3.D3Parameters], to be enforced
    where the potential is configured rather than asserted here.

    This kernel validates nothing. It assumes
    [D3ReferenceData][kups.potential.dispersion.d3.D3ReferenceData] was checked
    by the layer that loads and prepares it, and that every particle atomic
    number is a valid index into those tables. A loader cannot establish the
    second -- it never sees a particle state -- so rejecting an unsupported
    element belongs to the state and configuration boundary that builds the
    potential.
    """
    graph = inp.graph
    parameters = inp.parameters
    reference = parameters.reference
    assert graph.edges.indices.shape[-1] == 2, "D3 consumes pair edges"

    # edge geometry and validity
    edge_indices = graph.edges.indices.indices
    valid = graph.edges.indices.valid_mask.all(axis=-1)
    distances = _masked_edge_distances(graph.edge_shifts[:, 0], valid)

    # elements per atom, and per edge with padded rows sent to the masked row zero
    atomic_numbers = graph.particles.data.atomic_numbers
    edge_atomic_numbers = jnp.where(valid[:, None], atomic_numbers[edge_indices], 0)

    # per-system cutoffs, resolved onto edges
    edge_systems = graph.edge_batch_mask
    cutoff = Table.broadcast_to(parameters.cutoff, graph.systems)[edge_systems]
    cn_cutoff = Table.broadcast_to(parameters.cn_cutoff, graph.systems)[edge_systems]

    # node-level quantities: coordination numbers, then reference weights
    coordination_numbers = d3_coordination_numbers(
        distances,
        reference.covalent_radii[edge_atomic_numbers],
        edge_indices[:, 0],
        len(graph.particles),
        valid & (distances < cn_cutoff),
    )
    reference_weights = _reference_weights(
        coordination_numbers,
        reference.reference_cn[atomic_numbers],
        reference.reference_mask[atomic_numbers],
    )

    # gather weights onto edges, then the dispersion coefficients and pair energy
    edge_reference_weights = jnp.where(
        valid[:, None, None], reference_weights[edge_indices], 0.0
    )
    c6 = d3_c6_coefficients(
        edge_reference_weights,
        reference.reference_c6[edge_atomic_numbers[:, 0], edge_atomic_numbers[:, 1]],
    )
    r4r2_pairs = reference.r4r2[edge_atomic_numbers]
    c8 = d3_c8_coefficients(c6, r4r2_pairs)
    edge_energy = d3_bj_edge_energy(distances, c6, c8, r4r2_pairs, parameters.damping)

    # strict cutoff, padding dropped, and each unordered pair counted twice
    edge_energy = jnp.where(valid & (distances < cutoff), edge_energy, 0.0)
    return WithPatch(edge_systems.sum_over(edge_energy) / 2, IdPatch[Any]())
