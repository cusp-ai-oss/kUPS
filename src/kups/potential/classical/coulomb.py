# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Coulomb electrostatic potential for vacuum/non-periodic systems.

This module provides a simple pairwise Coulomb potential for charged systems
without periodic boundary conditions. For periodic systems with long-range
electrostatics, use [Ewald summation][kups.potential.classical.ewald] instead.

Potential: $U = \\frac{1}{4\\pi\\epsilon_0} \\sum_{i<j} \\frac{q_i q_j}{r_{ij}}$
"""

from typing import Any, Literal, Protocol

import jax.numpy as jnp

from kups.core.cell import AnyPeriodicity, Vacuum
from kups.core.constants import BOHR, HARTREE
from kups.core.data import Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import (
    NeighborList,
)
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    Energy,
    Potential,
    PotentialOut,
)
from kups.core.typing import (
    HasCell,
    HasCharges,
    HasPositionsAndSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.energy import (
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    GraphConstructor,
    GraphPotentialInput,
    IsGraphProbe,
    IsRadiusGraphPoints,
    LocalGraphSumComposer,
)

TO_STANDARD_UNITS = HARTREE * BOHR


class IsCoulombGraphParticles(
    HasPositionsAndSystemIndex, HasCharges, IsRadiusGraphPoints, Protocol
): ...


type CoulombVacuumInput = GraphPotentialInput[
    Any, IsCoulombGraphParticles, HasCell[Vacuum], Literal[2]
]

# Boundary-mode-agnostic alias used by Ewald's exclusion-correction term,
# which calls the pairwise sum on a periodic cell where CoulombVacuumInput
# would not type-check.
type _PairwiseCoulombInput = GraphPotentialInput[
    Any, IsCoulombGraphParticles, HasCell[AnyPeriodicity], Literal[2]
]


def _pairwise_coulomb_energy(
    inp: _PairwiseCoulombInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    edg = inp.graph.particles[inp.graph.edges.indices]
    qij = edg.charges[:, 0] * edg.charges[:, 1]
    dists = jnp.linalg.norm(inp.graph.edge_shifts[:, 0], axis=-1)
    energies = inp.graph.reduce_edges_to_systems(qij / dists) / 2 * TO_STANDARD_UNITS
    assert len(energies) == inp.graph.batch_size
    return WithPatch(energies, IdPatch[Any]())


def coulomb_vacuum_energy(
    inp: CoulombVacuumInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute Coulomb electrostatic energy for vacuum systems.

    Calculates pairwise electrostatic energy using Coulomb's law over all
    charge pairs in each (vacuum) system. Accounts for double counting.

    Args:
        inp: Graph potential input.

    Returns:
        Total electrostatic energy per system.
    """
    return _pairwise_coulomb_energy(inp)


def make_coulomb_vacuum_potential[
    State,
    Ptch: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsCoulombGraphParticles]],
    systems_view: View[State, Table[SystemId, HasCell[Vacuum]]],
    neighborlist_view: View[State, NeighborList[Literal[2]]],
    probe: Probe[State, Ptch, IsGraphProbe[IsCoulombGraphParticles, Literal[2]]] | None,
    gradient_lens: Lens[CoulombVacuumInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create simple Coulomb potential for non-periodic systems.

    Computes pairwise electrostatic interactions using Coulomb's law. Suitable for
    gas-phase or cluster systems. For periodic/bulk systems, use
    [Ewald summation][kups.potential.classical.ewald] for proper treatment of
    long-range electrostatics.

    Args:
        particles_view: Extracts indexed particle data (positions, charges, system index)
        systems_view: Extracts indexed system data (cell)
        neighborlist_view: Extracts a cutoff-bound neighbor list
        probe: Grouped probe for incremental updates (particles, neighborlist_after, neighborlist_before)
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure (optional)
        out_cache_lens: Cache location lens (optional)

    Returns:
        Coulomb potential for vacuum.
    """
    radius_graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=neighborlist_view,
        probe=probe,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=radius_graph_fn,
        parameter_view=lambda _: None,
    )
    potential = PotentialFromEnergy(
        composer=composer,
        energy_fn=coulomb_vacuum_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential
