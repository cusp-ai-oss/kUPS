# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Harmonic potentials for bonded interactions.

This module provides harmonic bond and angle potentials commonly used in molecular
mechanics force fields. These terms maintain molecular geometry and are typically
applied to explicitly defined bonds and angles.

Bond potential: $U(r) = k(r - r_0)^2$
Angle potential: $U(\\theta) = k(\\theta - \\theta_0)^2$
"""

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table
from kups.core.lens import Lens, View
from kups.core.neighborlist import FixedEdgesNeighborList
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    Energy,
    Potential,
    PotentialOut,
)
from kups.core.typing import (
    HasCell,
    HasPositionsAndLabels,
    Label,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.core.utils.kahan import KahanSummand
from kups.potential.common.energy import (
    EnergyFunction,
    PotentialFromEnergy,
)
from kups.potential.common.graph import (
    GraphConstructor,
    GraphPotentialInput,
    IsGraphProbe,
    IsRadiusGraphPoints,
    LocalGraphSumComposer,
)


@runtime_checkable
class IsBondedParticles(HasPositionsAndLabels, IsRadiusGraphPoints, Protocol):
    """Particle data with positions, labels, and system index."""

    ...


@dataclass
class HarmonicBondParameters:
    """Harmonic bond potential parameters.

    Attributes:
        labels: Species labels, shape `(n_species,)`
        x0: Equilibrium bond lengths [Å], shape `(n_species, n_species)`
        k: Force constants [energy/Å²], shape `(n_species, n_species)`
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    x0: Array  # (n_species, n_species)
    k: Array  # (n_species, n_species)


type HarmonicBondInput = GraphPotentialInput[
    HarmonicBondParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[2]
]


def harmonic_bond_energy(
    inp: HarmonicBondInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute harmonic bond energy for all bonds.

    Calculates energy as k(r - r₀)² for each bond and sums over all systems.

    Args:
        inp: Graph potential input with harmonic bond parameters

    Returns:
        Total bond energy per system
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 2, (
        "Harmonic bond potential only supports pairwise interactions (order=2)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    x0 = inp.parameters.x0[edg_species[:, 0], edg_species[:, 1]]
    k = inp.parameters.k[edg_species[:, 0], edg_species[:, 1]]
    edge_energy = (jnp.linalg.norm(graph.edge_shifts[:, 0], axis=-1) - x0) ** 2 * k
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


@dataclass
class HarmonicAngleParameters:
    """Harmonic angle potential parameters.

    Attributes:
        labels: Species labels, shape `(n_species,)`
        theta0: Equilibrium angles [degrees], shape `(n_species, n_species, n_species)`
        k: Force constants [energy/degree²], shape `(n_species, n_species, n_species)`
    """

    labels: tuple[Label, ...] = field(static=True)  # (n_species,)
    theta0: Array  # (n_species, n_species, n_species)
    k: Array  # (n_species, n_species, n_species)


type HarmonicAngleInput = GraphPotentialInput[
    HarmonicAngleParameters, IsBondedParticles, HasCell[AnyPeriodicity], Literal[3]
]


def harmonic_angle_energy(
    inp: HarmonicAngleInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch[Any]]:
    """Compute harmonic angle energy for all angles.

    Calculates energy as k(θ - θ₀)² for each angle triplet and sums over all systems.
    Angles are computed in degrees.

    Args:
        inp: Graph potential input with harmonic angle parameters

    Returns:
        Total angle energy per system
    """
    graph = inp.graph
    assert graph.edges.indices.indices.shape[1] == 3, (
        "Harmonic angle potential only supports triplet interactions (order=3)."
    )
    edg_species = graph.particles[graph.edges.indices].labels.indices_in(
        inp.parameters.labels
    )
    theta0 = inp.parameters.theta0[
        edg_species[:, 0], edg_species[:, 1], edg_species[:, 2]
    ]
    k = inp.parameters.k[edg_species[:, 0], edg_species[:, 1], edg_species[:, 2]]
    v1, v2 = graph.edge_shifts[:, 0], graph.edge_shifts[:, 1]
    angle = jnp.arccos(
        jnp.einsum("ij,ij->i", v1, v2)
        / (jnp.linalg.norm(v1, axis=-1) * jnp.linalg.norm(v2, axis=-1))
    )
    angle = jnp.rad2deg(angle)
    edge_energy = (angle - theta0) ** 2 * k
    total_energies = graph.edge_batch_mask.sum_over(edge_energy)
    return WithPatch(total_energies, IdPatch[Any]())


def make_harmonic_bond_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, HarmonicBondParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[HarmonicBondInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create harmonic bond potential for explicitly defined bonds.

    Applies harmonic restraints to specified atom pairs (bonds). Bonds must be
    explicitly provided via the input_view edge set.

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edge_indices_view: Extracts bond connectivity
        systems_view: Extracts indexed system data (cell)
        parameter_view: Extracts [HarmonicBondParameters][kups.potential.classical.harmonic.HarmonicBondParameters]
        probe: Graph probe for incremental particle and neighbor-list updates
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Harmonic bond [Potential][kups.core.potential.Potential]
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[2]](
            edge_indices_view(state)
        ),
        probe=probe,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=graph_fn,
        parameter_view=parameter_view,
    )
    potential = PotentialFromEnergy(
        composer=composer,
        energy_fn=harmonic_bond_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential


def make_harmonic_angle_potential[
    State,
    P: Patch[Any],
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edge_indices_view: View[State, Index[ParticleId]],
    systems_view: View[State, Table[SystemId, HasCell[AnyPeriodicity]]],
    parameter_view: View[State, HarmonicAngleParameters],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]] | None,
    gradient_lens: Lens[HarmonicAngleInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, KahanSummand[PotentialOut[Gradients, Hessians]]]
    | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create harmonic angle potential for explicitly defined angles.

    Applies harmonic restraints to specified atom triplets (angles). Angles must be
    explicitly provided via the input_view edge set as triplets (i-j-k).

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edge_indices_view: Extracts angle connectivity (triplets)
        systems_view: Extracts indexed system data (cell)
        parameter_view: Extracts [HarmonicAngleParameters][kups.potential.classical.harmonic.HarmonicAngleParameters]
        probe: Graph probe for incremental particle and neighbor-list updates
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Harmonic angle [Potential][kups.core.potential.Potential]
    """
    graph_fn = GraphConstructor(
        particles=particles_view,
        systems=systems_view,
        neighborlist=lambda state: FixedEdgesNeighborList[Literal[3]](
            edge_indices_view(state)
        ),
        probe=probe,
    )
    composer = LocalGraphSumComposer(
        graph_constructor=graph_fn,
        parameter_view=parameter_view,
    )
    potential = PotentialFromEnergy(
        composer=composer,
        energy_fn=harmonic_angle_energy,
        gradient_lens=gradient_lens,
        hessian_lens=hessian_lens,
        hessian_idx_view=hessian_idx_view,
        cache_lens=out_cache_lens,
        patch_idx_view=patch_idx_view,
    )
    return potential


if TYPE_CHECKING:
    _hb: EnergyFunction[Any, HarmonicBondInput] = harmonic_bond_energy
    _ha: EnergyFunction[Any, HarmonicAngleInput] = harmonic_angle_energy
