# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Harmonic potentials for bonded interactions.

This module provides harmonic bond and angle potentials commonly used in molecular
mechanics force fields. These terms maintain molecular geometry and are typically
applied to explicitly defined bonds and angles.

Bond potential: $U(r) = k(r - r_0)^2$
Angle potential: $U(\\theta) = k(\\theta - \\theta_0)^2$
"""

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

import jax.numpy as jnp
from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens, SimpleLens, View
from kups.core.neighborlist import Edges
from kups.core.patch import IdPatch, Patch, Probe, WithPatch
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Energy,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import (
    HasCache,
    HasPositionsAndLabels,
    HasSystemIndex,
    HasUnitCell,
    Label,
    MaybeCached,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field
from kups.potential.common.energy import (
    EnergyFunction,
    PositionAndUnitCell,
    PotentialFromEnergy,
    position_and_unitcell_idx_view,
)
from kups.potential.common.graph import (
    EdgeSetGraphConstructor,
    GraphPotentialInput,
    IsEdgeSetGraphProbe,
    LocalGraphSumComposer,
)


@runtime_checkable
class IsBondedParticles(HasPositionsAndLabels, HasSystemIndex, Protocol):
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
    HarmonicBondParameters, IsBondedParticles, HasUnitCell, Literal[2]
]


def harmonic_bond_energy(
    inp: HarmonicBondInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
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
    return WithPatch(total_energies, IdPatch())


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
    HarmonicAngleParameters, IsBondedParticles, HasUnitCell, Literal[3]
]


def harmonic_angle_energy(
    inp: HarmonicAngleInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
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
    return WithPatch(total_energies, IdPatch())


def make_harmonic_bond_potential[
    State,
    P: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edges_view: View[State, Edges[Literal[2]]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, HarmonicBondParameters],
    probe: Probe[State, P, IsEdgeSetGraphProbe[IsBondedParticles, Literal[2]]] | None,
    gradient_lens: Lens[HarmonicBondInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create harmonic bond potential for explicitly defined bonds.

    Applies harmonic restraints to specified atom pairs (bonds). Bonds must be
    explicitly provided via the input_view edge set.

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edges_view: Extracts bond connectivity
        systems_view: Extracts indexed system data (unit cell)
        parameter_view: Extracts [HarmonicBondParameters][kups.potential.classical.harmonic.HarmonicBondParameters]
        probe: Grouped probe for incremental updates (particles, edges, capacity)
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Harmonic bond [Potential][kups.core.potential.Potential]
    """
    graph_fn = EdgeSetGraphConstructor(
        particles=particles_view,
        edges=edges_view,
        systems=systems_view,
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
    P: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsBondedParticles]],
    edges_view: View[State, Edges[Literal[3]]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    parameter_view: View[State, HarmonicAngleParameters],
    probe: Probe[State, P, IsEdgeSetGraphProbe[IsBondedParticles, Literal[3]]] | None,
    gradient_lens: Lens[HarmonicAngleInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, P]:
    """Create harmonic angle potential for explicitly defined angles.

    Applies harmonic restraints to specified atom triplets (angles). Angles must be
    explicitly provided via the input_view edge set as triplets (i-j-k).

    Args:
        particles_view: Extracts particle data (positions, species) with system index
        edges_view: Extracts angle connectivity (triplets)
        systems_view: Extracts indexed system data (unit cell)
        parameter_view: Extracts [HarmonicAngleParameters][kups.potential.classical.harmonic.HarmonicAngleParameters]
        probe: Grouped probe for incremental updates (particles, edges, capacity)
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure
        out_cache_lens: Cache location lens

    Returns:
        Harmonic angle [Potential][kups.core.potential.Potential]
    """
    graph_fn = EdgeSetGraphConstructor(
        particles=particles_view,
        edges=edges_view,
        systems=systems_view,
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


class HasBondedParticlesAndSystems(Protocol):
    """Protocol for states with indexed particles and systems containing a unit cell."""

    @property
    def particles(self) -> Table[ParticleId, IsBondedParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...


class IsHarmonicBondState[Params](HasBondedParticlesAndSystems, Protocol):
    """Protocol for states providing all inputs for the harmonic bond potential."""

    @property
    def bond_edges(self) -> Edges[Literal[2]]: ...
    @property
    def harmonic_bond_parameters(self) -> Params: ...


@overload
def make_harmonic_bond_from_state[State](
    state: Lens[
        State,
        IsHarmonicBondState[MaybeCached[HarmonicBondParameters, Any]],
    ],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch]: ...


@overload
def make_harmonic_bond_from_state[State](
    state: Lens[
        State,
        IsHarmonicBondState[MaybeCached[HarmonicBondParameters, Any]],
    ],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Patch]: ...


@overload
def make_harmonic_bond_from_state[State, P: Patch](
    state: Lens[
        State,
        IsHarmonicBondState[
            HasCache[HarmonicBondParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsEdgeSetGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_harmonic_bond_from_state[State, P: Patch](
    state: Lens[
        State,
        IsHarmonicBondState[
            HasCache[
                HarmonicBondParameters, PotentialOut[PositionAndUnitCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsEdgeSetGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_harmonic_bond_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a harmonic bond potential, optionally with incremental updates.

    Convenience wrapper around
    [make_harmonic_bond_potential][kups.potential.classical.harmonic.make_harmonic_bond_potential].
    When `probe` is `None`, builds a plain potential from
    [IsHarmonicBondState][kups.potential.classical.harmonic.IsHarmonicBondState].
    When a `probe` is provided, builds an incrementally-updated potential from
    a state with `HasCache`-wrapped parameters.

    Args:
        state: Lens into the sub-state providing particles, unit cell, edges,
            and harmonic bond parameters.
        probe: If provided, detects which particles and edges changed since the
            last step for incremental updates.
        compute_position_and_unitcell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors (for forces / stress).

    Returns:
        Configured harmonic bond [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[HarmonicBondInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    param_view = state.focus(
        lambda x: (
            x.harmonic_bond_parameters.data
            if isinstance(x.harmonic_bond_parameters, HasCache)
            else x.harmonic_bond_parameters
        )
    )
    cache_view = None
    if probe is not None:
        param_view = state.focus(lambda x: x.harmonic_bond_parameters.data)
        cache_view = state.focus(lambda x: x.harmonic_bond_parameters.cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_harmonic_bond_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.bond_edges),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
        cache_view,
    )


class IsHarmonicAngleState[Params](HasBondedParticlesAndSystems, Protocol):
    """Protocol for states providing all inputs for the harmonic angle potential."""

    @property
    def angle_edges(self) -> Edges[Literal[3]]: ...
    @property
    def harmonic_angle_parameters(self) -> Params: ...


@overload
def make_harmonic_angle_from_state[State](
    state: Lens[
        State,
        IsHarmonicAngleState[MaybeCached[HarmonicAngleParameters, Any]],
    ],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch]: ...


@overload
def make_harmonic_angle_from_state[State](
    state: Lens[
        State,
        IsHarmonicAngleState[MaybeCached[HarmonicAngleParameters, Any]],
    ],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Patch]: ...


@overload
def make_harmonic_angle_from_state[State, P: Patch](
    state: Lens[
        State,
        IsHarmonicAngleState[
            HasCache[HarmonicAngleParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsEdgeSetGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_harmonic_angle_from_state[State, P: Patch](
    state: Lens[
        State,
        IsHarmonicAngleState[
            HasCache[
                HarmonicAngleParameters, PotentialOut[PositionAndUnitCell, EmptyType]
            ]
        ],
    ],
    probe: Probe[State, P, IsEdgeSetGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_harmonic_angle_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a harmonic angle potential, optionally with incremental updates.

    Convenience wrapper around
    [make_harmonic_angle_potential][kups.potential.classical.harmonic.make_harmonic_angle_potential].
    When `probe` is `None`, builds a plain potential from
    [IsHarmonicAngleState][kups.potential.classical.harmonic.IsHarmonicAngleState].
    When a `probe` is provided, builds an incrementally-updated potential from
    a state with `HasCache`-wrapped parameters.

    Args:
        state: Lens into the sub-state providing particles, unit cell, edges,
            and harmonic angle parameters.
        probe: If provided, detects which particles and edges changed since the
            last step for incremental updates.
        compute_position_and_unitcell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors (for forces / stress).

    Returns:
        Configured harmonic angle [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[HarmonicAngleInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    param_view = state.focus(
        lambda x: (
            x.harmonic_angle_parameters.data
            if isinstance(x.harmonic_angle_parameters, HasCache)
            else x.harmonic_angle_parameters
        )
    )
    cache_view = None
    if probe is not None:
        param_view = state.focus(lambda x: x.harmonic_angle_parameters.data)
        cache_view = state.focus(lambda x: x.harmonic_angle_parameters.cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_harmonic_angle_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.angle_edges),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
        cache_view,
    )


if TYPE_CHECKING:
    _hb: EnergyFunction = harmonic_bond_energy
    _ha: EnergyFunction = harmonic_angle_energy
