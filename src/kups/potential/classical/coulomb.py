# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Coulomb electrostatic potential for vacuum/non-periodic systems.

This module provides a simple pairwise Coulomb potential for charged systems
without periodic boundary conditions. For periodic systems with long-range
electrostatics, use [Ewald summation][kups.potential.classical.ewald] instead.

Potential: $U = \\frac{1}{4\\pi\\epsilon_0} \\sum_{i<j} \\frac{q_i q_j}{r_{ij}}$
"""

from typing import Any, Literal, Protocol, overload

import jax.numpy as jnp
from jax import Array

from kups.core.constants import BOHR, HARTREE
from kups.core.data import Table
from kups.core.lens import Lens, SimpleLens, View
from kups.core.neighborlist import NearestNeighborList
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
    HasCharges,
    HasPositionsAndSystemIndex,
    HasUnitCell,
    ParticleId,
    SystemId,
)
from kups.potential.common.energy import (
    PositionAndUnitCell,
    PotentialFromEnergy,
    position_and_unitcell_idx_view,
)
from kups.potential.common.graph import (
    GraphPotentialInput,
    IsRadiusGraphPoints,
    IsRadiusGraphProbe,
    LocalGraphSumComposer,
    RadiusGraphConstructor,
)

TO_STANDARD_UNITS = HARTREE * BOHR


class IsCoulombGraphParticles(
    HasPositionsAndSystemIndex, HasCharges, IsRadiusGraphPoints, Protocol
): ...


type CoulombVacuumInput = GraphPotentialInput[
    Any, IsCoulombGraphParticles, HasUnitCell, Literal[2]
]


def coulomb_vacuum_energy(
    inp: CoulombVacuumInput,
) -> WithPatch[Table[SystemId, Energy], IdPatch]:
    """Compute Coulomb electrostatic energy for all charge pairs.

    Calculates pairwise electrostatic energy using Coulomb's law and sums
    over all systems. Accounts for double counting.

    Args:
        inp: Graph potential input

    Returns:
        Total electrostatic energy per system
    """
    edg = inp.graph.particles[inp.graph.edges.indices]
    qij = edg.charges[:, 0] * edg.charges[:, 1]
    dists = jnp.linalg.norm(inp.graph.edge_shifts[:, 0], axis=-1)
    energies = inp.graph.edge_batch_mask.sum_over(qij / dists) / 2 * TO_STANDARD_UNITS
    assert len(energies) == inp.graph.batch_size
    return WithPatch(energies, IdPatch())


def make_coulomb_vacuum_potential[
    State,
    Ptch: Patch,
    Gradients,
    Hessians,
](
    particles_view: View[State, Table[ParticleId, IsCoulombGraphParticles]],
    systems_view: View[State, Table[SystemId, HasUnitCell]],
    cutoffs_view: View[State, Table[SystemId, Array]],
    neighborlist_view: View[State, NearestNeighborList],
    probe: Probe[State, Ptch, IsRadiusGraphProbe[IsCoulombGraphParticles]] | None,
    gradient_lens: Lens[CoulombVacuumInput, Gradients],
    hessian_lens: Lens[Gradients, Hessians],
    hessian_idx_view: View[State, Hessians],
    patch_idx_view: View[State, PotentialOut[Gradients, Hessians]] | None = None,
    out_cache_lens: Lens[State, PotentialOut[Gradients, Hessians]] | None = None,
) -> Potential[State, Gradients, Hessians, Ptch]:
    """Create simple Coulomb potential for non-periodic systems.

    Computes pairwise electrostatic interactions using Coulomb's law. Suitable for
    gas-phase or cluster systems. For periodic/bulk systems, use
    [Ewald summation][kups.potential.classical.ewald] for proper treatment of
    long-range electrostatics.

    Args:
        particles_view: Extracts indexed particle data (positions, charges, system index)
        systems_view: Extracts indexed system data (unit cell)
        cutoffs_view: Extracts cutoff array from state
        neighborlist_view: Extracts neighbor list
        probe: Grouped probe for incremental updates (particles, neighborlist_after, neighborlist_before)
        gradient_lens: Specifies gradients to compute
        hessian_lens: Specifies Hessians to compute
        hessian_idx_view: Hessian index structure
        patch_idx_view: Cached output index structure (optional)
        out_cache_lens: Cache location lens (optional)

    Returns:
        Coulomb potential for vacuum.
    """
    radius_graph_fn = RadiusGraphConstructor(
        particles=particles_view,
        systems=systems_view,
        cutoffs=cutoffs_view,
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


class IsCoulombVacuumState(Protocol):
    """Protocol for states providing all inputs for the Coulomb vacuum potential."""

    @property
    def particles(self) -> Table[ParticleId, IsCoulombGraphParticles]: ...
    @property
    def systems(self) -> Table[SystemId, HasUnitCell]: ...
    @property
    def neighborlist(self) -> NearestNeighborList: ...
    @property
    def coulomb_cutoff(self) -> Table[SystemId, Array]: ...


@overload
def make_coulomb_vacuum_from_state[State](
    state: Lens[State, IsCoulombVacuumState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Any]: ...


@overload
def make_coulomb_vacuum_from_state[State](
    state: Lens[State, IsCoulombVacuumState],
    probe: None = None,
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, Any]: ...


@overload
def make_coulomb_vacuum_from_state[State, P: Patch](
    state: Lens[State, IsCoulombVacuumState],
    probe: Probe[State, P, IsRadiusGraphProbe[IsCoulombGraphParticles]],
    *,
    compute_position_and_unitcell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_coulomb_vacuum_from_state[State, P: Patch](
    state: Lens[State, IsCoulombVacuumState],
    probe: Probe[State, P, IsRadiusGraphProbe[IsCoulombGraphParticles]],
    *,
    compute_position_and_unitcell_gradients: Literal[True],
) -> Potential[State, PositionAndUnitCell, EmptyType, P]: ...


def make_coulomb_vacuum_from_state(
    state: Any,
    probe: Any = None,
    *,
    compute_position_and_unitcell_gradients: bool = False,
) -> Any:
    """Create a Coulomb vacuum potential from a typed state, optionally with incremental updates.

    Convenience wrapper around
    [make_coulomb_vacuum_potential][kups.potential.classical.coulomb.make_coulomb_vacuum_potential].
    When ``probe`` is ``None``, creates a plain potential for states satisfying
    [IsCoulombVacuumState][kups.potential.classical.coulomb.IsCoulombVacuumState].
    When a ``probe`` is provided, wires incremental patch-based updates for the same state type.

    Args:
        state: Lens into the sub-state providing particles, systems, and neighbor list.
        probe: Detects which particles and neighbor-list edges changed since the last step.
            Pass ``None`` (default) for a non-incremental potential.
        compute_position_and_unitcell_gradients: When ``True``, compute gradients
            w.r.t. particle positions and lattice vectors.

    Returns:
        Configured Coulomb vacuum [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_unitcell_gradients:
        gradient_lens = SimpleLens[CoulombVacuumInput, PositionAndUnitCell](
            lambda x: PositionAndUnitCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.unitcell),
            )
        )
        patch_idx_view = position_and_unitcell_idx_view
    if probe is not None:
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_coulomb_vacuum_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        state.focus(lambda x: x.coulomb_cutoff),
        state.focus(lambda x: x.neighborlist),
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
    )
