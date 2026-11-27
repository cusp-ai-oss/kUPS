# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructor for Lennard-Jones with Ewald Coulomb.

Both potentials have a pair term over the same neighbors.
[make_lennard_jones_ewald_from_state][kups.application.potential.classical.lennard_jones_ewald.make_lennard_jones_ewald_from_state]
evaluates the two as one sum of pair terms, so a force evaluation reads the
neighbor list and its geometry once. The reciprocal sum is direct or on a
particle mesh, as the parameters say.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from kups.application.potential.classical.ewald import (
    IsEwaldGraphState,
    make_ewald_from_state,
)
from kups.core import units
from kups.core.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.core.lens import Lens, const_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    NeighborList,
    NeighborListFactory,
)
from kups.core.patch import Patch
from kups.core.potential import EMPTY_LENS, EmptyType, Potential, sum_potentials
from kups.core.typing import IsState
from kups.potential.classical.ewald import (
    EwaldParameters,
    EwaldPotential,
    IsEwaldPointData,
    IsEwaldScreening,
    PMEParameters,
    ewald_short_range_pair,
    make_pme_potential,
)
from kups.potential.classical.lennard_jones import (
    IsLennardJonesParticles,
    IsLJGraphParticles,
    LennardJonesParameters,
    lennard_jones_pair,
)
from kups.potential.common.energy import LocalSumComposer, PotentialFromEnergy
from kups.potential.common.graph import (
    GRAPH_GEOMETRY,
    POINTCLOUD_GEOMETRY,
    GraphConstructor,
    GraphInputConstructor,
    GraphPairEnergy,
)


class IsLJEwaldParticles(IsLJGraphParticles, IsEwaldPointData, Protocol):
    """Particle data of both terms: labels, charges and neighbor-list indices."""


class IsLJEwaldGraphState(
    IsEwaldGraphState, IsState[IsLJEwaldParticles, Any], Protocol
):
    """Particles, systems and neighbor-list capacities for both terms."""


type _Parameters = tuple[LennardJonesParameters, IsEwaldScreening]


def _lennard_jones(parameters: _Parameters) -> LennardJonesParameters:
    return parameters[0]


def _ewald(parameters: _Parameters) -> IsEwaldScreening:
    return parameters[1]


def _labelled(particles: IsLJEwaldParticles) -> IsLennardJonesParticles:
    return particles


def _charged(particles: IsLJEwaldParticles) -> IsEwaldPointData:
    return particles


_PAIR = lennard_jones_pair.with_particles(_labelled).with_parameters(
    _lennard_jones
) + ewald_short_range_pair.with_particles(_charged).with_parameters(_ewald)


def make_lennard_jones_ewald_from_state[State](
    state: Lens[State, IsLJEwaldGraphState],
    *,
    lennard_jones: LennardJonesParameters,
    ewald: EwaldParameters | PMEParameters,
    gradient: Lens[Geometry, PositionsAndCell[units.Length]],
    neighborlist_factory: NeighborListFactory[Any] = AdaptiveNeighborList.from_state,
) -> Potential[State, PositionsAndCell[units.Force], EmptyType, Patch[Any]]:
    """Lennard-Jones plus Ewald with both pair terms in one pass over the neighbors.

    The sum equals ``make_lennard_jones_from_state`` plus ``make_ewald_from_state``
    without exclusion correction, or plus ``make_pme_potential`` for particle-mesh
    parameters. The neighbor list is built at the larger of the two cutoffs; each
    term keeps its own.

    Args:
        state: Lens into the sub-state with particles, systems and neighbor list.
        lennard_jones: Constant Lennard-Jones parameters.
        ewald: Constant Ewald or particle-mesh parameters.
        gradient: Filter selecting the degrees of freedom to differentiate.
        neighborlist_factory: Builds the neighbor list from the sub-state and cutoff.

    Returns:
        The summed potential.
    """
    parameters = const_lens((lennard_jones, ewald))
    cutoffs = _PAIR.cutoffs((lennard_jones, ewald))

    def neighborlist_view(s: State) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), cutoffs)

    gradient_lens: Any = GRAPH_GEOMETRY.nest(gradient)
    pairs: Potential[State, PositionsAndCell[units.Force], EmptyType, Patch[Any]]
    pairs = PotentialFromEnergy(
        composer=LocalSumComposer(
            GraphInputConstructor(
                graph_constructor=GraphConstructor(
                    particles=state.focus(lambda x: x.particles),
                    systems=state.focus(lambda x: x.systems),
                    neighborlist=neighborlist_view,
                    probe=None,
                ),
                parameter_view=parameters,
            )
        ),
        energy_fn=GraphPairEnergy(_PAIR),
        gradient_lens=gradient_lens,
        hessian_lens=EMPTY_LENS,
        hessian_idx_view=EMPTY_LENS,
        cache_lens=None,
        patch_idx_view=position_and_cell_idx_view,
    )
    rest: EwaldPotential[State, PositionsAndCell[units.Force], EmptyType, Patch[Any]]
    match ewald:
        case EwaldParameters():
            rest = make_ewald_from_state(
                state,
                parameters=ewald,
                gradient=gradient,
                neighborlist_factory=neighborlist_factory,
            )
        case PMEParameters():
            point_gradient: Any = POINTCLOUD_GEOMETRY.nest(gradient)
            rest = make_pme_potential(
                state.focus(lambda x: x.particles),
                state.focus(lambda x: x.systems),
                neighborlist_view,
                const_lens(ewald),
                point_gradient,
                EMPTY_LENS,
                EMPTY_LENS,
                patch_idx_view=position_and_cell_idx_view,
            )
    return sum_potentials(
        pairs, rest.long_range, rest.self_interaction, rest.net_charge
    )
