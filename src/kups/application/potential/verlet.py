# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Bind the Verlet-skin cache to the conventional simulation-state fields."""

from typing import Literal, Protocol

from jax import Array

from kups.core.data import Table
from kups.core.lens import Lens
from kups.core.neighborlist import (
    SKIN_PARAMS,
    AdaptiveNeighborList,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
    NeighborListPoints,
    NeighborListSystems,
    VerletNeighborList,
    VerletSkinState,
    refresh_skin,
)
from kups.core.propagator import CachePropagator, Propagator
from kups.core.typing import IsState, SystemId


class IsSkinRefreshState(IsState[NeighborListPoints, NeighborListSystems], Protocol):
    @property
    def verlet_skin(self) -> VerletSkinState: ...


def verlet_neighborlist_factory[S](
    cache: Lens[S, VerletSkinState],
) -> NeighborListFactory[S]:
    """Neighbor-list factory that reuses the skin cache when it covers a call.

    Args:
        cache: Lens to the skin cache.

    Returns:
        Factory building a [VerletNeighborList][kups.core.neighborlist.VerletNeighborList].
    """

    def factory(
        state: S,
        lens: Lens[S, IsUniversalNeighborlistParams],
        cutoffs: Table[SystemId, Array],
        /,
    ) -> NeighborList[Literal[2]]:
        return VerletNeighborList.new(state, lens, cache.get(state), cutoffs)

    return factory


def make_skin_refresh_from_state[S](
    state: Lens[S, IsSkinRefreshState],
    cutoffs: Table[SystemId, Array],
    skin: float,
) -> Propagator[S]:
    """Rebuild the skin cache once motion, deformation, or labels exhaust it.

    Compose it before each step inside a ``ResetOnErrorPropagator``, so a
    rebuild that overflows its capacities is rolled back with the step.

    Args:
        state: Lens to particles, systems, and the skin cache.
        cutoffs: Largest cutoffs of the potentials reading the cache (Å).
        skin: Skin width (Å); zero disables caching.

    Returns:
        Propagator writing the refreshed cache.
    """
    cache = state.focus(lambda s: s.verlet_skin)
    capacities = cache.nest(SKIN_PARAMS)
    particles = state.focus(lambda s: s.particles)
    systems = state.focus(lambda s: s.systems)

    def refreshed(key: Array, state: S) -> VerletSkinState:
        del key
        return refresh_skin(
            cache.get(state),
            particles.get(state),
            systems.get(state),
            cutoffs,
            skin,
            lambda radii: AdaptiveNeighborList.new(state, capacities, radii),
        )

    return CachePropagator(refreshed, cache.set)
