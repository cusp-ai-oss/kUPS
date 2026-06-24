# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for Coulomb vacuum potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, neighbor list, and cutoff into the
state-agnostic factories in [kups.potential.classical.coulomb][].

The cutoff may live on the state (``state.coulomb_cutoff``) or be passed directly
via ``parameters=``; in the latter case it is bound with a constant lens and the
state need not carry a cutoff field.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from jax import Array

from kups.core.cell import Vacuum
from kups.core.data import Table
from kups.core.lens import Lens, const_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
)
from kups.core.patch import Patch, Probe
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Potential,
    empty_patch_idx_view,
)
from kups.core.typing import HasCell, IsState, SystemId
from kups.potential.classical.coulomb import (
    IsCoulombGraphParticles,
    make_coulomb_vacuum_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, IsGraphProbe


class IsCoulombVacuumGraphState(
    IsState[IsCoulombGraphParticles, HasCell[Vacuum]],
    IsNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, systems, and neighbor list for a Coulomb vacuum graph (no cutoff)."""


class IsCoulombVacuumState(IsCoulombVacuumGraphState, Protocol):
    """:class:`IsCoulombVacuumGraphState` that also carries the cutoff on the state."""

    @property
    def coulomb_cutoff(self) -> Table[SystemId, Array]: ...


@overload
def make_coulomb_vacuum_from_state[State](
    state: Lens[State, IsCoulombVacuumState],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumState] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_coulomb_vacuum_from_state[State](
    state: Lens[State, IsCoulombVacuumState],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumState] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_coulomb_vacuum_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCoulombVacuumState],
    probe: Probe[State, P, IsGraphProbe[IsCoulombGraphParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumState] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_coulomb_vacuum_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCoulombVacuumState],
    probe: Probe[State, P, IsGraphProbe[IsCoulombGraphParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumState] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_coulomb_vacuum_from_state[State](
    state: Lens[State, IsCoulombVacuumGraphState],
    probe: None = None,
    *,
    parameters: Table[SystemId, Array],
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumGraphState] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_coulomb_vacuum_from_state[State](
    state: Lens[State, IsCoulombVacuumGraphState],
    probe: None = None,
    *,
    parameters: Table[SystemId, Array],
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumGraphState] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_coulomb_vacuum_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCoulombVacuumGraphState],
    probe: Probe[State, P, IsGraphProbe[IsCoulombGraphParticles, Literal[2]]],
    *,
    parameters: Table[SystemId, Array],
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumGraphState] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_coulomb_vacuum_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCoulombVacuumGraphState],
    probe: Probe[State, P, IsGraphProbe[IsCoulombGraphParticles, Literal[2]]],
    *,
    parameters: Table[SystemId, Array],
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[IsCoulombVacuumGraphState] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_coulomb_vacuum_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: Table[SystemId, Array] | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
    neighborlist_factory: NeighborListFactory[Any] = AdaptiveNeighborList.from_state,
) -> Any:
    """Create a Coulomb vacuum potential from a typed state, optionally with incremental updates.

    Convenience wrapper around
    [make_coulomb_vacuum_potential][kups.potential.classical.coulomb.make_coulomb_vacuum_potential].
    When ``probe`` is ``None``, creates a plain potential for states satisfying
    [IsCoulombVacuumState][kups.potential.classical.coulomb.IsCoulombVacuumState].
    When a ``probe`` is provided, wires incremental patch-based updates for the same state type.

    Args:
        state: Lens into the sub-state providing particles, systems, and neighbor
            list (plus ``coulomb_cutoff`` when ``parameters`` is not given).
        probe: Detects which particles and neighbor-list edges changed since the last step.
            Pass ``None`` (default) for a non-incremental potential.
        parameters: Constant per-system cutoff. When given it is bound with a
            constant lens and the state need not carry ``coulomb_cutoff``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured Coulomb vacuum [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view
    if probe is not None:
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    if parameters is not None:
        cutoff_view: Any = const_lens(parameters)
    else:
        cutoff_view = state.focus(lambda x: x.coulomb_cutoff)

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), cutoff_view(s))

    return make_coulomb_vacuum_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
    )
