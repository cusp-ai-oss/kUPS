# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for jaxified MLIAP potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, neighbor list, and jaxified model into the
state-agnostic factories in [kups.potential.mliap.tojax][].

The jaxified model may live on the state (``state.jaxified_model``) or be passed
directly via ``parameters=``; in the latter case it is bound with a constant lens
and the state need not carry a model field.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from kups.core.cell import AnyPeriodicity
from kups.core.lens import Lens, SimpleLens, const_lens
from kups.core.neighborlist import (
    IsAdaptiveCutoffNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
    adaptive_cutoff_neighborlist_from_state,
)
from kups.core.patch import Patch
from kups.core.potential import EMPTY_LENS, EmptyType, Potential
from kups.core.typing import HasCell, IsState
from kups.potential.common.energy import PositionAndCell
from kups.potential.mliap.tojax import (
    IsTojaxedParticles,
    JaxifiedInput,
    TojaxedMliap,
    make_tojaxed_potential,
)


class IsTojaxedGraphState(
    IsState[IsTojaxedParticles, HasCell[AnyPeriodicity]],
    IsAdaptiveCutoffNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, systems, and neighbor list for a jaxified graph (no model)."""


class IsTojaxedState(IsTojaxedGraphState, Protocol):
    """:class:`IsTojaxedGraphState` that also carries the jaxified model."""

    @property
    def jaxified_model(self) -> TojaxedMliap: ...


@overload
def make_tojaxed_from_state[State](
    state: Lens[State, IsTojaxedState],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
    neighborlist_factory: NeighborListFactory[IsTojaxedState] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_tojaxed_from_state[State](
    state: Lens[State, IsTojaxedState],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
    neighborlist_factory: NeighborListFactory[IsTojaxedState] = ...,
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_tojaxed_from_state[State](
    state: Lens[State, IsTojaxedGraphState],
    *,
    parameters: TojaxedMliap,
    compute_position_and_cell_gradients: Literal[False] = ...,
    neighborlist_factory: NeighborListFactory[IsTojaxedGraphState] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_tojaxed_from_state[State](
    state: Lens[State, IsTojaxedGraphState],
    *,
    parameters: TojaxedMliap,
    compute_position_and_cell_gradients: Literal[True],
    neighborlist_factory: NeighborListFactory[IsTojaxedGraphState] = ...,
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


def make_tojaxed_from_state(
    state: Any,
    *,
    parameters: TojaxedMliap | None = None,
    compute_position_and_cell_gradients: bool = False,
    neighborlist_factory: NeighborListFactory[
        Any
    ] = adaptive_cutoff_neighborlist_from_state,
) -> Any:
    """Create a jaxified potential from a typed state.

    Args:
        state: Lens into the sub-state providing particles, cell, and neighbor
            list (plus ``jaxified_model`` when ``parameters`` is not given).
        parameters: Constant jaxified model. When given it is bound with a
            constant lens and the state need not carry ``jaxified_model``.
        compute_position_and_cell_gradients: When ``True``, compute
            gradients w.r.t. particle positions and cell
            (for forces / stress).

    Returns:
        Configured jaxified [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    if compute_position_and_cell_gradients:
        gradient_lens = SimpleLens[JaxifiedInput, PositionAndCell](
            lambda x: PositionAndCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.cell),
            )
        )
    if parameters is not None:
        model_view = const_lens(parameters)
    else:
        model_view = state.focus(lambda x: x.jaxified_model)

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), model_view(s).cutoff)

    return make_tojaxed_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        model_view,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
    )
