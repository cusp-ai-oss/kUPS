# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for torch MLFF potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, neighbor list, and torch MLFF model into
the state-agnostic factories in
[kups.potential.mliap.torch.interface][].

The model may live on the state (``state.torch_mliap_model``) or be passed
directly via ``parameters=``; in the latter case it is bound with a constant
lens and the state need not carry a model field.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from kups.core.cell import AnyPeriodicity
from kups.core.lens import Lens, const_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
)
from kups.core.patch import Patch
from kups.core.potential import EmptyType, Potential
from kups.core.typing import HasCell, IsState
from kups.potential.common.geometry import Geometry, PositionsAndCell
from kups.potential.mliap.torch.interface import (
    IsTorchMliapParticles,
    TorchMliap,
    make_torch_mliap_potential,
)


class IsTorchMliapGraphState(
    IsState[IsTorchMliapParticles, HasCell[AnyPeriodicity]],
    IsNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, systems, and neighbor list for a torch MLFF graph (no model)."""


class IsTorchMliapState(IsTorchMliapGraphState, Protocol):
    """:class:`IsTorchMliapGraphState` that also carries the model on the state."""

    @property
    def torch_mliap_model(self) -> TorchMliap: ...


@overload
def make_torch_mliap_from_state[State, Focus: IsTorchMliapState](
    state: Lens[State, Focus],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
    neighborlist_factory: NeighborListFactory[Focus] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_torch_mliap_from_state[State, Focus: IsTorchMliapGraphState](
    state: Lens[State, Focus],
    *,
    parameters: TorchMliap,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
    neighborlist_factory: NeighborListFactory[Focus] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


def make_torch_mliap_from_state(
    state: Any,
    *,
    parameters: TorchMliap | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
    neighborlist_factory: NeighborListFactory[Any] = AdaptiveNeighborList.from_state,
) -> Any:
    """Create a torch MLFF potential from a typed state.

    Args:
        state: Lens into a sub-state providing particles, systems, and neighbor
            list (plus ``torch_mliap_model`` when ``parameters`` is not given).
        parameters: Constant torch MLFF model. When given it is bound with a
            constant lens and the state need not carry ``torch_mliap_model``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs; the torch gradients are pulled back through ``gradient.set``
            into ``∂E/∂u``. When ``None`` (default), returns the raw
            ``PositionsAndCell`` (force + stress) gradients.
        neighborlist_factory: Neighbor list factory.

    Returns:
        Configured torch MLFF ``Potential``.
    """
    if parameters is not None:
        model_view = const_lens(parameters)
    else:
        model_view = state.focus(lambda x: x.torch_mliap_model)

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), model_view(s).cutoff)

    particles_view = state.focus(lambda x: x.particles)
    systems_view = state.focus(lambda x: x.systems)
    if gradient is not None:
        return make_torch_mliap_potential(
            particles_view,
            systems_view,
            neighborlist_view,
            model_view,
            gradient=gradient,
        )
    return make_torch_mliap_potential(
        particles_view,
        systems_view,
        neighborlist_view,
        model_view,
    )
