# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for blocking sphere potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, groups, systems, neighbor list, and parameters
into the state-agnostic factories in
[kups.potential.classical.blocking][].

Parameters may live on the state (``state.blocking_spheres_parameters``) or be
passed directly via ``parameters=``; in the latter case they are bound with a
constant lens and the state need not carry a parameter field.
"""

from __future__ import annotations

from typing import Any, Protocol, overload

from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import Lens, const_lens
from kups.core.neighborlist import (
    IsAdaptiveCutoffNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborListFactory,
    adaptive_cutoff_neighborlist_from_state,
)
from kups.core.patch import Patch, Probe
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Potential,
    empty_patch_idx_view,
)
from kups.core.typing import GroupId, HasCell, HasMotifIndex, IsState
from kups.potential.classical.blocking import (
    BlockingSpheresNeighborListFactory,
    BlockingSpheresParameters,
    IsBlockingSpheresProbe,
    _BlockingParticles,
    make_blocking_spheres_potential,
)


class IsBlockingSpheresGraphState(
    IsState[_BlockingParticles, HasCell[AnyPeriodicity]],
    IsAdaptiveCutoffNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, groups, systems, and neighbor list (no parameters)."""

    @property
    def groups(self) -> Table[GroupId, HasMotifIndex]: ...


class IsBlockingSpheresState(IsBlockingSpheresGraphState, Protocol):
    """:class:`IsBlockingSpheresGraphState` that also carries parameters on the state."""

    @property
    def blocking_spheres_parameters(self) -> BlockingSpheresParameters: ...


@overload
def make_blocking_spheres_from_state[State](
    state: Lens[State, IsBlockingSpheresState],
    probe: None = None,
    *,
    parameters: None = None,
    neighborlist_factory: NeighborListFactory[IsBlockingSpheresState] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_blocking_spheres_from_state[State, P: Patch[Any]](
    state: Lens[State, IsBlockingSpheresState],
    probe: Probe[State, P, IsBlockingSpheresProbe],
    *,
    parameters: None = None,
    neighborlist_factory: NeighborListFactory[IsBlockingSpheresState] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_blocking_spheres_from_state[State](
    state: Lens[State, IsBlockingSpheresGraphState],
    probe: None = None,
    *,
    parameters: BlockingSpheresParameters,
    neighborlist_factory: NeighborListFactory[IsBlockingSpheresGraphState] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_blocking_spheres_from_state[State, P: Patch[Any]](
    state: Lens[State, IsBlockingSpheresGraphState],
    probe: Probe[State, P, IsBlockingSpheresProbe],
    *,
    parameters: BlockingSpheresParameters,
    neighborlist_factory: NeighborListFactory[IsBlockingSpheresGraphState] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


def make_blocking_spheres_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: BlockingSpheresParameters | None = None,
    neighborlist_factory: NeighborListFactory[
        Any
    ] = adaptive_cutoff_neighborlist_from_state,
) -> Any:
    """Create a blocking spheres potential, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, groups, systems, and
            neighbor list (plus ``blocking_spheres_parameters`` when
            ``parameters`` is not given).
        probe: Probe returning a IsBlockingSpheresProbe; ``None`` for full
            recomputation.
        parameters: Constant blocking sphere parameters. When given they are
            bound with a constant lens and the state need not carry
            ``blocking_spheres_parameters``.
        neighborlist_factory: Builds a ``NeighborList[Literal[2]]`` from the
            sub-state and per-system cutoffs.

    Returns:
        Configured blocking spheres Potential.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if probe is not None:
        patch_idx_view = patch_idx_view or empty_patch_idx_view

    if parameters is not None:
        param_view: Any = const_lens(parameters)
    else:
        param_view = state.focus(lambda x: x.blocking_spheres_parameters)

    def neighborlist_view(s: Any) -> BlockingSpheresNeighborListFactory:
        return lambda cutoffs: neighborlist_factory(state(s), cutoffs)

    return make_blocking_spheres_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.groups),
        state.focus(lambda x: x.systems),
        param_view,
        neighborlist_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
    )
