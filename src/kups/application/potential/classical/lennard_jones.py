# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for Lennard-Jones potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, neighbor list, and parameters into the
state-agnostic factories in
[kups.potential.classical.lennard_jones][].

Parameters may live on the state (``state.lj_parameters``) or be passed directly
via ``parameters=``; in the latter case they are bound with a constant lens and
the state need not carry a parameter field. For incremental (probe) updates with
constant parameters, the cache is read from a conventional ``lj_cache`` attribute.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast, overload

from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Table
from kups.core.lens import Lens, const_lens, identity_lens
from kups.core.neighborlist import (
    IsAdaptiveCutoffNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
    adaptive_cutoff_neighborlist_from_state,
)
from kups.core.patch import Patch, Probe
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import HasCache, HasCell, IsState, MaybeCached, SystemId
from kups.potential.classical.lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    IsLJGraphParticles,
    LennardJonesParameters,
    make_global_lennard_jones_tail_correction_potential,
    make_global_lennard_jones_tail_correction_pressure,
    make_lennard_jones_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, IsGraphProbe


class HasLJParticlesAndSystems(
    IsState[IsLJGraphParticles, HasCell[AnyPeriodicity]], Protocol
): ...


class IsLJGraphState(
    HasLJParticlesAndSystems,
    IsAdaptiveCutoffNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, systems, and neighbor list for an LJ graph (no parameters)."""


class IsLJState[Params](IsLJGraphState, Protocol):
    """:class:`IsLJGraphState` that also carries LJ parameters on the state."""

    @property
    def lj_parameters(self) -> Params: ...


class IsCachedLJGraphState[Cache](IsLJGraphState, Protocol):
    """:class:`IsLJGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def lj_cache(self) -> Cache: ...


@overload
def make_lennard_jones_from_state[State](
    state: Lens[State, IsLJState[MaybeCached[LennardJonesParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_lennard_jones_from_state[State](
    state: Lens[State, IsLJState[MaybeCached[LennardJonesParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_lennard_jones_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsLJState[HasCache[LennardJonesParameters, PotentialOut[EmptyType, EmptyType]]],
    ],
    probe: Probe[State, P, IsGraphProbe[IsLJGraphParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_lennard_jones_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsLJState[
            HasCache[LennardJonesParameters, PotentialOut[PositionsAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsLJGraphParticles, Literal[2]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_lennard_jones_from_state[State](
    state: Lens[State, IsLJGraphState],
    probe: None = None,
    *,
    parameters: LennardJonesParameters,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_lennard_jones_from_state[State](
    state: Lens[State, IsLJGraphState],
    probe: None = None,
    *,
    parameters: LennardJonesParameters,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_lennard_jones_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedLJGraphState[PotentialOut[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsLJGraphParticles, Literal[2]]],
    *,
    parameters: LennardJonesParameters,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_lennard_jones_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedLJGraphState[PotentialOut[PositionsAndCell, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsLJGraphParticles, Literal[2]]],
    *,
    parameters: LennardJonesParameters,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_lennard_jones_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: LennardJonesParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
    neighborlist_factory: NeighborListFactory[
        Any
    ] = adaptive_cutoff_neighborlist_from_state,
) -> Any:
    """Create a LJ potential from a typed state, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, systems, and neighbor
            list (plus ``lj_parameters`` when ``parameters`` is not given).
        probe: Detects which particles changed since the last step.
            ``None`` for full recomputation.
        parameters: Constant LJ parameters. When given they are bound with a
            constant lens and the state need not carry ``lj_parameters``; with a
            ``probe``, the cache is read from ``state.lj_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured Lennard-Jones potential.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.lj_parameters.data
                if isinstance(x.lj_parameters, HasCache)
                else x.lj_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.lj_parameters.data)
            cache_view = state.focus(lambda x: x.lj_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.lj_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), param_view(s).cutoff)

    return make_lennard_jones_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
        cache_view,
    )


type IsGlobalTailCorrectedIsLJState = IsLJState[
    MaybeCached[
        GlobalTailCorrectedLennardJonesParameters,
        PotentialOut[Any, Any],
    ]
]


@overload
def make_lennard_jones_tail_correction_from_state[
    InState,
    State: IsGlobalTailCorrectedIsLJState,
](
    state: Lens[InState, State],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[InState, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_lennard_jones_tail_correction_from_state[
    InState,
    State: IsGlobalTailCorrectedIsLJState,
](
    state: Lens[InState, State],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[InState, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_lennard_jones_tail_correction_from_state[State](
    state: Lens[State, IsLJGraphState],
    *,
    parameters: GlobalTailCorrectedLennardJonesParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_lennard_jones_tail_correction_from_state[State](
    state: Lens[State, IsLJGraphState],
    *,
    parameters: GlobalTailCorrectedLennardJonesParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


def make_lennard_jones_tail_correction_from_state(
    state: Any,
    *,
    parameters: GlobalTailCorrectedLennardJonesParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a global tail-corrected LJ potential from a typed state.

    Args:
        state: Lens into the sub-state providing particles and systems (plus
            ``lj_parameters`` when ``parameters`` is not given).
        parameters: Constant tail-correction parameters, bound with a constant
            lens when given (state need not carry ``lj_parameters``).
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured tail-corrected Lennard-Jones potential.
    """
    gradient_lens: Any = EMPTY_LENS
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
    if parameters is not None:
        param_view: Any = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.lj_parameters.data
                if isinstance(x.lj_parameters, HasCache)
                else x.lj_parameters
            )
        )
    return make_global_lennard_jones_tail_correction_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        cast(Any, param_view),
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
    )


def global_lennard_jones_tail_correction_pressure_from_state(
    key: Array,
    state: IsGlobalTailCorrectedIsLJState,
    *,
    parameters: GlobalTailCorrectedLennardJonesParameters | None = None,
) -> Table[SystemId, Array]:
    """Create long-range pressure correction from a typed state.

    When ``parameters`` is given it is bound with a constant lens and the state
    need not carry ``lj_parameters``.
    """
    state_lens = identity_lens(type(state))
    if parameters is not None:
        param_view: Any = const_lens(parameters)
    else:
        param_view = state_lens.focus(
            lambda x: (
                x.lj_parameters.data
                if isinstance(x.lj_parameters, HasCache)
                else x.lj_parameters
            )
        )
    return make_global_lennard_jones_tail_correction_pressure(
        state_lens.focus(lambda x: x.particles),
        state_lens.focus(lambda x: x.systems),
        cast(Any, param_view),
    )(key, state)
