# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for Ewald potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, neighbor list, and parameters into the
state-agnostic factories in [kups.potential.classical.ewald][].

Parameters may live on the state (``state.ewald_parameters``) or be passed
directly via ``parameters=``; in the latter case they are bound with a constant
lens and the state need not carry a parameter field. For incremental (probe)
updates with constant parameters, the cache is read from a conventional
``ewald_cache`` attribute.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from kups.core.cell import Periodic3D
from kups.core.lens import Lens, SimpleLens, const_lens
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
    empty_patch_idx_view,
)
from kups.core.typing import HasCache, HasCell, IsState, MaybeCached
from kups.potential.classical.ewald import (
    EwaldCache,
    EwaldParameters,
    EwaldPotential,
    IsEwaldPointData,
    make_ewald_potential,
)
from kups.potential.common.energy import PositionAndCell, position_and_cell_idx_view
from kups.potential.common.graph import IsGraphProbe, PointCloud


class IsEwaldGraphState(
    IsState[IsEwaldPointData, HasCell[Periodic3D]],
    IsAdaptiveCutoffNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles, systems, and neighbor list for an Ewald graph (no parameters)."""


class IsEwaldState[Params](IsEwaldGraphState, Protocol):
    """:class:`IsEwaldGraphState` that also carries Ewald parameters on the state."""

    @property
    def ewald_parameters(self) -> Params: ...


class IsCachedEwaldGraphState[Cache](IsEwaldGraphState, Protocol):
    """:class:`IsEwaldGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def ewald_cache(self) -> Cache: ...


@overload
def make_ewald_from_state[State](
    state: Lens[State, IsEwaldState[MaybeCached[EwaldParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[
        IsEwaldState[MaybeCached[EwaldParameters, Any]]
    ] = ...,
) -> EwaldPotential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_ewald_from_state[State](
    state: Lens[State, IsEwaldState[MaybeCached[EwaldParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[
        IsEwaldState[MaybeCached[EwaldParameters, Any]]
    ] = ...,
) -> EwaldPotential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_ewald_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsEwaldState[HasCache[EwaldParameters, EwaldCache[EmptyType, EmptyType]]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsEwaldPointData, Literal[2]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[
        IsEwaldState[HasCache[EwaldParameters, EwaldCache[EmptyType, EmptyType]]]
    ] = ...,
) -> EwaldPotential[State, EmptyType, EmptyType, P]: ...


@overload
def make_ewald_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsEwaldState[HasCache[EwaldParameters, EwaldCache[PositionAndCell, EmptyType]]],
    ],
    probe: Probe[State, P, IsGraphProbe[IsEwaldPointData, Literal[2]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[
        IsEwaldState[HasCache[EwaldParameters, EwaldCache[PositionAndCell, EmptyType]]]
    ] = ...,
) -> EwaldPotential[State, PositionAndCell, EmptyType, P]: ...


@overload
def make_ewald_from_state[State](
    state: Lens[State, IsEwaldGraphState],
    probe: None = None,
    *,
    parameters: EwaldParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[IsEwaldGraphState] = ...,
) -> EwaldPotential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_ewald_from_state[State](
    state: Lens[State, IsEwaldGraphState],
    probe: None = None,
    *,
    parameters: EwaldParameters,
    compute_position_and_cell_gradients: Literal[True],
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[IsEwaldGraphState] = ...,
) -> EwaldPotential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_ewald_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedEwaldGraphState[EwaldCache[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsEwaldPointData, Literal[2]]],
    *,
    parameters: EwaldParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[
        IsCachedEwaldGraphState[EwaldCache[EmptyType, EmptyType]]
    ] = ...,
) -> EwaldPotential[State, EmptyType, EmptyType, P]: ...


@overload
def make_ewald_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedEwaldGraphState[EwaldCache[PositionAndCell, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsEwaldPointData, Literal[2]]],
    *,
    parameters: EwaldParameters,
    compute_position_and_cell_gradients: Literal[True],
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[
        IsCachedEwaldGraphState[EwaldCache[PositionAndCell, EmptyType]]
    ] = ...,
) -> EwaldPotential[State, PositionAndCell, EmptyType, P]: ...


def make_ewald_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: EwaldParameters | None = None,
    compute_position_and_cell_gradients: bool = False,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[
        Any
    ] = adaptive_cutoff_neighborlist_from_state,
) -> Any:
    """Create an Ewald potential from a typed state, optionally with incremental updates.

    When ``probe`` is ``None``, builds a static potential by extracting
    components directly from the state.  When a probe is provided, the
    potential supports incremental (cached) evaluation via the probe's
    patch mechanism.

    Args:
        state: Lens focusing on the Ewald state (particles, systems,
            neighborlist, plus ``ewald_parameters`` when ``parameters`` is not
            given).
        probe: Probe for incremental updates. ``None`` for a static
            potential.
        parameters: Constant Ewald parameters. When given they are bound with a
            constant lens and the state need not carry ``ewald_parameters``;
            with a ``probe``, the cache is read from ``state.ewald_cache``.
        compute_position_and_cell_gradients: When ``True``, the
            returned potential computes gradients w.r.t. particle
            positions and lattice vectors (for forces / stress).
            Gradient type becomes ``PositionAndCell``.
        include_exclusion_mask: Whether to include the exclusion
            correction term in the returned potential.

    Returns:
        An ``EwaldPotential`` combining short-range, long-range,
        self-energy, and (optionally) exclusion-correction terms.
        Gradient type is ``PositionAndCell`` when gradients are
        requested, ``EmptyType`` otherwise.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view = empty_patch_idx_view
    if compute_position_and_cell_gradients:
        gradient_lens = SimpleLens[
            PointCloud[IsEwaldPointData, HasCell[Periodic3D]], PositionAndCell
        ](
            lambda pc: PositionAndCell(
                pc.particles.map_data(lambda p: p.positions),
                pc.systems.map_data(lambda s: s.cell),
            )
        )
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.ewald_parameters.data
                if isinstance(x.ewald_parameters, HasCache)
                else x.ewald_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.ewald_parameters.data)
            cache_view = state.focus(lambda x: x.ewald_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.ewald_cache)

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), param_view(s).cutoff)

    return make_ewald_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        param_view,
        cache_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        include_exclusion_mask=include_exclusion_mask,
    )
