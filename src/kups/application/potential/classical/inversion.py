# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for UFF-style inversion potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, inversion connectivity, and parameters
into the state-agnostic factories in
[kups.potential.classical.inversion][].

Parameters may live on the state (``state.inversion_parameters``) or be passed
directly via ``parameters=``; in the latter case they are bound with a constant
lens and the state need not carry a parameter field. For incremental (probe)
updates with constant parameters, the cache is read from a conventional
``inversion_cache`` attribute.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index
from kups.core.lens import Lens, SimpleLens, const_lens
from kups.core.patch import Patch, Probe
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import HasCache, HasCell, IsState, MaybeCached, ParticleId
from kups.potential.classical.inversion import (
    InversionInput,
    InversionParameters,
    IsBondedParticles,
    make_inversion_potential,
)
from kups.potential.common.energy import PositionAndCell, position_and_cell_idx_view
from kups.potential.common.graph import IsGraphProbe


class IsInversionGraphState(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
):
    """Particles, systems, and inversion connectivity (no parameters)."""

    @property
    def inversion_edge_indices(self) -> Index[ParticleId]: ...


class IsInversionState[Params](IsInversionGraphState, Protocol):
    """:class:`IsInversionGraphState` that also carries inversion parameters."""

    @property
    def inversion_parameters(self) -> Params: ...


class IsCachedInversionGraphState[Cache](IsInversionGraphState, Protocol):
    """:class:`IsInversionGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def inversion_cache(self) -> Cache: ...


@overload
def make_inversion_from_state[State](
    state: Lens[
        State,
        IsInversionState[MaybeCached[InversionParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_inversion_from_state[State](
    state: Lens[
        State,
        IsInversionState[MaybeCached[InversionParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_inversion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsInversionState[
            HasCache[InversionParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_inversion_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsInversionState[
            HasCache[InversionParameters, PotentialOut[PositionAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


@overload
def make_inversion_from_state[State](
    state: Lens[State, IsInversionGraphState],
    probe: None = None,
    *,
    parameters: InversionParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_inversion_from_state[State](
    state: Lens[State, IsInversionGraphState],
    probe: None = None,
    *,
    parameters: InversionParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_inversion_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedInversionGraphState[PotentialOut[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: InversionParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_inversion_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedInversionGraphState[PotentialOut[PositionAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: InversionParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


def make_inversion_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: InversionParameters | None = None,
    compute_position_and_cell_gradients: bool = False,
) -> Any:
    """Create an inversion potential, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, cell, inversion
            indices, and inversion parameters (the latter only when
            ``parameters`` is not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
            Those neighbor lists carry any required update capacity.
        parameters: Constant inversion parameters. When given they are bound
            with a constant lens and the state need not carry
            ``inversion_parameters``; with a ``probe``, the cache is read from
            ``state.inversion_cache``.
        compute_position_and_cell_gradients: When True, computes gradients
            w.r.t. particle positions and lattice vectors.

    Returns:
        Configured inversion [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_cell_gradients:
        gradient_lens = SimpleLens[InversionInput, PositionAndCell](
            lambda x: PositionAndCell(
                x.graph.particles.map_data(lambda p: p.positions),
                x.graph.systems.map_data(lambda s: s.cell),
            )
        )
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(
            lambda x: (
                x.inversion_parameters.data
                if isinstance(x.inversion_parameters, HasCache)
                else x.inversion_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.inversion_parameters.data)
            cache_view = state.focus(lambda x: x.inversion_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.inversion_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_inversion_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.inversion_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )
