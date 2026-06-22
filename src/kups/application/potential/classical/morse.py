# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for Morse bond potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, bond indices, and parameters into the
state-agnostic factories in
[kups.potential.classical.morse][].

Parameters may live on the state (``state.morse_bond_parameters``) or be passed
directly via ``parameters=``; in the latter case they are bound with a constant
lens and the state need not carry a parameter field. For incremental (probe)
updates with constant parameters, the cache is read from a conventional
``morse_bond_cache`` attribute.
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
from kups.potential.classical.morse import (
    IsBondedParticles,
    MorseBondInput,
    MorseBondParameters,
    make_morse_bond_potential,
)
from kups.potential.common.energy import PositionAndCell, position_and_cell_idx_view
from kups.potential.common.graph import IsGraphProbe


class IsMorseBondGraphState(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
):
    """Particles, systems, and bond indices for a Morse bond (no parameters)."""

    @property
    def bond_edge_indices(self) -> Index[ParticleId]: ...


class IsMorseBondState[Params](IsMorseBondGraphState, Protocol):
    """:class:`IsMorseBondGraphState` that also carries Morse bond parameters."""

    @property
    def morse_bond_parameters(self) -> Params: ...


class IsCachedMorseBondGraphState[Cache](IsMorseBondGraphState, Protocol):
    """:class:`IsMorseBondGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def morse_bond_cache(self) -> Cache: ...


@overload
def make_morse_bond_from_state[State](
    state: Lens[State, IsMorseBondState[MaybeCached[MorseBondParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_morse_bond_from_state[State](
    state: Lens[State, IsMorseBondState[MaybeCached[MorseBondParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_morse_bond_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsMorseBondState[
            HasCache[MorseBondParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_morse_bond_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsMorseBondState[
            HasCache[MorseBondParameters, PotentialOut[PositionAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


@overload
def make_morse_bond_from_state[State](
    state: Lens[State, IsMorseBondGraphState],
    probe: None = None,
    *,
    parameters: MorseBondParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_morse_bond_from_state[State](
    state: Lens[State, IsMorseBondGraphState],
    probe: None = None,
    *,
    parameters: MorseBondParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_morse_bond_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedMorseBondGraphState[PotentialOut[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: MorseBondParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_morse_bond_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedMorseBondGraphState[PotentialOut[PositionAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: MorseBondParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


def make_morse_bond_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: MorseBondParameters | None = None,
    compute_position_and_cell_gradients: bool = False,
) -> Any:
    """Create a Morse bond potential from a typed state, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, cell, and bond
            indices (plus ``morse_bond_parameters`` when ``parameters`` is not
            given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
            Those neighbor lists carry any required update capacity.
        parameters: Constant Morse bond parameters. When given they are bound
            with a constant lens and the state need not carry
            ``morse_bond_parameters``; with a ``probe``, the cache is read from
            ``state.morse_bond_cache``.
        compute_position_and_cell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors (for forces / stress).

    Returns:
        Configured Morse bond [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_cell_gradients:
        gradient_lens = SimpleLens[MorseBondInput, PositionAndCell](
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
                x.morse_bond_parameters.data
                if isinstance(x.morse_bond_parameters, HasCache)
                else x.morse_bond_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.morse_bond_parameters.data)
            cache_view = state.focus(lambda x: x.morse_bond_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.morse_bond_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_morse_bond_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.bond_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )
