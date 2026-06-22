# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for harmonic bonded potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, bond/angle indices, and parameters into
the state-agnostic factories in
[kups.potential.classical.harmonic][].

Parameters may live on the state (``state.harmonic_bond_parameters`` /
``state.harmonic_angle_parameters``) or be passed directly via ``parameters=``;
in the latter case they are bound with a constant lens and the state need not
carry a parameter field. For incremental (probe) updates with constant
parameters, the cache is read from a conventional ``harmonic_bond_cache`` /
``harmonic_angle_cache`` attribute.
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
from kups.potential.classical.harmonic import (
    HarmonicAngleInput,
    HarmonicAngleParameters,
    HarmonicBondInput,
    HarmonicBondParameters,
    IsBondedParticles,
    make_harmonic_angle_potential,
    make_harmonic_bond_potential,
)
from kups.potential.common.energy import PositionAndCell, position_and_cell_idx_view
from kups.potential.common.graph import IsGraphProbe


class HasBondedParticlesAndSystems(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
): ...


class IsHarmonicBondGraphState(HasBondedParticlesAndSystems, Protocol):
    """Particles, systems, and bond indices for a harmonic bond graph (no parameters)."""

    @property
    def harmonic_bond_indices(self) -> Index[ParticleId]: ...


class IsHarmonicBondState[Params](IsHarmonicBondGraphState, Protocol):
    """:class:`IsHarmonicBondGraphState` that also carries bond parameters on the state."""

    @property
    def harmonic_bond_parameters(self) -> Params: ...


class IsCachedHarmonicBondState[Cache](IsHarmonicBondGraphState, Protocol):
    """:class:`IsHarmonicBondGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def harmonic_bond_cache(self) -> Cache: ...


@overload
def make_harmonic_bond_from_state[State](
    state: Lens[
        State,
        IsHarmonicBondState[MaybeCached[HarmonicBondParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_bond_from_state[State](
    state: Lens[
        State,
        IsHarmonicBondState[MaybeCached[HarmonicBondParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_bond_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsHarmonicBondState[
            HasCache[HarmonicBondParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_harmonic_bond_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsHarmonicBondState[
            HasCache[HarmonicBondParameters, PotentialOut[PositionAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


@overload
def make_harmonic_bond_from_state[State](
    state: Lens[State, IsHarmonicBondGraphState],
    probe: None = None,
    *,
    parameters: HarmonicBondParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_bond_from_state[State](
    state: Lens[State, IsHarmonicBondGraphState],
    probe: None = None,
    *,
    parameters: HarmonicBondParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_bond_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedHarmonicBondState[PotentialOut[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: HarmonicBondParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_harmonic_bond_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedHarmonicBondState[PotentialOut[PositionAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[2]]],
    *,
    parameters: HarmonicBondParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


def make_harmonic_bond_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: HarmonicBondParameters | None = None,
    compute_position_and_cell_gradients: bool = False,
) -> Any:
    """Create a harmonic bond potential, optionally with incremental updates.

    Convenience wrapper around
    [make_harmonic_bond_potential][kups.potential.classical.harmonic.make_harmonic_bond_potential].
    When `probe` is `None`, builds a plain potential from
    [IsHarmonicBondState][kups.application.potential.classical.harmonic.IsHarmonicBondState].
    When a `probe` is provided, builds an incrementally-updated potential from
    a state with `HasCache`-wrapped parameters.

    Args:
        state: Lens into the sub-state providing particles, cell, and bond
            indices (plus ``harmonic_bond_parameters`` when ``parameters`` is
            not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
            Those neighbor lists carry any required update capacity.
        parameters: Constant harmonic bond parameters. When given they are bound
            with a constant lens and the state need not carry
            ``harmonic_bond_parameters``; with a ``probe``, the cache is read
            from ``state.harmonic_bond_cache``.
        compute_position_and_cell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors (for forces / stress).

    Returns:
        Configured harmonic bond [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_cell_gradients:
        gradient_lens = SimpleLens[HarmonicBondInput, PositionAndCell](
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
                x.harmonic_bond_parameters.data
                if isinstance(x.harmonic_bond_parameters, HasCache)
                else x.harmonic_bond_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.harmonic_bond_parameters.data)
            cache_view = state.focus(lambda x: x.harmonic_bond_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.harmonic_bond_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_harmonic_bond_potential(
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


class IsHarmonicAngleGraphState(HasBondedParticlesAndSystems, Protocol):
    """Particles, systems, and angle indices for a harmonic angle graph (no parameters)."""

    @property
    def harmonic_angle_indices(self) -> Index[ParticleId]: ...


class IsHarmonicAngleState[Params](IsHarmonicAngleGraphState, Protocol):
    """:class:`IsHarmonicAngleGraphState` that also carries angle parameters on the state."""

    @property
    def harmonic_angle_parameters(self) -> Params: ...


class IsCachedHarmonicAngleState[Cache](IsHarmonicAngleGraphState, Protocol):
    """:class:`IsHarmonicAngleGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def harmonic_angle_cache(self) -> Cache: ...


@overload
def make_harmonic_angle_from_state[State](
    state: Lens[
        State,
        IsHarmonicAngleState[MaybeCached[HarmonicAngleParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_angle_from_state[State](
    state: Lens[
        State,
        IsHarmonicAngleState[MaybeCached[HarmonicAngleParameters, Any]],
    ],
    probe: None = None,
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_angle_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsHarmonicAngleState[
            HasCache[HarmonicAngleParameters, PotentialOut[EmptyType, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_harmonic_angle_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsHarmonicAngleState[
            HasCache[HarmonicAngleParameters, PotentialOut[PositionAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: None = None,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


@overload
def make_harmonic_angle_from_state[State](
    state: Lens[State, IsHarmonicAngleGraphState],
    probe: None = None,
    *,
    parameters: HarmonicAngleParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_angle_from_state[State](
    state: Lens[State, IsHarmonicAngleGraphState],
    probe: None = None,
    *,
    parameters: HarmonicAngleParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_harmonic_angle_from_state[State, P: Patch[Any]](
    state: Lens[State, IsCachedHarmonicAngleState[PotentialOut[EmptyType, EmptyType]]],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: HarmonicAngleParameters,
    compute_position_and_cell_gradients: Literal[False] = ...,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_harmonic_angle_from_state[State, P: Patch[Any]](
    state: Lens[
        State, IsCachedHarmonicAngleState[PotentialOut[PositionAndCell, EmptyType]]
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[3]]],
    *,
    parameters: HarmonicAngleParameters,
    compute_position_and_cell_gradients: Literal[True],
) -> Potential[State, PositionAndCell, EmptyType, P]: ...


def make_harmonic_angle_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: HarmonicAngleParameters | None = None,
    compute_position_and_cell_gradients: bool = False,
) -> Any:
    """Create a harmonic angle potential, optionally with incremental updates.

    Convenience wrapper around
    [make_harmonic_angle_potential][kups.potential.classical.harmonic.make_harmonic_angle_potential].
    When `probe` is `None`, builds a plain potential from
    [IsHarmonicAngleState][kups.application.potential.classical.harmonic.IsHarmonicAngleState].
    When a `probe` is provided, builds an incrementally-updated potential from
    a state with `HasCache`-wrapped parameters.

    Args:
        state: Lens into the sub-state providing particles, cell, and angle
            indices (plus ``harmonic_angle_parameters`` when ``parameters`` is
            not given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
            Those neighbor lists carry any required update capacity.
        parameters: Constant harmonic angle parameters. When given they are
            bound with a constant lens and the state need not carry
            ``harmonic_angle_parameters``; with a ``probe``, the cache is read
            from ``state.harmonic_angle_cache``.
        compute_position_and_cell_gradients: When ``True``, the returned
            potential computes gradients w.r.t. particle positions and lattice
            vectors (for forces / stress).

    Returns:
        Configured harmonic angle [Potential][kups.core.potential.Potential].
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if compute_position_and_cell_gradients:
        gradient_lens = SimpleLens[HarmonicAngleInput, PositionAndCell](
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
                x.harmonic_angle_parameters.data
                if isinstance(x.harmonic_angle_parameters, HasCache)
                else x.harmonic_angle_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.harmonic_angle_parameters.data)
            cache_view = state.focus(lambda x: x.harmonic_angle_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.harmonic_angle_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_harmonic_angle_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.angle_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )
