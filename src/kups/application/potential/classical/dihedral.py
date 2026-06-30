# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructors for dihedral/torsion potentials.

These adapters take a :class:`~kups.core.lens.Lens` into a concrete simulation
state and wire its particles, systems, dihedral indices, and parameters into the
state-agnostic factories in [kups.potential.classical.dihedral][].

Parameters may live on the state (``state.dihedral_parameters``) or be passed
directly via ``parameters=``; in the latter case they are bound with a constant
lens and the state need not carry a parameter field. For incremental (probe)
updates with constant parameters, the cache is read from a conventional
``dihedral_cache`` attribute.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index
from kups.core.lens import Lens, const_lens
from kups.core.patch import Patch, Probe
from kups.core.potential import (
    EMPTY_LENS,
    EmptyType,
    Potential,
    PotentialOut,
    empty_patch_idx_view,
)
from kups.core.typing import HasCache, HasCell, IsState, MaybeCached, ParticleId
from kups.core.utils.kahan import KahanSummand
from kups.potential.classical.dihedral import (
    DihedralParameters,
    IsBondedParticles,
    make_dihedral_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, IsGraphProbe


class IsDihedralGraphState(
    IsState[IsBondedParticles, HasCell[AnyPeriodicity]], Protocol
):
    """Particles, systems, and dihedral indices (no parameters)."""

    @property
    def dihedral_edge_indices(self) -> Index[ParticleId]: ...


class IsDihedralState[Params](IsDihedralGraphState, Protocol):
    """:class:`IsDihedralGraphState` that also carries dihedral parameters."""

    @property
    def dihedral_parameters(self) -> Params: ...


class IsCachedDihedralGraphState[Cache](IsDihedralGraphState, Protocol):
    """:class:`IsDihedralGraphState` carrying an incremental-update cache (params passed in)."""

    @property
    def dihedral_cache(self) -> Cache: ...


@overload
def make_dihedral_from_state[State](
    state: Lens[State, IsDihedralState[MaybeCached[DihedralParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_dihedral_from_state[State](
    state: Lens[State, IsDihedralState[MaybeCached[DihedralParameters, Any]]],
    probe: None = None,
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsDihedralState[
            HasCache[
                DihedralParameters, KahanSummand[PotentialOut[EmptyType, EmptyType]]
            ]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: None = None,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsDihedralState[
            HasCache[
                DihedralParameters,
                KahanSummand[PotentialOut[PositionsAndCell, EmptyType]],
            ]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


@overload
def make_dihedral_from_state[State](
    state: Lens[State, IsDihedralGraphState],
    probe: None = None,
    *,
    parameters: DihedralParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_dihedral_from_state[State](
    state: Lens[State, IsDihedralGraphState],
    probe: None = None,
    *,
    parameters: DihedralParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCachedDihedralGraphState[KahanSummand[PotentialOut[EmptyType, EmptyType]]],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: DihedralParameters,
    gradient: None = None,
) -> Potential[State, EmptyType, EmptyType, P]: ...


@overload
def make_dihedral_from_state[State, P: Patch[Any]](
    state: Lens[
        State,
        IsCachedDihedralGraphState[
            KahanSummand[PotentialOut[PositionsAndCell, EmptyType]]
        ],
    ],
    probe: Probe[State, P, IsGraphProbe[IsBondedParticles, Literal[4]]],
    *,
    parameters: DihedralParameters,
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, P]: ...


def make_dihedral_from_state(
    state: Any,
    probe: Any = None,
    *,
    parameters: DihedralParameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
) -> Any:
    """Create a dihedral potential from a typed state, optionally with incremental updates.

    Args:
        state: Lens into the sub-state providing particles, cell, and dihedral
            indices (plus ``dihedral_parameters`` when ``parameters`` is not
            given).
        probe: If provided, detects particle changes and supplies the
            before/after fixed-edge neighbor lists for incremental updates.
            Those neighbor lists carry any required update capacity.
        parameters: Constant dihedral parameters. When given they are bound with
            a constant lens and the state need not carry ``dihedral_parameters``;
            with a ``probe``, the cache is read from ``state.dihedral_cache``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting the
            optimizer DOFs. ``None`` computes no gradients (the default). Composed with
            ``GRAPH_GEOMETRY`` into the potential's gradient lens.

    Returns:
        Configured dihedral [Potential][kups.core.potential.Potential].
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
                x.dihedral_parameters.data
                if isinstance(x.dihedral_parameters, HasCache)
                else x.dihedral_parameters
            )
        )
    cache_view = None
    if probe is not None:
        if parameters is None:
            param_view = state.focus(lambda x: x.dihedral_parameters.data)
            cache_view = state.focus(lambda x: x.dihedral_parameters.cache)
        else:
            cache_view = state.focus(lambda x: x.dihedral_cache)
        patch_idx_view = patch_idx_view or empty_patch_idx_view
    return make_dihedral_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.dihedral_edge_indices),
        state.focus(lambda x: x.systems),
        param_view,
        probe,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        out_cache_lens=cache_view,
    )
