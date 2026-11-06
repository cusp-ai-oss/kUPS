# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructor for smooth particle-mesh Ewald potentials.

The adapter takes a :class:`~kups.core.lens.Lens` into a concrete simulation state
and wires its particles, systems, neighbor list and parameters into
[make_pme_potential][kups.potential.classical.ewald.make_pme_potential].

Parameters may live on the state (``state.pme_parameters``) or be passed directly
via ``parameters=``. PME evaluates whole systems, so there is no probe and no cache;
MC uses direct Ewald for its incremental updates.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from kups.application.potential.classical.ewald import IsEwaldGraphState
from kups.core.lens import Lens, const_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    NeighborList,
    NeighborListFactory,
)
from kups.core.patch import Patch
from kups.core.potential import EMPTY_LENS, EmptyType, empty_patch_idx_view
from kups.potential.classical.ewald import (
    EwaldPotential,
    PMEParameters,
    make_pme_potential,
)
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import POINTCLOUD_GEOMETRY


class IsPMEState(IsEwaldGraphState, Protocol):
    """:class:`IsEwaldGraphState` that also carries PME parameters on the state."""

    @property
    def pme_parameters(self) -> PMEParameters: ...


@overload
def make_pme_from_state[State](
    state: Lens[State, IsPMEState],
    *,
    parameters: None = None,
    gradient: None = None,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[IsPMEState] = ...,
) -> EwaldPotential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_pme_from_state[State](
    state: Lens[State, IsPMEState],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[IsPMEState] = ...,
) -> EwaldPotential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_pme_from_state[State](
    state: Lens[State, IsEwaldGraphState],
    *,
    parameters: PMEParameters,
    gradient: None = None,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[IsEwaldGraphState] = ...,
) -> EwaldPotential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_pme_from_state[State](
    state: Lens[State, IsEwaldGraphState],
    *,
    parameters: PMEParameters,
    gradient: Lens[Geometry, PositionsAndCell],
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[IsEwaldGraphState] = ...,
) -> EwaldPotential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


def make_pme_from_state(
    state: Any,
    *,
    parameters: PMEParameters | None = None,
    gradient: Lens[Geometry, Any] | None = None,
    include_exclusion_mask: bool = False,
    neighborlist_factory: NeighborListFactory[Any] = AdaptiveNeighborList.from_state,
) -> Any:
    """Create a PME potential from a typed state.

    Args:
        state: Lens focusing on the PME state (particles, systems and neighbor-list
            parameters, plus ``pme_parameters`` when ``parameters`` is not given).
        parameters: Constant PME parameters. When given they are bound with a
            constant lens and the state need not carry ``pme_parameters``.
        gradient: Relaxation filter ``Lens[Geometry, PositionsAndCell]`` selecting
            the variables to differentiate. Composed with the
            ``PointCloud``-to-``Geometry`` adapter; ``None`` computes no gradients.
        include_exclusion_mask: Whether to subtract molecular-exclusion pair
            energies.
        neighborlist_factory: Builds the real-space neighbor list from the root
            state, the ``state`` lens and the PME cutoffs.

    Returns:
        An ``EwaldPotential`` with the Ewald real-space, self and background terms
        and the PME reciprocal term.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view = empty_patch_idx_view
    if gradient is not None:
        gradient_lens = POINTCLOUD_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view
    if parameters is not None:
        param_view = const_lens(parameters)
    else:
        param_view = state.focus(lambda x: x.pme_parameters)

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(s, state, param_view(s).cutoff)

    return make_pme_potential(
        state.focus(lambda x: x.particles),
        state.focus(lambda x: x.systems),
        neighborlist_view,
        param_view,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view=patch_idx_view,
        include_exclusion_mask=include_exclusion_mask,
    )
