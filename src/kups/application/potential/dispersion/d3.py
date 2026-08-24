# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""State-binding constructor for the D3 dispersion correction.

Takes a [Lens][kups.core.lens.Lens] into a concrete simulation state and wires its
particles, systems and neighbor list into the state-agnostic factory in
[kups.potential.dispersion.d3][].

Parameters may live on the state (``state.d3_parameters``) or be passed directly
via ``parameters=``; in the latter case they are bound with a constant lens and the
state need not carry a parameter field. Compose the result with an MLIP or a
classical force field using
[sum_potentials][kups.core.potential.sum_potentials], giving both terms the *same*
``gradient`` filter so their gradient pytrees match.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

from jax import Array

from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table
from kups.core.lens import Lens, const_lens
from kups.core.neighborlist import (
    AdaptiveNeighborList,
    IsNeighborListState,
    IsUniversalNeighborlistParams,
    NeighborList,
    NeighborListFactory,
)
from kups.core.patch import Patch
from kups.core.potential import EMPTY_LENS, EmptyType, Potential
from kups.core.typing import (
    ExclusionId,
    HasCell,
    InclusionId,
    IsState,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY
from kups.potential.dispersion.d3 import (
    D3Parameters,
    IsD3GraphParticles,
    make_d3_potential,
)

__all__ = ["IsD3GraphState", "IsD3State", "make_d3_from_state"]


@dataclass
class _D3ParticleData:
    """Particle view with D3's own inclusion/exclusion semantics.

    Dispersion acts between *every* pair in a system, including atoms of the same
    molecule. A state's native ``exclusion`` index often means something else --
    in an MCMC state it is the molecular group, used to remove intramolecular
    Coulomb — and
    [ExclusionMask][kups.core.neighborlist.masks.ExclusionMask] drops the
    minimum-image pair of anything sharing an exclusion id. Handing D3 the state's
    exclusion index would therefore delete every intramolecular dispersion
    interaction, silently and without error.

    So D3 rebuilds the view the way Ewald's atomic term does: ``inclusion`` is the
    system (pairs never cross systems) and ``exclusion`` is unique per particle,
    which suppresses exactly the ``i == j``, ``T == 0`` self pair while leaving
    self-images intact.
    """

    positions: Array
    atomic_numbers: Array
    system: Index[SystemId]
    inclusion: Index[InclusionId]
    exclusion: Index[ExclusionId]


def _d3_particles(
    particles: Table[ParticleId, Any],
) -> Table[ParticleId, _D3ParticleData]:
    """Re-index particles with D3's inclusion/exclusion semantics."""
    data = particles.data
    return Table(
        particles.keys,
        _D3ParticleData(
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            system=data.system,
            inclusion=data.system.to_cls(InclusionId),
            exclusion=Index.arange(len(particles), label=ExclusionId),
        ),
    )


class HasD3ParticlesAndSystems(
    IsState[IsD3GraphParticles, HasCell[AnyPeriodicity]], Protocol
): ...


class IsD3GraphState(
    HasD3ParticlesAndSystems,
    IsNeighborListState[IsUniversalNeighborlistParams],
    Protocol,
):
    """Particles (with atomic numbers), systems, and neighbor-list capacities."""


class IsD3State(IsD3GraphState, Protocol):
    """:class:`IsD3GraphState` that also carries D3 parameters on the state."""

    @property
    def d3_parameters(self) -> D3Parameters: ...


@overload
def make_d3_from_state[State](
    state: Lens[State, IsD3State],
    *,
    parameters: None = None,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_d3_from_state[State](
    state: Lens[State, IsD3State],
    *,
    parameters: None = None,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


@overload
def make_d3_from_state[State](
    state: Lens[State, IsD3GraphState],
    *,
    parameters: D3Parameters,
    gradient: None = None,
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, EmptyType, EmptyType, Patch[Any]]: ...


@overload
def make_d3_from_state[State](
    state: Lens[State, IsD3GraphState],
    *,
    parameters: D3Parameters,
    gradient: Lens[Geometry, PositionsAndCell],
    neighborlist_factory: NeighborListFactory[Any] = ...,
) -> Potential[State, PositionsAndCell, EmptyType, Patch[Any]]: ...


def make_d3_from_state(
    state: Any,
    *,
    parameters: D3Parameters | None = None,
    gradient: Lens[Geometry, PositionsAndCell] | None = None,
    neighborlist_factory: NeighborListFactory[Any] = AdaptiveNeighborList.from_state,
) -> Any:
    """Create a D3(BJ) dispersion potential from a typed state.

    The neighbor list is built at ``parameters.cutoff``; because ``cn_cutoff`` is
    capped at ``cutoff``, that single list also serves the coordination numbers.
    Particles are re-indexed through ``_D3ParticleData`` first, so a state whose
    ``exclusion`` index carries molecular groups (as MCMC states do) does not lose
    its intramolecular dispersion.
    Size ``state.neighborlist_params`` from the **largest** cutoff among all
    composed potentials — D3's is normally the largest by a wide margin.

    There is no ``probe`` argument: the coordination number couples atoms beyond
    the pair being moved, so D3 does not support incremental Monte-Carlo updates
    and always recomputes in full.

    **Scope.** This is the pairwise (two-body) D3 term with Becke-Johnson damping,
    which is what the published per-functional ``s6``/``s8``/``a1``/``a2`` fits are
    fitted to. The Axilrod-Teller-Muto three-body term is not implemented; it is
    small for dense molecular solids but not always negligible in porous
    frameworks.

    Args:
        state: Lens into the sub-state providing particles, systems and
            neighbor-list capacities (plus ``d3_parameters`` when ``parameters``
            is not given).
        parameters: Constant D3 parameters, normally from
            [D3Parameters.from_functional][kups.potential.dispersion.d3.D3Parameters.from_functional].
            When given they are bound with a constant lens and the state need not
            carry ``d3_parameters``.
        gradient: Relaxation filter selecting the optimizer degrees of freedom.
            ``None`` computes no gradients. Composed with ``GRAPH_GEOMETRY`` into
            the potential's gradient lens.
        neighborlist_factory: Strategy for building the pair neighbor list.

    Returns:
        A configured D3 potential.
    """
    gradient_lens: Any = EMPTY_LENS
    patch_idx_view: Any = None
    if gradient is not None:
        gradient_lens = GRAPH_GEOMETRY.nest(gradient)
        patch_idx_view = position_and_cell_idx_view

    if parameters is not None:
        param_view: Any = const_lens(parameters)
    else:
        param_view = state.focus(lambda x: x.d3_parameters)

    def particles_view(s: Any) -> Table[ParticleId, _D3ParticleData]:
        return _d3_particles(state(s).particles)

    def neighborlist_view(s: Any) -> NeighborList[Literal[2]]:
        return neighborlist_factory(state(s), param_view(s).cutoff)

    return make_d3_potential(
        particles_view,
        state.focus(lambda x: x.systems),
        neighborlist_view,
        param_view,
        gradient_lens,
        EMPTY_LENS,
        EMPTY_LENS,
        patch_idx_view,
        None,
    )
