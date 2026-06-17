# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Geometric-view types shared by gradient filters and the gradient machinery.

``Geometry`` is the ``Table``-based projection of any carrier's particles and
systems (per-particle position + system index, per-system cell). A relaxation
filter is a ``Lens[Geometry, U]`` selecting the optimizer's degrees of freedom;
the concrete filters live in [kups.application.potential.filter][], while the
adapters that build a ``Geometry`` from a carrier live with their carriers
(``GRAPH_GEOMETRY`` in [kups.potential.common.graph][], ``pointcloud_geometry``
in [kups.potential.classical.ewald][]).
"""

from typing import NamedTuple, no_type_check

from jax import Array

from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.potential import EMPTY, EmptyType, PotentialOut, empty_patch_idx_view
from kups.core.typing import HasPositionsAndSystemIndex, IsState, ParticleId, SystemId


class PositionsAndSystemIndex(NamedTuple):
    """Per-particle geometric data: cartesian positions + system assignment."""

    positions: Array
    system: Index[SystemId]


class Geometry(NamedTuple):
    """Shared domain of geometric gradient lenses, built from any carrier."""

    particles: Table[ParticleId, PositionsAndSystemIndex]
    systems: Table[SystemId, Cell[AnyPeriodicity]]


class PositionsAndCell(NamedTuple):
    """Optimizer DOFs / gradient payload of the standard cell filters.

    ``positions`` is a per-particle table of DOF coordinates (``∂E/∂r`` when this
    carries a gradient); ``cell`` is a per-system table of the cell (its frame
    parameters carrying ``∂E/∂h`` for a gradient).
    """

    positions: Table[ParticleId, Array]
    cell: Table[SystemId, Cell[AnyPeriodicity]]


type IsStateWithParticlesAndCell = IsState[
    HasPositionsAndSystemIndex, Cell[AnyPeriodicity]
]


@no_type_check
def position_and_cell_idx_view(
    state: IsStateWithParticlesAndCell,
) -> PotentialOut[PositionsAndCell, EmptyType]:
    """Patch index structure matching the ``PositionsAndCell`` filter codomain.

    The per-particle system index masks the position rows and the systems index
    masks every ``Cell`` leaf.
    """
    return PotentialOut(
        empty_patch_idx_view(state).total_energies,
        PositionsAndCell(state.particles.data.system, state.systems.index),
        EMPTY,
    )
