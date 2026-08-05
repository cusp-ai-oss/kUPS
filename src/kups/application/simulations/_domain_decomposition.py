# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared wiring for the domain-decomposed LJ drivers.

Everything the MD and relaxation DD apps have in common: re-tagging particles
with their owner device, the ``OriginDeviceId`` mesh, the shard-mapped
propagator wrapper, the mesh-max'd cell-list view, and the sharded LJ
potential. The apps keep only what genuinely differs (integrator vs optimizer,
gradient filters, configs).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Literal

import jax
import numpy as np
from jax import Array

from kups.core.capacity import Capacity, FixedCapacity
from kups.core.cell import AnyPeriodicity
from kups.core.data import Index, Table
from kups.core.domain import (
    IsDecomposedParticle,
    MeshMaxCapacity,
    MortonPartitioner,
    make_sharded_radius_graph_from_state,
)
from kups.core.lens import Lens, View, const_lens
from kups.core.neighborlist import (
    CellListNeighborList,
    NeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.potential import EMPTY_LENS, EmptyType, Potential, ShardedPotential
from kups.core.propagator import Propagator
from kups.core.sharding import shard_axis
from kups.core.typing import (
    HasCell,
    HasPositionsAndSystemIndex,
    IsState,
    OriginDeviceId,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass, field, shard_map
from kups.potential.classical.lennard_jones import (
    LennardJonesParameters,
    lennard_jones_energy,
)
from kups.potential.common.energy import PotentialFromEnergy
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    position_and_cell_idx_view,
)
from kups.potential.common.graph import GRAPH_GEOMETRY, LocalGraphSumComposer

_AXIS = shard_axis(OriginDeviceId)
_REPL = jax.sharding.PartitionSpec()


def origin_mesh() -> jax.sharding.Mesh:
    """The one-axis device mesh the DD apps run on (all local devices)."""
    return jax.sharding.Mesh(np.array(jax.devices()), axis_names=(_AXIS,))


def with_origin[P, D: IsDecomposedParticle](
    particles: Table[ParticleId, P],
    origin: Index[OriginDeviceId],
    cls: Callable[..., D],
) -> Table[ParticleId, D]:
    """Re-tag a particle table as ``cls`` (the same fields plus ``origin``)."""
    d = particles.data
    values = {f.name: getattr(d, f.name) for f in dataclasses.fields(type(d))}
    return particles.set_data(cls(**values, origin=origin))


def partition[P: HasPositionsAndSystemIndex, S: HasCell[AnyPeriodicity]](
    particles: Table[ParticleId, P],
    systems: Table[SystemId, S],
    n_devices: int,
) -> tuple[Index[OriginDeviceId], FixedCapacity[int]]:
    """Morton-partition the particles; also return the owned-buffer capacity.

    ``cap_owned`` (the max atoms any device owns) must be known before the
    potential is built. The Morton cut is balanced, so it is exact.
    """
    origin = MortonPartitioner()(particles, systems, n_devices)
    cap_owned = FixedCapacity(int(origin.counts.data.max()))
    return origin, cap_owned


def mesh_max_cell_list_view[State](
    params_lens: Lens[State, UniversalNeighborlistParameters],
    cutoffs: Table[SystemId, Array],
) -> View[State, NeighborList[Literal[2]]]:
    """A ``CellListNeighborList`` view with resizable, mesh-max'd capacities.

    ``CellListNeighborList.new`` wires ``LensCapacity`` resizing from the
    state's neighbor-list parameters; wrapping each capacity in
    ``MeshMaxCapacity`` makes the recorded requirements device-invariant so
    they leave the shard-mapped step through the assertion interpreter.
    """

    def view(state: State) -> NeighborList[Literal[2]]:
        nl = CellListNeighborList.new(state, params_lens, cutoffs)
        return CellListNeighborList(
            avg_candidates=MeshMaxCapacity(nl.avg_candidates),
            avg_edges=MeshMaxCapacity(nl.avg_edges),
            cells=MeshMaxCapacity(nl.cells),
            avg_image_candidates=MeshMaxCapacity(nl.avg_image_candidates),
            cutoffs=nl.cutoffs,
        )

    return view


def make_sharded_lj_potential[State: IsState[Any, Any]](
    state_lens: Lens[State, State],
    parameters: LennardJonesParameters,
    neighborlist: View[State, NeighborList[Literal[2]]],
    cap_owned: Capacity[int],
    gradient: Lens[Geometry, PositionsAndCell],
) -> Potential[State, PositionsAndCell, EmptyType, Any]:
    """The DD Lennard-Jones potential: owned-incident shard + energy ``psum``.

    Hand-assembled (rather than ``make_lennard_jones_from_state``, which
    hardwires the whole-graph constructor): only the graph constructor and the
    ``ShardedPotential`` wrapper differ from the stock LJ potential.
    """
    gradient_lens: Any = GRAPH_GEOMETRY.nest(gradient)
    graph_constructor: Any = make_sharded_radius_graph_from_state(
        state_lens, neighborlist, cap_owned
    )
    composer = LocalGraphSumComposer(
        graph_constructor=graph_constructor,
        parameter_view=const_lens(parameters),
    )
    inner = PotentialFromEnergy(
        composer=composer,
        energy_fn=lennard_jones_energy,
        gradient_lens=gradient_lens,
        hessian_lens=EMPTY_LENS,
        hessian_idx_view=EMPTY_LENS,
        cache_lens=None,
        patch_idx_view=position_and_cell_idx_view,
    )
    return ShardedPotential(inner)


@dataclass
class ShardMappedPropagator[State]:
    """Run each propagator step under ``shard_map`` over a replicated state.

    The stock run loops (``run_md``/``run_relax``) then work unchanged: the
    step is a plain ``Propagator``, its runtime assertions are device-invariant
    (``MeshMaxCapacity`` / ``owned_subset``) so the assertion interpreter
    extracts them through the manual region, and capacity fixes resize the
    replicated state as usual.
    """

    propagator: Propagator[State] = field(static=True)
    mesh: jax.sharding.Mesh = field(static=True)

    def __call__(self, key: Array, state: State) -> State:
        return shard_map(
            self.propagator, mesh=self.mesh, in_specs=(_REPL, _REPL), out_specs=_REPL
        )(key, state)
