# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared builders for the neighbor list test suite.

Single source of truth for the concrete ``NeighborListPoints`` /
``NeighborListSystems`` tables, ``Edges``, ``PipelineContext``, and
``CandidateBatch`` values the unit and integration tests construct. Positions
are taken as-is — callers are responsible for the fractional convention where
it matters.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core import units
from kups.core.capacity import Capacity
from kups.core.cell import Cell, OrthogonalFrame, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist.common import Candidates, replicate_for_images
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.types import (
    CandidateBatch,
    FractionalPoints,
    NeighborList,
    NeighborListSystems,
    PipelineContext,
)
from kups.core.result import Result, as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass
from test.units import A, as_length


@dataclass
class SamplePoints:
    """Concrete ``NeighborListPoints`` for testing."""

    positions: Array
    system: Index
    inclusion: Index
    exclusion: Index


@dataclass
class SampleSystems:
    """Concrete ``NeighborListSystems`` for testing."""

    cell: Cell


def make_lh(
    positions: Array,
    batch_mask: Array,
    exclusion_ids: Array | None = None,
    *,
    inclusion_max_count: int | None = None,
) -> Table[ParticleId, SamplePoints]:
    """Create the self-graph/output particle table from positions and a batch mask.

    Args:
        positions: ``(n, 3)`` particle positions.
        batch_mask: ``(n,)`` per-particle system id (also used as the inclusion
            segment id).
        exclusion_ids: Optional per-particle exclusion segment ids. Defaults to
            ``arange(n)`` so every particle is in its own exclusion segment.
        inclusion_max_count: When set, builds the inclusion ``Index`` with this
            ``max_count`` (required by ``all_connected_neighborlist``).
    """
    n = len(positions)
    n_sys = int(jnp.max(batch_mask)) + 1 if n > 0 else 1
    sys_keys = tuple(range(n_sys))
    pi_keys = tuple(ParticleId(i) for i in range(n))
    if exclusion_ids is None:
        exclusion_ids = jnp.arange(n)
    inclusion = Index(sys_keys, batch_mask.astype(int), max_count=inclusion_max_count)
    return Table(
        pi_keys,
        SamplePoints(
            positions=positions,
            system=Index(sys_keys, batch_mask.astype(int)),
            inclusion=inclusion,
            exclusion=Index.integer(exclusion_ids.astype(int)),
        ),
    )


def make_systems(
    cell: Cell, cutoffs: Array | units.Length
) -> tuple[Table[SystemId, SampleSystems], Table[SystemId, units.Length]]:
    """Create a systems table from a ``Cell``, alongside the cutoff table.

    Args:
        cell: The cell every system shares.
        cutoffs: One cutoff per system, in Angstrom if given as magnitudes.

    Returns:
        A tuple of ``(systems, cutoffs)`` tables keyed by ``SystemId``.
    """
    cuts = as_length(cutoffs)
    sys_keys = tuple(SystemId(i) for i in range(len(cuts)))
    indexed_systems = Table(sys_keys, SampleSystems(cell=cell))
    indexed_cutoffs = Table(sys_keys, cuts)
    return indexed_systems, indexed_cutoffs


def systems_from_lvecs(
    lvecs: Array | units.Length, cutoffs: Array | units.Length
) -> tuple[Table[SystemId, SampleSystems], Table[SystemId, units.Length]]:
    """Create a systems table from raw lattice vectors, alongside cutoffs.

    Args:
        lvecs: Lattice vectors, in Angstrom if given as magnitudes.
        cutoffs: One cutoff per system, in Angstrom if given as magnitudes.

    Returns:
        A tuple of ``(systems, cutoffs)`` tables keyed by ``SystemId``.
    """
    n = len(cutoffs)
    lv = as_length(lvecs)
    if lv.shape[0] == 1 and n > 1:
        lv = units.apply(jnp.repeat, lv, n, axis=0)
    cell = PeriodicCell(TriclinicFrame.from_matrix(lv))
    return make_systems(cell, cutoffs)


def cutoff_table(cutoffs: Array | units.Length) -> Table[SystemId, units.Length]:
    """Create a cutoff table with canonical ``SystemId`` keys.

    Args:
        cutoffs: One cutoff per system, in Angstrom if given as magnitudes.

    Returns:
        The cutoffs as a table keyed by ``SystemId``.
    """
    cuts = as_length(cutoffs)
    return Table(tuple(SystemId(i) for i in range(len(cuts))), cuts)


def make_rh(
    lh: Table[ParticleId, SamplePoints],
    update_positions: Array,
    update_batch_mask: Array,
    queried_keys: Array,
    exclusion_ids: Array | None = None,
) -> tuple[Table[ParticleId, SamplePoints], Index[ParticleId]]:
    """Create proposed particle data and the affected ``lh`` ids for testing."""
    n_rh = len(update_positions)
    n_sys = int(jnp.max(update_batch_mask)) + 1
    sys_keys = tuple(range(n_sys))
    rh_pi_keys = tuple(ParticleId(i) for i in range(n_rh))
    if exclusion_ids is None:
        exclusion_ids = queried_keys
    rh_points = SamplePoints(
        positions=update_positions,
        system=Index(sys_keys, update_batch_mask.astype(int)),
        inclusion=Index(sys_keys, update_batch_mask.astype(int)),
        exclusion=Index.integer(exclusion_ids.astype(int)),
    )
    rh_indexed = Table(rh_pi_keys, rh_points)
    queried_keys_idx = Index(lh.keys, queried_keys.astype(int))
    return rh_indexed, queried_keys_idx


def after[D](
    lh: Table[ParticleId, Any], rh: Table[ParticleId, D]
) -> Table[ParticleId, D]:
    """``rh`` with particle ids following those of ``lh``, for bipartite queries."""
    return Table(tuple(ParticleId(lh.size + i) for i in range(rh.size)), rh.data)


def make_edges(
    lh_indices: Array,
    rh_indices: Array,
    n_particles: int | None = None,
    shifts: Array | None = None,
) -> Edges:
    """Create ``Edges`` with an ``Index`` first column for testing."""
    raw = jnp.stack([lh_indices, rh_indices], axis=-1)
    if n_particles is None:
        n_particles = int(max(lh_indices.max(), rh_indices.max())) + 1
    if shifts is None:
        shifts = jnp.zeros((len(raw), 1, 3), dtype=int)
    else:
        shifts = shifts.reshape(len(raw), 1, 3)
    return Edges(Index(tuple(ParticleId(i) for i in range(n_particles)), raw), shifts)


def make_pipeline_ctx(
    lh: Table[ParticleId, SamplePoints],
    rh: Table[ParticleId, SamplePoints] | None = None,
    cell: Cell | None = None,
    queried_keys: Array | Index[ParticleId] | None = None,
) -> PipelineContext:
    """Build a ``PipelineContext`` directly for unit-level mask/compactor tests.

    Positions are taken as-is — the caller is responsible for the fractional
    convention. Using a unit cell (``eye(3)``) keeps real == fractional so
    ``DistanceCutoffMask`` produces the same numbers either way.
    """
    if queried_keys is not None and not isinstance(queried_keys, Index):
        queried_keys = Index(lh.keys, queried_keys)
    assert rh is None or queried_keys is None
    if cell is None:
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * A))
    systems, _ = make_systems(cell, jnp.array([1.0]))
    if queried_keys is not None:
        queried_keys = queried_keys.update_labels(lh.keys)
    return PipelineContext(
        keys=lh, queries=rh, systems=systems, queried_keys=queried_keys
    )


def make_batch(
    lh_keys: tuple[ParticleId, ...],
    lh_idx: Array,
    rh_idx: Array,
    shifts: Array | None = None,
    is_minimum_image: Array | None = None,
) -> CandidateBatch:
    """Build a ``CandidateBatch`` with one ``Index`` key set for both sides."""
    n = lh_idx.shape[0]
    if shifts is None:
        shifts = jnp.zeros((n, 1, 3))
    if is_minimum_image is None:
        is_minimum_image = jnp.ones((n,), dtype=bool)
    indices_2d = jnp.stack([lh_idx, rh_idx], axis=-1)
    return CandidateBatch(
        edges=Edges(Index(lh_keys, indices_2d), shifts),
        is_minimum_image=is_minimum_image,
    )


def valid_ids(index: Index[ParticleId]) -> np.ndarray:
    """Integer ids of the rows of ``index`` whose entries are all in bounds."""
    valid = np.asarray(index.valid_mask).reshape(len(index), -1).all(axis=-1)
    return np.asarray(index[valid]).astype(int).reshape(-1, *index.shape[1:])


def valid_pairs(
    key_idx: Index[ParticleId], query_idx: Index[ParticleId]
) -> list[tuple[int, int]]:
    """Integer id pairs of the candidates whose two sides are in bounds."""
    return [
        tuple(p)
        for p in valid_ids(Index.concatenate(key_idx, query_idx).reshape(2, -1).T)
    ]


def valid_edge_set(edges: Edges) -> set[tuple[int, ...]]:
    """Set of id tuples of the non-padding edges."""
    return set(map(tuple, valid_ids(edges.indices).tolist()))


@dataclass
class EvalState:
    """Minimal state satisfying ``IsNeighborListState``: exposes particles,
    systems, and neighbor-list capacity hints so ``from_state`` works."""

    particles: Table
    systems: Table
    neighborlist_params: UniversalNeighborlistParameters


def make_adaptive_state(n_particles: int, n_systems: int) -> EvalState:
    """Build an ``EvalState`` with ``n_particles`` split across ``n_systems``."""
    positions = jnp.zeros((n_particles, 3))
    per_sys = max(1, n_particles // n_systems)
    batch_mask = jnp.minimum(
        jnp.arange(n_particles) // per_sys, jnp.array(n_systems - 1)
    )
    lh = make_lh(positions, batch_mask)
    systems, _ = make_systems(
        PeriodicCell(
            OrthogonalFrame(jnp.tile(jnp.array([10.0, 10.0, 10.0]), (n_systems, 1)) * A)
        ),
        jnp.full((n_systems,), 2.0),
    )
    params = UniversalNeighborlistParameters(
        avg_edges=8, avg_candidates=8, avg_image_candidates=8, cells=32
    )
    return EvalState(particles=lh, systems=systems, neighborlist_params=params)


def pad(
    lh: Table[ParticleId, SamplePoints],
    systems: Table[SystemId, SampleSystems],
    cutoffs: Table[SystemId, units.Length],
    queried_keys: Index[ParticleId] | None = None,
    *,
    n_particles: int,
    n_systems: int,
    n_queried: int = 0,
) -> tuple[
    Table[ParticleId, SamplePoints],
    Table[SystemId, SampleSystems],
    Table[SystemId, units.Length],
    Index[ParticleId] | None,
]:
    """Pad to ``n_particles`` particles, ``n_systems`` systems, ``n_queried`` queried keys.

    Padding particles belong to the out-of-range system and inclusion segment and
    padding queried keys are out of range, so both are inactive; padding systems
    repeat the last one and hold no particles. The edges between real particles
    are unchanged.
    """
    labels = tuple(range(n_systems))
    # Rows past ``lh`` gather out of bounds: zero positions, out-of-range indices.
    rows = lh[Index(lh.keys, jnp.minimum(jnp.arange(n_particles), lh.size))]
    points = SamplePoints(
        positions=rows.positions,
        system=rows.system.update_labels(labels),
        inclusion=rows.inclusion.update_labels(labels),
        exclusion=rows.exclusion,
    )
    padded = Table(tuple(ParticleId(i) for i in range(n_particles)), points)
    if queried_keys is not None:
        n_missing = n_queried - len(queried_keys)
        assert n_missing >= 0
        queried_keys = Index.concatenate(
            queried_keys.update_labels(padded.keys),
            Index(padded.keys, jnp.full(n_missing, n_particles)),
        )
    system_keys = tuple(SystemId(i) for i in range(n_systems))

    def repeat_last(x: Array) -> Array:
        return jnp.concatenate([x, jnp.repeat(x[-1:], n_systems - len(x), axis=0)])

    return (
        padded,
        Table(system_keys, jax.tree.map(repeat_last, systems.data)),
        Table(system_keys, jax.tree.map(repeat_last, cutoffs.data)),
        queried_keys,
    )


@jax.jit
def _run_nl(
    nl: NeighborList[Literal[2]],
    keys: Table[ParticleId, SamplePoints],
    systems: Table[SystemId, SampleSystems],
    queried_keys: Index[ParticleId] | None,
) -> Result[Any, Edges[Literal[2]]]:
    return as_result_function(nl)(keys=keys, systems=systems, queried_keys=queried_keys)


def run_nl(
    nl: NeighborList[Literal[2]],
    keys: Table[ParticleId, SamplePoints],
    systems: Table[SystemId, SampleSystems],
    queried_keys: Index[ParticleId] | None = None,
) -> Result[Any, Edges[Literal[2]]]:
    """Run a neighbor list under jit, with its cutoffs as runtime inputs.

    Calls with equal capacities, shapes and labels share one compiled program;
    passing every argument positionally keeps them on one cache entry.
    """
    return _run_nl(nl, keys, systems, queried_keys)


@jax.jit
def _replicate(
    candidates: Candidates,
    keys: Table[ParticleId, FractionalPoints],
    queries: Table[ParticleId, FractionalPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, units.Length],
    image_candidates: Capacity[int] | None,
) -> Result[Any, CandidateBatch[Literal[2]]]:
    return as_result_function(replicate_for_images)(
        candidates, keys, queries, systems, cutoffs, image_candidates
    )


def replicate(
    candidates: Candidates,
    keys: Table[ParticleId, FractionalPoints],
    queries: Table[ParticleId, FractionalPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, units.Length],
    image_candidates: Capacity[int] | None,
) -> CandidateBatch[Literal[2]]:
    """``replicate_for_images`` under jit, raising its runtime assertions.

    Cutoffs and positions are runtime inputs and the image capacity is static
    (pytree metadata), so calls with equal shapes and capacity share one
    compiled program.
    """
    result = _replicate(candidates, keys, queries, systems, cutoffs, image_candidates)
    result.raise_assertion()
    return result.value


def call_with_retry(nl, lh, systems, queried_keys=None):
    """Run a neighbor list, growing its capacities until no assertion fails."""
    while (result := run_nl(nl, lh, systems, queried_keys)).failed_assertions:
        nl = result.fix_or_raise(nl)
    result.raise_assertion()
    return result.value
