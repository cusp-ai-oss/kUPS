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

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.cell import Cell, OrthogonalFrame, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.types import CandidateBatch, PipelineContext
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass


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
    cell: Cell, cutoffs: Array
) -> tuple[Table[SystemId, SampleSystems], Table[SystemId, Array]]:
    """Create a systems table from a ``Cell``, alongside the cutoff table.

    Returns:
        A tuple of ``(systems, cutoffs)`` tables keyed by ``SystemId``.
    """
    n = len(cutoffs)
    sys_keys = tuple(SystemId(i) for i in range(n))
    indexed_systems = Table(sys_keys, SampleSystems(cell=cell))
    indexed_cutoffs = Table(sys_keys, cutoffs)
    return indexed_systems, indexed_cutoffs


def systems_from_lvecs(
    lvecs: Array, cutoffs: Array
) -> tuple[Table[SystemId, SampleSystems], Table[SystemId, Array]]:
    """Create a systems table from raw lattice vectors, alongside cutoffs."""
    n = len(cutoffs)
    lv = jnp.asarray(lvecs)
    if lv.shape[0] == 1 and n > 1:
        lv = jnp.repeat(lv, n, axis=0)
    cell = PeriodicCell(TriclinicFrame.from_matrix(lv))
    return make_systems(cell, cutoffs)


def cutoff_table(cutoffs: Array) -> Table[SystemId, Array]:
    """Create a cutoff table with canonical ``SystemId`` keys."""
    return Table(tuple(SystemId(i) for i in range(len(cutoffs))), cutoffs)


def make_rh(
    lh: Table[ParticleId, SamplePoints],
    update_positions: Array,
    update_batch_mask: Array,
    for_indices: Array,
    exclusion_ids: Array | None = None,
) -> tuple[Table[ParticleId, SamplePoints], Index[ParticleId]]:
    """Create proposed particle data and the affected ``lh`` ids for testing."""
    n_rh = len(update_positions)
    n_sys = int(jnp.max(update_batch_mask)) + 1
    sys_keys = tuple(range(n_sys))
    rh_pi_keys = tuple(ParticleId(i) for i in range(n_rh))
    if exclusion_ids is None:
        exclusion_ids = for_indices
    rh_points = SamplePoints(
        positions=update_positions,
        system=Index(sys_keys, update_batch_mask.astype(int)),
        inclusion=Index(sys_keys, update_batch_mask.astype(int)),
        exclusion=Index.integer(exclusion_ids.astype(int)),
    )
    rh_indexed = Table(rh_pi_keys, rh_points)
    for_indices_idx = Index(lh.keys, for_indices.astype(int))
    return rh_indexed, for_indices_idx


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
    for_indices: Array | Index[ParticleId] | None = None,
) -> PipelineContext:
    """Build a ``PipelineContext`` directly for unit-level mask/compactor tests.

    Positions are taken as-is — the caller is responsible for the fractional
    convention. Using a unit cell (``eye(3)``) keeps real == fractional so
    ``DistanceCutoffMask`` produces the same numbers either way.
    """
    if for_indices is not None and not isinstance(for_indices, Index):
        for_indices = Index(lh.keys, for_indices)
    assert rh is None or for_indices is None
    if cell is None:
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None]))
    systems, _ = make_systems(cell, jnp.array([1.0]))
    for_indices_raw = (
        for_indices.indices_in(lh.keys) if for_indices is not None else None
    )
    return PipelineContext(lh=lh, rh=rh, systems=systems, for_indices=for_indices_raw)


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


def valid_edge_set(edges: Edges, n_particles: int) -> set[tuple[int, int]]:
    """Set of ``(i, j)`` edges with both endpoints below ``n_particles`` (non-padding)."""
    raw = edges.indices.indices
    in_range = (raw[:, 0] < n_particles) & (raw[:, 1] < n_particles)
    return {(int(a), int(b)) for a, b in np.asarray(raw[in_range])}


@dataclass
class EvalState:
    """Minimal state satisfying ``IsNeighborListState`` /
    ``IsAdaptiveCutoffNeighborListState``: exposes particles, systems, and
    neighbor-list capacity hints so ``from_state`` and the adaptive factory work."""

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
            OrthogonalFrame(jnp.tile(jnp.array([10.0, 10.0, 10.0]), (n_systems, 1)))
        ),
        jnp.full((n_systems,), 2.0),
    )
    params = UniversalNeighborlistParameters(
        avg_edges=8, avg_candidates=8, avg_image_candidates=8, cells=32
    )
    return EvalState(particles=lh, systems=systems, neighborlist_params=params)


def call_with_retry(nl, lh, systems, for_indices=None):
    """Run a neighbor list, growing its capacities until no assertion fails."""
    from kups.core.result import as_result_function

    while (
        result := jax.jit(as_result_function(nl))(
            lh=lh, systems=systems, for_indices=for_indices
        )
    ).failed_assertions:
        nl = result.fix_or_raise(nl)
    result.raise_assertion()
    return result.value
