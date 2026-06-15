# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Verlet neighbor list with a skin.

Builds a *conservative* neighbor list at ``cutoff + skin`` once, stores its edges
in the MD state, and reuses them for many steps via
:class:`~kups.core.neighborlist.refine.RefineCutoffNeighborList` (which re-masks to
the true cutoff and recomputes minimum-image shifts for the current — possibly
rescaled — cell). The expensive O(N²/K²) dense build then runs only when a
PBC-aware displacement/strain trigger fires, amortizing it over the rebuild window.

A pair absent from the stored skin list had build-time distance ``> cutoff + skin``;
it can only intrude inside the true cutoff if the total inward change exceeds ``skin``.
The two sources of change are atom motion (``≤ 2·δ_max``) and isotropic cell strain
(a pair at the skin radius sees ``(cutoff+skin)·max_a|f_a−1|``), giving the guarantee

    completeness holds  ⇔  2·δ_max + (cutoff+skin)·max_a|f_a − 1|  <  skin

where ``δ_max`` is the max **minimum-image** atom displacement since the last build and
``f_a`` is the per-axis ``perpendicular_lengths`` ratio (current / reference).

Reuse is exact only while ``(cutoff+skin)/perpendicular_length ≤ 0.5`` on every axis
(single-image regime); above that the dense build replicates periodic images that the
refine path collapses to one minimum image. :func:`skin_mic_ratio` exposes this so the
caller can guard (force a rebuild / widen the box) before it is violated.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.capacity import FixedCapacity
from kups.core.data import Table
from kups.core.neighborlist.cell_list import CellListNeighborList
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.neighborlist.edges import Edges
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from kups.core.neighborlist.refine import RefineCutoffNeighborList
from kups.core.neighborlist.types import (
    NearestNeighborList,
    NeighborListPoints,
    NeighborListSystems,
)
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field


def skin_cutoffs(
    cutoffs: Table[SystemId, Array], skin: float
) -> Table[SystemId, Array]:
    """Per-system ``cutoff + skin`` table (the radius the skin list is built at)."""
    return cutoffs.map_data(lambda c: c + skin)


def estimate_skin_params(
    particles_per_system: Table[SystemId, Array],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: float,
) -> UniversalNeighborlistParameters:
    """Capacities sized consistently for the ``cutoff + skin`` sphere.

    Re-estimates *all* capacities at the enlarged radius (never hand-scale a single
    one — mismatched candidate/image buffers trigger a pathological dense-build path).
    """
    return UniversalNeighborlistParameters.estimate(
        particles_per_system, systems, skin_cutoffs(cutoffs, skin)
    )


def dense_skin_nl(
    skin_params: UniversalNeighborlistParameters,
) -> DenseNearestNeighborList:
    """Dense ``O(N²/K²)`` builder for the skin list — fastest when box ~ cutoff+skin."""
    return DenseNearestNeighborList(
        avg_candidates=FixedCapacity(skin_params.avg_candidates),
        avg_edges=FixedCapacity(skin_params.avg_edges),
        avg_image_candidates=FixedCapacity(skin_params.avg_image_candidates),
    )


def cell_skin_nl(
    skin_params: UniversalNeighborlistParameters,
) -> CellListNeighborList:
    """Cell-list ``O(N)`` builder for the skin list — for large boxes (box ≫ cutoff+skin),
    where the dense build is quadratic. Capacities are pinned at the cutoff+skin radius."""
    return CellListNeighborList(
        avg_candidates=FixedCapacity(skin_params.avg_candidates),
        avg_edges=FixedCapacity(skin_params.avg_edges),
        cells=FixedCapacity(skin_params.cells),
        avg_image_candidates=FixedCapacity(skin_params.avg_image_candidates),
    )


def build_skin_edges(
    particles: Table[ParticleId, NeighborListPoints],
    systems: Table[SystemId, NeighborListSystems],
    cutoffs: Table[SystemId, Array],
    skin: float,
    skin_nl: NearestNeighborList,
) -> Edges[Literal[2]]:
    """One build at ``cutoff + skin`` via the *injected* neighbor list (e.g.
    :func:`dense_skin_nl` or :func:`cell_skin_nl`) — used to seed and to rebuild the skin
    list. The reuse path (:func:`skin_neighborlist`) is independent of this choice."""
    return skin_nl(particles, None, systems, skin_cutoffs(cutoffs, skin))


def skin_neighborlist(
    stored_edges: Edges[Literal[2]], avg_edges: int
) -> RefineCutoffNeighborList:
    """The cheap per-step reuse path: refine stored skin edges to the true cutoff.

    ``avg_edges`` is the per-particle true-cutoff edge capacity (e.g.
    ``neighborlist_params.avg_edges``); ``RefineCutoffNeighborList`` multiplies it by
    the particle count internally.
    """
    return RefineCutoffNeighborList(
        candidates=stored_edges, avg_edges=FixedCapacity(avg_edges)
    )


def skin_mic_ratio(
    systems: Table[SystemId, NeighborListSystems], cutoff: float, skin: float
) -> Array:
    """``max_a (cutoff+skin)/perpendicular_length_a`` over axes & systems.

    Reuse is bitwise-exact vs a fresh dense build only while this is ``< 0.5``.
    """
    perp = systems.data.cell.perpendicular_lengths  # (n_sys, 3)
    return jnp.max((cutoff + skin) / perp)


def should_rebuild(
    positions: Array,
    reference_positions: Array,
    system_index: Array,
    perp_now: Array,
    perp_ref: Array,
    cutoff: float,
    skin: float,
) -> Array:
    """PBC-aware Verlet rebuild trigger (scalar bool), reduced over all systems.

    Args:
        positions: current cartesian positions, ``(N, 3)``.
        reference_positions: positions at the last rebuild, ``(N, 3)``.
        system_index: per-particle system index, ``(N,)`` — selects each atom's box.
        perp_now / perp_ref: ``perpendicular_lengths`` now / at last rebuild, ``(n_sys, 3)``.
        cutoff: true cutoff (Å).
        skin: skin width (Å).

    Returns:
        Scalar boolean: ``True`` when the completeness margin is exhausted.
    """
    lp = perp_now[system_index]  # (N, 3) — each atom's box edges
    d = positions - reference_positions
    d_mic = d - lp * jnp.round(d / lp)  # minimum image (atoms wrap the box)
    delta_max = jnp.max(jnp.linalg.norm(d_mic, axis=-1))
    f = perp_now / perp_ref
    cell_term = (cutoff + skin) * jnp.max(jnp.abs(f - 1.0))
    return (2.0 * delta_max + cell_term) >= skin


class IsVerletState(Protocol):
    """State carrying the stored skin list and rebuild references."""

    @property
    def stored_skin_edges(self) -> Edges[Literal[2]]: ...
    @property
    def reference_positions(self) -> Array: ...
    @property
    def reference_cell(self) -> object: ...  # Cell; has .perpendicular_lengths
    @property
    def should_rebuild(self) -> Array: ...


@dataclass
class RebuildSkinStep:
    """Propagator step that rebuilds the stored skin list (a build at cutoff+skin via the
    injected ``skin_nl``) and resets the rebuild references. Run at step start, before the
    force eval.

    State-generic: updates the verlet fields via :func:`dataclasses.replace`, so it works
    on any MD state carrying ``stored_skin_edges``/``reference_positions``/
    ``reference_cell``/``should_rebuild`` plus ``particles``/``systems``. The skin builder
    (dense vs cell) is *injected*, not selected by flag — see :func:`dense_skin_nl` /
    :func:`cell_skin_nl`.
    """

    cutoff: float = field(static=True)
    skin: float = field(static=True)
    skin_nl: NearestNeighborList = field(static=True)

    def __call__(self, key: Array, state):
        del key
        n = len(state.systems.keys)
        cutoffs = Table(state.systems.keys, jnp.full((n,), self.cutoff))
        edges = build_skin_edges(
            state.particles, state.systems, cutoffs, self.skin, self.skin_nl
        )
        return dataclasses.replace(
            state,
            stored_skin_edges=edges,
            reference_positions=state.particles.data.positions + 0.0,  # distinct buffer
            reference_cell=jax.tree.map(lambda x: x + 0.0, state.systems.data.cell),
            should_rebuild=jnp.array(False),
        )


@dataclass
class TriggerStep:
    """Propagator step that computes the PBC-aware rebuild trigger and stores it in
    ``state.should_rebuild`` (read host-side by the run loop to dispatch the next step).
    Cheap: O(N) displacement + per-axis strain, no neighbor build."""

    cutoff: float = field(static=True)
    skin: float = field(static=True)

    def __call__(self, key: Array, state):
        del key
        sr = should_rebuild(
            state.particles.data.positions,
            state.reference_positions,
            state.particles.data.system.indices,
            state.systems.data.cell.perpendicular_lengths,
            state.reference_cell.perpendicular_lengths,
            self.cutoff,
            self.skin,
        )
        return dataclasses.replace(state, should_rebuild=sr)
