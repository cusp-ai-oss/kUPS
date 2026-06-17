# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Concrete relaxation filters as gradient lenses over the shared ``Geometry``.

A relaxation filter answers "which degrees of freedom does the optimizer see?".
Each is a ``Lens[Geometry, PositionsAndCell]`` selecting the optimizer DOFs from
the shared geometric view; compose a carrier adapter (e.g. ``GRAPH_GEOMETRY``)
with a filter via ``NestedLens`` to obtain a potential ``gradient_lens``.

The geometric-view types live in [kups.potential.common.geometry][].
"""

import jax
import jax.numpy as jnp

from kups.core.lens import LambdaLens, Lens, lens
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndSystemIndex,
)

POSITIONS_AND_CELL: Lens[Geometry, PositionsAndCell] = lens(
    lambda g: PositionsAndCell(g.particles.map_data(lambda p: p.positions), g.systems)
)
"""Positions + cell DOFs with atoms pinned (today's default cell gradient)."""


def _positions_only_set(g: Geometry, value: PositionsAndCell) -> Geometry:
    particles = g.particles.map_data(
        lambda p: PositionsAndSystemIndex(value.positions.data, p.system)
    )
    cell = jax.lax.stop_gradient(value.cell.data)
    return Geometry(particles, g.systems.set_data(cell))


POSITIONS_ONLY: Lens[Geometry, PositionsAndCell] = LambdaLens(
    lambda g: PositionsAndCell(g.particles.map_data(lambda p: p.positions), g.systems),
    _positions_only_set,
)
"""Positions-only DOFs (``optimize_cell=False``): the cell rides along in the
``PositionsAndCell`` codomain but is stop-gradiented on set, so ``∂E/∂cell`` is zero
while the DOF pytree matches the cell filters."""


def _frechet_filter_get(g: Geometry) -> PositionsAndCell:
    idx = g.particles.data.system.indices_in(g.systems.keys)
    cell = g.systems.data
    reference = cell.frame.reference_vectors
    s = jnp.einsum("ni,nij->nj", g.particles.data.positions, cell.inverse_vectors[idx])
    q = jnp.einsum("ni,nij->nj", s, reference[idx])
    return PositionsAndCell(g.particles.set_data(q), g.systems)


def _frechet_filter_set(g: Geometry, value: PositionsAndCell) -> Geometry:
    idx = g.particles.data.system.indices
    cell = value.cell.data
    reference_inv = jax.lax.stop_gradient(jnp.linalg.inv(cell.frame.reference_vectors))
    m = jnp.array(cell.periodic)[:, None]
    h = cell.vectors
    h_eff = m * h + (1 - m) * jax.lax.stop_gradient(h)
    s = jnp.einsum("ni,nij->nj", value.positions.data, reference_inv[idx])
    positions = jnp.einsum("ni,nij->nj", s, h_eff[idx])
    particles = g.particles.map_data(
        lambda p: PositionsAndSystemIndex(positions, p.system)
    )
    return Geometry(particles, g.systems.set_data(cell))


FRECHET_FILTER: Lens[Geometry, PositionsAndCell] = LambdaLens(
    _frechet_filter_get, _frechet_filter_set
)
"""Atoms-ride-the-cell filter; the frame chooses the conditioning.

DOFs are ``(q, cell)`` with ``q = (r @ h^-1) @ R`` for the cell's fixed reference
basis ``R = frame.reference_vectors``. On set, ``r = (q @ R^-1) @ h`` so atoms ride
the cell at fixed fractional coordinates (the virial coupling falls out of autodiff);
the per-axis periodicity mask ``h_eff = m*h + (1-m)*stop_gradient(h)`` keeps
non-periodic axes out of the riding derivative without disturbing the primal.

The reference is the frame's: a [DeformedFrame][kups.core.cell.DeformedFrame] returns
its ``base``, so ``q`` is reference-cartesian (ASE ``FrechetCellFilter`` conditioning);
any other frame returns the identity, so ``q`` is fractional (ASE ``UnitCellFilter``
conditioning).
"""
