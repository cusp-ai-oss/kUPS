# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Concrete relaxation filters as gradient lenses over the shared ``Geometry``.

A relaxation filter answers "which degrees of freedom does the optimizer see?".
Each is a ``Lens[Geometry, PositionsAndCell]`` selecting the optimizer DOFs from
the shared geometric view; compose a carrier adapter (e.g. ``GRAPH_GEOMETRY``)
with a filter via ``NestedLens`` to obtain a potential ``gradient_lens``.

The geometric-view types live in [kups.core.geometry][].
"""

import jax
import jax.numpy as jnp

from kups.core import units
from kups.core.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndCellTree,
    PositionsAndSystemIndex,
)
from kups.core.lens import LambdaLens, Lens, lens

POSITIONS_AND_CELL: Lens[Geometry, PositionsAndCell[units.Length]] = lens(
    lambda g: PositionsAndCellTree(
        g.particles.map_data(lambda p: p.positions), g.systems
    )
)
"""Positions + cell DOFs with atoms pinned (today's default cell gradient)."""


def _frechet_filter_get(g: Geometry) -> PositionsAndCell[units.Length]:
    cell = g.systems.data
    inverse, reference = g.systems.set_data(
        (cell.inverse_vectors, cell.frame.reference_vectors)
    )[g.particles.data.system]
    s = units.einsum("ni,nij->nj", g.particles.data.positions, inverse)
    q = units.einsum("ni,nij->nj", s, reference)
    return PositionsAndCellTree(g.particles.set_data(q), g.systems)


def _frechet_filter_set(g: Geometry, value: PositionsAndCell[units.Length]) -> Geometry:
    cell = value.cell.data
    reference_inv = jax.lax.stop_gradient(units.inv(cell.frame.reference_vectors))
    m = jnp.array(cell.periodic)[:, None]
    h = cell.vectors
    h_eff = m * h + (1 - m) * jax.lax.stop_gradient(h)
    dh = h_eff - g.systems.data.vectors
    reference_inv, h_eff, dh = value.cell.set_data((reference_inv, h_eff, dh))[
        g.particles.data.system
    ]
    # r = (q R^-1) h, written as an increment on g: unchanged DOFs return g's
    # positions exactly, with the same map and derivative at u = get(g).
    q = _frechet_filter_get(g).positions.data
    dq = units.einsum("ni,nij->nj", value.positions.data - q, reference_inv)
    s = units.einsum("ni,nij->nj", q, reference_inv)
    positions = (
        g.particles.data.positions
        + units.einsum("ni,nij->nj", dq, h_eff)
        + units.einsum("ni,nij->nj", s, dh)
    )
    particles = g.particles.map_data(
        lambda p: PositionsAndSystemIndex(positions, p.system)
    )
    return Geometry(particles, g.systems.set_data(cell))


FRECHET_FILTER: Lens[Geometry, PositionsAndCell[units.Length]] = LambdaLens(
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
