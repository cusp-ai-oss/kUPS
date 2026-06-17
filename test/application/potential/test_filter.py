"""Tests for relaxation filters (``src/kups/potential/filter.py``)."""

from collections.abc import Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from kups.application.potential.filter import (
    FRECHET_FILTER,
    POSITIONS_AND_CELL,
    POSITIONS_ONLY,
)
from kups.core.cell import (
    Cell,
    DeformedFrame,
    LogTriclinicFrame,
    PeriodicCell,
    TriclinicFrame,
)
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import Lens
from kups.core.typing import ParticleId, SystemId
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndSystemIndex,
)

jax.config.update("jax_enable_x64", True)


def _frechet_cell(
    h: Array,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    cell_factor: float | Array = 1.0,
) -> Cell[Any]:
    """Build a batched Cell with DeformedFrame from a stack of basis matrices."""
    base = TriclinicFrame.from_matrix(h)
    frame = DeformedFrame.from_frame(base, cell_factor=cell_factor)
    if periodic == (True, True, True):
        return PeriodicCell(frame)
    return Cell(frame, periodic=periodic)


def _geometry(positions: Array, system_ids: Sequence[int], cell: Cell[Any]) -> Geometry:
    """Assemble a Table-based Geometry from raw positions, ids and a batched cell."""
    sys_index = Index.new([SystemId(s) for s in system_ids])
    particles = Table.arange(
        PositionsAndSystemIndex(positions, sys_index), label=ParticleId
    )
    systems = Table(tuple(SystemId(k) for k in sorted(set(system_ids))), cell)
    return Geometry(particles, systems)


def _single() -> Geometry:
    h = jnp.array([[[3.8, 0.0, 0.0], [0.4, 3.7, 0.0], [0.2, 0.3, 3.9]]])
    r = jnp.array([[0.1, 0.2, 0.3], [2.0, 1.0, 0.5], [1.0, 2.5, 3.0]])
    return _geometry(r, [0, 0, 0], _frechet_cell(h, cell_factor=3.0))


def _multi() -> Geometry:
    h = jnp.array(
        [
            [[3.8, 0.0, 0.0], [0.4, 3.7, 0.0], [0.2, 0.3, 3.9]],
            [[4.1, 0.0, 0.0], [0.1, 4.0, 0.0], [0.0, 0.2, 4.2]],
        ]
    )
    r = jnp.array([[0.1, 0.2, 0.3], [2.0, 1.0, 0.5], [1.0, 2.5, 3.0], [0.5, 0.5, 1.5]])
    return _geometry(
        r, [0, 0, 1, 1], _frechet_cell(h, cell_factor=jnp.array([3.0, 2.0]))
    )


def _slab() -> Geometry:
    h = jnp.array([[[5.3, 0.0, 0.0], [0.0, 5.3, 0.0], [0.0, 0.0, 12.0]]])
    r = jnp.array([[0.5, 0.5, 4.0], [2.0, 1.0, 5.0], [3.0, 3.5, 6.0]])
    cell = _frechet_cell(h, periodic=(True, True, False), cell_factor=3.0)
    return _geometry(r, [0, 0, 0], cell)


_GEOMETRIES = {"single": _single, "multi": _multi, "slab": _slab}


@pytest.mark.parametrize("name", _GEOMETRIES)
@pytest.mark.parametrize("filt", [POSITIONS_AND_CELL, POSITIONS_ONLY, FRECHET_FILTER])
def test_lens_law_roundtrip(name: str, filt: Lens[Geometry, Any]):
    """get/set round-trips: set(g, get(g)) leaves the Geometry unchanged."""
    g = _GEOMETRIES[name]()
    g2 = filt.set(g, filt.get(g))
    np.testing.assert_allclose(
        g2.particles.data.positions, g.particles.data.positions, atol=1e-12
    )
    np.testing.assert_allclose(
        g2.systems.data.vectors, g.systems.data.vectors, atol=1e-12
    )


def test_frechet_get_is_inverse_of_set():
    """get(set(g, u)) == u for CELL_FILTER DOFs (lens law on the codomain)."""
    g = _multi()
    filt = FRECHET_FILTER
    u = PositionsAndCell(
        g.particles.set_data(g.particles.data.positions + 0.3), g.systems
    )
    u_back = filt.get(filt.set(g, u))
    np.testing.assert_allclose(u_back.positions.data, u.positions.data, atol=1e-12)


def _toy_energy(positions: Array, cell: Cell[Any]) -> Array:
    """Toy energy reading positions and the cell (mirrors the notebook)."""
    g_r = jnp.arange(positions.size, dtype=positions.dtype).reshape(positions.shape)
    w = jnp.arange(cell.vectors.size, dtype=positions.dtype).reshape(cell.vectors.shape)
    return jnp.sum(0.1 * g_r * positions) + jnp.sum(0.05 * w * cell.vectors)


def _filter_gradient(filt: Lens[Geometry, Any], g: Geometry) -> tuple[Array, Any]:
    """dE/du via vjp of the energy through the filter's set (mirrors energy.py)."""
    u0 = filt.get(g)

    def e_of_u(u: Any) -> Array:
        gg = filt.set(g, u)
        return _toy_energy(gg.particles.data.positions, gg.systems.data)

    e, vjp = jax.vjp(e_of_u, u0)
    (grad,) = vjp(jnp.ones_like(e))
    return e, grad


def test_frechet_total_gradient_matches_jax_grad():
    """CELL_FILTER cell gradient == jax.grad of E(s0 @ h(A), h(A)) at fixed s0."""
    g = _single()
    cell = g.systems.data
    frame = cast(DeformedFrame, cell.frame)
    deformation = cast(LogTriclinicFrame, frame.deformation)
    r = g.particles.data.positions
    s0 = r @ jnp.linalg.inv(cell.vectors[0])

    _, grad = _filter_gradient(FRECHET_FILTER, g)

    def e_total(tril: Array) -> Array:
        f = DeformedFrame(
            frame.base, LogTriclinicFrame(tril[None], deformation.cell_factor)
        )
        h = f.vectors[0]
        return _toy_energy(s0 @ h, PeriodicCell(TriclinicFrame.from_matrix(h[None])))

    truth = jax.grad(e_total)(deformation.tril[0])
    grad_frame = cast(DeformedFrame, grad.cell.data.frame)
    np.testing.assert_allclose(
        cast(LogTriclinicFrame, grad_frame.deformation).tril[0], truth, atol=1e-10
    )


def test_frechet_dE_dq_equals_force_at_A_zero():
    """At A=0, dE/dq equals the physical force dE/dr (q == r)."""
    g = _single()
    _, grad_fr = _filter_gradient(FRECHET_FILTER, g)
    _, grad_id = _filter_gradient(POSITIONS_AND_CELL, g)
    np.testing.assert_allclose(
        grad_fr.positions.data, grad_id.positions.data, atol=1e-10
    )


def test_slab_mask_leaves_primal_untouched():
    """The periodicity stop_gradient splice does not move primal positions."""
    g = _slab()
    filt = FRECHET_FILTER
    g2 = filt.set(g, filt.get(g))
    np.testing.assert_allclose(
        g2.particles.data.positions, g.particles.data.positions, atol=1e-12
    )
