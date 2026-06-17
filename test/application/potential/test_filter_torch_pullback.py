"""Tests for the direct-MLIAP filter pullback (``mliap.direct.filter_pullback``).

Exercises ``filter_pullback`` on fabricated geometry and hand-made physical
gradients (no real torch model). The pullback maps a model's direct
``(∂E/∂r, ∂E/∂h|_r)`` outputs to the gradient of the optimizer's filter DOFs by
pulling the cotangent back through ``filter.set``. The model's partial cell
gradient is carried in a frame's parameter leaves, so the physical ``∂E/∂h`` here
is lower-triangular (the 6-DOF triclinic subspace the frame can represent).
"""

from collections.abc import Sequence
from typing import Any, NamedTuple

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
from kups.core.typing import ParticleId, SystemId
from kups.potential.common.geometry import (
    Geometry,
    PositionsAndCell,
    PositionsAndSystemIndex,
)
from kups.potential.common.graph import GRAPH_GEOMETRY
from kups.potential.mliap.direct import filter_pullback

jax.config.update("jax_enable_x64", True)


class _SysData(NamedTuple):
    """System data carrying a cell, as ``GRAPH_GEOMETRY`` expects."""

    cell: Cell[Any]


class _PointCloud(NamedTuple):
    """Minimal graph: particle/system tables ``GRAPH_GEOMETRY`` reads."""

    particles: Table[ParticleId, PositionsAndSystemIndex]
    systems: Table[SystemId, _SysData]


class _Input(NamedTuple):
    """Fabricated direct-MLIAP input exposing only ``.graph``."""

    graph: _PointCloud


def _frechet_cell(
    h: Array,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    cf: Array | float = 1.0,
) -> Cell[Any]:
    """Build a batched DeformedFrame-backed Cell from stacked basis matrices."""
    frame = DeformedFrame.from_frame(TriclinicFrame.from_matrix(h), cell_factor=cf)
    if periodic == (True, True, True):
        return PeriodicCell(frame)
    return Cell(frame, periodic=periodic)


def _make_input(positions: Array, system_ids: Sequence[int], cell: Cell[Any]) -> _Input:
    """Assemble a fabricated graph input from positions, ids and a batched cell."""
    sys_index = Index.new([SystemId(s) for s in system_ids])
    particles = Table.arange(
        PositionsAndSystemIndex(positions, sys_index), label=ParticleId
    )
    systems = Table(tuple(SystemId(k) for k in sorted(set(system_ids))), _SysData(cell))
    return _Input(_PointCloud(particles, systems))


def _direct_gradients(
    cell: Cell[Any], system_keys: tuple[SystemId, ...], g_r: Array, dE_dh: Array
) -> PositionsAndCell:
    """Pack physical ``(∂E/∂r, ∂E/∂h|_r)`` as direct-model ``PositionsAndCell``.

    The partial cell gradient ``dE_dh`` (lower-triangular) is stored in a
    gradient frame's parameter leaves via ``parameter_gradient``, matching how a
    direct model emits its cell cotangent.
    """
    grad_frame = cell.frame.parameter_gradient(dE_dh)
    grad_cell: Cell[Any] = (
        PeriodicCell(grad_frame)
        if cell.periodic == (True, True, True)
        else Cell(grad_frame, periodic=cell.periodic)
    )
    return PositionsAndCell(
        Table.arange(g_r, label=ParticleId),
        Table(system_keys, grad_cell),
    )


def _physical_gradients(geom: Geometry) -> tuple[Array, Array]:
    """Hand-made ``(∂E/∂r, ∂E/∂h|_r)`` with lower-triangular cell block."""
    r = geom.particles.data.positions
    h = geom.systems.data.vectors
    g_r = 0.1 * jnp.arange(r.size, dtype=r.dtype).reshape(r.shape)
    dE_dh = 0.05 * jnp.arange(h.size, dtype=h.dtype).reshape(h.shape)
    return g_r, dE_dh * jnp.tril(jnp.ones((3, 3)))


def _single() -> tuple[_Input, Array, Array]:
    h = jnp.array([[[3.8, 0.0, 0.0], [0.4, 3.7, 0.0], [0.2, 0.3, 3.9]]])
    r = jnp.array([[0.1, 0.2, 0.3], [2.0, 1.0, 0.5], [1.0, 2.5, 3.0]])
    inp = _make_input(r, [0, 0, 0], _frechet_cell(h, cf=3.0))
    g_r, dE_dh = _physical_gradients(GRAPH_GEOMETRY.get(inp))
    return inp, g_r, dE_dh


def _multi() -> tuple[_Input, Array, Array]:
    h = jnp.array(
        [
            [[3.8, 0.0, 0.0], [0.4, 3.7, 0.0], [0.2, 0.3, 3.9]],
            [[4.1, 0.0, 0.0], [0.1, 4.0, 0.0], [0.0, 0.2, 4.2]],
        ]
    )
    r = jnp.array([[0.1, 0.2, 0.3], [2.0, 1.0, 0.5], [1.0, 2.5, 3.0], [0.5, 0.5, 1.5]])
    inp = _make_input(r, [0, 0, 1, 1], _frechet_cell(h, cf=jnp.array([3.0, 2.0])))
    g_r, dE_dh = _physical_gradients(GRAPH_GEOMETRY.get(inp))
    return inp, g_r, dE_dh


_CASES = {"single": _single, "multi": _multi}


def _gradients_for(inp: _Input, g_r: Array, dE_dh: Array) -> PositionsAndCell:
    geom = GRAPH_GEOMETRY.get(inp)
    return _direct_gradients(geom.systems.data, inp.graph.systems.keys, g_r, dE_dh)


@pytest.mark.parametrize("name", _CASES)
def test_frechet_pullback_matches_total_lattice_gradient(name: str):
    """CELL_FILTER pullback: cell grad == total lattice grad, pos block == dE/dr.

    The total lattice gradient is ``d/dA E(s0 @ h(A), h(A))`` at fixed fractional
    ``s0`` -- the virial coupling from atoms riding the cell, computed
    independently via ``jax.grad`` of a linear energy with the given partials.
    """
    inp, g_r, dE_dh = _CASES[name]()
    geom = GRAPH_GEOMETRY.get(inp)
    cell = geom.systems.data
    frame = cell.frame
    assert isinstance(frame, DeformedFrame)
    deformation = frame.deformation
    assert isinstance(deformation, LogTriclinicFrame)

    g_u = filter_pullback(
        GRAPH_GEOMETRY.get(inp), _gradients_for(inp, g_r, dE_dh), FRECHET_FILTER
    )
    np.testing.assert_allclose(g_u.positions.data, g_r, atol=1e-9)

    s0 = jnp.einsum(
        "ni,nij->nj",
        geom.particles.data.positions,
        cell.inverse_vectors[geom.particles.data.system.indices],
    )
    idx = geom.particles.data.system.indices

    def e_total(tril: Array) -> Array:
        f = DeformedFrame(frame.base, LogTriclinicFrame(tril, deformation.cell_factor))
        h = f.vectors
        r = jnp.einsum("ni,nij->nj", s0, h[idx])
        return jnp.sum(g_r * r) + jnp.sum(dE_dh * h)

    truth = jax.grad(e_total)(deformation.tril)
    out_frame = g_u.cell.data.frame
    assert isinstance(out_frame, DeformedFrame)
    out_deformation = out_frame.deformation
    assert isinstance(out_deformation, LogTriclinicFrame)
    np.testing.assert_allclose(out_deformation.tril, truth, atol=1e-8)


@pytest.mark.parametrize("name", _CASES)
def test_positions_only_pullback_equals_force(name: str):
    """POSITIONS_ONLY pullback returns dE/dr unchanged (no cell coupling)."""
    inp, g_r, dE_dh = _CASES[name]()
    g_u = filter_pullback(
        GRAPH_GEOMETRY.get(inp), _gradients_for(inp, g_r, dE_dh), POSITIONS_ONLY
    )
    np.testing.assert_allclose(g_u.positions.data, g_r, atol=1e-9)


@pytest.mark.parametrize("name", _CASES)
def test_position_and_cell_pullback_is_partial(name: str):
    """POSITION_AND_CELL pullback: pos block == dE/dr, cell == partial dE/dh|_r.

    With atoms pinned, the cell DOF gradient is the raw partial cell gradient,
    recovered from the output gradient frame via ``vectors_gradient``.
    """
    inp, g_r, dE_dh = _CASES[name]()
    geom = GRAPH_GEOMETRY.get(inp)
    g_u = filter_pullback(
        GRAPH_GEOMETRY.get(inp), _gradients_for(inp, g_r, dE_dh), POSITIONS_AND_CELL
    )
    np.testing.assert_allclose(g_u.positions.data, g_r, atol=1e-9)
    dE_dh_back = geom.systems.data.frame.vectors_gradient(g_u.cell.data.frame)
    np.testing.assert_allclose(dE_dh_back, dE_dh, atol=1e-8)
