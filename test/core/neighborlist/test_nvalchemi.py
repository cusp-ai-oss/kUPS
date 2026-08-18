# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``nvalchemiops``-backed neighbor lists.

``nvalchemiops`` is a CUDA/Warp dependency whose kernels are jit-traced through,
so they cannot be replaced by a NumPy stand-in. Correctness is validated by
running the actual kernels on a GPU and cross-checking against
``DenseNearestNeighborList`` (the ``real_nvalchemi`` tests, skipped without a
CUDA device). The toolkit-free behaviours -- bipartite rejection and the
missing-dependency hint -- run anywhere.
"""

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.lens import identity_lens
from kups.core.neighborlist import (
    DenseNearestNeighborList,
    NvalchemiCellListNeighborList,
    NvalchemiNaiveNeighborList,
)
from kups.core.neighborlist.edges import Edges
from kups.core.result import as_result_function
from kups.core.utils.jax import dataclass, field

from ._builders import make_lh, systems_from_lvecs


def _edge_signature(edges: Edges, n: int) -> set[tuple[int, int, int, int, int]]:
    """Set of ``(i, j, *shift)`` for non-padding edges (shifts rounded to int)."""
    idx = np.asarray(edges.indices.indices)
    shifts = np.asarray(jnp.round(edges.shifts[:, 0, :])).astype(int)
    keep = (idx[:, 0] < n) & (idx[:, 1] < n)
    return {
        (int(a), int(b), int(s[0]), int(s[1]), int(s[2]))
        for a, b, s in zip(idx[keep, 0], idx[keep, 1], shifts[keep])
    }


# Cubic and triclinic (rows = lattice vectors; catches a cell transpose).
_CUBIC = jnp.array([[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]])
_TRICLINIC = jnp.array([[[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 3.0, 8.0]]])


def test_bipartite_queries_rejected():
    """Bipartite ``queries`` raise before any toolkit call (no GPU needed)."""

    n = 6
    lh = make_lh(jnp.zeros((n, 3)), jnp.zeros(n, dtype=int))
    systems, _ = systems_from_lvecs(_CUBIC, jnp.array([3.0]))
    nl = NvalchemiNaiveNeighborList(
        cutoff=3.0,
        max_neighbors=FixedCapacity(8),
        avg_edges=FixedCapacity(64),
        max_shifts=FixedCapacity(8),
    )
    with pytest.raises(NotImplementedError, match="bipartite"):
        nl(lh, systems, queries=lh)


def test_missing_dependency_raises():
    """Without ``nvalchemiops`` installed, calling raises a clear install hint."""
    if importlib.util.find_spec("nvalchemiops") is not None:
        pytest.skip("nvalchemiops is installed; missing-dependency path not exercised")

    n = 4
    lh = make_lh(jnp.zeros((n, 3)), jnp.zeros(n, dtype=int))
    systems, _ = systems_from_lvecs(_CUBIC, jnp.array([3.0]))
    nl = NvalchemiNaiveNeighborList(
        cutoff=3.0,
        max_neighbors=FixedCapacity(8),
        avg_edges=FixedCapacity(64),
        max_shifts=FixedCapacity(8),
    )
    with pytest.raises(ImportError, match="nvalchemiops"):
        nl(lh, systems)


# --- Real-toolkit validation (GPU + nvalchemiops only) -----------------------
# Run the actual CUDA/Warp kernels and cross-check against the dense reference.
# Capacities (max_total_cells / max_shifts) are sized in-trace from the cell
# geometry, so no host-side estimate is passed; a generous FixedCapacity guess
# stands in for the resizable state-backed capacity used in production.

_HAS_REAL_NVALCHEMI = (
    importlib.util.find_spec("nvalchemiops") is not None
    and jax.default_backend() == "gpu"
)
real_nvalchemi = pytest.mark.skipif(
    not _HAS_REAL_NVALCHEMI,
    reason="requires a CUDA device with nvalchemiops installed",
)


@real_nvalchemi
@pytest.mark.parametrize("method", ["naive", "cell_list"])
@pytest.mark.parametrize("lvecs", [_CUBIC, _TRICLINIC], ids=["cubic", "triclinic"])
def test_real_matches_dense_single_system(method, lvecs):

    rng = np.random.default_rng(0)
    n, cutoff = 96, 4.0
    positions = jnp.asarray(rng.uniform(0.0, 12.0, size=(n, 3)))
    lh = make_lh(positions, jnp.zeros(n, dtype=int))
    systems, _ = systems_from_lvecs(lvecs * 1.2, jnp.array([cutoff]))

    dense = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n * n),
        avg_edges=FixedCapacity(n * 128),
        avg_image_candidates=FixedCapacity(n * n),
        cutoff=cutoff,
    )
    if method == "naive":
        nl = NvalchemiNaiveNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            avg_edges=FixedCapacity(n * 128),
            max_shifts=FixedCapacity(64),
        )
    else:
        nl = NvalchemiCellListNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            max_total_cells=FixedCapacity(512),
            avg_edges=FixedCapacity(n * 128),
        )
    expected = _edge_signature(dense(lh, systems), n)
    got = _edge_signature(nl(lh, systems), n)
    assert got == expected and len(got) > 0


@real_nvalchemi
def test_real_matches_dense_with_queried_keys():
    from kups.core.data.index import Index

    rng = np.random.default_rng(2)
    n, cutoff = 64, 4.0
    positions = jnp.asarray(rng.uniform(0.0, 12.0, size=(n, 3)))
    lh = make_lh(positions, jnp.zeros(n, dtype=int))
    systems, _ = systems_from_lvecs(_CUBIC * 1.2, jnp.array([cutoff]))
    queried = Index(lh.keys, jnp.array([2, 5, 11], dtype=int))

    dense = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n * n),
        avg_edges=FixedCapacity(n * 128),
        avg_image_candidates=FixedCapacity(n * n),
        cutoff=cutoff,
    )
    nl = NvalchemiCellListNeighborList(
        cutoff=cutoff,
        max_neighbors=FixedCapacity(128),
        max_total_cells=FixedCapacity(512),
        avg_edges=FixedCapacity(n * 128),
    )
    expected = _edge_signature(dense(lh, systems, queried_keys=queried), n)
    got = _edge_signature(nl(lh, systems, queried_keys=queried), n)
    assert got == expected and len(got) > 0


@real_nvalchemi
@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_real_matches_dense_multi_system(method):
    """Exercise the real batched kernels (batch_ptr, system sort, shared grid)."""

    rng = np.random.default_rng(4)
    # cutoff with grid margin: >=4 cells/axis (no promotion) and cell_size > cutoff
    # so the batched radius-1 search is exact (no float32 tight-boundary misses).
    n, cutoff = 96, 2.2
    positions = jnp.asarray(rng.uniform(0.0, 12.0, size=(n, 3)))
    batch = jnp.array([0] * 40 + [1] * 56)  # contiguous two-system layout
    lh = make_lh(positions, batch)
    lvecs = jnp.concatenate([_CUBIC, _TRICLINIC], axis=0) * 1.2
    systems, _ = systems_from_lvecs(lvecs, jnp.array([cutoff, cutoff]))

    dense = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n * n),
        avg_edges=FixedCapacity(n * 128),
        avg_image_candidates=FixedCapacity(n * n),
        cutoff=cutoff,
    )
    if method == "naive":
        nl = NvalchemiNaiveNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            avg_edges=FixedCapacity(n * 128),
            max_shifts=FixedCapacity(64),
        )
    else:
        nl = NvalchemiCellListNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            max_total_cells=FixedCapacity(512),
            avg_edges=FixedCapacity(n * 128),
        )
    expected = _edge_signature(dense(lh, systems), n)
    got = _edge_signature(nl(lh, systems), n)
    assert got == expected and len(got) > 0


@real_nvalchemi
def test_real_multi_system_cell_list_promotion_raises():
    """Multi-system cell_list with <4 cells/axis raises rather than silently miss."""

    rng = np.random.default_rng(5)
    n, cutoff = 60, 4.0  # box 12 -> 3 cells/axis -> grid would be promoted
    positions = jnp.asarray(rng.uniform(0.0, 12.0, size=(n, 3)))
    batch = jnp.array([0] * 30 + [1] * 30)
    lh = make_lh(positions, batch)
    systems, _ = systems_from_lvecs(
        jnp.concatenate([_CUBIC, _CUBIC]) * 1.2, jnp.array([cutoff, cutoff])
    )
    nl = NvalchemiCellListNeighborList(
        cutoff=cutoff,
        max_neighbors=FixedCapacity(128),
        max_total_cells=FixedCapacity(512),
        avg_edges=FixedCapacity(n * 128),
    )
    with pytest.raises(ValueError, match="at least 4 cells"):
        as_result_function(nl)(keys=lh, systems=systems).raise_assertion()


@real_nvalchemi
@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_real_multi_system_noncontiguous(method):
    """Interleaved atom->system order exercises the batched sort + inverse remap.

    The contiguous tests leave ``argsort`` an identity; alternating systems
    forces the real reorder/remap so a permutation bug would diverge from dense.
    """

    rng = np.random.default_rng(6)
    n, cutoff = (
        150,
        2.2,
    )  # grid margin: cell_size > cutoff -> batched cell_list exact  # cutoff <= perp/4 keeps the multi-system grid unpromoted
    positions = jnp.asarray(rng.uniform(0.0, 12.0, size=(n, 3)))
    batch = jnp.asarray(np.arange(n) % 2)  # alternating: maximally non-contiguous
    lh = make_lh(positions, batch)
    lvecs = jnp.concatenate([_CUBIC, _TRICLINIC], axis=0) * 1.2
    systems, _ = systems_from_lvecs(lvecs, jnp.array([cutoff, cutoff]))

    dense = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n * n),
        avg_edges=FixedCapacity(n * 128),
        avg_image_candidates=FixedCapacity(n * n),
        cutoff=cutoff,
    )
    if method == "naive":
        nl = NvalchemiNaiveNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            avg_edges=FixedCapacity(n * 128),
            max_shifts=FixedCapacity(64),
        )
    else:
        nl = NvalchemiCellListNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            max_total_cells=FixedCapacity(512),
            avg_edges=FixedCapacity(n * 128),
        )
    expected = _edge_signature(dense(lh, systems), n)
    got = _edge_signature(nl(lh, systems), n)
    assert got == expected and len(got) > 0


@real_nvalchemi
@pytest.mark.parametrize("method", ["naive", "cell_list"])
def test_real_multi_system_queried_keys(method):
    """Batched ``queried_keys`` over a non-contiguous batch: target_indices are
    remapped through the system sort, affected atoms span both systems."""
    from kups.core.data.index import Index

    rng = np.random.default_rng(7)
    n, cutoff = 150, 2.2  # grid margin: cell_size > cutoff -> batched cell_list exact
    positions = jnp.asarray(rng.uniform(0.0, 12.0, size=(n, 3)))
    batch = jnp.asarray(np.arange(n) % 2)
    lh = make_lh(positions, batch)
    lvecs = jnp.concatenate([_CUBIC, _TRICLINIC], axis=0) * 1.2
    systems, _ = systems_from_lvecs(lvecs, jnp.array([cutoff, cutoff]))
    # Affected atoms in both systems (even -> system 0, odd -> system 1).
    queried = Index(lh.keys, jnp.array([4, 9, 22, 37, 50, 63, 88, 101, 124, 139]))

    dense = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n * n),
        avg_edges=FixedCapacity(n * 128),
        avg_image_candidates=FixedCapacity(n * n),
        cutoff=cutoff,
    )
    if method == "naive":
        nl = NvalchemiNaiveNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            avg_edges=FixedCapacity(n * 128),
            max_shifts=FixedCapacity(64),
        )
    else:
        nl = NvalchemiCellListNeighborList(
            cutoff=cutoff,
            max_neighbors=FixedCapacity(128),
            max_total_cells=FixedCapacity(512),
            avg_edges=FixedCapacity(n * 128),
        )
    expected = _edge_signature(dense(lh, systems, queried_keys=queried), n)
    got = _edge_signature(nl(lh, systems, queried_keys=queried), n)
    assert got == expected and len(got) > 0


# --- Capacity assertions (every capacity grows via as_result_function) --------


@dataclass
class _NvalParams:
    """Growable capacities for both nvalchemi lists, focused by ``identity_lens``."""

    max_neighbors: int = field(static=True)
    avg_edges: int = field(static=True)
    max_total_cells: int = field(static=True)
    max_shifts: int = field(static=True)


_GENEROUS = dict(max_neighbors=64, avg_edges=64, max_total_cells=256, max_shifts=64)
# Each list's three independently-asserted capacities.
_CAPACITIES = {
    "cell_list": ("max_neighbors", "avg_edges", "max_total_cells"),
    "naive": ("max_neighbors", "avg_edges", "max_shifts"),
}


def _make_nl(method, params, cutoff):
    cls = (
        NvalchemiCellListNeighborList
        if method == "cell_list"
        else NvalchemiNaiveNeighborList
    )
    return cls.new(params, identity_lens(_NvalParams), cutoff)


def _result(make, params, lh, systems):
    """Collect runtime assertions from a jitted ``make(params)`` run.

    The list is built outside jit (``.new`` reads a concrete cutoff), then its
    call is run under ``jax.jit(as_result_function(...))`` with the tables passed
    as traced arguments; its ``LensCapacity`` fixes target ``params``.
    """
    nl = make(params)
    run = jax.jit(as_result_function(lambda k, s: nl(k, s)))
    return run(lh, systems)


def _capacity_system(seed):
    rng = np.random.default_rng(seed)
    n, cutoff = 64, 4.0
    positions = jnp.asarray(rng.uniform(0.0, 12.0, size=(n, 3)))
    lh = make_lh(positions, jnp.zeros(n, dtype=int))
    systems, _ = systems_from_lvecs(_CUBIC * 1.2, jnp.array([cutoff]))
    return n, lh, systems, cutoff


@real_nvalchemi
@pytest.mark.parametrize(
    "method,capacity",
    [(m, c) for m, caps in _CAPACITIES.items() for c in caps],
)
def test_real_capacity_assertion_fires(method, capacity):
    """A generous build passes all assertions; undersizing any one capacity to 1
    trips its runtime assertion under ``as_result_function``."""
    _, lh, systems, cutoff = _capacity_system(seed=8)
    make = lambda p: _make_nl(method, p, cutoff)  # noqa: E731

    good = _result(make, _NvalParams(**_GENEROUS), lh, systems)
    assert good.all_assertions_pass

    bad = _result(make, _NvalParams(**{**_GENEROUS, capacity: 1}), lh, systems)
    assert not bad.all_assertions_pass


@real_nvalchemi
@pytest.mark.parametrize("method", ["cell_list", "naive"])
def test_real_capacity_grow_converges(method):
    """From all-tiny capacities, the standard fix loop grows every capacity until
    assertions pass, and the converged list matches the brute-force reference."""
    n, lh, systems, cutoff = _capacity_system(seed=9)
    make = lambda p: _make_nl(method, p, cutoff)  # noqa: E731

    params = _NvalParams(max_neighbors=1, avg_edges=1, max_total_cells=1, max_shifts=1)
    result = _result(make, params, lh, systems)
    assert not result.all_assertions_pass  # tiny capacities overflow
    for _ in range(30):
        if result.all_assertions_pass:
            break
        params = result.fix_or_raise(params)
        result = _result(make, params, lh, systems)
    assert result.all_assertions_pass

    dense = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n * n),
        avg_edges=FixedCapacity(n * 128),
        avg_image_candidates=FixedCapacity(n * n),
        cutoff=cutoff,
    )
    assert _edge_signature(result.value, n) == _edge_signature(dense(lh, systems), n)
