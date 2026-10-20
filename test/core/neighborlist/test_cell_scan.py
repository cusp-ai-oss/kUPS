# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""``CellScanNeighborList`` against the dense reference on shift-resolved edge sets: batched
skewed systems, bipartite queries, exclusions under periodic images and cutoffs wider than the
cell, plus its runtime guards."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data import Index, Table
from kups.core.neighborlist import CellScanNeighborList, DenseNearestNeighborList
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId

from ._builders import SamplePoints, make_lh, make_systems


def _edges_with_shifts(nl, keys, systems, queries=None):
    result = jax.jit(as_result_function(nl))(
        keys=keys, systems=systems, queries=queries
    )
    result.raise_assertion()
    edges = result.value
    valid = np.asarray(edges.indices.valid_mask).all(-1)
    raw = np.asarray(edges.indices.raw_indices)[valid]
    shifts = np.asarray(edges.shifts)[valid, 0]
    return {(int(a), int(b), *map(int, np.round(s))) for (a, b), s in zip(raw, shifts)}


def _both(cutoffs, capacity, image_capacity=None):
    scan = CellScanNeighborList(
        avg_edges=FixedCapacity(capacity), cells=FixedCapacity(512), cutoffs=cutoffs
    )
    dense = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(capacity),
        avg_edges=FixedCapacity(capacity),
        avg_image_candidates=FixedCapacity(image_capacity or capacity * 8),
        cutoffs=cutoffs,
    )
    return scan, dense


def _triclinic(scale):
    lvecs = jnp.array([[1.0, 0.0, 0.0], [0.3, 0.9, 0.0], [-0.2, 0.25, 1.1]]) * scale
    return PeriodicCell(TriclinicFrame.from_matrix(lvecs[None]))


def test_batched_skewed_systems_match_dense():
    """Systems of different sizes and shapes in one call, unsorted by system."""
    rng = np.random.default_rng(3)
    scales = np.array([9.0, 12.0, 15.0])
    base = np.array([[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [-0.3, 0.2, 1.0]])
    counts = np.array([30, 60, 90])
    system = rng.permutation(np.repeat(np.arange(3), counts))
    frac = rng.uniform(0, 1, (system.size, 3))
    positions = jnp.asarray(
        np.einsum("ni,nij->nj", frac, scales[system, None, None] * base)
    )
    keys = make_lh(positions, jnp.asarray(system))
    cell = PeriodicCell(
        TriclinicFrame.from_matrix(jnp.asarray(scales[:, None, None] * base))
    )
    systems, cutoffs = make_systems(cell, jnp.array([3.0, 3.5, 4.0]))
    scan, dense = _both(cutoffs, 200)
    assert _edges_with_shifts(scan, keys, systems) == _edges_with_shifts(
        dense, keys, systems
    )


def test_pairs_just_inside_the_cutoff_match_dense():
    """Pairs within float32 round-off of the cutoff survive the float32 kernel."""
    rng = np.random.default_rng(5)
    cutoff, n = 4.0, 50
    anchors = rng.uniform(300.0, 330.0, (n, 3))  # unwrapped, far from the home cell
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    partners = anchors + (cutoff - 1e-9) * direction
    positions = jnp.asarray(np.concatenate([anchors, partners]))
    keys = make_lh(positions, jnp.zeros(2 * n, int))
    systems, cutoffs = make_systems(_triclinic(30.0), jnp.array([cutoff]))
    scan, dense = _both(cutoffs, 128)
    expected = _edges_with_shifts(dense, keys, systems)
    assert len(expected) >= 2 * n
    assert _edges_with_shifts(scan, keys, systems) == expected


def test_empty_queries():
    keys = make_lh(jnp.ones((5, 3)), jnp.zeros(5, int))
    none = np.zeros(0, int)
    queries = Table(
        (),
        SamplePoints(
            positions=jnp.zeros((0, 3)),
            system=Index.integer(none, n=1, label=SystemId),
            inclusion=Index.integer(none, n=1, label=SystemId),
            exclusion=Index.integer(none, n=1),
        ),
        _cls=ParticleId,
    )
    systems, cutoffs = make_systems(_triclinic(10.0), jnp.array([3.0]))
    scan, _ = _both(cutoffs, 8)
    assert _edges_with_shifts(scan, keys, systems, queries) == set()
    assert _edges_with_shifts(scan, queries, systems) == set()


@pytest.mark.parametrize(
    ("cutoff", "stencil_radius", "message"),
    [(3.0, 1, "stencil_radius too small"), (jnp.inf, 1, "finite cutoffs")],
)
def test_runtime_guards(cutoff, stencil_radius, message):
    keys = make_lh(jnp.ones((4, 3)), jnp.zeros(4, int))
    systems, cutoffs = make_systems(_triclinic(2.0), jnp.array([cutoff]))
    scan = CellScanNeighborList(
        avg_edges=FixedCapacity(64),
        cells=FixedCapacity(8),
        cutoffs=cutoffs,
        stencil_radius=stencil_radius,
    )
    result = jax.jit(as_result_function(scan))(keys=keys, systems=systems)
    with pytest.raises(Exception, match=message):
        result.raise_assertion()


def test_bipartite_queries_match_dense():
    rng = np.random.default_rng(0)
    cell = _triclinic(10.0)
    keys = make_lh(jnp.asarray(rng.uniform(0, 10, (80, 3))), jnp.zeros(80, int))
    queries = make_lh(jnp.asarray(rng.uniform(-2, 12, (17, 3))), jnp.zeros(17, int))
    queries = Table(tuple(ParticleId(80 + i) for i in range(17)), queries.data)
    systems, cutoffs = make_systems(cell, jnp.array([3.0]))
    scan, dense = _both(cutoffs, 80)
    expected = _edges_with_shifts(dense, keys, systems, queries)
    assert expected and _edges_with_shifts(scan, keys, systems, queries) == expected


def test_exclusions_keep_non_minimum_images():
    """Excluded pairs drop only their minimum image; in a box narrower than twice the
    cutoff the other images survive, so ``is_minimum_image`` must be exact."""
    rng = np.random.default_rng(1)
    n = 24
    positions = jnp.asarray(rng.uniform(0, 4.0, (n, 3)))
    molecules = jnp.arange(n) // 3  # exclusion groups of three
    keys = make_lh(positions, jnp.zeros(n, int), molecules)
    systems, cutoffs = make_systems(_triclinic(4.0), jnp.array([3.0]))
    scan, dense = _both(cutoffs, 400)
    assert _edges_with_shifts(scan, keys, systems) == _edges_with_shifts(
        dense, keys, systems
    )


def test_cutoff_wider_than_cell_with_larger_stencil():
    rng = np.random.default_rng(2)
    keys = make_lh(jnp.asarray(rng.uniform(0, 2.0, (6, 3))), jnp.zeros(6, int))
    systems, cutoffs = make_systems(_triclinic(2.0), jnp.array([3.0]))
    _, dense = _both(cutoffs, 2000, image_capacity=200_000)
    scan = CellScanNeighborList(
        avg_edges=FixedCapacity(2000),
        cells=FixedCapacity(512),
        cutoffs=cutoffs,
        stencil_radius=3,
    )
    assert _edges_with_shifts(scan, keys, systems) == _edges_with_shifts(
        dense, keys, systems
    )
