# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Mixed batches whose capacities come from ``estimate``, as the applications
size them: skewed cells, an open axis, and a batch whose estimate itself has as
many image rows as candidate rows. The reference is a brute-force image search.
"""

from collections import Counter
from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.cell import Cell, TriclinicFrame
from kups.core.data import Table
from kups.core.neighborlist import CellListNeighborList, DenseNearestNeighborList
from kups.core.neighborlist.adaptive import AdaptiveNeighborList
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from test.units import A

from .._builders import EvalState, make_lh, make_systems, run_nl, valid_ids

_CUTOFF = 3.0
_WIDE = [[9.0, 0.0, 0.0], [3.6, 8.1, 0.0], [-2.7, 1.8, 9.0]]
_WIDER = [[12.0, 0.0, 0.0], [4.8, 10.8, 0.0], [-3.6, 2.4, 12.0]]
_NARROW = [[5.5, 0.0, 0.0], [1.0, 5.2, 0.0], [0.5, -0.8, 5.6]]  # < 2 x cutoff
# (atoms, cell) per system, periodic axes
_SKEWED = ((30, _WIDE), (60, _WIDER), (20, _NARROW))
# Two bins per axis in the cube and three images along the narrow axis of the
# small cell, a particle's own included: the estimate is 32 candidate rows and
# 32 image rows per particle.
_EQUAL_ROWS = ((44, np.diag([8.0, 8.0, 8.0])), (2, np.diag([2.9, 8.0, 8.0])))
_BATCHES = {
    "skewed": (_SKEWED, (True, True, True)),
    "skewed, open axis": (_SKEWED, (True, True, False)),
    "equal rows": (_EQUAL_ROWS, (True, True, True)),
}


def _brute_force(positions, lattices, system, periodic) -> Counter[tuple[int, int]]:
    """Count of in-range periodic images per ordered pair, self images included."""
    shifts = np.array(list(product(range(-3, 4), repeat=3)))
    shifts = shifts[~(shifts[:, ~np.asarray(periodic)] != 0).any(axis=1)]
    pairs: Counter[tuple[int, int]] = Counter()
    for i, j in product(range(len(positions)), repeat=2):
        if system[i] != system[j]:
            continue
        deltas = positions[j] - positions[i] + shifts @ lattices[system[i]]
        dist = np.linalg.norm(deltas, axis=-1)
        if count := int(((dist > 0) & (dist < _CUTOFF)).sum()):
            pairs[(i, j)] = count
    return pairs


@pytest.mark.parametrize("batch", _BATCHES)
@pytest.mark.parametrize(
    "neighbor_list",
    [DenseNearestNeighborList, CellListNeighborList, AdaptiveNeighborList],
)
def test_estimated_mixed_batch_matches_brute_force(neighbor_list, batch):
    boxes, periodic = _BATCHES[batch]
    lattices = np.array([lattice for _, lattice in boxes], dtype=float)
    system = np.repeat(np.arange(len(boxes)), [n for n, _ in boxes])
    frac = np.random.default_rng(0).uniform(0.0, 1.0, (system.size, 3))
    positions = np.einsum("ni,nij->nj", frac, lattices[system])
    lh = make_lh(jnp.asarray(positions) * A, jnp.asarray(system))
    frame = TriclinicFrame.from_matrix(jnp.asarray(lattices) * A)
    systems, cutoffs = make_systems(
        Cell.from_pbc(frame, periodic), jnp.full(len(boxes), _CUTOFF) * A
    )
    counts = Table(systems.keys, lh.data.system.counts.data)
    params = UniversalNeighborlistParameters.estimate(counts, systems, cutoffs)
    state = EvalState(particles=lh, systems=systems, neighborlist_params=params)

    while (
        result := run_nl(neighbor_list.from_state(state, cutoffs), lh, systems)
    ).failed_assertions:
        state = result.fix_or_raise(state)

    edges = Counter(map(tuple, valid_ids(result.value.indices).tolist()))
    reference = _brute_force(positions, lattices, system, periodic)
    assert any(count > 1 for count in reference.values())
    assert edges == reference
