# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""An image buffer no larger than the candidate buffer keeps only the minimum
image, which is exact only while no candidate reaches a second image.

A batch that mixes a minimum-image system with one whose cutoff reaches past
half its cell has more candidates than image rows to spare; the capacity check
must then grow the image buffer instead of letting the minimum-image shortcut
drop the images of the small system.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import CapacityError, FixedCapacity
from kups.core.data.index import Index
from kups.core.neighborlist.cell_list import CellListNeighborList
from kups.core.neighborlist.common import Candidates
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
from test.units import A

from ._builders import (
    EvalState,
    make_lh,
    replicate,
    run_nl,
    systems_from_lvecs,
    valid_ids,
)

_CUTOFF = 3.0
# Cu fcc conventional cell: the cutoff reaches past half the 3.61 Å cell, so
# every pair has eight candidate images and each atom twelve neighbors.
_FCC = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
_SMALL = 3.61 * np.eye(3)
_BIG = 9.0 * np.eye(3)
_N_BIG = 36


def _batch():
    big_frac = np.asarray(jax.random.uniform(jax.random.key(0), (_N_BIG, 3)))
    fracs = [big_frac, _FCC]
    matrices = [_BIG, _SMALL]
    positions = jnp.asarray(np.concatenate([f @ m for f, m in zip(fracs, matrices)]))
    system_ids = jnp.repeat(jnp.arange(2), jnp.array([_N_BIG, len(_FCC)]))
    lh = make_lh(positions * A, system_ids)
    systems, cutoffs = systems_from_lvecs(
        jnp.asarray(np.stack(matrices)), jnp.array([_CUTOFF, _CUTOFF])
    )
    return lh, systems, cutoffs, fracs, matrices


def _brute_force(fracs, matrices) -> set[tuple[int, int, tuple[int, ...]]]:
    """Directed ``(i, j, n)`` within the cutoff, per system, global particle ids."""
    edges, offset = set(), 0
    for f, m in zip(fracs, matrices):
        for n in itertools.product(range(-3, 4), repeat=3):
            d = (f[:, None, :] - f[None, :, :] - np.asarray(n)) @ m
            d2 = (d * d).sum(-1)
            ii, jj = np.nonzero((d2 < _CUTOFF**2) & (d2 > 1e-12))
            edges.update((offset + int(i), offset + int(j), n) for i, j in zip(ii, jj))
        offset += len(f)
    return edges


@pytest.mark.parametrize("nl_cls", [DenseNearestNeighborList, CellListNeighborList])
def test_batch_keeps_images_when_capacities_are_equal(nl_cls):
    lh, systems, cutoffs, fracs, matrices = _batch()
    # Equal candidate and image capacities select the minimum-image shortcut,
    # which is what the capacity fix loop reaches from the estimate
    # (32 / 64 -> 64 / 64) on this batch.
    params = UniversalNeighborlistParameters(
        avg_edges=64, avg_candidates=64, avg_image_candidates=64, cells=8
    )
    state = EvalState(particles=lh, systems=systems, neighborlist_params=params)
    first = run_nl(nl_cls.from_state(state, cutoffs), lh, systems)
    assert first.failed_assertions, "dropping the images must not pass silently"
    for _ in range(5):
        result = run_nl(nl_cls.from_state(state, cutoffs), lh, systems)
        if not result.failed_assertions:
            break
        state = result.fix_or_raise(state)
    result.raise_assertion()
    assert state.neighborlist_params.avg_image_candidates > 64
    edges = result.value
    keep = np.asarray(edges.indices.valid_mask.all(axis=-1))
    shifts = np.rint(np.asarray(edges.shifts[:, 0, :])).astype(int)[keep]
    got = {
        (int(i), int(j), tuple(int(s) for s in sh))
        for (i, j), sh in zip(valid_ids(edges.indices), shifts)
    }
    assert got == _brute_force(fracs, matrices)


def test_replicate_rejects_a_candidate_sized_buffer_with_padding():
    # One real candidate, whose cell gives it eight images, padded to sixteen
    # candidates: the summed image count (8) fits sixteen rows, but a buffer no
    # larger than the candidates takes the minimum-image path, so it must fail.
    fracs = np.concatenate([np.zeros((_N_BIG, 3)), _FCC])
    system_ids = jnp.repeat(jnp.arange(2), jnp.array([_N_BIG, len(_FCC)]))
    lh = make_lh(jnp.asarray(fracs), system_ids)
    systems, cutoffs = systems_from_lvecs(
        jnp.asarray(np.stack([_BIG, _SMALL])), jnp.array([_CUTOFF, _CUTOFF])
    )
    n_padding = 15
    key = jnp.concatenate([jnp.array([_N_BIG]), jnp.full(n_padding, lh.size)])
    query = jnp.concatenate([jnp.array([_N_BIG + 1]), jnp.full(n_padding, lh.size)])
    candidates = Candidates(
        key_idx=Index(lh.keys, key), query_idx=Index(lh.keys, query)
    )
    with pytest.raises(CapacityError):
        replicate(candidates, lh, lh, systems, cutoffs, FixedCapacity(1 + n_padding))
