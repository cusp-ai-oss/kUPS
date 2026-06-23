# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Brute-force completeness and minimum-image tests for the anchored-window image
replication on skewed/acute/anisotropic triclinic cells.

These exercise the runtime pipeline (resize, vmap shapes, segment-argmin flag)
that a pure-NumPy geometry check cannot: the dense neighbor list is run via
``call_with_retry`` and its emitted ``(i, j, shift)`` edge set is compared to a
converged brute-force enumeration; ``replicate_for_images`` is checked directly
for the exact closest-image flag.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.common import (
    Candidates,
    candidate_image_counts,
    replicate_for_images,
)
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.result import as_result_function

from ._builders import cutoff_table, make_lh, make_systems

# (name, lattice matrix [rows = lattice vectors], cutoff). Cover the orthogonal
# baseline, strong monoclinic skew, an acute 35-degree cell (where round-based
# minimum image can pick the wrong copy), and an anisotropic short-axis cell.
CELLS = [
    ("orthogonal", [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], 2.2),
    ("monoclinic", [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [5.0, 0.0, 5.0]], 4.0),
    ("acute35", [[6.0, 0.0, 0.0], [4.915, 3.441, 0.0], [0.0, 0.0, 7.0]], 3.0),
    ("aniso_short", [[8.0, 0.0, 0.0], [2.5, 7.5, 0.0], [1.5, 1.0, 2.2]], 4.0),
]
IDS = [c[0] for c in CELLS]

type Matrix = list[list[float]]
type EdgeSet = set[tuple[int, int, tuple[int, ...]]]


def _real_positions(n: int, seed: int, matrix: Matrix) -> jax.Array:
    """Random real positions spanning the cell (the neighbor list takes real coords)."""
    return jax.random.uniform(jax.random.key(seed), (n, 3)) @ jnp.asarray(matrix)


def _brute_force(real: jax.Array, matrix: Matrix, cutoff: float) -> EdgeSet:
    """All ``(i, j, offset)`` with ``0 < |r_i - r_j - n @ A| < cutoff``.

    Enumerates a span provably wider than any in-range image and asserts the
    result is converged (one wider shell adds nothing).
    """
    r, A = np.asarray(real), np.asarray(matrix)
    perp = 1.0 / np.linalg.norm(np.linalg.inv(A), axis=0)  # column norms
    span = (np.ceil(cutoff / perp).astype(int) + 2).tolist()

    def collect(extra: int) -> EdgeSet:
        edges = set()
        ranges = [range(-(s + extra), s + extra + 1) for s in span]
        for n in itertools.product(*ranges):
            d = (r[:, None, :] - r[None, :, :]) - np.array(n) @ A
            d2 = (d * d).sum(-1)
            ii, jj = np.nonzero((d2 < cutoff**2) & (d2 > 1e-12))
            edges.update((int(i), int(j), tuple(n)) for i, j in zip(ii, jj))
        return edges

    edges = collect(0)
    assert edges == collect(1), "brute force not converged"
    return edges


def _dense_nl_edges(real: jax.Array, matrix: Matrix, cutoff: float, n: int) -> EdgeSet:
    """Run the dense neighbor list and return its ``(i, j, shift)`` edge set."""
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.asarray(matrix)[None]))
    lh = make_lh(jnp.asarray(real), jnp.zeros(n, dtype=int))
    systems, _ = make_systems(cell, jnp.array([cutoff]))
    images = int(candidate_image_counts(cell, jnp.array([cutoff])).prod())
    nl = DenseNearestNeighborList(
        avg_candidates=FixedCapacity(n),
        avg_edges=FixedCapacity(n * images),
        avg_image_candidates=FixedCapacity(n * images),
        cutoffs=cutoff_table(jnp.array([cutoff])),
    )
    result = jax.jit(as_result_function(nl))(keys=lh, systems=systems)
    result.raise_assertion()
    edges = result.value
    raw = np.asarray(edges.indices.indices)
    shifts = np.asarray(edges.shifts[:, 0, :])
    keep = (raw[:, 0] < n) & (raw[:, 1] < n)
    return {
        (int(i), int(j), tuple(int(s) for s in sh))
        for (i, j), sh in zip(raw[keep], shifts[keep])
    }


@pytest.mark.parametrize("name,matrix,cutoff", CELLS, ids=IDS)
def test_edges_match_brute_force(name: str, matrix: Matrix, cutoff: float):
    n = 6
    real = _real_positions(n, 0, matrix)
    truth = _brute_force(real, matrix, cutoff)
    got = _dense_nl_edges(real, matrix, cutoff, n)
    assert got == truth, f"missing={truth - got} extra={got - truth}"


@pytest.mark.parametrize("name,matrix,cutoff", CELLS, ids=IDS)
def test_minimum_image_flag_is_true_closest(name: str, matrix: Matrix, cutoff: float):
    # replicate_for_images operates on the internal fractional representation
    # (the pipeline folds real -> fractional upstream), so positions are in [0, 1).
    n = 5
    frac = jax.random.uniform(jax.random.key(1), (n, 3))
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.asarray(matrix)[None]))
    lh = make_lh(frac, jnp.zeros(n, dtype=int))
    systems, _ = make_systems(cell, jnp.array([cutoff]))

    grid_i, grid_j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    candidates = Candidates(
        key_idx=Index(lh.keys, jnp.asarray(grid_i.ravel())),
        query_idx=Index(lh.keys, jnp.asarray(grid_j.ravel())),
    )
    per_pair = int(candidate_image_counts(cell, jnp.array([cutoff])).prod())
    batch = replicate_for_images(
        candidates,
        lh,
        lh,
        systems,
        cutoff_table(jnp.array([cutoff])),
        FixedCapacity(n * n * per_pair),
    )

    raw = np.asarray(batch.edges.indices.indices)
    shifts = np.asarray(batch.edges.shifts[:, 0, :])
    is_min = np.asarray(batch.is_minimum_image)
    f, A = np.asarray(frac), np.asarray(matrix)

    flagged: dict[tuple[int, int], list[np.ndarray]] = {}
    for (i, j), sh, m in zip(raw, shifts, is_min):
        if i < n and j < n and m:
            flagged.setdefault((int(i), int(j)), []).append(sh)

    for i in range(n):
        for j in range(n):
            copies = flagged.get((i, j), [])
            assert len(copies) == 1, f"pair ({i},{j}) flagged {len(copies)} times"
            d_flag = float(np.linalg.norm((f[i] - f[j] - copies[0]) @ A))
            d_min = min(
                float(np.linalg.norm((f[i] - f[j] - np.array(off)) @ A))
                for off in itertools.product(range(-3, 4), repeat=3)
            )
            # The window provably brackets the global closest only for in-range
            # pairs; out-of-range pairs are fully distance-masked, so the flag on
            # them is a don't-care.
            if d_min < cutoff:
                assert d_flag == pytest.approx(d_min, abs=1e-5), f"pair ({i},{j})"
