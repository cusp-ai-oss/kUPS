# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive coverage for the anchored-window periodic-image replication.

The neighbor list replaces a symmetric ``2*ceil(ratio)+1`` image stencil with a
per-pair *anchored window*: each candidate pair is replicated over
``candidate_image_counts`` consecutive integer shifts per axis
(``width = max(ceil(2*cutoff/perp), 1)``) anchored at ``ceil(separation - ratio)``,
and the minimum-image copy is flagged by a real-distance segment-argmin rather
than by rounding. This module tests that machinery across the full space of
cutoff/lattice combinations, in four layers of increasing integration:

1. ``TestWindowMathSpec`` — the pure integer spec of the anchor+width formula
   (window brackets every in-range image; width is the tight open-interval
   count). Vectorized NumPy, hammering float boundaries.
2. ``TestCandidateImageCounts`` / ``TestGenerateImageOffsets`` — the real width
   and offset-enumeration primitives across ratio regimes, anisotropy, frame
   types, partial periodicity and non-finite ratios.
3. ``TestReplicateForImagesBruteForce`` — ``replicate_for_images`` over a rich
   cell x cutoff x periodicity x positions matrix, asserting the emitted window
   is an exact superset of the brute-force in-range images and the min-image
   flag is the exact global closest (NumPy f64 oracle; no distance-boundary
   fuzz).
4. ``TestEndToEnd*`` — the full pipeline (masks, compaction, mirror, single-cell
   fast path) for ``Dense`` / ``CellList`` / ``AllDense``, compared to the same
   brute force with a boundary guard band, plus symmetry, determinism, and
   translation invariance.
"""

from __future__ import annotations

import collections
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.capacity import FixedCapacity
from kups.core.cell import Cell, OrthogonalFrame, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.neighborlist.all_dense import AllDenseNearestNeighborList
from kups.core.neighborlist.cell_list import CellListNeighborList, _cell_list_subselect
from kups.core.neighborlist.common import (
    Candidates,
    _generate_image_offsets,
    candidate_image_counts,
    num_cells,
    replicate_for_images,
)
from kups.core.neighborlist.dense import DenseNearestNeighborList
from kups.core.result import as_result_function

from ._builders import cutoff_table, make_lh, make_systems

# --------------------------------------------------------------------------- #
# Cell zoo: (name, 3x3 matrix rows=lattice vectors, is_triclinic-only?)
# Covers the orthogonal baseline, anisotropic short-axis, monoclinic skew, an
# acute 35-degree cell (round-based minimum image picks the wrong copy),
# a general triclinic, and a strongly sheared small-perp cell.
# --------------------------------------------------------------------------- #
type Matrix = list[list[float]]
type Vec3 = tuple[int, int, int]
type Edge = tuple[int, int, Vec3]

CELLS: list[tuple[str, Matrix]] = [
    ("ortho_cubic", [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]),
    ("ortho_aniso", [[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 8.0]]),
    ("monoclinic", [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [5.0, 0.0, 5.0]]),
    ("acute35", [[6.0, 0.0, 0.0], [4.915, 3.441, 0.0], [0.0, 0.0, 7.0]]),
    ("triclinic", [[8.0, 0.0, 0.0], [2.5, 7.5, 0.0], [1.5, 1.0, 6.2]]),
    ("sheared", [[5.0, 0.0, 0.0], [3.8, 3.2, 0.0], [3.5, 2.9, 3.3]]),
]
CELL_IDS = [c[0] for c in CELLS]

# Ratios (relative to the *shortest* perpendicular length) chosen to be
# non-commensurate with any lattice plane so brute-force distances rarely land
# on the cutoff, and to sweep the regimes: minimum-image, mild replication,
# ~one image, and deep (>1) replication.
RATIOS = [0.37, 0.63, 0.88, 1.19]


def _perp(matrix: Matrix) -> np.ndarray:
    """Per-axis perpendicular lengths ``perp_k = 1/|column k of inv(A)|``."""
    return 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(matrix)), axis=0)


def _face_lengths(matrix: Matrix) -> np.ndarray:
    """Cell-list binning length ``1/|row k of inv(A)|`` (what ``num_cells`` uses).

    Differs from ``_perp`` (column norm) on skewed cells; a cutoff above
    ``max(face)/2`` forces exactly one bin per axis, triggering the single-cell
    fast path in ``_cell_list_subselect``.
    """
    return 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(matrix)), axis=1)


def _cutoff_for(matrix: Matrix, ratio: float) -> float:
    """Cutoff giving ``ratio`` on the shortest-perpendicular axis."""
    return float(ratio * _perp(matrix).min())


def _brute_force(frac: np.ndarray, matrix: Matrix, cutoff: float, pbc) -> set[Edge]:
    """All directed ``(i, j, n)`` with ``0 < |(f_i - f_j - n) @ A| < cutoff``.

    ``n`` ranges over integer shifts on periodic axes only; the span is provably
    wider than any in-range image and convergence is asserted by widening it.
    """
    f, A = np.asarray(frac, dtype=np.float64), np.asarray(matrix, dtype=np.float64)
    perp = _perp(matrix)
    span = [int(np.ceil(cutoff / perp[a])) + 2 if pbc[a] else 0 for a in range(3)]

    def collect(extra: int) -> set[Edge]:
        edges: set[Edge] = set()
        # Widen only periodic axes; non-periodic axes never leave n = 0.
        ranges = [
            range(-(span[a] + extra), span[a] + extra + 1) if pbc[a] else range(0, 1)
            for a in range(3)
        ]
        for n in itertools.product(*ranges):
            # real delta = ((f_i - f_j) - n) @ A  (subtract the integer shift in
            # fractional space, then map to real).
            d = ((f[:, None, :] - f[None, :, :]) - np.asarray(n)) @ A
            d2 = (d * d).sum(-1)
            ii, jj = np.nonzero((d2 < cutoff**2) & (d2 > 1e-9))
            edges.update((int(i), int(j), tuple(n)) for i, j in zip(ii, jj))
        return edges

    edges = collect(0)
    assert edges == collect(1), "brute force not converged"
    return edges


# Closest-image squared distances for every pair at once. Independent of cutoff,
# so one vectorized pass is shared across all ratios of a (cell, positions) —
# far cheaper than the per-pair Python loop the flag test would otherwise run.
_CLOSEST_CACHE: dict = {}


def _all_closest_d2(frac: np.ndarray, matrix: Matrix, pbc) -> np.ndarray:
    """``(n, n)`` array of the minimum squared image distance for each pair."""
    key = (frac.tobytes(), frac.shape, tuple(map(tuple, matrix)), tuple(pbc))
    cached = _CLOSEST_CACHE.get(key)
    if cached is not None:
        return cached
    f, A = np.asarray(frac, dtype=np.float64), np.asarray(matrix, dtype=np.float64)
    ranges = [range(-3, 4) if pbc[a] else range(0, 1) for a in range(3)]
    best = np.full((f.shape[0], f.shape[0]), np.inf)
    for n in itertools.product(*ranges):
        d = (f[:, None, :] - f[None, :, :] - np.asarray(n)) @ A
        best = np.minimum(best, (d * d).sum(-1))
    _CLOSEST_CACHE[key] = best
    return best


def _make_cell(matrix: Matrix, pbc=(True, True, True)) -> Cell:
    frame = TriclinicFrame.from_matrix(jnp.asarray(matrix)[None])
    return Cell.from_pbc(frame, pbc)


# ============================================================================ #
# Layer 1: pure-math spec of the anchor + width formula.
# ============================================================================ #
class TestWindowMathSpec:
    """The intended integer math of the anchored window, independent of JAX.

    ``candidate_image_counts`` gives ``width = max(ceil(2r), 1)`` and
    ``_get_candidate_images`` anchors the window at ``anchor = ceil(s - r)``.
    The window ``[anchor, anchor + width - 1]`` must bracket every integer ``n``
    with ``|s - n| < r`` (the reciprocal-projection bound), for *all* real
    separations ``s`` and ratios ``r``. These assertions reimplement the formula
    to lock the spec; the real code is checked in later layers.
    """

    @staticmethod
    def _window(s: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width = np.maximum(np.ceil(2.0 * r), 1.0).astype(np.int64)
        anchor = np.ceil(s - r).astype(np.int64)
        return anchor, anchor + width - 1

    def _assert_superset(self, s: np.ndarray, r: np.ndarray) -> None:
        anchor, top = self._window(s, r)
        # No in-range integer may sit just below the window or just above it;
        # since in-range integers are contiguous, checking a margin of 2 on each
        # side proves the window brackets the whole run (robust to fp error).
        for m in (anchor - 2, anchor - 1):
            assert bool((np.abs(s - m) >= r).all()), "in-range image below window"
        for m in (top + 1, top + 2):
            assert bool((np.abs(s - m) >= r).all()), "in-range image above window"

    def test_superset_random_f64(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(-4.0, 4.0, 400_000)
        r = rng.uniform(1e-3, 25.0, 400_000)
        self._assert_superset(s, r)

    def test_superset_boundary_grid(self):
        # Adversarial: s near integers/half-integers, (s - r) near an integer
        # (the ceil flip), r near 0.5 / 1.0 / integers, tiny and huge r.
        eps = np.array([-1e-6, -1e-9, 0.0, 1e-9, 1e-6, 0.5, -0.5])
        bases = np.arange(-3, 4, dtype=np.float64)
        s_vals = np.concatenate(
            [bases[:, None] + eps, bases[:, None] + 0.5 + eps]
        ).ravel()
        r_vals = np.concatenate(
            [
                np.array(
                    [
                        1e-7,
                        1e-3,
                        0.25,
                        0.4999999,
                        0.5,
                        0.5000001,
                        0.9999,
                        1.0,
                        1.0001,
                        2.0,
                        3.5,
                        50.0,
                    ]
                ),
                (np.arange(1, 8) / 2.0)[:, None].ravel() + eps.reshape(-1)[0],
            ]
        )
        s, r = (a.ravel() for a in np.meshgrid(s_vals, r_vals))
        r = np.abs(r) + 1e-9
        self._assert_superset(s, r)

    def test_superset_float32_boundary(self):
        # The real code computes the window in float32; a wrong f32 anchor/width
        # off by one is caught here at the same boundaries.
        eps = np.array([-1e-4, 0.0, 1e-4, 0.5], dtype=np.float32)
        bases = np.arange(-3, 4, dtype=np.float32)
        s = (bases[:, None] + eps).ravel().astype(np.float32)
        r = np.array([0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 4.5], dtype=np.float32)
        ss, rr = (a.ravel() for a in np.meshgrid(s, r))
        self._assert_superset(ss.astype(np.float64), rr.astype(np.float64))

    def test_width_is_tight_open_interval_count(self):
        # width must equal the maximum number of integers in an open interval of
        # length 2r (never fewer -> no missed images; never larger than needed).
        r = np.concatenate(
            [np.linspace(0.01, 6.0, 2000), np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
        )
        width = np.maximum(np.ceil(2.0 * r), 1.0).astype(np.int64)
        offsets = np.linspace(0.0, 1.0, 257)[:, None]  # sweep interval placement
        lo = np.floor(offsets - r) + 1  # smallest integer strictly > offset - r
        hi = np.ceil(offsets + r) - 1  # largest integer strictly < offset + r
        count = np.maximum(hi - lo + 1, 0).astype(np.int64)
        max_count = count.max(axis=0)
        npt.assert_array_equal(max_count, width)
        assert bool((count <= width).all())  # never under-sized for any placement


# ============================================================================ #
# Layer 2: real width and offset-enumeration primitives.
# ============================================================================ #
class TestCandidateImageCounts:
    @pytest.mark.parametrize(
        "ratio,expected",
        [
            (0.1, 1),
            (0.4999, 1),
            (0.5, 1),  # boundary: strict mask -> only the minimum image
            (0.5001, 2),
            (0.8, 2),
            (0.9999, 2),
            (1.0, 2),  # integer boundary: ceil(2) = 2
            (1.0001, 3),
            (1.5, 3),
            (2.0, 4),
            (2.5, 5),
        ],
    )
    def test_width_matches_ceil_formula(self, ratio, expected):
        # Cubic cell: perp = L on every axis; cutoff = ratio * L.
        cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * 4.0))
        cutoff = jnp.array([ratio * 4.0])
        images = candidate_image_counts(cell, cutoff)
        npt.assert_array_equal(np.asarray(images), np.full((1, 3), expected))

    def test_anisotropic_per_axis_widths(self):
        # perp = (3, 5, 8); cutoff 4 -> ratios (1.33, 0.8, 0.5) -> (3, 2, 1).
        cell = PeriodicCell(OrthogonalFrame(jnp.array([[3.0, 5.0, 8.0]])))
        images = candidate_image_counts(cell, jnp.array([4.0]))
        npt.assert_array_equal(np.asarray(images), np.array([[3, 2, 1]]))

    def test_orthogonal_and_triclinic_frames_agree(self):
        # Same diagonal geometry through both frame parameterisations.
        ortho = PeriodicCell(OrthogonalFrame(jnp.array([[3.0, 5.0, 8.0]])))
        tri = PeriodicCell(
            TriclinicFrame.from_matrix(jnp.diag(jnp.array([3.0, 5.0, 8.0]))[None])
        )
        cutoff = jnp.array([4.0])
        npt.assert_array_equal(
            np.asarray(candidate_image_counts(ortho, cutoff)),
            np.asarray(candidate_image_counts(tri, cutoff)),
        )

    @pytest.mark.parametrize(
        "pbc,expected",
        [
            ((True, True, False), [2, 2, 1]),
            ((True, False, False), [2, 1, 1]),
            ((False, False, False), [1, 1, 1]),
        ],
    )
    def test_open_axes_use_single_image(self, pbc, expected):
        # ratio 0.8 on every axis, but open axes must never replicate.
        cell = Cell.from_pbc(TriclinicFrame.from_matrix(jnp.eye(3)[None]), pbc)
        images = candidate_image_counts(cell, jnp.array([0.8]))
        npt.assert_array_equal(np.asarray(images), np.array([expected]))

    def test_multi_system_heterogeneous(self):
        cell = PeriodicCell(
            OrthogonalFrame(jnp.array([[10.0, 10.0, 10.0], [1.0, 1.0, 1.0]]))
        )
        images = candidate_image_counts(cell, jnp.array([2.0, 0.8]))
        npt.assert_array_equal(np.asarray(images), np.array([[1, 1, 1], [2, 2, 2]]))

    def test_nonfinite_ratio_clamps_to_one(self):
        class Cells:
            perpendicular_lengths = jnp.array([[0.0, 4.0, jnp.nan]])
            periodic = (True, True, True)

        images = candidate_image_counts(Cells(), jnp.array([6.0]))
        npt.assert_array_equal(np.asarray(images), np.array([[1, 3, 1]]))


class TestGenerateImageOffsets:
    @staticmethod
    def _blocks(images: np.ndarray) -> list[set[Vec3]]:
        return [
            {(i, j, k) for i in range(w[0]) for j in range(w[1]) for k in range(w[2])}
            for w in images
        ]

    @pytest.mark.parametrize(
        "images",
        [
            [[3, 3, 3]],
            [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
            [[4, 2, 3]],
            [[1, 1, 1], [3, 1, 2], [2, 2, 2]],
            [[5, 1, 1]],
        ],
    )
    def test_enumerates_exact_cartesian_product_per_block(self, images):
        images = np.array(images)
        total = int(images.prod(axis=1).sum())
        coords = np.asarray(
            _generate_image_offsets(jnp.asarray(images), FixedCapacity(total))
        )
        assert coords.shape == (total, 3)
        cursor = 0
        for w, expected in zip(images, self._blocks(images)):
            n = int(np.prod(w))
            block = coords[cursor : cursor + n]
            got = {tuple(int(c) for c in row) for row in block}
            assert got == expected, f"block width {w.tolist()}"
            # Center-first: the (w-1)//2 element leads each block.
            center = tuple(int((wi - 1) // 2) for wi in w)
            assert tuple(int(c) for c in block[0]) == center
            cursor += n

    def test_padding_rows_are_appended_not_interleaved(self):
        images = np.array([[2, 2, 1]])  # 4 real rows
        coords = np.asarray(
            _generate_image_offsets(jnp.asarray(images), FixedCapacity(8))
        )
        assert coords.shape == (8, 3)
        real = {tuple(int(c) for c in row) for row in coords[:4]}
        assert real == self._blocks(images)[0]


# ============================================================================ #
# Layer 3: replicate_for_images vs an exact NumPy oracle (no distance-boundary
# fuzz — the emitted window is compared to the in-range set as exact integers,
# and the flag to the exact global closest).
# ============================================================================ #
# The superset and min-image-flag tests call this with identical arguments (same
# seeded positions, cell, cutoff), so memoize the result to compute each
# (cell, cutoff, positions) once instead of twice.
_REPLICATE_CACHE: dict = {}
# ``replicate_for_images`` under jit: cutoff and positions are runtime inputs, so
# one compile per (particle count, image count, periodicity) serves every cell,
# cutoff and seed sharing that signature (the capacity must stay static). Layer 3
# uses generic non-integer ratios, so no float32 anchor boundary is in play and
# the jitted result is bit-identical to eager for these inputs.
_REPLICATE_COMPILED: dict = {}


def _replicate_core(n: int, images: int, pbc):
    key = (n, images, tuple(pbc))
    fn = _REPLICATE_COMPILED.get(key)
    if fn is None:
        cap = FixedCapacity(n * n * images)

        @jax.jit
        def fn(lh, systems, cutoffs, gi, gj):
            candidates = Candidates(
                key_idx=Index(lh.keys, gi), query_idx=Index(lh.keys, gj)
            )
            batch = replicate_for_images(candidates, lh, lh, systems, cutoffs, cap)
            return (
                batch.edges.indices.indices,
                batch.edges.shifts[:, 0, :],
                batch.is_minimum_image,
            )

        _REPLICATE_COMPILED[key] = fn
    return fn


def _replicate_edges(frac: np.ndarray, matrix: Matrix, cutoff: float, pbc):
    """Run ``replicate_for_images`` on the full self all-pairs candidate set.

    Returns ``(edge_set, flag_by_pair)`` where ``edge_set`` is every emitted
    ``(i, j, shift)`` (window before distance masking) and ``flag_by_pair`` maps
    each ``(i, j)`` to the list of flagged ``is_minimum_image`` shifts.
    """
    key = (
        frac.tobytes(),
        frac.shape,
        tuple(map(tuple, matrix)),
        round(float(cutoff), 10),
        tuple(pbc),
    )
    if key in _REPLICATE_CACHE:
        return _REPLICATE_CACHE[key]
    n = frac.shape[0]
    cell = _make_cell(matrix, pbc)
    lh = make_lh(jnp.asarray(frac), jnp.zeros(n, dtype=int))
    systems, _ = make_systems(cell, jnp.array([cutoff]))
    images = int(candidate_image_counts(cell, jnp.array([cutoff])).prod())
    gi, gj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    # Exact capacity -> no padding rows, so every emitted row is a real image.
    raw, shifts, is_min = _replicate_core(n, images, pbc)(
        lh,
        systems,
        cutoff_table(jnp.array([cutoff])),
        jnp.asarray(gi.ravel()),
        jnp.asarray(gj.ravel()),
    )
    raw = np.asarray(raw)
    shifts = np.rint(np.asarray(shifts)).astype(int)
    is_min = np.asarray(is_min)

    edge_set: set[Edge] = set()
    flag_by_pair: dict[tuple[int, int], list[Vec3]] = {}
    for (i, j), sh, m in zip(raw, shifts, is_min):
        edge_set.add((int(i), int(j), tuple(int(s) for s in sh)))
        if m:
            flag_by_pair.setdefault((int(i), int(j)), []).append(
                tuple(int(s) for s in sh)
            )
    _REPLICATE_CACHE[key] = (edge_set, flag_by_pair)
    return edge_set, flag_by_pair


_POS_SEEDS = [0, 1, 2]


@pytest.mark.parametrize("cell_name,matrix", CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("seed", _POS_SEEDS)
class TestReplicateForImagesBruteForce:
    n = 5

    def _frac(self, seed: int) -> np.ndarray:
        return np.asarray(jax.random.uniform(jax.random.key(seed), (self.n, 3)))

    def test_window_is_superset_of_in_range_images(
        self, cell_name, matrix, ratio, seed
    ):
        cutoff = _cutoff_for(matrix, ratio)
        frac = self._frac(seed)
        truth = _brute_force(frac, matrix, cutoff, (True, True, True))
        emitted, _ = _replicate_edges(frac, matrix, cutoff, (True, True, True))
        missing = truth - emitted
        assert not missing, (
            f"{cell_name}@{ratio}: {len(missing)} in-range images not emitted"
        )

    def test_min_image_flag_is_global_closest(self, cell_name, matrix, ratio, seed):
        cutoff = _cutoff_for(matrix, ratio)
        frac = self._frac(seed)
        A = np.asarray(matrix, dtype=np.float64)
        _, flags = _replicate_edges(frac, matrix, cutoff, (True, True, True))
        closest = _all_closest_d2(frac, matrix, (True, True, True))
        for i in range(self.n):
            for j in range(self.n):
                copies = flags.get((i, j), [])
                assert len(copies) == 1, f"pair ({i},{j}) flagged {len(copies)} times"
                d2_min = float(closest[i, j])
                # The window provably brackets the global closest only for pairs
                # with an in-range image; out-of-range flags are distance-masked
                # don't-cares. Compare distances, not shift tuples: equidistant
                # images (e.g. exact half-box separations) are a valid tie that
                # kups and the oracle may resolve to different copies.
                if d2_min < cutoff**2:
                    d = ((frac[i] - frac[j]) - np.asarray(copies[0])) @ A
                    assert float(d @ d) == pytest.approx(d2_min, abs=1e-5), (
                        f"pair ({i},{j}) flagged copy is not a global closest"
                    )


@pytest.mark.parametrize(
    "pbc", [(True, True, False), (True, False, False)], ids=["slab", "wire"]
)
@pytest.mark.parametrize("cell_name,matrix", CELLS, ids=CELL_IDS)
class TestReplicatePartialPeriodicity:
    """Open axes must never replicate across a non-periodic face."""

    n = 5
    ratio = 0.95

    def test_window_superset_on_periodic_axes_only(self, cell_name, matrix, pbc):
        cutoff = _cutoff_for(matrix, self.ratio)
        frac = np.asarray(jax.random.uniform(jax.random.key(7), (self.n, 3)))
        truth = _brute_force(frac, matrix, cutoff, pbc)
        emitted, _ = _replicate_edges(frac, matrix, cutoff, pbc)
        assert not (truth - emitted)
        # No emitted shift may be nonzero on an open axis.
        for _, _, sh in emitted:
            for a in range(3):
                if not pbc[a]:
                    assert sh[a] == 0


# ============================================================================ #
# Layer 4: full pipeline (masks, compaction, mirror, single-cell) vs brute
# force, with a boundary guard band (kups masks in float32; the exact
# under-replication direction is already pinned by layer 3's superset test).
# ============================================================================ #
# The neighbor-list matrix (cell) and positions are runtime inputs to the jitted
# pipeline, so one XLA compile serves every geometry that shares the same static
# signature (nl class, particle count, buffer capacities, periodicity, and the
# closed-over cutoff). Caching the jitted callable lets the module-scoped
# ``clear_cache`` fixture reuse compilations across every test in the file —
# without it each call rebuilt a fresh ``jax.jit`` wrapper and recompiled.
_NL_COMPILED: dict = {}


def _compiled_nl(nl_cls, n: int, images: int, bins: int, pbc, cutoff: float):
    key = (nl_cls.__name__, n, images, bins, tuple(pbc), round(float(cutoff), 10))
    fn = _NL_COMPILED.get(key)
    if fn is None:
        edge_cap = FixedCapacity(max(n * images, n))
        image_cap = FixedCapacity(max(n * images, n))
        cutoffs = cutoff_table(jnp.array([cutoff]))
        if nl_cls is CellListNeighborList:
            nl = CellListNeighborList(
                avg_candidates=FixedCapacity(n),
                avg_edges=edge_cap,
                cells=FixedCapacity(max(bins, 1)),
                avg_image_candidates=image_cap,
                cutoffs=cutoffs,
            )
        elif nl_cls is AllDenseNearestNeighborList:
            nl = AllDenseNearestNeighborList(
                avg_edges=edge_cap, avg_image_candidates=image_cap, cutoffs=cutoffs
            )
        else:
            nl = DenseNearestNeighborList(
                avg_candidates=FixedCapacity(n),
                avg_edges=edge_cap,
                avg_image_candidates=image_cap,
                cutoffs=cutoffs,
            )
        fn = jax.jit(as_result_function(nl))
        _NL_COMPILED[key] = fn
    return fn


def _nl_edges(
    nl_cls, real: jax.Array, matrix: Matrix, cutoff: float, pbc, n: int
) -> set[Edge]:
    """Run a neighbor list and return its ``(i, j, integer-shift)`` edge set."""
    cell = _make_cell(matrix, pbc)
    lh = make_lh(jnp.asarray(real), jnp.zeros(n, dtype=int))
    systems, _ = make_systems(cell, jnp.array([cutoff]))
    images = int(candidate_image_counts(cell, jnp.array([cutoff])).prod())
    bins = int(np.asarray(num_cells(systems.data, jnp.array([cutoff])).prod()))
    result = _compiled_nl(nl_cls, n, images, bins, pbc, cutoff)(
        keys=lh, systems=systems
    )
    result.raise_assertion()
    edges = result.value
    raw = np.asarray(edges.indices.indices)
    shifts = np.rint(np.asarray(edges.shifts[:, 0, :])).astype(int)
    keep = (raw[:, 0] < n) & (raw[:, 1] < n)
    return {
        (int(i), int(j), tuple(int(s) for s in sh))
        for (i, j), sh in zip(raw[keep], shifts[keep])
    }


def _assert_matches_with_band(
    got: set[Edge],
    frac: np.ndarray,
    matrix: Matrix,
    cutoff: float,
    pbc,
    band: float = 2e-3,
) -> None:
    """Compare an emitted edge set to brute force, ignoring a thin distance band
    around the cutoff (float32 vs float64 rounding). Clearly-inside edges must
    all be present; clearly-outside edges must all be absent."""
    f, A = np.asarray(frac, np.float64), np.asarray(matrix, np.float64)

    def dist(edge: Edge) -> float:
        i, j, n = edge
        d = ((f[i] - f[j]) - np.asarray(n)) @ A
        return float(np.sqrt(d @ d))

    truth = _brute_force(frac, matrix, cutoff, pbc)
    inside = {e for e in truth if dist(e) < cutoff * (1 - band)}
    missing = inside - got
    assert not missing, f"missing clearly-inside edges: {sorted(missing)[:5]}"
    extra = {e for e in got if dist(e) > cutoff * (1 + band)}
    assert not extra, f"extra clearly-outside edges: {sorted(extra)[:5]}"


# Curated end-to-end matrix: the nastiest geometries at the mild and deep
# replication regimes. Kept small because each entry is a JIT compile.
_E2E_CELLS = [
    c
    for c in CELLS
    if c[0] in {"ortho_cubic", "ortho_aniso", "acute35", "triclinic", "sheared"}
]
_E2E_RATIOS = [0.63, 1.19]


@pytest.mark.parametrize("cell_name,matrix", _E2E_CELLS, ids=[c[0] for c in _E2E_CELLS])
@pytest.mark.parametrize("ratio", _E2E_RATIOS)
class TestEndToEndDenseBruteForce:
    n = 6

    def _run(self, matrix, ratio, seed=0, pbc=(True, True, True)):
        cutoff = _cutoff_for(matrix, ratio)
        frac = np.asarray(jax.random.uniform(jax.random.key(seed), (self.n, 3)))
        real = jnp.asarray(frac) @ jnp.asarray(matrix)
        got = _nl_edges(DenseNearestNeighborList, real, matrix, cutoff, pbc, self.n)
        return frac, cutoff, got

    def test_matches_brute_force(self, cell_name, matrix, ratio):
        frac, cutoff, got = self._run(matrix, ratio)
        _assert_matches_with_band(got, frac, matrix, cutoff, (True, True, True))

    def test_edge_set_is_symmetric(self, cell_name, matrix, ratio):
        # Full self-list: (i, j, n) present iff (j, i, -n) present.
        _, _, got = self._run(matrix, ratio)
        mirror = {(j, i, tuple(-c for c in n)) for i, j, n in got}
        assert got == mirror

    def test_deterministic(self, cell_name, matrix, ratio):
        assert self._run(matrix, ratio)[2] == self._run(matrix, ratio)[2]

    def test_translation_invariance_unfolded_positions(self, cell_name, matrix, ratio):
        # Shifting an atom by a whole lattice vector must not change the physical
        # neighbor set. Positions become unfolded (fractional outside [0, 1)),
        # exercising a large anchor; the emitted edge set must still match brute
        # force computed from the same unfolded fractional coordinates.
        cutoff = _cutoff_for(matrix, ratio)
        frac = np.array(
            jax.random.uniform(jax.random.key(3), (self.n, 3))
        )  # writable copy
        frac[0] += np.array([2.0, -1.0, 3.0])  # translate atom 0 by (2,-1,3) cells
        real = jnp.asarray(frac) @ jnp.asarray(matrix)
        got = _nl_edges(
            DenseNearestNeighborList, real, matrix, cutoff, (True, True, True), self.n
        )
        _assert_matches_with_band(got, frac, matrix, cutoff, (True, True, True))


@pytest.mark.parametrize(
    "pbc",
    [(True, True, False), (True, False, False), (False, False, False)],
    ids=["slab", "wire", "vacuum"],
)
class TestEndToEndPartialPeriodicity:
    n = 6
    matrix = [[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.8, 0.5, 4.0]]

    def test_matches_brute_force(self, pbc):
        cutoff = _cutoff_for(self.matrix, 0.9)
        # Keep positions inside the box so open axes are well-defined.
        frac = np.asarray(jax.random.uniform(jax.random.key(5), (self.n, 3)))
        real = jnp.asarray(frac) @ jnp.asarray(self.matrix)
        got = _nl_edges(
            DenseNearestNeighborList, real, self.matrix, cutoff, pbc, self.n
        )
        _assert_matches_with_band(got, frac, self.matrix, cutoff, pbc)


@pytest.mark.parametrize(
    "nl_cls",
    [CellListNeighborList, AllDenseNearestNeighborList],
    ids=["cell_list", "all_dense"],
)
@pytest.mark.parametrize(
    "cell_name,matrix",
    [("ortho_cubic", CELLS[0][1]), ("acute35", CELLS[3][1])],
    ids=["ortho_cubic", "acute35"],
)
@pytest.mark.parametrize("ratio", _E2E_RATIOS)
class TestEndToEndOtherSelectors:
    n = 6

    def test_matches_brute_force(self, nl_cls, cell_name, matrix, ratio):
        cutoff = _cutoff_for(matrix, ratio)
        frac = np.asarray(jax.random.uniform(jax.random.key(11), (self.n, 3)))
        real = jnp.asarray(frac) @ jnp.asarray(matrix)
        got = _nl_edges(nl_cls, real, matrix, cutoff, (True, True, True), self.n)
        _assert_matches_with_band(got, frac, matrix, cutoff, (True, True, True))


# ============================================================================ #
# cell_list single-cell fast path (max_num_cells.size == 1) vs the general
# stencil path, across geometries and cutoffs.
# ============================================================================ #
@pytest.mark.parametrize("cell_name,matrix", CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("ratio", [0.55, 0.8, 1.2])
class TestCellListSingleCellPath:
    n = 5

    def _candidate_pairs(self, max_cells, matrix, cutoff):
        frac = np.asarray(jax.random.uniform(jax.random.key(9), (self.n, 3)))
        lh = make_lh(jnp.asarray(frac), jnp.zeros(self.n, dtype=int))
        systems, _ = make_systems(_make_cell(matrix), jnp.array([cutoff]))
        c = _cell_list_subselect(
            lh,
            lh,
            systems,
            cutoffs=jnp.array([cutoff]),
            max_num_cells=max_cells,
            max_num_candidates=FixedCapacity(self.n * self.n * 2),
        )
        return {
            (int(a), int(b))
            for a, b in zip(c.key_idx.indices.tolist(), c.query_idx.indices.tolist())
            if a < self.n and b < self.n
        }

    def test_single_cell_matches_multi_cell_and_is_all_pairs(
        self, cell_name, matrix, ratio
    ):
        # cutoff > max(face)/2 -> exactly 1 bin/axis, so the single-cell fast path
        # applies. It must emit the same all-pairs candidate set as forcing a
        # larger cell capacity (the general stencil + dedup path).
        cutoff = float(ratio * _face_lengths(matrix).max())
        systems, _ = make_systems(_make_cell(matrix), jnp.array([cutoff]))
        bins = int(np.asarray(num_cells(systems.data, jnp.array([cutoff])).prod()))
        assert bins == 1, (
            f"{cell_name}@{ratio}: expected a single cell, got {bins} bins"
        )
        single = self._candidate_pairs(FixedCapacity(1), matrix, cutoff)
        multi = self._candidate_pairs(FixedCapacity(64), matrix, cutoff)
        assert single == multi
        assert single == {(a, b) for a in range(self.n) for b in range(self.n)}


# ============================================================================ #
# float32 anchor boundary: the anchor ceil(separation - ratio) is evaluated in
# float32, so it can round off by one exactly when (separation - ratio) lands
# within a float32 ULP of an integer -- only at (near-)integer ratios with tiny
# separations. This test pins the practical guarantee: any image the f32 anchor
# drops sits within a float32 ULP of the cutoff distance (a neighbor the f32
# DistanceCutoffMask treats ambiguously anyway), so no image with a distance
# safely below the cutoff is ever missed.
# ============================================================================ #
# Cutoff and positions are runtime inputs to the jitted replication, so one XLA
# compile per distinct image-count serves every (ratio, separation). Running it
# under jit also matches production, where ``replicate_for_images`` is jitted.
_XWINDOW_COMPILED: dict = {}


def _x_window_core(images: int):
    fn = _XWINDOW_COMPILED.get(images)
    if fn is None:
        cap = FixedCapacity(images)

        @jax.jit
        def fn(lh, systems, cutoffs):
            candidates = Candidates(
                key_idx=Index(lh.keys, jnp.array([1])),
                query_idx=Index(lh.keys, jnp.array([0])),
            )
            batch = replicate_for_images(candidates, lh, lh, systems, cutoffs, cap)
            return batch.edges.shifts[:, 0, :]

        _XWINDOW_COMPILED[images] = fn
    return fn


def _x_window(box: float, cutoff: float, sep: float) -> set[int]:
    """Emitted x-axis integer shifts for one pair separated by ``sep`` in x
    (cubic box, so perp = box). Runs the real float32 ``replicate_for_images``."""
    cell = PeriodicCell(TriclinicFrame.from_matrix(jnp.eye(3)[None] * box))
    frac = jnp.array([[0.0, 0.3, 0.7], [sep, 0.3, 0.7]])  # query, key
    lh = make_lh(frac, jnp.zeros(2, dtype=int))
    systems, _ = make_systems(cell, jnp.array([cutoff]))
    images = int(candidate_image_counts(cell, jnp.array([cutoff])).prod())
    shifts = _x_window_core(images)(lh, systems, cutoff_table(jnp.array([cutoff])))
    return {int(s[0]) for s in np.rint(np.asarray(shifts)).astype(int)}


class TestFloat32AnchorBoundary:
    # ``sep`` includes the adversarial negatives an independent audit flagged as
    # a window-level superset "miss" (e.g. -1.9999999 at ratio 1.0): float32
    # rounds ``separation - ratio`` onto an integer and the anchor lands one too
    # low. These are the exact reproducers, pinned to stay benign.
    @pytest.mark.parametrize("ratio", [1.0, 2.0, 3.0, 5.0, 0.9999, 2.0001])
    @pytest.mark.parametrize(
        "sep", [1e-7, -1e-7, 1.9999999, -1.9999999, 0.9999999, -0.9999999, 0.5, 1e-4]
    )
    def test_only_within_ulp_of_cutoff_may_be_dropped(self, ratio, sep):
        box = 9.0
        cutoff = ratio * box
        emitted = _x_window(box, cutoff, sep)
        # In-range x-images by exact (f64) arithmetic.
        reach = int(np.ceil(ratio)) + 2
        for n in range(-reach, reach + 1):
            d = abs(sep - n) * box  # real distance (cubic, x only)
            if d >= cutoff:
                continue  # out of range: absence is expected, presence harmless
            if n not in emitted:
                # A dropped in-range image must be within a float32 ULP of the
                # cutoff sphere (benign: the f32 distance mask drops it too).
                assert (cutoff - d) / cutoff < 5e-6, (
                    f"ratio={ratio} sep={sep}: dropped image n={n} at "
                    f"dist={d} is not within a float32 ULP of cutoff={cutoff}"
                )
            if d < cutoff * (1 - 1e-4):
                # The practical guarantee: everything safely inside is emitted.
                assert n in emitted, (
                    f"ratio={ratio} sep={sep}: image n={n} at dist={d} is safely "
                    f"inside the cutoff but was not emitted"
                )

    @pytest.mark.parametrize("box,cutoff", [(9.0, 9.0), (9.0, 27.0), (4.0, 8.0)])
    @pytest.mark.parametrize(
        "xs",
        [
            [0.0, 1.9999999],
            [0.0, -1.9999999],  # the audit's exact reproducer
            [0.0, 1e-7],
            [0.0, 0.9999999],
            [0.1, 2.0999999, 3.9999999],
        ],
        ids=["s2", "s-2", "tiny", "s1", "triple"],
    )
    def test_no_clearly_inside_neighbor_dropped_end_to_end(self, box, cutoff, xs):
        # The window-level miss is NOT observable through the full pipeline: real
        # positions round consistently in float32, both orientations are emitted,
        # and the only affected image sits within a ULP of the cutoff. Compare the
        # DenseNL edge set against an f64 oracle built from the SAME float32-rounded
        # positions the pipeline actually uses; nothing clearly inside may be lost.
        n = len(xs)
        matrix = [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]]
        frac = np.array([[x, 0.13, 0.57] for x in xs])
        real = jnp.asarray(frac) * box
        got = _nl_edges(
            DenseNearestNeighborList, real, matrix, cutoff, (True, True, True), n
        )
        # Oracle from the float32-rounded positions the pipeline actually saw.
        rf = np.asarray(real, dtype=np.float64) / box
        A = np.diag([box, box, box]).astype(float)
        reach = int(np.ceil(cutoff / box)) + 2
        inside = set()
        for i in range(n):
            for j in range(n):
                for a in range(-reach, reach + 1):
                    d = ((rf[i] - rf[j]) - np.array([a, 0, 0])) @ A
                    if 1e-6 < float(np.sqrt(d @ d)) < cutoff * (1 - 1e-4):
                        inside.add((i, j, (a, 0, 0)))
        assert not (inside - got), (
            f"dropped clearly-inside edges: {sorted(inside - got)}"
        )


# ============================================================================ #
# queried_keys self-image dedup: a single atom in a cubic box has only
# self-image neighbors (0, 0, n). Replication emits each pair in both
# orientations (n and -n), so QueriedKeysDedupMask must keep exactly one for
# MirrorPairEdges to restore, or every self-image row is double-counted.
# ============================================================================ #
def _queried_nl_multiset(
    nl_cls, real: jax.Array, matrix: Matrix, cutoff: float, n: int, queried: bool
):
    """Edge multiset ``Counter[(i, j, shift)]`` of a full or queried-keys run."""
    cell = _make_cell(matrix)
    lh = make_lh(jnp.asarray(real), jnp.zeros(n, dtype=int))
    systems, _ = make_systems(cell, jnp.array([cutoff]))
    images = int(candidate_image_counts(cell, jnp.array([cutoff])).prod())
    bins = int(np.asarray(num_cells(systems.data, jnp.array([cutoff])).prod()))
    fn = _compiled_nl(nl_cls, n, images, bins, (True, True, True), cutoff)
    if queried:
        result = fn(
            keys=lh, systems=systems, queried_keys=Index(lh.keys, jnp.arange(n))
        )
    else:
        result = fn(keys=lh, systems=systems)
    result.raise_assertion()
    edges = result.value
    raw = np.asarray(edges.indices.indices)
    shifts = np.rint(np.asarray(edges.shifts[:, 0, :])).astype(int)
    keep = (raw[:, 0] < n) & (raw[:, 1] < n)
    return collections.Counter(
        (int(i), int(j), tuple(int(s) for s in sh))
        for (i, j), sh in zip(raw[keep], shifts[keep])
    )


@pytest.mark.parametrize(
    "nl_cls",
    [DenseNearestNeighborList, CellListNeighborList],
    ids=["dense", "cell_list"],
)
# 2.3 puts the cutoff above twice the lattice length (second-shell images).
@pytest.mark.parametrize("ratio", [1.19, 2.3])
class TestQueriedKeysSingleAtomSelfImages:
    box = 4.0

    def test_each_in_range_self_image_appears_exactly_once(self, nl_cls, ratio):
        matrix: Matrix = (np.eye(3) * self.box).tolist()
        cutoff = ratio * self.box
        real = jnp.array([[1.3, 2.1, 0.7]])
        # Exact truth: one edge per nonzero integer shift with |n| * box < cutoff.
        span = range(-int(np.ceil(ratio)), int(np.ceil(ratio)) + 1)
        expected = collections.Counter(
            (0, 0, n)
            for n in itertools.product(span, span, span)
            if 0.0 < float(np.linalg.norm(n)) < ratio
        )
        for queried in (False, True):
            got = _queried_nl_multiset(nl_cls, real, matrix, cutoff, 1, queried)
            assert got == expected, (
                f"ratio={ratio}, queried={queried}: "
                f"{dict((got - expected) + (expected - got))}"
            )


# ============================================================================ #
# Buffer-size safety: the estimate must never under-provision the replicated
# candidate buffer (which relies on the tight image count).
# ============================================================================ #
class TestParametersEstimateSufficiency:
    @pytest.mark.parametrize("cell_name,matrix", CELLS, ids=CELL_IDS)
    @pytest.mark.parametrize("ratio", RATIOS + [1.6, 2.1])
    def test_image_count_covers_actual_replication(self, cell_name, matrix, ratio):
        # The per-system image product feeding estimate.avg_image_candidates must
        # be >= the actual number of replicated rows per candidate.
        from kups.core.data import Table
        from kups.core.neighborlist.parameters import UniversalNeighborlistParameters
        from kups.core.typing import SystemId

        cutoff = _cutoff_for(matrix, ratio)
        cell = _make_cell(matrix)
        systems, cutoffs = make_systems(cell, jnp.array([cutoff]))
        n_particles = 8
        ppc_table = Table((SystemId(0),), jnp.array([n_particles]))
        params = UniversalNeighborlistParameters.estimate(ppc_table, systems, cutoffs)
        images = int(candidate_image_counts(cell, jnp.array([cutoff])).prod())
        # estimate replicates the rounded candidate buffer by the image product;
        # avg_image_candidates must be at least that product (per candidate).
        assert params.avg_image_candidates >= images
        assert params.avg_image_candidates >= params.avg_candidates
