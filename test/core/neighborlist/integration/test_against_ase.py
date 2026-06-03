# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Compare kups neighbor lists against ase.neighborlist as a reference.

Builds canonical structures with ase.build (cubic Cu, hexagonal Mg, a triclinic
single-atom cell, and a 500-atom Cu supercell) and asserts that the directed
edge sets produced by ``CellListNeighborList`` and ``DenseNearestNeighborList``
agree with ``ase.neighborlist.neighbor_list`` across cutoffs that are smaller,
larger, and much larger than the perpendicular cell length.
"""

import ase
import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from ase.neighborlist import neighbor_list as ase_neighbor_list

from kups.core.cell import Cell, PeriodicCell, TriclinicFrame, to_lower_triangular
from kups.core.data.table import Table
from kups.core.neighborlist import (
    CellListNeighborList,
    DenseNearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.result import as_result_function

from .._builders import EvalState, make_lh, make_systems


def _atoms_to_cell(atoms: ase.Atoms) -> Cell:
    """Build a kups ``PeriodicCell`` from an ``ase.Atoms`` whose lattice
    matrix is lower-triangular. Use ``_ensure_lower_triangular`` to bring
    arbitrary ASE inputs onto this branch first."""
    lvecs = jnp.asarray(np.asarray(atoms.cell.array))[None]
    return PeriodicCell(TriclinicFrame.from_matrix(lvecs))


def _ensure_lower_triangular(atoms: ase.Atoms) -> ase.Atoms:
    """Rigidly rotate ``atoms`` so its lattice matrix is lower-triangular.

    ``TriclinicFrame.from_matrix`` projects onto the lower-triangular block
    (a documented gradient-wrapping behaviour), so an ASE lattice that is
    not already lower-triangular would be silently truncated. Rotating up
    front via the same QR reduction kups uses internally
    (``kups.core.cell.to_lower_triangular``) puts the input on the supported
    branch and is a no-op for cells that are already lower-triangular.
    """
    cell = jnp.asarray(np.asarray(atoms.cell.array))
    L, uc_transform = to_lower_triangular(cell)
    rotated_positions = np.asarray(
        uc_transform(jnp.asarray(np.asarray(atoms.positions)))
    )
    out = atoms.copy()
    out.set_cell(np.asarray(L), scale_atoms=False)
    out.set_positions(rotated_positions)
    return out


def _build_state(atoms: ase.Atoms, cutoff: float) -> tuple[EvalState, Table]:
    """Construct an ``EvalState`` from an ``ase.Atoms`` and a scalar cutoff."""
    positions = jnp.asarray(np.asarray(atoms.get_positions()))
    particles = make_lh(positions, jnp.zeros(len(atoms), dtype=int))
    systems, cutoff_table = make_systems(
        _atoms_to_cell(atoms), jnp.array([float(cutoff)])
    )
    particles_per_system = Table(systems.keys, jnp.array([len(atoms)]))
    nl_params = UniversalNeighborlistParameters.estimate(
        particles_per_system, systems, cutoff_table
    )
    state = EvalState(
        particles=particles, systems=systems, neighborlist_params=nl_params
    )
    return state, cutoff_table


def _kups_edges(nl_cls, atoms: ase.Atoms, cutoff: float):
    """Run an NL built from an ``EvalState`` via ``from_state``, growing its
    LensCapacity-backed capacities through ``fix_or_raise`` until no assertion
    fails. Returns ``(i, j, displacement)`` for valid (non-padding) directed edges.
    """
    atoms = _ensure_lower_triangular(atoms)
    n = len(atoms)
    state, cutoff_table = _build_state(atoms, cutoff)

    result = None
    for _ in range(2):
        nl = nl_cls.from_state(state, cutoff_table)
        result = jax.jit(as_result_function(nl))(
            lh=state.particles,
            systems=state.systems,
        )
        if not result.failed_assertions:
            break
        state = result.fix_or_raise(state)
    assert result is not None
    result.raise_assertion()

    edges = result.value
    raw = np.asarray(edges.indices.indices)
    disp = np.asarray(edges.difference_vectors(state.particles, state.systems))[:, 0, :]
    valid = (raw[:, 0] < n) & (raw[:, 1] < n)
    return raw[valid, 0], raw[valid, 1], disp[valid]


def _ase_edges(atoms: ase.Atoms, cutoff: float):
    """Reference directed edges from ase.neighborlist as (i, j, displacement)."""
    atoms = _ensure_lower_triangular(atoms)
    i, j, S = ase_neighbor_list("ijS", atoms, float(cutoff))
    pos = np.asarray(atoms.get_positions())
    cell = np.asarray(atoms.cell.array)
    disp = pos[j] - pos[i] + S @ cell
    return np.asarray(i), np.asarray(j), disp


def _canonical(i, j, disp, atol: float = 1e-3):
    """Sorted multiset of (i, j, quantized-displacement) tuples.

    Quantizing the real-space displacement vector lets us compare two neighbor
    lists without relying on the shift-integer / lattice-vector orientation
    being identical between ase and kups.
    """
    rounded = np.round(disp / atol).astype(np.int64)
    return sorted(
        (int(a), int(b), int(r[0]), int(r[1]), int(r[2]))
        for a, b, r in zip(i, j, rounded)
    )


def _make_cubic_cu() -> ase.Atoms:
    return ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)


def _make_hcp_mg() -> ase.Atoms:
    return ase.build.bulk("Mg", "hcp", a=3.2, c=5.2)


def _make_triclinic() -> ase.Atoms:
    return ase.Atoms(
        "Cu",
        positions=[[0.0, 0.0, 0.0]],
        cell=[[5.0, 0.0, 0.0], [1.5, 4.5, 0.0], [0.7, 0.4, 5.2]],
        pbc=True,
    )


def _make_cu_supercell() -> ase.Atoms:
    """Interior-bulk fcc Cu supercell for the minimum-image, multi-shell regime.

    A 3x3x3 cubic supercell (108 atoms, box 10.8 A) is the smallest repeat that
    reproduces the *relative* conditions that matter here, rather than raw size:
    at cutoff 5.0 A the ratio to the half-box is 5.0/5.4 < 1 (so neighbors are
    found by the minimum image, with no periodic image replication), the cutoff
    spans ~3 coordination shells (2.55, 3.6, 4.41 A), and the central cell's
    neighbors all come from adjacent cells. A 2x2x2 box (7.2 A) would fall below
    2*cutoff and flip into the replication regime, changing the test.
    """
    return ase.build.bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3))


def _make_fcc_primitive_222() -> ase.Atoms:
    """FCC primitive 2x2x2 — non-lower-triangular ASE lattice, exercises the
    cell-list path at ``cutoff > perp`` (perpendicular cell width ~4.17 A)."""
    return ase.build.bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))


def _build_bulk_al() -> ase.Atoms:
    at = ase.build.bulk("Al", "fcc", a=4.05, cubic=True)
    at.rattle(0.5)
    at.wrap()
    return at


# Cutoff sweeps are chosen by the *relative* regime they exercise, not absolute
# size. Cost is dominated by the periodic-image replication depth,
# ceil(cutoff / perpendicular_length): each axis emits (2*ceil+1) images, so a
# 3-shell cutoff replicates a (multi-atom) candidate set ~343x. Because the
# replication / stencil code (``candidate_image_counts``,
# ``_generate_image_offsets``) is geometry-agnostic, the deepest (3-shell)
# regime only needs covering once — and the cheapest place is the 1-atom,
# non-orthogonal ``triclinic`` cell (also the hardest geometry). The multi-atom
# crystals therefore stop at their 2-shell cutoff; deeper cutoffs there were
# pure cost, not new coverage.
_CASES = [
    ("cubic_Al", _build_bulk_al, (6.0,)),  # rattled fcc, 2-shell replication
    ("cubic_Cu", _make_cubic_cu, (2.5, 5.0)),  # 1- and 2-shell
    ("hcp_Mg", _make_hcp_mg, (2.5, 5.0)),  # 1- and 2-shell
    ("triclinic", _make_triclinic, (3.0, 6.0, 12.0)),  # deep 3-shell guard (1 atom)
    ("Cu_3x3x3", _make_cu_supercell, (5.0,)),  # large interior-bulk, minimum-image
    ("fcc_primitive_222", _make_fcc_primitive_222, (3.0, 6.0)),  # non-lower-triangular
]

# Flatten (system, cutoff) into one item per cutoff so the xdist scheduler can
# distribute the expensive per-cutoff compilations across workers instead of
# serializing a whole cutoff sweep inside a single test.
_FLAT_CASES = [
    (name, builder, cutoff) for name, builder, cutoffs in _CASES for cutoff in cutoffs
]


class TestEnsureLowerTriangular:
    """``_ensure_lower_triangular`` is a rigid QR rotation that brings the
    ASE lattice onto the lower-triangular branch required by kups'
    ``TriclinicFrame``. Without pre-rotating, ``TriclinicFrame.from_matrix``
    silently *projects* the arbitrary basis matrix onto its lower-triangular
    block (a documented behaviour used elsewhere for ``∂E/∂h`` gradient
    wrapping), which for FCC primitive ASE conventions corrupts the cell into
    a degenerate basis.
    """

    def test_rotates_fcc_primitive_to_lower_triangular(self):
        atoms = ase.build.bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))
        npt.assert_(
            not np.allclose(np.triu(np.asarray(atoms.cell.array), k=1), 0.0),
            "FCC primitive ASE cell is expected to be non-lower-triangular",
        )
        rotated = _ensure_lower_triangular(atoms)
        upper = np.triu(np.asarray(rotated.cell.array), k=1)
        npt.assert_allclose(upper, 0.0, atol=1e-5)

    def test_no_op_on_already_lower_triangular(self):
        atoms = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
        rotated = _ensure_lower_triangular(atoms)
        npt.assert_allclose(
            np.asarray(rotated.cell.array), np.asarray(atoms.cell.array), atol=1e-6
        )
        npt.assert_allclose(
            np.asarray(rotated.positions), np.asarray(atoms.positions), atol=1e-6
        )

    def test_preserves_volume(self):
        atoms = ase.build.bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))
        v_before = abs(np.linalg.det(np.asarray(atoms.cell.array)))
        rotated = _ensure_lower_triangular(atoms)
        v_after = abs(np.linalg.det(np.asarray(rotated.cell.array)))
        npt.assert_allclose(v_after, v_before, rtol=1e-6)

    def test_preserves_pairwise_distances(self):
        atoms = ase.build.bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))
        rotated = _ensure_lower_triangular(atoms)
        p0 = np.asarray(atoms.positions)
        p1 = np.asarray(rotated.positions)
        d0 = np.linalg.norm(p0[:, None] - p0[None, :], axis=-1)
        d1 = np.linalg.norm(p1[:, None] - p1[None, :], axis=-1)
        npt.assert_allclose(d1, d0, atol=1e-5)

    def test_preserves_periodic_image_distances(self):
        atoms = ase.build.bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))
        rotated = _ensure_lower_triangular(atoms)
        i0, j0, S0 = ase_neighbor_list("ijS", atoms, 6.0)
        i1, j1, S1 = ase_neighbor_list("ijS", rotated, 6.0)
        d0 = np.linalg.norm(
            np.asarray(atoms.positions)[j0]
            - np.asarray(atoms.positions)[i0]
            + S0 @ np.asarray(atoms.cell.array),
            axis=-1,
        )
        d1 = np.linalg.norm(
            np.asarray(rotated.positions)[j1]
            - np.asarray(rotated.positions)[i1]
            + S1 @ np.asarray(rotated.cell.array),
            axis=-1,
        )
        npt.assert_allclose(np.sort(d0), np.sort(d1), atol=1e-5)

    def test_atoms_to_cell_produces_finite_perpendicular_lengths(self):
        atoms = ase.build.bulk("Cu", "fcc", a=3.61).repeat((2, 2, 2))
        rotated = _ensure_lower_triangular(atoms)
        cell = _atoms_to_cell(rotated)
        perp = np.asarray(cell.perpendicular_lengths)
        npt.assert_(np.all(np.isfinite(perp)), f"perp not finite: {perp}")
        npt.assert_(np.all(perp > 0), f"perp not strictly positive: {perp}")


@pytest.mark.parametrize(
    "case", _FLAT_CASES, ids=[f"{n}@{c}" for n, _, c in _FLAT_CASES]
)
@pytest.mark.parametrize(
    "nl_cls",
    [DenseNearestNeighborList, CellListNeighborList],
    ids=lambda cls: cls.__name__,
)
class TestNeighborListAgainstASE:
    """kups neighbor lists must agree with ase.neighborlist on every (system,
    cutoff) pair we test. Cutoffs cover smaller-than, larger-than, and
    much-larger-than the perpendicular cell length."""

    def test_matches_ase(self, case, nl_cls):
        name, builder, cutoff = case
        atoms = builder()
        ki, kj, kdisp = _kups_edges(nl_cls, atoms, cutoff)
        ai, aj, adisp = _ase_edges(atoms, cutoff)

        kups_canon = _canonical(ki, kj, kdisp)
        ase_canon = _canonical(ai, aj, adisp)

        assert kups_canon == ase_canon, (
            f"{name} @ cutoff={cutoff} {nl_cls.__name__}: "
            f"kups produced {len(ki)} directed edges, "
            f"ase produced {len(ai)} — displacement multisets differ"
        )
