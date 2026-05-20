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
import pytest
from ase.neighborlist import neighbor_list as ase_neighbor_list

from kups.core.cell import Cell, PeriodicCell, TriclinicFrame
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.neighborlist import (
    CellListNeighborList,
    DenseNearestNeighborList,
    UniversalNeighborlistParameters,
)
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass

from ..clear_cache import clear_cache  # noqa: F401


@dataclass
class _SamplePoints:
    positions: jax.Array
    system: Index
    inclusion: Index
    exclusion: Index


@dataclass
class _SampleSystems:
    cell: Cell


def _make_lh(positions: jax.Array) -> Table:
    n = len(positions)
    sys_keys = (0,)
    pi_keys = tuple(ParticleId(i) for i in range(n))
    batch = jnp.zeros(n, dtype=int)
    return Table(
        pi_keys,
        _SamplePoints(
            positions=positions,
            system=Index(sys_keys, batch),
            inclusion=Index(sys_keys, batch),
            exclusion=Index.integer(jnp.arange(n, dtype=int)),
        ),
    )


def _make_systems(cell: Cell, cutoff: float) -> tuple[Table, Table]:
    sys_keys = (SystemId(0),)
    systems = Table(sys_keys, _SampleSystems(cell=cell))
    cutoffs = Table(sys_keys, jnp.array([float(cutoff)]))
    return systems, cutoffs


def _atoms_to_cell(atoms: ase.Atoms) -> Cell:
    lvecs = jnp.asarray(np.asarray(atoms.cell.array))[None]
    return PeriodicCell(TriclinicFrame.from_matrix(lvecs))


@dataclass
class _EvalState:
    """Minimal state carrying particles, systems, and neighbor-list capacity
    hints, matching the ``IsNeighborListState`` protocol so ``from_state``
    builds an NL with ``LensCapacity`` fields wired into ``neighborlist_params``.
    """

    particles: Table
    systems: Table
    neighborlist_params: UniversalNeighborlistParameters


def _build_state(atoms: ase.Atoms, cutoff: float) -> tuple[_EvalState, Table]:
    """Construct ``_EvalState`` from an ``ase.Atoms`` and a scalar cutoff."""
    positions = jnp.asarray(np.asarray(atoms.get_positions()))
    particles = _make_lh(positions)
    systems, cutoff_table = _make_systems(_atoms_to_cell(atoms), cutoff)
    particles_per_system = Table(systems.keys, jnp.array([len(atoms)]))
    nl_params = UniversalNeighborlistParameters.estimate(
        particles_per_system, systems, cutoff_table
    )
    state = _EvalState(
        particles=particles, systems=systems, neighborlist_params=nl_params
    )
    return state, cutoff_table


def _kups_edges(nl_cls, atoms: ase.Atoms, cutoff: float):
    """Run an NL built from a ``_EvalState`` via ``from_state``, growing its
    LensCapacity-backed capacities through ``fix_or_raise`` until no assertion
    fails. Returns ``(i, j, displacement)`` for valid (non-padding) directed
    edges.
    """
    n = len(atoms)
    state, cutoff_table = _build_state(atoms, cutoff)

    # Loop with a generous bound to avoid pathological growth.
    for _ in range(2):
        nl = nl_cls.from_state(state)
        result = jax.jit(as_result_function(nl))(
            lh=state.particles,
            rh=None,
            systems=state.systems,
            cutoffs=cutoff_table,
            rh_index_remap=None,
        )
        if not result.failed_assertions:
            break
        state = result.fix_or_raise(state)
    result.raise_assertion()

    edges = result.value
    raw = np.asarray(edges.indices.indices)
    disp = np.asarray(edges.difference_vectors(state.particles, state.systems))[:, 0, :]
    valid = (raw[:, 0] < n) & (raw[:, 1] < n)
    return raw[valid, 0], raw[valid, 1], disp[valid]


def _ase_edges(atoms: ase.Atoms, cutoff: float):
    """Reference directed edges from ase.neighborlist as (i, j, displacement)."""
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
    )  # .repeat((2,2,2))


def _make_cu_supercell() -> ase.Atoms:
    return ase.build.bulk("Cu", "fcc", a=3.6, cubic=True).repeat((5, 5, 5))


def _build_bulk_al() -> ase.Atoms:
    at = ase.build.bulk("Al", "fcc", a=4.05, cubic=True)
    at.rattle(0.5)
    at.wrap()
    return at


_CASES = [
    ("cubic_Al", _build_bulk_al, (6.0, 12.0)),
    ("cubic_Cu", _make_cubic_cu, (2.5, 5.0, 10.0)),
    ("hcp_Mg", _make_hcp_mg, (2.5, 5.0, 10.0)),
    ("triclinic", _make_triclinic, (3.0, 6.0, 12.0)),
    ("Cu_5x5x5", _make_cu_supercell, (5.0,)),
]


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[c[0] for c in _CASES],
)
@pytest.mark.parametrize(
    "nl_cls",
    [DenseNearestNeighborList, CellListNeighborList],
    ids=lambda cls: cls.__name__,
)
class TestNeighborListAgainstASE:
    """kups neighbor lists must agree with ase.neighborlist on every (system,
    cutoff) pair we test. Cutoff sweep covers smaller-than, larger-than, and
    much-larger-than the perpendicular cell length."""

    def test_matches_ase(self, case, nl_cls):
        name, builder, cutoffs = case
        atoms = builder()
        for cutoff in cutoffs:
            ki, kj, kdisp = _kups_edges(nl_cls, atoms, cutoff)
            ai, aj, adisp = _ase_edges(atoms, cutoff)

            kups_canon = _canonical(ki, kj, kdisp)
            ase_canon = _canonical(ai, aj, adisp)

            assert kups_canon == ase_canon, (
                f"{name} @ cutoff={cutoff} {nl_cls.__name__}: "
                f"kups produced {len(ki)} directed edges, "
                f"ase produced {len(ai)} — displacement multisets differ"
            )
