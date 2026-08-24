# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Reference systems shared by the D3 tests and the reference generator.

Kept free of kUPS imports so the generator script can import it from an
environment that only has ASE and NumPy (see ``generate_reference.py``).
Positions are Å, cells are row-vector lattice matrices.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class System(NamedTuple):
    """A test structure in plain NumPy."""

    numbers: np.ndarray  # (n,) int
    positions: np.ndarray  # (n, 3) float [Å]
    cell: np.ndarray | None  # (3, 3) float [Å], rows are lattice vectors
    pbc: tuple[bool, bool, bool]


def _molecule(numbers: list[int], positions: list[list[float]]) -> System:
    return System(
        np.array(numbers, dtype=int),
        np.array(positions, dtype=float),
        None,
        (False, False, False),
    )


def _build() -> dict[str, System]:
    from ase.build import bulk, molecule

    systems: dict[str, System] = {}

    # --- molecular -------------------------------------------------------
    systems["ar_atom"] = _molecule([18], [[0.0, 0.0, 0.0]])
    systems["ar2"] = _molecule([18, 18], [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    systems["hf"] = _molecule([9, 1], [[0.0, 0.0, 0.0], [0.917, 0.0, 0.0]])

    for name, formula in [("water", "H2O"), ("co2", "CO2"), ("benzene", "C6H6")]:
        atoms = molecule(formula)
        systems[name] = _molecule(
            list(atoms.get_atomic_numbers()), atoms.positions.tolist()
        )

    water = molecule("H2O")
    dimer = water.copy()
    shifted = water.copy()
    shifted.translate([2.9, 0.0, 0.0])
    dimer += shifted
    systems["water_dimer"] = _molecule(
        list(dimer.get_atomic_numbers()), dimer.positions.tolist()
    )

    benzene = molecule("C6H6")
    stacked = benzene.copy()
    stacked.translate([1.6, 0.0, 3.4])  # parallel-displaced, the canonical D3 case
    pair = benzene + stacked
    systems["benzene_dimer"] = _molecule(
        list(pair.get_atomic_numbers()), pair.positions.tolist()
    )

    # --- periodic --------------------------------------------------------
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    systems["si_diamond"] = System(
        np.array(si.get_atomic_numbers()),
        si.positions.copy(),
        si.cell.array.copy(),
        (True, True, True),
    )

    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    systems["nacl"] = System(
        np.array(nacl.get_atomic_numbers()),
        nacl.positions.copy(),
        nacl.cell.array.copy(),
        (True, True, True),
    )

    # a genuinely triclinic cell, to exercise the non-orthogonal to_real path
    sheared = bulk("Si", "diamond", a=5.43, cubic=True)
    shear = np.array([[1.0, 0.0, 0.0], [0.12, 1.0, 0.0], [0.07, 0.09, 1.0]])
    sheared.set_cell(shear @ sheared.cell.array, scale_atoms=True)
    systems["si_sheared"] = System(
        np.array(sheared.get_atomic_numbers()),
        sheared.positions.copy(),
        sheared.cell.array.copy(),
        (True, True, True),
    )

    # --- transition metals ------------------------------------------------
    # The main-group systems above all sit close to a tabulated reference CN.
    # Metals do not, and each of these stresses a different part of the C6
    # interpolation. They are also the elements MOF nodes are built from.
    def _periodic(atoms) -> System:
        return System(
            np.array(atoms.get_atomic_numbers()),
            atoms.positions.copy(),
            atoms.cell.array.copy(),
            (True, True, True),
        )

    # CN ~ 8.7 against references [0, 0.96]: nine times the largest, so the
    # Gaussian weights are pure extrapolation and every reference but the top
    # one underflows. This is where a normalisation that underflows to zero, or
    # a fallback that picks the wrong reference, would show up.
    systems["cu_fcc"] = _periodic(bulk("Cu", "fcc", a=3.61, cubic=True))
    # CN ~ 11.2 against [0, 1.83, 10.62] -- the widest reference gap in the
    # whole table, and the only regime where interpolating across it matters.
    systems["cr_bcc"] = _periodic(bulk("Cr", "bcc", a=2.88, cubic=True))
    # references [0, 1.79, 6.55, 6.29] are *not* sorted; slot order must not
    # leak into the result.
    systems["ni_fcc"] = _periodic(bulk("Ni", "fcc", a=3.52, cubic=True))
    # a metal and a light element in one cell, so the cross-element C6 block is
    # exercised in both index orders -- the block a transpose error would hit.
    systems["zno"] = _periodic(bulk("ZnO", "wurtzite", a=3.25, c=5.20))
    # hcp: a non-orthogonal lattice with a 120 degree angle, distinct from the
    # sheared cubic above. Zr is the UiO-66 node metal.
    systems["zr_hcp"] = _periodic(bulk("Zr", "hcp", a=3.23, c=5.15))

    # a slab: periodic in x/y, open in z
    slab = bulk("Si", "diamond", a=5.43, cubic=True)
    cell = slab.cell.array.copy()
    cell[2, 2] = 20.0
    slab.set_cell(cell)
    systems["si_slab"] = System(
        np.array(slab.get_atomic_numbers()),
        slab.positions.copy(),
        cell,
        (True, True, False),
    )

    return systems


SYSTEMS: dict[str, System] = _build()

MOLECULAR = (
    "ar_atom",
    "ar2",
    "hf",
    "water",
    "water_dimer",
    "co2",
    "benzene",
    "benzene_dimer",
)
PERIODIC = ("si_diamond", "nacl", "si_sheared", "si_slab")
TRANSITION_METAL = ("cu_fcc", "cr_bcc", "ni_fcc", "zno", "zr_hcp")

# Damping parameters used throughout the D3 tests: PBE-D3(BJ),
# Grimme, Ehrlich & Goerigk, J. Comput. Chem. 32 (2011) 1456, doi:10.1002/jcc.21759.
# a2 is quoted in Bohr upstream; tests convert it.
PBE_D3BJ_BOHR = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}

# simple-dftd3's own real-space defaults, in Bohr.
SDFTD3_DISP2_BOHR = 60.0
SDFTD3_CN_BOHR = 40.0
