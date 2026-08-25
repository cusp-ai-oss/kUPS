# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared builders for the application-level Lennard-Jones integration tests.

Single source of truth for the fcc-argon input structure the MD, relaxation and
Verlet-skin smoke tests run on, and for the LBFGS optimizer spec the relaxation
ones relax with.
"""

from __future__ import annotations

import tempfile

import ase.build

from kups.relaxation.config import TransformationConfig

#: LBFGS direction, clamped step length, descent sign — the spec every
#: Lennard-Jones relaxation test relaxes with.
LBFGS_OPTIMIZER: TransformationConfig = [
    {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
    {"transform": "max_step_size", "max_step_size": 0.2},
    {"transform": "scale", "step_size": -1},
]


def ar_cif(rattle: float = 0.0, *, cubic: bool = False) -> str:
    """Write an fcc-argon supercell as a P1 CIF and return the file path.

    Labels are kept uniform (``Ar``) so they match the LJ parameter table;
    ASE's CIF writer would otherwise uniquify them to ``Ar1``, ``Ar2``, ... The
    rattle gives an optimizer nonzero forces to act on.

    Args:
        rattle: Stdev (Å) of the Gaussian displacement applied to every atom;
            ``0`` leaves the ideal lattice.
        cubic: Build the conventional cubic cell (32 atoms, ~10.6 Å box, so
            ``(cutoff + skin) / box`` stays inside the single-image limit for
            the Verlet-skin tests) instead of the fcc primitive one.
    """
    atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=cubic) * (2, 2, 2)
    atoms.rattle(rattle, seed=1)
    a, b, c, al, be, ga = atoms.cell.cellpar()
    rows = "\n".join(
        f"Ar Ar {x:.6f} {y:.6f} {z:.6f}" for x, y, z in atoms.get_scaled_positions()
    )
    cif = (
        f"data_ar\n_cell_length_a {a:.6f}\n_cell_length_b {b:.6f}\n"
        f"_cell_length_c {c:.6f}\n_cell_angle_alpha {al:.6f}\n"
        f"_cell_angle_beta {be:.6f}\n_cell_angle_gamma {ga:.6f}\n"
        "_symmetry_space_group_name_H-M 'P 1'\n_symmetry_Int_Tables_number 1\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        f"_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n{rows}\n"
    )
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.write(cif)
    f.close()
    return f.name


def tmp_h5() -> str:
    """Path of a fresh, closed temporary ``.h5`` file for a run's output."""
    f = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    f.close()
    return f.name
