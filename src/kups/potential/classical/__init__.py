# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Classical molecular mechanics force fields.

This module provides standard force field components used in molecular simulations:
non-bonded interactions (Lennard-Jones, Coulomb, Ewald) and bonded terms (harmonic
bonds, angles, dihedrals). All potentials support tail corrections, cutoffs, and
neighbor lists.

## Available Potentials

- **[Lennard-Jones][kups.potential.classical.lennard_jones]**: Van der Waals interactions with optional tail corrections
- **[Coulomb][kups.potential.classical.coulomb]**: Electrostatic interactions
- **[Ewald][kups.potential.classical.ewald]**: Long-range electrostatics via Ewald summation
- **[Harmonic][kups.potential.classical.harmonic]**: Bonded interactions (bonds, angles)
- **[Cosine Angle][kups.potential.classical.cosine_angle]**: UFF-style cosine angle bending
- **[Morse][kups.potential.classical.morse]**: Anharmonic bond stretching with proper dissociation
- **[Dihedral][kups.potential.classical.dihedral]**: Torsion potentials (UFF-style)
- **[Inversion][kups.potential.classical.inversion]**: Out-of-plane/improper potentials (UFF-style)

Each potential provides a `make_*_potential` factory function that constructs a
configured [Potential][kups.core.potential.Potential] instance.
"""

from .cosine_angle import (
    CosineAngleParameters,
    cosine_angle_energy,
    make_cosine_angle_potential,
)
from .dihedral import DihedralParameters, make_dihedral_potential
from .ewald import EwaldParameters, make_ewald_potential
from .harmonic import (
    HarmonicAngleParameters,
    HarmonicBondParameters,
    make_harmonic_angle_potential,
    make_harmonic_bond_potential,
)
from .inversion import (
    InversionParameters,
    inversion_energy,
    make_inversion_potential,
)
from .lennard_jones import (
    GlobalTailCorrectedLennardJonesParameters,
    LennardJonesParameters,
    PairTailCorrectedLennardJonesParameters,
    make_global_lennard_jones_tail_correction_potential,
    make_lennard_jones_potential,
    make_pair_tail_corrected_lennard_jones_potential,
)
from .morse import MorseBondParameters, make_morse_bond_potential

__all__ = [
    "make_cosine_angle_potential",
    "make_dihedral_potential",
    "make_ewald_potential",
    "make_harmonic_angle_potential",
    "make_harmonic_bond_potential",
    "make_inversion_potential",
    "make_morse_bond_potential",
    "make_lennard_jones_potential",
    "make_global_lennard_jones_tail_correction_potential",
    "make_pair_tail_corrected_lennard_jones_potential",
    "CosineAngleParameters",
    "cosine_angle_energy",
    "DihedralParameters",
    "EwaldParameters",
    "HarmonicAngleParameters",
    "HarmonicBondParameters",
    "InversionParameters",
    "inversion_energy",
    "MorseBondParameters",
    "LennardJonesParameters",
    "GlobalTailCorrectedLennardJonesParameters",
    "PairTailCorrectedLennardJonesParameters",
]
