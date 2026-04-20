# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Physical constants and unit conversions for molecular simulations.

This module provides fundamental physical constants and unit conversion factors
commonly used in molecular dynamics and quantum chemistry simulations. All
constants are sourced from ASE (Atomic Simulation Environment) for consistency.

The default unit system uses:

- Length: Angstrom (Å)
- Energy: electron volt (eV)
- Mass: atomic mass unit (amu)
- Time: derived from the above units

Common conversions:

- 1 eV = 1 (energy unit)
- 1 Å = 1 (length unit)
- Temperature in Kelvin × BOLTZMANN_CONSTANT = energy in eV
"""

import numpy as np

# Constants are taken from ASE
_SPEED_OF_LIGHT = 299792458.0  # m/s
_MU_0 = 4e-7 * np.pi
_GRAV = 6.67408e-11  # m^3/(kg*s^2)
_PLANCK_CONSTANT = 6.626070040e-34  # J*s
_ELECTRON_CHARGE = 1.6021766208e-19  # C
_ELECTRON_MASS = 9.10938356e-31  # kg
_PROTON_MASS = 1.672621898e-27  # kg
_AVOGADRO_NUMBER = 6.022140857e23  # mol^-1
_BOLTZMANN_CONSTANT = 1.38064852e-23  # J/K
_ATOMIC_MASS_UNIT = 1.660539040e-27  # kg


EPSILON_0 = 1 / (_MU_0 * _SPEED_OF_LIGHT**2)  # F/m
HBAR = _PLANCK_CONSTANT / (2 * np.pi)  # J*s

MEGA = 1e6
KILO = 1e3
MILLI = 1e-3
MICRO = 1e-6
NANO = 1e-9
PICO = 1e-12
FEMTO = 1e-15

KELVIN = 1
ANGSTROM = 1
METER = 1e10 * ANGSTROM
BOHR = 4e10 * np.pi * EPSILON_0 * HBAR**2 / _ELECTRON_MASS / _ELECTRON_CHARGE**2
ELECTRON_VOLT = 1
HARTREE = _ELECTRON_MASS * _ELECTRON_CHARGE**3 / 16 / np.pi**2 / EPSILON_0**2 / HBAR**2
JOULE = 1 / _ELECTRON_CHARGE
KILO_JOULE = 1e3 / _ELECTRON_CHARGE
KILO_CALORIE = 4.184 * KILO_JOULE
MOL = _AVOGADRO_NUMBER
RYDBERG = 0.5 * HARTREE

SECOND = 1e10 * (_ELECTRON_CHARGE / _ATOMIC_MASS_UNIT) ** 0.5
FEMTO_SECOND = FEMTO * SECOND

BOLTZMANN_CONSTANT = _BOLTZMANN_CONSTANT / _ELECTRON_CHARGE  # eV/K
GAS_CONSTANT = BOLTZMANN_CONSTANT * MOL  # eV/(mol K)
PASCAL = 1 / (METER**3 * _ELECTRON_CHARGE)
BAR = 1e5 * PASCAL

DEBYE = 1e-11 / _ELECTRON_CHARGE / _SPEED_OF_LIGHT
ALPHA = 100 * _SPEED_OF_LIGHT * _PLANCK_CONSTANT / _ELECTRON_CHARGE
INV_CM = 100 * _SPEED_OF_LIGHT / _PLANCK_CONSTANT / _ELECTRON_CHARGE  # cm^-1 to eV

KILOGRAM = 1.0 / _ATOMIC_MASS_UNIT
AMPERE = 1.0 / (_ELECTRON_CHARGE * SECOND)
COULOMB = 1 / _ELECTRON_CHARGE
