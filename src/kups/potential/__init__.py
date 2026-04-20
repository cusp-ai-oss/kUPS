# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Potential energy implementations for molecular simulations.

This package provides energy functions for various interaction types in molecular
systems. All potentials implement the [Potential][kups.core.potential.Potential]
protocol and can be composed using [sum_potentials][kups.core.potential.sum_potentials].

## Submodules

- **[classical][kups.potential.classical]**: Classical force fields (Lennard-Jones, Coulomb, Ewald, harmonic bonds/angles)
- **[mliap][kups.potential.mliap]**: Machine learning interatomic potentials (MACE)
- **[common][kups.potential.common]**: Shared utilities for graph construction and energy computation

## Usage Pattern

```python
from kups.core.potential import sum_potentials
from kups.potential.classical import make_lennard_jones_potential, make_ewald_potential
from kups.potential.mliap.interface import make_mliap_potential

# Compose multiple potentials
total_potential = sum_potentials(
    make_lennard_jones_potential(...),
    make_ewald_potential(...),
    make_mliap_potential(...)
)

# Evaluate energy
result = total_potential(state)
energy = result.data.total_energies
forces = result.data.gradients.positions
```

See submodule documentation for specific potential implementations.
"""
