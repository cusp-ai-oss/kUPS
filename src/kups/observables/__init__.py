# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Observable property calculations for molecular simulations.

This package provides tools for computing thermodynamic and structural properties
from simulation trajectories. Observables can be computed on-the-fly during
simulations or from stored trajectory data.

## Module Organization

- **[pressure][kups.observables.pressure]**: Pressure calculations from stress tensors and ideal gas law
- **[stress][kups.observables.stress]**: Stress tensor calculations via the virial theorem
- **[radial_distribution_function][kups.observables.radial_distribution_function]**: Pair correlation functions $g(r)$

## Key Concepts

### Stress and Pressure

The stress tensor relates internal forces to system deformation. Pressure is
the isotropic component: $P = -\\text{Tr}(\\sigma)/3$. The stress is computed
via the virial theorem, combining the atomic outer-product term with the
symmetric cell virial $h^T \\cdot \\partial U/\\partial h$:

$$\\sigma = -\\frac{1}{V}\\left[ \\mathrm{sym}\\left(\\sum_i \\partial U/\\partial r_i \\otimes r_i\\right) + h^T \\cdot \\partial U/\\partial h \\right]$$

Only the 6 lower-triangular entries of $\\partial U/\\partial h$ are stored
(the cell's parameter DoF). The lower triangle of $h^T \\cdot \\partial U/\\partial h$
depends only on those 6 entries; symmetry of the stress tensor fills the
upper triangle.

**Molecular stress** differs from atomic stress by using center-of-mass positions,
avoiding spurious intramolecular contributions (following RASPA convention).

### Radial Distribution Function

The pair correlation function $g(r)$ measures the probability of finding a particle
at distance $r$ from another particle, normalized by the bulk density. It reveals
structural information like coordination numbers and phase transitions.

## Common Patterns

```python
from kups.observables.stress import stress_via_virial_theorem
from kups.observables.pressure import ideal_gas_pressure

# Compute stress from virial theorem
stress = stress_via_virial_theorem(particles, systems)

# Compute ideal gas pressure
pressure = ideal_gas_pressure(counts, systems)

# For offline analysis
from kups.observables.radial_distribution_function import offline_radial_distribution_function
rdf = offline_radial_distribution_function(positions, cell, rmax=10.0, bins=200)
```

See individual module documentation for detailed API references:

- [ideal_gas_pressure][kups.observables.pressure.ideal_gas_pressure]
- [stress_via_virial_theorem][kups.observables.stress.stress_via_virial_theorem]
- [RadialDistributionFunction][kups.observables.radial_distribution_function.RadialDistributionFunction]
"""
