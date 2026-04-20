# Simulation Ensembles

*k*UPS supports multiple statistical ensembles for Monte Carlo simulations, each suited for different types of molecular systems and scientific questions.

## Supported Ensembles

| Ensemble | Description | Use Cases |
|----------|-------------|-----------|
| **NVT** (Canonical) | Fixed number of particles, volume, and temperature | Bulk properties, phase behavior |
| **GCMC** (Grand Canonical) | Fixed pressure, volume, and temperature | Adsorption isotherms, pore filling |

## NVT (Canonical Ensemble)

The canonical ensemble maintains constant:

- **N**: Number of particles
- **V**: Volume
- **T**: Temperature

This ensemble is ideal for studying:

- Bulk phase properties
- Phase behavior and transitions
- Equilibrium properties of closed systems
- Structural properties of molecular systems

## GCMC (Grand Canonical Monte Carlo)

The grand canonical ensemble maintains constant:

- **μ**: Chemical potential (controlled via fugacity)
- **V**: Volume
- **T**: Temperature

This ensemble is particularly useful for:

- Adsorption isotherms in porous materials
- Gas uptake in metal-organic frameworks (MOFs)
- Pore filling mechanisms
- Gas mixture separations

> GCMC simulations involve particle insertion and deletion moves, allowing the system to exchange particles with an implicit reservoir at fixed chemical potential.

## Current Focus

> **Current Development Status**: *k*UPS currently focuses on rigid-body molecules. Support for flexible molecules and additional ensembles is planned for future releases.

## Related Topics

- [Monte Carlo Moves](mc_moves.md): Learn about the sampling moves used in each ensemble
- [Force Fields](force_fields.md): Understand the energy calculations that drive acceptance probabilities
