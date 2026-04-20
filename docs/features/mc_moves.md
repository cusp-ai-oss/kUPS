# Monte Carlo Moves

*k*UPS implements several types of Monte Carlo moves optimized for molecular simulations. Each move type serves a specific purpose in sampling the configurational space efficiently.

## Supported Move Types

| Move Type | Description | Use Cases |
|-----------|-------------|-----------|
| **Translation** | Random displacement of molecule center-of-mass | Position sampling in all ensembles |
| **Rigid-Body Rotation** | Random rotation around molecule center-of-mass | Orientational sampling for anisotropic molecules |
| **Reinsertion** | Remove and reinsert molecule at random position/orientation | Enhanced sampling in confined systems |
| **Exchange (Swap)** | Deletion/insertion moves for changing particle number | GCMC simulations, chemical potential control |

## Translation Moves

Translation moves displace the center-of-mass of a molecule by a random vector:

- **Purpose**: Sample different positions in the simulation box
- **Used in**: All ensemble types (NVT, GCMC)
- **Key parameter**: Maximum displacement distance

Translation moves are essential for exploring the positional degrees of freedom and are typically the most frequent move type in bulk simulations.

## Rigid-Body Rotation Moves

Rotation moves randomly rotate a molecule around its center-of-mass:

- **Purpose**: Sample orientational configurations
- **Used in**: All ensemble types, particularly important for anisotropic molecules
- **Key parameter**: Maximum rotation angle

For molecules with directional interactions (e.g., dipoles, quadrupoles), rotation moves are crucial for proper sampling of orientational distributions.

## Reinsertion Moves

Reinsertion moves remove a molecule from its current position and reinsert it at a completely random position and orientation:

- **Purpose**: Overcome energy barriers and enhance sampling
- **Used in**: All ensemble types, especially useful in confined systems
- **Benefits**: Can escape local energy minima more effectively than small translation/rotation moves

These moves are particularly valuable in simulations of adsorption in porous materials, where molecules may become trapped in local minima.

## Exchange (Swap) Moves

Exchange moves insert or delete molecules to maintain equilibrium with a reservoir at fixed chemical potential:

- **Purpose**: Sample particle number fluctuations
- **Used in**: GCMC simulations only
- **Types**:
    - **Insertion**: Add a molecule at random position/orientation
    - **Deletion**: Remove a randomly selected molecule
- **Acceptance**: Governed by chemical potential (fugacity) and energy change

Exchange moves are the defining feature of grand canonical Monte Carlo and enable simulations of adsorption, absorption, and gas mixture equilibria.

## Move Selection and Optimization

The relative frequencies of different move types can significantly impact sampling efficiency. *k*UPS allows users to specify:

- Probability of each move type
- Move-specific parameters (e.g., maximum displacement, rotation angle)
- Adaptive tuning of parameters to achieve target acceptance rates

## Related Topics

- [Simulation Ensembles](ensembles.md): Learn about the ensembles that use these moves
- [Force Fields](force_fields.md): Understand how energy changes are calculated for move acceptance
