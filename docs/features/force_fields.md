# Force Fields and Potentials

*k*UPS supports a range of force fields and potentials for calculating molecular energies, from classical pairwise potentials to modern machine learning models.

## Supported Potentials

| Potential | Full Computation | Efficient Updates | Applications |
|-----------|:----------------:|:-----------------:|--------------|
| **Lennard-Jones** | ✅ | ✅ | Van der Waals interactions |
| **Coulomb (Ewald)** | ✅ | ✅ | Electrostatic interactions |
| **MACE** | ✅ | ❌ | Machine learning potentials |

## Lennard-Jones Potential

The Lennard-Jones (LJ) potential models van der Waals interactions between neutral atoms or molecules:

$$V_{LJ}(r) = 4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$

- **Parameters**:
    - $\epsilon$: Depth of the potential well
    - $\sigma$: Distance at which the potential is zero
- **Applications**: Non-bonded interactions, noble gases, united-atom models
- **Performance**: Highly optimized with efficient energy update calculations

## Coulomb Potential (Ewald Summation)

The Coulomb potential handles long-range electrostatic interactions using Ewald summation:

$$V_{Coulomb}(r) = \frac{q_i q_j}{4\pi\epsilon_0 r}$$

- **Method**: Ewald summation for periodic boundary conditions
- **Components**:
    - Real-space sum with error function damping
    - Reciprocal-space sum in Fourier space
    - Self-energy correction
- **Applications**: Charged systems, ionic liquids, polar molecules
- **Performance**: Efficient updates for Monte Carlo moves

For more details on the implementation, see the [Potentials documentation](../potentials.md).

## MACE (Machine Learning Potential)

MACE is a state-of-the-art machine learning potential based on equivariant graph neural networks:

- **Advantages**:
    - Near quantum-mechanical accuracy
    - Transferable across different systems
    - Captures many-body effects
- **Performance**: Full system energy computation required for each evaluation due to MACE's graph neural network architecture, which needs complete atomic environment information
- **Applications**: Complex molecular systems where classical force fields are inadequate

> **Note**: Efficient energy updates are not possible with MACE. The graph neural network architecture requires information about the entire atomic environment to compute energies, making partial updates infeasible. Full energy computations are performed for every Monte Carlo move.

## Performance Considerations

### Efficient Energy Updates

> **Performance Note**: Efficient updates enable rapid sampling by computing only energy changes for proposed moves.

For Monte Carlo simulations, calculating only the energy change $\Delta E$ for a proposed move is much faster than recomputing the total system energy:

- **Translation/Rotation**: Only interactions involving the moved molecule need recalculation
- **Exchange moves**: Energy change involves only the inserted/deleted molecule
- **Speedup**: Typically 10-1000x faster depending on system size

### Current Implementation Status

| Potential | Update Strategy | Scaling |
|-----------|-----------------|---------|
| Lennard-Jones | Pairwise differences | O(N) per move |
| Coulomb (Ewald) | Fourier space updates | O(N log N) per move |
| MACE | Full recomputation | O(E) per move |

## Combining Potentials

Multiple potentials can be combined additively to model complex systems:

$$V_{total} = V_{LJ} + V_{Coulomb} + V_{other}$$

This allows flexible modeling of systems with both van der Waals and electrostatic interactions.

## Related Topics

- [Potentials Architecture](../potentials.md): Detailed implementation of potential calculations
- [Monte Carlo Moves](mc_moves.md): How energy calculations are used in move acceptance
- [Simulation Ensembles](ensembles.md): Different simulation types using these potentials
