# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Core library for JAX-based particle simulations.

This package provides the foundational components for building molecular dynamics
and Monte Carlo simulations with JAX. The architecture emphasizes composability,
type safety, and compatibility with JAX transformations.

## Module Organization

### Data Structures and Manipulation
- **[data][kups.core.data]**: PyTree wrappers ([Batched][kups.core.data.Batched], [Table][kups.core.data.Table]) for structured data
- **[lens][kups.core.lens]**: Functional lenses for accessing and modifying nested structures
- **[unitcell][kups.core.unitcell]**: Periodic boundary conditions and crystallographic unit cells

### Computation and Simulation
- **[potential][kups.core.potential]**: Energy calculations with gradients and Hessians
- **[propagator][kups.core.propagator]**: State evolution (MD integrators, MC moves, composition)
- **[neighborlist][kups.core.neighborlist]**: Efficient neighbor search algorithms (cell lists, dense)

### System Management
- **[assertion][kups.core.assertion]**: Runtime validation with automatic fixing capabilities
- **[capacity][kups.core.capacity]**: Dynamic array resizing with automatic capacity management
- **[patch][kups.core.patch]**: State modification protocols and implementations
- **[result][kups.core.result]**: Result types with runtime assertion tracking

### Utilities
- **[parameter_scheduler][kups.core.parameter_scheduler]**: Adaptive parameter tuning (step sizes, temperatures)
- **[constants][kups.core.constants]**: Physical constants and unit conversions
- **[utils][kups.core.utils]**: Mathematical utilities, JAX helpers, functional programming tools
- **[storage][kups.core.storage]**: HDF5-based trajectory logging

## Design Philosophy

1. **Composability**: Small, focused components that combine easily
2. **Type Safety**: Extensive use of generics and protocols
3. **JAX Integration**: All components work with jit, grad, vmap
4. **Immutability**: Functional updates via lenses and patches
5. **Performance**: Optimized algorithms with automatic capacity management

## Common Patterns

```python
# Build a potential from components
potential = sum_potentials(
    bonded_potential,
    lennard_jones_potential,
    coulomb_potential
)

# Compose propagators
propagator = compose_propagators(
    neighbor_list_update,
    monte_carlo_move,
    parameter_adjustment
)

# Run simulation with logging
with h5py.File("traj.h5", "w") as f:
    writer = HDF5StorageWriter.init(f, config, state, total_steps)
    with writer.background_writer() as bg:
        for step in range(total_steps):
            state = propagator(key, state)
            bg.write(state, step)
```

See individual module documentation for detailed API references:

- [Potential API][kups.core.potential.Potential]
- [Propagator API][kups.core.propagator.Propagator]
- [Neighbor Lists][kups.core.neighborlist.CellListNeighborList]
- [Lens System][kups.core.lens.Lens]
"""
