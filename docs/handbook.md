# Handbook

The handbook is a tour of *why* <em>k</em>UPS is shaped the way it is. Each chapter introduces one primitive, motivates it from the gap left by the previous chapter, and ends with the reader able to read and extend that layer of the framework's own code.

This is not an API reference — for function signatures and class attributes, use the API Reference tab. For CLI-ready, packaged simulations, see [Simulations](simulations.md). All code here uses the unit system described in [Units](units.md), and assumes you are comfortable with JAX pytrees and `jax.jit`.

## Three requirements that usually fight

Molecular simulation has to satisfy three things simultaneously, and each one alone rules out solutions that would satisfy the others:

- **Hardware acceleration.** Force evaluations dominate cost. They have to run on CPU, GPU, and TPU with one set of kernels — which rules out a Python object per particle.
- **Composability.** Real research mixes MD with MC, custom potentials, online analysis, and new ensembles. Methods have to snap together — which rules out a monolithic simulator per ensemble.
- **JIT correctness.** JAX compiles to fixed-shape computation graphs. Simulations have variable data — particles inserted and deleted, neighbor lists that grow and shrink — which rules out naive dynamic allocation.

<em>k</em>UPS is a chain of small primitives that resolve all three together. The rest of the handbook walks that chain.

## The chain

Each chapter uses only the vocabulary of the ones before it. Read in order on the first pass.

1. **[Tables](notebooks/tables.md)** — keyed containers and typed foreign-key indices. Flat arrays get hardware acceleration for free but lose relationships; [`Table`][kups.core.data.table.Table] and [`Index`][kups.core.data.index.Index] restore them at compile time. [`Table.union`][kups.core.data.table.Table.union] flattens many independent systems into one vectorized computation — the batched-chains thread that runs through the rest of the handbook. [`Buffered`][kups.core.data.buffered.Buffered] pre-allocates free slots inside a fixed-capacity array so GCMC can insert and delete without recompilation.

2. **[Lenses](notebooks/lens.md)** — get-and-update pairs. Tables fix the data model, but every primitive still has to work against user-defined state layouts. Lenses adapt generic code to specific fields without mutation, so state layout and algorithm stay decoupled.

3. **[Runtime Assertions](notebooks/runtime_assertions.md)** — side-channel checks inside JIT. Buffer sizes can't always be known ahead of time; [`runtime_assert`][kups.core.assertion.runtime_assert] fails cleanly from inside compiled kernels, and the host-side retry loop [`propagate_and_fix`][kups.core.propagator.propagate_and_fix] resizes and re-enters.

4. **[Propagators](notebooks/propagators.md)** — the evolution primitive: `(key, state) → state`. MD integrators, MC moves, neighbor-list refreshes, and logging all share this shape. Sequential, loop, and switch composition build a full MD step from momentum-step and position-step primitives.

5. **[Conventions](notebooks/conventions.md)** — `Has*`/`Is*` protocols, dataclasses, `make_*_from_state` factories, `@property` for derived quantities. Instead of a framework base class, states structurally satisfy protocols — so new simulations keep exactly the fields they need.

6. **[Patches](notebooks/patches.md)** — conditional local state changes: "if accepted, write these bytes here." Patches make incremental Monte Carlo possible — build the patch, score it, accept or reject, and commit the change together with its cache dependencies atomically, per batched chain independently.

7. **[Neighbor Lists](notebooks/neighborlist.md)** — which pairs sit within `r_cut`. Naively O(N²) and breaks the JIT fixed-shape contract. The neighbor-list layer hides cell lists, refinement, capacity growth, and incremental updates behind one protocol.

8. **[Potentials](notebooks/potentials.md)** — energy as a composable object. Potentials compose by *summation* (LJ + Coulomb + bonded); propagators compose by *sequencing*. [`CachedPotential`][kups.core.potential.CachedPotential] stores the last full evaluation and evaluates only the delta when it sees a patch, which is how a Metropolis-Hastings step over thousands of particles stays cheap.

That's the chain. Every <em>k</em>UPS simulation — MD, MC, relaxation, batched GCMC, ML-potential dynamics — is built from these concepts.

## A worked example: `md_lj`

The shortest complete simulation in the repo is [`kups.application.simulations.md_lj`][kups.application.simulations.md_lj], CLI-exposed as `kups_md_lj`. The file is about a hundred lines end-to-end; its `run` function is ten of them. Every concept above appears once.

**State definition.** The user picks the fields; nothing inherits from a framework base.

```python
@dataclass
class LjMdState:
    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    step: Array
    lj_parameters: LennardJonesParameters
```

Structurally satisfies `IsMdState` (ch. 5); the `Table[ParticleId, ...]` and `Table[SystemId, ...]` carry relational data with typed foreign-key indices (ch. 1); `neighborlist_params` is what the retry loop grows when buffers overflow (ch. 3, ch. 7); `lj_parameters` is the potential's parameters (ch. 8).

**State construction.** Read a standard file, build the two tables, estimate initial capacities.

```python
particles, systems = md_state_from_ase(config.inp_file, config.md, key=mb_key)
neighborlist_params = UniversalNeighborlistParameters.estimate(
    particles.data.system.counts, systems, lj_params.cutoff
)
```

`md_state_from_ase` accepts xyz, cif, or lammps input; [`UniversalNeighborlistParameters.estimate`][kups.core.neighborlist.UniversalNeighborlistParameters.estimate] guesses initial capacities from geometry (ch. 7). Nothing here has to be exact — warmup will grow what is too small via the fix-and-retry loop (ch. 3).

**Wiring potential and propagator.** One lens, one factory, one composed propagator.

```python
state_lens = identity_lens(LjMdState)
potential = make_lennard_jones_from_state(
    state_lens, compute_position_and_unitcell_gradients=True
)
propagator = make_md_propagator(state_lens, config.md.integrator, potential)
```

[`make_lennard_jones_from_state`][kups.potential.classical.lennard_jones.make_lennard_jones_from_state] is a convention-following factory that reads particles, systems, and LJ parameters through a single state lens (ch. 2, ch. 5). [`make_md_propagator`][kups.application.md.simulation.make_md_propagator] composes a [`PotentialAsPropagator`][kups.core.potential.PotentialAsPropagator], the integrator's momentum and position steps, a step counter, and a [`ResetOnErrorPropagator`][kups.core.propagator.ResetOnErrorPropagator] — all in one [`SequentialPropagator`][kups.core.propagator.SequentialPropagator] (ch. 4).

**Running.** The loop lives on the host side.

```python
state = run_md(next(chain), propagator, state, config.run)
```

`run_md` does two things: a warmup phase calls [`propagate_and_fix`][kups.core.propagator.propagate_and_fix] until buffer capacities stabilize (ch. 3); a production phase then runs the compiled propagator with an HDF5 logger and a progress bar. Each step is one JIT call; buffer donation lets JAX reuse the input state's memory for the output so allocation stays flat.

Every line in the file reaches into one chapter, and the chapters fit together because they share the same primitives.

## Where to go from here

Run a packaged simulation from [Simulations](simulations.md) and trace it back against this handbook. If GPU utilization, GPU memory, or a <em>k</em>UPS-specific error gets in the way, [Troubleshooting](troubleshooting.md) covers the sharp edges.
