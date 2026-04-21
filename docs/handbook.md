# Handbook

A tour of the primitives <em>k</em>UPS is built from. Each chapter covers one primitive: what it is and what forced it into the design.

This is not an API reference — function signatures live under the API Reference tab, and CLI-ready packaged simulations under [Simulations](simulations.md). Code samples assume familiarity with [JAX pytrees](https://docs.jax.dev/en/latest/pytrees.html) and [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html), and use the conventions in [Units](units.md).

<em>k</em>UPS is a toolkit for batched, differentiable molecular simulations on GPU. One composable interface covers molecular dynamics, Monte Carlo, geometry optimization, classical force fields, and machine-learning potentials (via [Tojax](https://github.com/cusp-ai-oss/tojax)), with thousands of independent systems running as a single vectorized computation.

## Three requirements that usually fight

A molecular-simulation framework has to satisfy three things at once, and the naive solution to each breaks the other two.

- **Hardware throughput.** Force evaluations dominate cost, and a real workflow — ensemble sampling, parameter sweeps, Monte Carlo chains — needs thousands of independent simulations running on a single GPU at once. Structure-of-arrays layout with coalesced access is the only thing that saturates the hardware; a Python object per particle cannot be compiled into a GPU kernel, and a loop that holds one system at a time leaves the GPU idle. The price: every primitive operates on batched arrays, not on single atoms or systems.
- **Composability.** Real research mixes MD with MC, custom potentials, online analysis, and new ensembles. A monolithic simulator per ensemble forks the code for every new method and inherits no performance work from the others. The price: a shared abstraction every method has to express itself through, even when a hand-tuned one-off would be faster.
- **Per-step latency.** Step N+1 reads step N's state, so a simulation is sequential by construction and per-step latency sets the wall-clock cost. Compiling the whole step into a single [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) kernel brings it down to what classical C++ engines deliver. The price: shapes are fixed at compile time — every buffer, neighbor list, and loop count sized up front.

<em>k</em>UPS resolves the three together: the primitives below compose freely, operate on batched arrays, and fit inside a fixed-shape compiled kernel — all at once.

## Primitives

The chapters are organized in four pairs. Tables and Lenses are prerequisite vocabulary for everything after; the remaining pairs are largely independent.

**Data layout — batched arrays that still carry relational structure.**

1. **[Tables](notebooks/tables.md).** Keyed containers and typed foreign-key indices. Flattens many independent systems into one vectorized computation.
2. **[Lenses](notebooks/lens.md).** Generic get-and-update pairs that let primitives operate on arbitrary user-defined state layouts.

**Control flow — staying inside the JIT kernel, even when things go wrong.**

3. **[Runtime Assertions](notebooks/runtime_assertions.md).** Side-channel checks that survive JIT, plus a host-side retry loop that resizes buffers and re-enters.
4. **[Propagators](notebooks/propagators.md).** The evolution primitive: `(key, state) → state`. Integrators, MC moves, neighbor-list refreshes, and logging all share this signature.

**Composition — decoupling state and updates from the primitives that operate on them.**

5. **[Conventions](notebooks/conventions.md).** Structural `Has*` and `Is*` protocols on plain dataclasses. No framework base class; a state carries only the fields it uses.
6. **[Patches](notebooks/patches.md).** Conditional, atomic local state changes — the abstraction behind batched Monte Carlo where each chain accepts or rejects independently.

**Interactions — energy, forces, and the pair lists that make them tractable.**

7. **[Neighbor Lists](notebooks/neighborlist.md).** Which particle pairs sit within `r_cut`. Cell lists, refinement, and capacity growth live behind a single protocol.
8. **[Potentials](notebooks/potentials.md).** Energy as a composable, differentiable object. Classical terms and ML force fields compose by summation; cached evaluations make patched MC steps cheap.

MD, MC, relaxation, GCMC, and ML-potential dynamics are all assembled from these eight pieces. A GCMC step, for example, runs translation, rotation, and exchange as propagators (ch. 4) that construct patches (ch. 6) scored by a cached potential (ch. 8) over a fixed-capacity buffered table (ch. 1). Same primitives, different composition.

## A worked example: `md_lj`

[kups.application.simulations.md_lj][kups.application.simulations.md_lj] (CLI: `kups_md_lj`) is the shortest complete simulation in the repo — about a hundred lines, with a ten-line `run`.

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

The state structurally satisfies `IsMdState`. Both tables carry relational data via typed foreign-key indices. `neighborlist_params` is resized by the retry loop on overflow.

**State construction.** Read a standard file, build the two tables, pick initial capacities.

```python
particles, systems = md_state_from_ase(config.inp_file, config.md, key=mb_key)
neighborlist_params = UniversalNeighborlistParameters.estimate(
    particles.data.system.counts, systems, lj_params.cutoff
)
```

`md_state_from_ase` accepts xyz, cif, or lammps input. [UniversalNeighborlistParameters.estimate][kups.core.neighborlist.UniversalNeighborlistParameters.estimate] guesses initial capacities from geometry — it does not have to be exact, because warmup grows what is too small.

**Wiring potential and propagator.** Factories take a single state lens and fan it out to the fields they need.

```python
state_lens = identity_lens(LjMdState)
potential = make_lennard_jones_from_state(
    state_lens, compute_position_and_unitcell_gradients=True
)
propagator = make_md_propagator(state_lens, config.md.integrator, potential)
```

[make_lennard_jones_from_state][kups.potential.classical.lennard_jones.make_lennard_jones_from_state] reads particles, systems, and LJ parameters through the state lens. [make_md_propagator][kups.application.md.simulation.make_md_propagator] composes a [PotentialAsPropagator][kups.core.potential.PotentialAsPropagator], the integrator's momentum and position steps, a step counter, and a [ResetOnErrorPropagator][kups.core.propagator.ResetOnErrorPropagator] inside one [SequentialPropagator][kups.core.propagator.SequentialPropagator].

**Running.** The loop lives on the host side.

```python
state = run_md(next(chain), propagator, state, config.run)
```

`run_md` has two phases. Warmup calls [propagate_and_fix][kups.core.propagator.propagate_and_fix] until buffer capacities stabilize. Production runs the compiled propagator with an HDF5 logger and a progress bar. Each step is one JIT call, and [buffer donation](https://docs.jax.dev/en/latest/faq.html#buffer-donation) lets JAX reuse the input state's memory for the output so the step allocates nothing new.

## Extending <em>k</em>UPS

Two more chapters, coming soon, build on the primitives:

- **Custom MCMC moves.** Writing your own propagator from scratch, worked through with Hamiltonian Monte Carlo.
- **Custom force fields.** Adding a new energy term that composes and caches correctly.

## Where to go next

Pick a packaged simulation from [Simulations](simulations.md) and trace it back through the relevant chapters. [Troubleshooting](troubleshooting.md) covers the GPU and JIT errors that come up most often.
