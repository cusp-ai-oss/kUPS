---
icon: lucide/house
---

# Handbook

A tour of the primitives <em>k</em>UPS is built from. Each chapter covers one primitive: what it is and why it's included in the design.

This is not an API reference. Function signatures live under the API Reference tab, and CLI-ready packaged simulations under [Simulations](simulations.md). Code samples assume familiarity with [JAX pytrees](https://docs.jax.dev/en/latest/pytrees.html) and [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html), and use the conventions in [Units](units.md).

<em>k</em>UPS is a toolkit for batched, differentiable molecular simulations on GPU. One composable interface covers molecular dynamics, Monte Carlo, geometry optimization, classical force fields, and machine-learning potentials (via [Tojax](https://github.com/cusp-ai-oss/tojax)), with thousands of independent systems running as a single vectorized computation.

## Three requirements that usually fight

A molecular-simulation framework has to satisfy three things at once, and the naive solution to each breaks the other two.

- **Hardware throughput.** Force evaluations dominate cost, and a real workflow (ensemble sampling, parameter sweeps, Monte Carlo chains) needs thousands of independent simulations running on a single GPU at once. Structure-of-arrays layout with coalesced access is the only thing that saturates the hardware; a Python object per particle cannot be compiled into a GPU kernel, and a loop that holds one system at a time leaves the GPU idle. The price: every primitive operates on batched arrays, not on single atoms or systems.
- **Composability.** Real research mixes MD with MC, custom potentials, online analysis, and new ensembles. A monolithic simulator per ensemble forks the code for every new method and inherits no performance work from the others. The price: a shared abstraction every method has to express itself through, even when a hand-tuned one-off would be faster.
- **Per-step latency.** Step N+1 reads step N's state, so a simulation is sequential by construction and per-step latency sets the wall-clock cost. Compiling the whole step into a single [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) kernel brings it down to what classical C++ engines deliver. The price: shapes are fixed at compile time, with every buffer, neighbor list, and loop count sized up front.

<em>k</em>UPS resolves the three together. The primitives below compose freely, operate on batched arrays, and fit inside a fixed-shape compiled kernel, all at once.

## Primitives

The chapters are organized in five pairs. Tables and Lenses are prerequisite vocabulary for everything after; the remaining pairs are largely independent.

**Data layout: batched arrays that still carry relational structure.**

1. **[Tables](notebooks/tables.md).** Keyed containers and typed foreign-key indices. Flattens many independent systems into one vectorized computation.
2. **[Lenses](notebooks/lens.md).** Generic get-and-update pairs that let primitives operate on arbitrary user-defined state layouts.

**Control flow: staying inside the JIT kernel, even when things go wrong.**

3. **[Runtime Assertions](notebooks/runtime_assertions.md).** Side-channel checks that survive JIT, plus a host-side retry loop that resizes buffers and re-enters.
4. **[Propagators](notebooks/propagators.md).** The evolution primitive: `(key, state) -> state`. Integrators, MC moves, neighbor-list refreshes, and logging all share this signature.

**Composition: decoupling state and updates from the primitives that operate on them.**

5. **[Conventions](notebooks/conventions.md).** Structural `Has*` and `Is*` protocols on plain dataclasses. No framework base class; a state carries only the fields it uses.
6. **[Patches](notebooks/patches.md).** Conditional, atomic local state changes. The abstraction behind batched Monte Carlo where each chain accepts or rejects independently.

**Interactions: energy, forces, and the pair lists that make them tractable.**

7. **[Neighbor Lists](notebooks/neighborlist.md).** Which particle pairs sit within `r_cut`. Cell lists, refinement, and capacity growth live behind a single protocol.
8. **[Potentials](notebooks/potentials.md).** Energy as a composable, differentiable object. Classical terms and ML force fields compose by summation; cached evaluations make patched MC steps cheap.

**Sampling and observability: what sits around the compiled step.**

9. **[Monte Carlo Moves](notebooks/mc_moves.md).** Batched Metropolis-Hastings on top of the integrator stack. Every system is an independent Markov chain with its own per-system acceptance, step widths, and move statistics.
10. **[Logging](notebooks/logging.md).** Host-side observability around the pure step function: HDF5 writers, counters, progress bars, and profiler hooks that stay outside `jax.jit`.

MD, MC, relaxation, GCMC, and ML-potential dynamics are all assembled from these ten pieces. A GCMC step, for example, runs translation, rotation, and exchange as propagators (ch. 4) that construct patches (ch. 6) scored by a cached potential (ch. 8) over a fixed-capacity buffered table (ch. 1), with per-system acceptance and step-width tuning handled by the Monte Carlo machinery (ch. 9). Same primitives, different composition.

## A worked example: `md`

[kups.application.simulations.md][kups.application.simulations.md] (CLI: `kups_md`) is the shortest complete simulation in the repo: about a hundred lines, with a ten-line `run`. A discriminated `potential` config selects the force field, so one state and one driver serve Lennard-Jones, Lennard-Jones plus Coulomb, and every MLFF backend.

**State definition.** The user picks the fields; nothing inherits from a framework base.

```python
@dataclass
class MdState:
    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    skin_list: SkinList
    step: Array
```

The state structurally satisfies `IsMdState` and carries no force-field field; parameters are bound into the potential at construction. Both tables carry relational data via typed foreign-key indices. `skin_list` keeps the neighbor pairs across steps ([Neighbor Lists](notebooks/neighborlist.md)). `neighborlist_params` and the skin list's capacities are resized by the retry loop on overflow.

**Wiring potential and propagator.** The `potential` config builds the potential directly with its parameters, given a state lens, the gradient filter, and the particle and system tables, from which a config can size parameters that depend on the input (`lj_coulomb` estimates its Coulomb screening from the charges and cells); factories fan the lens out to the fields they need. `run` reads the input into the tables first (see State construction below).

```python
state_lens = identity_lens(MdState)
potential, cutoff = config.potential.build(
    state_lens, POSITIONS_AND_CELL, particles, systems
)
evaluate_forces, propagator = make_md_propagator(
    state_lens, config.md.integrator, potential
)
```

For the LJ backend, `build` delegates to [make_lennard_jones_row_from_state][kups.application.potential.classical.lennard_jones.make_lennard_jones_row_from_state], wiring particles, systems, the skin list, and the LJ parameters through the state lens and returning the cutoff the skin list serves. The potential reads the skin list through [read_skin_list][kups.core.neighborlist.read_skin_list] and evaluates LJ over its neighbor rows. It checks the skin list at every force evaluation, rebuilds it when a pair may be missing, and returns a patch that stores the list it read. [make_md_propagator][kups.application.md.simulation.make_md_propagator] returns the force evaluation, a [PotentialAsPropagator][kups.core.potential.PotentialAsPropagator] that applies this patch with the forces, and the MD step, which composes the integrator's momentum and position steps around that evaluation with a step counter inside one [SequentialPropagator][kups.core.propagator.SequentialPropagator]. Each comes in a [ResetOnErrorPropagator][kups.core.propagator.ResetOnErrorPropagator] that keeps the skin list when it fails.

**State construction.** Read a standard file, build the two tables, seed the rest from them.

```python
particles, systems = md_state_from_ase(config.inp_files[0], config.md, key=mb_key)
state = MdState.create(particles, systems, cutoff, config.md.skin)
```

`md_state_from_ase` accepts xyz, cif, or lammps input. `MdState.create` guesses initial capacities from geometry with [UniversalNeighborlistParameters.estimate][kups.core.neighborlist.UniversalNeighborlistParameters.estimate] and builds the skin list; the estimate does not have to be exact, because warmup grows what is too small.

`run` loads `kups_md` output files with `md_state_from_run`, which starts from the final state of the earlier run.

**Running.** The loop lives on the host side.

```python
state = propagate_and_fix(make_cycle_function(evaluate_forces), next(chain), state)
state = run_md(next(chain), propagator, state, config.run)
```

The state starts without forces, so they are evaluated once before the first step. `run_md` has two phases. Warmup calls [propagate_and_fix][kups.core.propagator.propagate_and_fix] until buffer capacities stabilize. Production runs the compiled propagator with an HDF5 logger and a progress bar. Each step is one JIT call, and [buffer donation](https://docs.jax.dev/en/latest/faq.html#buffer-donation) lets JAX reuse the input state's memory for the output so the step allocates nothing new.

## Where to go next

Pick a packaged simulation from [Simulations](simulations.md) and trace it back through the relevant chapters. [Troubleshooting](troubleshooting.md) covers the GPU and JIT errors that come up most often.
