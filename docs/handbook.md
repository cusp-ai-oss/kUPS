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

[kups.application.simulations.md][kups.application.simulations.md] (CLI: `kups_md`) is the shortest complete simulation in the repo: about a hundred lines, with a ten-line `run`. A discriminated `potential` config selects the force field, so one state and one driver serve Lennard-Jones and every MLFF backend.

**State definition.** The user picks the fields; nothing inherits from a framework base.

```python
@dataclass
class MdState:
    particles: Table[ParticleId, MDParticles]
    systems: Table[SystemId, MDSystems]
    neighborlist_params: UniversalNeighborlistParameters
    step: Array
```

The state structurally satisfies `IsMdState` and carries no force-field field; parameters are bound into the potential at construction. Both tables carry relational data via typed foreign-key indices. `neighborlist_params` is resized by the retry loop on overflow.

**Wiring potential and propagator.** The `potential` config builds the potential directly with its parameters, given a state lens and the gradient filter; factories fan the lens out to the fields they need.

```python
state_lens = identity_lens(MdState)
potential, cutoff = config.potential.build(state_lens, POSITIONS_AND_CELL)
propagator = make_md_propagator(state_lens, config.md.integrator, potential)
```

For the LJ backend, `build` delegates to [make_lennard_jones_from_state][kups.application.potential.classical.lennard_jones.make_lennard_jones_from_state], wiring particles, systems, and the LJ parameters through the state lens and returning the cutoff used to size the neighbor list. [make_md_propagator][kups.application.md.simulation.make_md_propagator] composes a [PotentialAsPropagator][kups.core.potential.PotentialAsPropagator], the integrator's momentum and position steps, a step counter, and a [ResetOnErrorPropagator][kups.core.propagator.ResetOnErrorPropagator] inside one [SequentialPropagator][kups.core.propagator.SequentialPropagator].

**State construction.** Read a standard file, build the two tables, pick initial capacities.

```python
particles, systems = md_state_from_ase(config.inp_files[0], config.md, key=mb_key)
neighborlist_params = UniversalNeighborlistParameters.estimate(
    particles.data.system.counts, systems, cutoff
)
```

`md_state_from_ase` accepts xyz, cif, or lammps input. [UniversalNeighborlistParameters.estimate][kups.core.neighborlist.UniversalNeighborlistParameters.estimate] guesses initial capacities from geometry; it does not have to be exact, because warmup grows what is too small.

**Running.** The loop lives on the host side.

```python
state = run_md(next(chain), propagator, state, config.run)
```

`run_md` has two phases. Warmup calls [propagate_and_fix][kups.core.propagator.propagate_and_fix] until buffer capacities stabilize. Production runs the compiled propagator with an HDF5 logger and a progress bar. Each step is one JIT call, and [buffer donation](https://docs.jax.dev/en/latest/faq.html#buffer-donation) lets JAX reuse the input state's memory for the output so the step allocates nothing new.

## Adding a dispersion correction

Semi-local DFT misses long-range London dispersion, and MLIPs trained on it inherit that gap: layer spacings come out too large, molecular crystals under-bind, physisorption is too weak. [D3][kups.potential.dispersion.d3] adds it back.

What kUPS implements is the **pairwise (two-body) D3 term with Becke-Johnson damping** — the `C₆`/`C₈` sum, which is what the published per-functional `s6`/`s8`/`a1`/`a2` fits are fitted to. The Axilrod-Teller-Muto three-body term and the zero-damping D3(0) variant are **not** implemented. ATM is small for dense molecular solids but not always negligible in porous frameworks, so if you are chasing adsorption enthalpies to better than a few kJ/mol, budget for it being absent.

D3 is a *correction*, not a force field, so it composes rather than replaces. Give both terms the **same** gradient filter — [sum_potentials][kups.core.potential.sum_potentials] merges gradient pytrees element-wise, so their structures have to match — and add them:

```python
from kups.application.potential.dispersion.d3 import make_d3_from_state
from kups.core.potential import sum_potentials
from kups.potential.dispersion import D3Parameters

d3 = make_d3_from_state(
    state_lens,
    parameters=D3Parameters.from_functional("pbe"),
    gradient=POSITIONS_AND_CELL,
)
total = sum_potentials(mlip, d3)
```

That is all that is required. Energies, forces and cell gradients are additive, so stress via [stress_via_virial_theorem][kups.observables.stress.stress_via_virial_theorem] is correct with no extra wiring. Pick the parameterisation with `from_functional` to match whatever functional the MLIP's training data used, or pass `s6`/`s8`/`a1`/`a2` explicitly to `D3Parameters.from_damping`. As with the other force fields, parameters may instead live on the state as `d3_parameters`, which is what you want if they change during a run.

**Size the neighbor list from the largest cutoff.** Each potential builds its own neighbor list from its own cutoff, but they share `state.neighborlist_params`. D3's cutoff is normally far larger than an MLIP's 5–6 Å, so it has to drive the estimate:

```python
cutoff = jax.tree.map(jnp.maximum, mlip_cutoff, d3.cutoff)
neighborlist_params = UniversalNeighborlistParameters.estimate(
    particles.data.system.counts, systems, cutoff
)
```

Getting this wrong costs recompiles, not correctness — the capacity retry loop grows what is too small.

**The cutoff is the cost.** D3's real-space sum converges absolutely, so no Ewald machinery is needed, but it is genuinely long-ranged. `simple-dftd3` defaults to 60 a₀ (31.75 Å), which at condensed-phase density means of order 10⁴ neighbours per atom. kUPS defaults to a much cheaper 15 Å; the neglected tail is roughly `4πρ⟨C₆⟩/(3R³)` per atom and varies slowly, so forces converge far faster with cutoff than energies do. Raise `cutoff` when absolute energies or pressures matter, or when reproducing published D3 numbers.

The truncation is abrupt — no shift, no switching function, matching `simple-dftd3`. The step as a pair crosses the cutoff is tiny (2.6 µeV per carbon pair at 15 Å, or order 2 µeV/atom per 0.01 Å of motion at a condensed-phase density) but non-zero, so it sets a floor on NVE energy conservation and on a meaningful force-convergence threshold. Raising the cutoff shrinks the step; it does not remove it.

D3 always recomputes in full: the coordination number couples atoms beyond any single moved pair, so the incremental Monte-Carlo path that Lennard-Jones and Ewald use would be wrong here rather than merely slow.

## Where to go next

Pick a packaged simulation from [Simulations](simulations.md) and trace it back through the relevant chapters. [Troubleshooting](troubleshooting.md) covers the GPU and JIT errors that come up most often.
