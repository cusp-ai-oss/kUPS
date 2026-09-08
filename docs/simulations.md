# Packaged Simulations

<em>k</em>UPS ships with several ready-to-use simulation applications as CLI tools. Each is a thin layer built on the core primitives (propagators, potentials, lenses, tables) and serves as both a useful tool and a reference implementation. All commands take a YAML configuration file via and use [nanoargs](https://github.com/cusp-ai-oss/nanoargs) for argument parsing, so any configuration value can also be overridden from the command line. Example configurations are provided in the [`examples/`](https://github.com/cusp-ai-oss/kups/tree/main/examples) directory and should be run from there.

## Molecular Dynamics

Run molecular dynamics trajectories in the NVE, NVT, or NPT ensemble with `kups_md`. The force field is selected by the `potential.backend` field:

| `potential.backend` | Force Field | Description |
|---------------------|-------------|-------------|
| `lj` | Lennard-Jones | Classical pair potential with configurable mixing rules |
| `tojax` | MACE, UMA, ORB | Machine-learned interatomic potentials loaded via [Tojax](https://github.com/cusp-ai-oss/tojax) |
| `mace`, `uma` | MACE, UMA | Native PyTorch MLFFs bridged into JAX (see [Direct PyTorch Bridge](#direct-pytorch-bridge)) |

```sh
cd examples
kups_md md_lj_argon_nvt.yaml
kups_md md_lj_argon_nve.yaml
kups_md md_mace.yaml
kups_md md_orb.yaml
```

**Ensembles and integrators:**

- **NVE** — velocity Verlet. Constant energy, useful for validating energy conservation.
- **NVT** — Langevin thermostat (BAOAB splitting) or canonical sampling via velocity rescaling (CSVR). Constant temperature.
- **NPT** — CSVR thermostat with stochastic cell rescaling barostat (`csvr_npt`), or fully-flexible-cell BAOAB NPT Langevin (`baoab_npt_langevin`, Gao–Fang–Wang JCP 2016). Constant temperature and pressure.

All integrators are built from the same composable propagator primitives described in the Propagators tutorial.

## Geometry Optimization

Relax atomic positions (and optionally lattice vectors) to a local energy minimum with `kups_relax`. The force field is selected by the `potential.backend` field:

| `potential.backend` | Force Field | Description |
|---------------------|-------------|-------------|
| `lj` | Lennard-Jones | Classical relaxation |
| `tojax` | MACE, UMA, ORB | Machine-learned force field relaxation via JAX-exported models ([Tojax](https://github.com/cusp-ai-oss/tojax)) |
| `mace`, `uma` | MACE, UMA | Native PyTorch MLFFs (MACE checkpoints, Meta's UMA via [fairchem-core](https://github.com/facebookresearch/fairchem)) bridged into JAX |

```sh
cd examples
kups_relax relax_mace.yaml
kups_relax relax_orb.yaml
kups_relax relax_torch_mace.yaml   # native MACE .model checkpoint
kups_relax relax_torch_uma.yaml    # native UMA .pt checkpoint (fairchem)
```

**Optimizers:**

- **FIRE** — fast inertial relaxation engine. Adaptive timestep, robust for rough energy landscapes.
- **L-BFGS** — limited-memory quasi-Newton method. Fast convergence near the minimum.
- Any **Optax** optimizer (Adam, SGD, etc.) can be plugged in via the same interface.

Relaxation converges when the maximum force on any atom drops below a configurable tolerance.

### Streaming batches as a library

Streaming is assembled from the same propagators and runtime assertions as other
simulations. `SlotLayout` reserves a fixed number of particle rows per system;
`make_streaming_relax_propagator` returns the numerical step, optimizer initializer,
and per-system reset function. `make_streaming_relax_state` creates idle slots
requesting their first refill.

The application supplies a host refill function. It reads completed structures
through `slot_payload_lens`, packs replacements with `payload_from_structures`,
and calls `replace_slots`. The replacement payload has one row per slot; only
requested rows are installed. At end of input, supply `idle_payload` rows with
ordinal `-1`. Replacement resets the selected optimizer state and clears the
serviced requests, preserving other slots.

```python
import jax.numpy as jnp
from jax import Array

from kups.application.relaxation.streaming import StreamingRelaxState
from kups.application.utils.propagate import make_cycle_function
from kups.core.propagator import LoopPropagator, SequentialPropagator, propagate_and_fix
from kups.core.stream import RefillPropagator

# numerical_step comes from make_streaming_relax_propagator.
# refill(state, requested) is the application's input/output adapter.
def repetitions(state: StreamingRelaxState) -> Array:
    run = ~state.finished.data.any() & (state.slot_ordinal >= 0).any()
    return jnp.where(run, 8, 0)

cycle = make_cycle_function(SequentialPropagator((
    RefillPropagator(lambda state: state.finished, refill),
    LoopPropagator(numerical_step, repetitions),
)))

state = propagate_and_fix(cycle, key, state)
```

The refill gate runs **before** the numerical step's error-recovery wrapper.
When refill is requested, the numerical step cannot commit; the host applies the
fix and retries. The zero repetition count avoids evaluating the potential on
repair-only attempts or after input is exhausted. Completed payloads remain in
state, rather than travelling in
`fix_args`, which compiled loops reduce by an elementwise maximum.

Run one such cycle per host iteration. Refill happens on the host, not inside an
uninterrupted device loop; using the bounded repair loop to consume an entire
stream would exhaust its retry budget. Input/output adapters should not retain
references to donated state arrays: extract completed results before replacing
slots. The stream ends when all slot ordinals are `-1` after servicing refill.

Bounded device blocks amortize host synchronization. Completed structures stay
frozen until the block ends; eight steps is an example, not a universal optimum.
Smaller blocks refill sooner, while larger blocks reduce dispatch overhead.
Both the block and slot count should be measured on the intended workload.

Refill is a performance boundary: prepare payloads in bounded batches and JIT a
closure around the pure `replace_slots` operation, not the host callback itself.
Transfer request masks to the host once per refill rather than reading GPU
scalars in a Python loop. The integration test below demonstrates correctness;
throughput also depends on packing, result collection, and host synchronization.

`max_steps` counts parameter updates. One final evaluation records consistent
energy and gradients even when a structure exhausts that budget. FIRE, FIRE2,
L-BFGS, line searches, and stateless Optax transforms support slot reset.
Stateful Optax transforms require an explicit per-system reset adapter.
Custom optimizers advertise that capability through `SupportsReset`; ordinary
fixed-batch optimization only requires the existing `Optimizer` interface.
Non-finite active gradients fail a runtime assertion instead of counting as
convergence. As in fixed-batch relaxation, choose appropriate optimizer controls
(for example, `MaxStepSize`) for strongly distorted structures.

The [streaming integration test](https://github.com/cusp-ai-oss/kUPS/blob/main/test/application/test_relax_streaming.py)
shows a complete source, collector, and refill callback.

## Grand-Canonical Monte Carlo (GCMC)

Simulate adsorption of rigid molecules in a host framework at constant chemical potential, volume, and temperature (μVT ensemble).

| Command | Force Field | Description |
|---------|-------------|-------------|
| `kups_mcmc_rigid` | Lennard-Jones + Ewald | Rigid-body GCMC for gas adsorption in porous materials |

```sh
cd examples
kups_mcmc_rigid mcmc_rigid.yaml
```

**Monte Carlo moves:**

- **Translation** — displace a molecule by a random vector.
- **Rotation** — rotate a molecule about its center of mass.
- **Reinsertion** — delete a molecule and reinsert it at a random position and orientation.
- **Exchange** — insert or delete a molecule based on the chemical potential (fugacity computed via the Peng-Robinson equation of state).

Move probabilities and step sizes are configurable. The simulation supports multiple adsorbate species (CO₂, CH₄, H₂O, N₂, etc.) with pre-defined molecular geometries.

## Widom Test-Particle Insertion

Compute the excess chemical potential, Henry coefficient, and isosteric heat of adsorption for a rigid adsorbate in a host framework. Ghost insertions accumulate Boltzmann factors $\exp(-\beta\Delta U)$ alongside an optional NVT displacement chain over real adsorbates.

| Command | Force Field | Description |
|---------|-------------|-------------|
| `kups_mcmc_widom` | Lennard-Jones + Ewald | Widom ghost insertion for $\mu^\mathrm{ex}$, $K_H$, $q_\mathrm{st}$ |

```sh
cd examples
kups_mcmc_widom mcmc_widom.yaml
```

Per-cycle [`WidomStatistics`][kups.mcmc.widom.WidomStatistics] snapshots are written to the HDF5 output file. [`analyze_widom_file`][kups.application.mcmc.analysis.analyze_widom_file] reduces them post-hoc into block-averaged $\mu^\mathrm{ex}$, $K_H$, and $q_\mathrm{st}$ with standard errors (Vlugt 2008 eq. 16, $N=0$).

# Machine-learning Force Fields

CuspAI publishes JAX exports of MACE and Orb on the Hugging Face Hub — one repository per model so each retains its upstream license:

| Model | Hugging Face repository | License |
|-------|-------------------------|---------|
| [MACE](https://github.com/ACEsuit/mace-foundations) | [CuspAI/kUPS-mace-jax](https://huggingface.co/CuspAI/kUPS-mace-jax) | MIT |
| [MatterSim](https://github.com/microsoft/mattersim) | [CuspAI/kUPS-mattersim-jax](https://huggingface.co/CuspAI/kUPS-mattersim-jax) | MIT |
| [Orb](https://github.com/orbital-materials/orb-models) | [CuspAI/kUPS-orb-jax](https://huggingface.co/CuspAI/kUPS-orb-jax) | Apache 2.0 |

These are re-exports (via [Tojax](https://github.com/cusp-ai-oss/tojax)), not retrainings — weights and architectures are unchanged from upstream.

> Meta's [UMA](https://huggingface.co/facebook/UMA) model is not redistributed by CuspAI. Two routes to run it with <em>k</em>UPS: (1) download the PyTorch checkpoint from Hugging Face and run it natively via [`kups_relax`](#direct-pytorch-bridge) (`potential.backend: uma`); or (2) port it to JAX using [Tojax](https://github.com/cusp-ai-oss/tojax) following the instructions [here](notebooks/potentials.md#tojax-machine-learned-force-fields) and run via `kups_relax` (`potential.backend: tojax`).

Any `potential.model_path:` field accepts either an `hf://<owner>/<repo>/<filename>` URI (fetched via `huggingface_hub.hf_hub_download` and cached on first use) or a local filesystem path to a Tojax-exported `.zip`:

```yaml
potential:
  backend: tojax
  # Remote (HF Hub, requires pip install kups[hf])
  model_path: hf://CuspAI/kUPS-mace-jax/mace-mpa-0-medium_32.zip
  # ...or local (anything readable by TojaxedMliap.from_zip_file)
  # model_path: ./my_model.zip
```

The `hf://` scheme requires the optional `huggingface_hub` dependency: `pip install kups[hf]`. Local paths work without it.

## Direct PyTorch Bridge

The `mace` and `uma` backends (for both `kups_relax` and `kups_md`) load native PyTorch MLFF checkpoints (no Tojax conversion) and run them from JAX via a [DLPack-based bridge](https://pytorch.org/docs/stable/dlpack.html). Functionality across the two paths is identical (forces, stress, optimisation, HDF5 trajectory output) — the differences are about *how* the model is plumbed in.

**Pros**

- **No JAX-traceability requirement.** Any `torch.nn.Module` plugs in as-is: you write one adapter forward that consumes the universal [`AtomGraphInput`][kups.potential.mliap.torch.interface.AtomGraphInput] dict and you're done. The model never has to be expressible in pure JAX ops or jaxprs.
- **Run upstream checkpoints unmodified** (MACE `.model`, UMA `.pt`) — no export step, no risk of conversion drift, easy to diff against the reference implementation.
- **Full upstream feature surface** — every inference toggle the original library exposes is still accessible. For UMA that means tunable [`InferenceSettings`](https://github.com/facebookresearch/fairchem) and all five dataset heads (`omat`/`omol`/`oc20`/`odac`/`omc`) selected per-call, without re-exporting one zip per head.
- **Custom CUDA kernels** (e.g. [cuequivariance](https://github.com/NVIDIA/cuEquivariance) for MACE/UMA) execute on the original PyTorch path, no JAX reimplementation required.
- **UMA turbo is faster than its Tojax export.** `inference_settings: turbo` merges the mixture-of-linear-experts at lazy-init, beating the JAX-exported equivalent on the same model.

**Cons**

- **Per-call JAX↔PyTorch hand-off.** Zero-copy via DLPack but a serial dispatch — adds latency on every potential evaluation.
- **MACE is slower than its Tojax export.** For MACE specifically the JAX build wins; use the bridge mainly for unconverted checkpoints or reference diffs.
- **Extra runtime dependency** — you install the model's own PyTorch package (`mace-torch`, `fairchem-core`, …) alongside kUPS.
- **Python-version constraints** from upstream propagate (e.g. `fairchem-core>=2.0` caps at Python ≤3.13).

**Backends** are selected via the discriminated `potential:` union in the YAML config:

```yaml
# MACE — native .model checkpoint
potential:
  backend: mace
  model_path: ./mace-mpa-0-medium.model
  device: cuda
  dtype: float32           # or float64
```

```yaml
# UMA — native .pt checkpoint via fairchem-core
potential:
  backend: uma
  model_path: ./uma-s-1p2.pt
  device: cuda
  task_name: omat          # one of: omat | omol | oc20 | odac | omc
  inference_settings: default   # or "turbo" (compiled MOLE merge — same composition every call)
```

**Optional extras** install the model's own PyTorch package alongside kUPS:

```sh
pip install kups[mace]   # mace-torch
pip install kups[uma]    # fairchem-core (≥2.0; Python ≤3.13)
```

Under the hood both backends share the universal [`TorchMliap`][kups.potential.mliap.torch.interface.TorchMliap] container — adding a third backend means writing one `torch.nn.Module` that consumes the [`AtomGraphInput`][kups.potential.mliap.torch.interface.AtomGraphInput] schema and returns `{"energy", "position_gradients", "cell_gradients"}`. See `kups.potential.mliap.torch.mace` and `kups.potential.mliap.torch.uma` for the adapter pattern.
