# Packaged Simulations

<em>k</em>UPS ships with several ready-to-use simulation applications as CLI tools. Each is a thin layer built on the core primitives (propagators, potentials, lenses, tables) and serves as both a useful tool and a reference implementation. All commands take a YAML configuration file via and use [nanoargs](https://github.com/cusp-ai-oss/nanoargs) for argument parsing, so any configuration value can also be overridden from the command line. Example configurations are provided in the [`examples/`](https://github.com/cusp-ai-oss/kups/tree/main/examples) directory and should be run from there.

## Molecular Dynamics

Run molecular dynamics trajectories in the NVE, NVT, or NPT ensemble with `kups_md`. The force field is selected by the `potential.backend` field:

| `potential.backend` | Force Field | Description |
|---------------------|-------------|-------------|
| `lj` | Lennard-Jones | Classical pair potential with configurable mixing rules |
| `lj_coulomb` | Lennard-Jones + Coulomb | Lennard-Jones plus Ewald or particle-mesh Ewald electrostatics of the input charges |
| `tojax` | MACE, UMA, ORB | Machine-learned interatomic potentials loaded via [Tojax](https://github.com/cusp-ai-oss/tojax) |
| `mace`, `uma` | MACE, UMA | Native PyTorch MLFFs bridged into JAX (see [Direct PyTorch Bridge](#direct-pytorch-bridge)) |

```sh
cd examples
kups_md md_lj_argon_nvt.yaml
kups_md md_lj_argon_nve.yaml
kups_md md_lj_argon_npt.yaml
kups_md md_lj_argon_nvt_from_npt.yaml  # continues the NPT run
kups_md md_mace.yaml
kups_md md_orb.yaml
```

**Ensembles and integrators:**

- **NVE** — velocity Verlet. Constant energy, useful for validating energy conservation.
- **NVT** — Langevin thermostat (BAOAB splitting) or canonical sampling via velocity rescaling (CSVR). Constant temperature.
- **NPT** — CSVR thermostat with stochastic cell rescaling barostat (`csvr_npt`), or fully-flexible-cell BAOAB NPT Langevin (`baoab_npt_langevin`, Gao–Fang–Wang JCP 2016). Constant temperature and pressure.

All integrators are built from the same composable propagator primitives described in the Propagators tutorial.

**Electrostatics:** `lj_coulomb` adds the Coulomb interactions of the input charges (`_atom_site_charge` or `_atom_type_partial_charge` in CIF files, else the atoms' initial charges, as extxyz files store them) to Lennard-Jones. Every pair of particles interacts, with no bonded exclusions, so it suits ionic systems without bonds. `coulomb.method` sums the reciprocal part on a particle mesh (`pme`, the default) or over k-vectors (`ewald`, faster only for cells of up to a few hundred atoms). `coulomb.precision` (default 1e-6, as `ewald.precision` in GCMC) is the target of the estimated real- and reciprocal-space truncation errors, not a force error: at the default and an 8 Å cutoff, Coulomb forces on rattled rock salt with charges of ±1 e are within about 3e-5 eV/Å (RMS) of a converged sum, and the Coulomb pressure within about 10 bar. The screening and the k-vectors or the mesh are estimated from it once, at the input cells, so a cell that changes a lot under NPT or in a cell relaxation loses some accuracy. The real-space sum uses `lj.cutoff` and the same skin list. `lj.kernel_dtype` sets the precision of the Lennard-Jones pair terms only; the Coulomb terms run in that of the positions. Cells must be periodic, and every system needs charges. GCMC and Widom keep direct Ewald, whose structure factors update incrementally.

```yaml
potential:
  backend: lj_coulomb
  lj:
    cutoff: 10.0
    parameters:
      Na: [2.25, 0.01]  # (sigma [Å], epsilon [eV])
      Cl: [4.0, 0.01]
    mixing_rule: lorentz_berthelot
  coulomb:
    method: pme
    precision: 1.e-6
```

**Output:** `run.out_file` is an HDF5 file with four groups. `step` holds the energies, stress and volume of each system every `run.thermo_every` steps; `frames` holds positions (wrapped into the cell), momenta and cells every `run.frames_every` steps; `init` and `final` hold the particle and system tables after the first and the last block. `run.num_steps` and both intervals are multiples of `run.block_size`, the steps fused per dispatch; the intervals default to it. Rows and frames are taken at the end of each interval and carry their MD step, warmup included. Particles keep the input order.

```yaml
run:
  block_size: 10
  thermo_every: 10   # energies every 10 steps
  frames_every: 500  # positions and momenta every 500 steps
```

```python
from kups.application.md.logging import MDLoggedData
from kups.core.constants import FEMTO_SECOND
from kups.core.storage import HDF5StorageReader

with HDF5StorageReader[MDLoggedData]("md_nvt_lj_argon.h5") as reader:
    frames = reader.focus_group(lambda s: s.frames)[:]
    masses = reader.focus_group(lambda s: s.init).read(select=lambda d: d.atoms.data.masses)
velocities = frames.momenta / masses[:, None] * FEMTO_SECOND  # Å/fs
```

**Continuing a run:** an `inp_files` entry may be the output of an earlier `kups_md` run. The new run starts from its final state, the positions, momenta and cells of all its systems in the same particle order, and takes everything else, the index and float dtypes included, from its own settings; forces are evaluated again before the first step. `md.rescale_to_mean_volume: true` first scales each cell and its positions to the mean volume of the earlier run's rows, which keeps the cell shape and the fractional coordinates. The mean covers all rows, so equilibrate in `run.num_warmup_steps` of the earlier run. `md.initialize_momenta: true` draws new momenta. Otherwise a Verlet run continues the earlier one to round-off; stochastic integrators draw new noise, and BAOAB NPT derives its barostat mass from the starting cell, as for a structure file. To run NVT at the mean density of an NPT run (`md_lj_argon_npt.yaml`, then `md_lj_argon_nvt_from_npt.yaml` in `examples/`):

```yaml
md:
  integrator: csvr
  rescale_to_mean_volume: true

inp_files:
  - md_npt_lj_argon.h5
```

## Geometry Optimization

Relax atomic positions (and optionally lattice vectors) to a local energy minimum with `kups_relax`. The force field is selected by the `potential.backend` field:

| `potential.backend` | Force Field | Description |
|---------------------|-------------|-------------|
| `lj` | Lennard-Jones | Classical relaxation |
| `lj_coulomb` | Lennard-Jones + Coulomb | Classical relaxation with electrostatics, as in `kups_md` |
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
