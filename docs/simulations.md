# Packaged Simulations

<em>k</em>UPS ships with several ready-to-use simulation applications as CLI tools. Each is a thin layer built on the core primitives (propagators, potentials, lenses, tables) and serves as both a useful tool and a reference implementation. All commands take a YAML configuration file via and use [nanoargs](https://github.com/cusp-ai-oss/nanoargs) for argument parsing, so any configuration value can also be overridden from the command line. Example configurations are provided in the [`examples/`](https://github.com/cusp-ai-oss/kups/tree/main/examples) directory and should be run from there.

## Molecular Dynamics

Run molecular dynamics trajectories in the NVE, NVT, or NPT ensemble.

| Command | Force Field | Description |
|---------|-------------|-------------|
| `kups_md_lj` | Lennard-Jones | Classical pair potential with optional tail corrections and mixing rules |
| `kups_md_mlff` | MACE, UMA, ORB | Machine-learned interatomic potentials loaded via [Tojax](https://github.com/cusp-ai-oss/tojax) |

```sh
cd examples
kups_md_lj md_lj_argon_nvt.yaml
kups_md_lj md_lj_argon_nve.yaml
kups_md_mlff md_mace.yaml
kups_md_mlff md_uma.yaml
kups_md_mlff md_orb.yaml
```

**Ensembles and integrators:**

- **NVE** — velocity Verlet. Constant energy, useful for validating energy conservation.
- **NVT** — Langevin thermostat (BAOAB splitting) or canonical sampling via velocity rescaling (CSVR). Constant temperature.
- **NPT** — CSVR thermostat with stochastic cell rescaling barostat. Constant temperature and pressure.

All integrators are built from the same composable propagator primitives described in the Propagators tutorial.

## Geometry Optimization

Relax atomic positions (and optionally lattice vectors) to a local energy minimum.

| Command | Force Field | Description |
|---------|-------------|-------------|
| `kups_relax_lj` | Lennard-Jones | Classical relaxation |
| `kups_relax_mlff` | MACE, UMA, ORB | Machine-learned force field relaxation |

```sh
cd examples
kups_relax_mlff relax_mace.yaml
kups_relax_mlff relax_orb.yaml
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

# Machine-learning Force Fields

CuspAI publishes JAX exports of MACE, UMA, and Orb on the Hugging Face Hub — one repository per model so each retains its upstream license:

| Model | Hugging Face repository | License |
|-------|-------------------------|---------|
| [MACE](https://github.com/ACEsuit/mace-foundations) | [CuspAI/kUPS-mace-jax](https://huggingface.co/CuspAI/kUPS-mace-jax) | MIT |
| [UMA](https://huggingface.co/facebook/UMA) | [CuspAI/kUPS-uma-jax](https://huggingface.co/CuspAI/kUPS-uma-jax) (gated) | FAIR Chemistry License v1 |
| [Orb](https://github.com/orbital-materials/orb-models) | [CuspAI/kUPS-orb-jax](https://huggingface.co/CuspAI/kUPS-orb-jax) | Apache 2.0 |

These are re-exports (via [Tojax](https://github.com/cusp-ai-oss/tojax)), not retrainings — weights and architectures are unchanged from upstream. UMA inherits Meta's gated distribution: request access on both the [upstream page](https://huggingface.co/facebook/UMA) and [CuspAI/kUPS-uma-jax](https://huggingface.co/CuspAI/kUPS-uma-jax), then `huggingface-cli login` before downloading.

Any `model_path:` field accepts either an `hf://<owner>/<repo>/<filename>` URI (fetched via `huggingface_hub.hf_hub_download` and cached on first use) or a local filesystem path to a Tojax-exported `.zip`:

```yaml
# Remote (HF Hub, requires pip install kups[hf])
model_path: hf://CuspAI/kUPS-mace-jax/mace-mpa-0-medium_32.zip
model_path: hf://CuspAI/kUPS-uma-jax/uma-s-1p2_omat.zip        # heads: oc20, odac, omat, omc, omol
model_path: hf://CuspAI/kUPS-orb-jax/orb_v3_conservative_inf_omat.zip

# Local (anything readable by TojaxedMliap.from_zip_file)
model_path: ./my_model.zip
model_path: /absolute/path/to/my_model.zip
```

The `hf://` scheme requires the optional `huggingface_hub` dependency: `pip install kups[hf]`. Local paths work without it.
