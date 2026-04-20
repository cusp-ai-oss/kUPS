# Packaged Simulations

<em>k</em>UPS ships with several ready-to-use simulation applications as CLI tools. Each is a thin layer built on the core primitives (propagators, potentials, lenses, tables) and serves as both a useful tool and a reference implementation. All commands take a YAML configuration file via `--config` and use [nanoargs](https://github.com/cusp-ai-oss/nanoargs) for argument parsing, so any configuration value can also be overridden from the command line. Example configurations are provided in the [`examples/`](https://github.com/cusp-ai-oss/kups/tree/main/examples) directory and should be run from there.

## Molecular Dynamics

Run molecular dynamics trajectories in the NVE, NVT, or NPT ensemble.

| Command | Force Field | Description |
|---------|-------------|-------------|
| `kups_md_lj` | Lennard-Jones | Classical pair potential with optional tail corrections and mixing rules |
| `kups_md_mlff` | MACE, UMA, ORB | Machine-learned interatomic potentials loaded via [Tojax](https://github.com/cusp-ai-oss/tojax) |

```sh
cd examples
kups_md_lj --config md_lj_argon_nvt.yaml
kups_md_lj --config md_lj_argon_nve.yaml
kups_md_mlff --config md_mace.yaml
kups_md_mlff --config md_uma.yaml
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
kups_relax_mlff --config relax_mace.yaml
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
kups_mcmc_rigid --config mcmc_rigid.yaml
```

**Monte Carlo moves:**

- **Translation** — displace a molecule by a random vector.
- **Rotation** — rotate a molecule about its center of mass.
- **Reinsertion** — delete a molecule and reinsert it at a random position and orientation.
- **Exchange** — insert or delete a molecule based on the chemical potential (fugacity computed via the Peng-Robinson equation of state).

Move probabilities and step sizes are configurable. The simulation supports multiple adsorbate species (CO₂, CH₄, H₂O, N₂, etc.) with pre-defined molecular geometries.
