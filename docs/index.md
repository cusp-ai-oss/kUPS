# kUPS { .visually-hidden }

<p align="center">
  <img src="media/logo/logo.svg" alt="kUPS" width="240">
</p>

<em>k</em>UPS is a toolkit for building high-performance molecular simulations on JAX. It provides composable, differentiable primitives — samplers, potentials, and propagators — with hardware acceleration on CPU, GPU, and TPU.

## Installation

```sh
pip install kups
```

For GPU support, install JAX with CUDA separately:

```sh
pip install jax[cuda]
```

## Quick Start

The repository includes example applications built with <em>k</em>UPS. To try them, run from the `examples/` directory:

```sh
cd examples
kups_mcmc_rigid --config gcmc_co2_30box.yaml
```

```sh
cd examples
kups_md_lj --config md_lj_argon_nvt.yaml
```

See the [examples/](https://github.com/cusp-ai-oss/kups/tree/main/examples) directory for more configurations.

## Features

- **Composable** — every operation is a propagator with a shared interface; methods and potentials snap together freely
- **Monte Carlo** — NVT and GCMC ensembles with translation, rotation, reinsertion, and exchange moves
- **Molecular dynamics** — NVE, NVT, NPT ensembles
- **Geometry optimization** — FIRE and L-BFGS relaxation
- **Force fields** — Lennard-Jones, Coulomb (Ewald summation), harmonic bonds/angles, Morse, MACE, UMA
- **Differentiable** — full automatic differentiation through simulations via JAX
- **Batched** — run thousands of independent simulations as a single vectorized computation
- **GPU-native** — JIT-compiled on CPU, GPU, and TPU with no code changes
- **PyTorch interop** — bring any PyTorch model into JAX via [Tojax](https://github.com/cusp-ai-oss/tojax)

## License

Apache License 2.0 — see [LICENSE](https://github.com/cusp-ai-oss/kups/blob/main/LICENSE).
