# *k*UPS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A toolkit for building high-performance molecular simulations on JAX. *k*UPS provides composable, differentiable primitives — samplers, potentials, and propagators — with hardware acceleration on CPU, GPU, and TPU.

## Installation

```sh
pip install kups
```

For GPU support:

```sh
pip install kups[cuda]
```

For development from source:

```sh
git clone https://github.com/cusp-ai-oss/kups.git
cd kups
uv sync
```

## Quick Start

The repository includes example applications built with *k*UPS. To try them, run from the `examples/` directory:

```sh
cd examples
kups_mcmc_rigid --config gcmc_co2_30box.yaml
```

```sh
cd examples
kups_md_lj --config md_lj_argon_nvt.yaml
```

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

## Documentation

Full documentation is available at [cusp-ai-oss.github.io/kups](https://cusp-ai-oss.github.io/kups/).

## Citation

```bibtex
@software{kups2026,
  author = {{Cusp AI}},
  title = {kUPS},
  year = {2026},
  url = {https://github.com/cusp-ai-oss/kups}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
