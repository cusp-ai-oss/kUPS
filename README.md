<div align="center">

<img src="docs/media/logo/logo-readme.svg" alt="kUPS" width="240">
<br>
<img src="docs/media/video/boltzmann_k_cell.gif" width="300" alt="kUPS demo">


**A toolkit for building high-performance molecular simulations on JAX**

*k*UPS provides composable, differentiable primitives — samplers, potentials, and propagators — with hardware acceleration on CPU, GPU, and TPU.

[Documentation](https://cusp-ai-oss.github.io/kups/) | [Quick Start](#quick-start) | [Features](#features) | [Examples](examples/)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-powered-orange.svg)](https://github.com/google/jax)


</div>

---

## Installation

<table>
<tr>
<td><b>Standard Installation</b></td>
<td>

```bash
pip install kups
```

</td>
</tr>
<tr>
<td><b>GPU Support</b></td>
<td>

```bash
pip install kups[cuda]
```

</td>
</tr>
<tr>
<td><b>Development</b></td>
<td>

```bash
git clone https://github.com/cusp-ai-oss/kups.git
cd kups
uv sync
```

</td>
</tr>
</table>

## Quick Start

> [!TIP]
> The repository includes example applications built with *k*UPS in the `examples/` directory.

<details>
<summary><b>Monte Carlo Simulation (GCMC)</b></summary>

```bash
cd examples
kups_mcmc_rigid gcmc_co2_30box.yaml
```

</details>

<details>
<summary><b>Molecular Dynamics (Lennard-Jones)</b></summary>

```bash
cd examples
kups_md_lj md_lj_argon_nvt.yaml
```

</details>

## Features

<table>
<tr>
<td width="50%">

**Simulation Methods**
- **Monte Carlo** — NVT and GCMC ensembles with translation, rotation, reinsertion, and exchange moves
- **Molecular Dynamics** — NVE, NVT, NPT ensembles
- **Geometry Optimization** — FIRE and L-BFGS relaxation

</td>
<td width="50%">

**Force Fields & Potentials**
- **Lennard-Jones** potential
- **Coulomb** interactions (Ewald summation)
- **Harmonic** bonds and angles
- **Morse** potential
- **MACE** and **UMA** ML force fields

</td>
</tr>
<tr>
<td width="50%">

**Core Capabilities**
- **Composable** — shared propagator interface; methods and potentials snap together freely
- **Batched** — run thousands of independent simulations as vectorized computations

</td>
<td width="50%">

**Performance & Integration**
- **GPU-native** — JIT-compiled on CPU, GPU, and TPU with no code changes
- **Differentiable** — full automatic differentiation via JAX
- **PyTorch interop** — bring PyTorch models into JAX via [Tojax](https://github.com/cusp-ai-oss/tojax)

</td>
</tr>
</table>

## Documentation

Full documentation is available at **[cusp-ai-oss.github.io/kups](https://cusp-ai-oss.github.io/kups/)**.

### Building docs locally

To build the docs locally run `./docs/scripts/build.sh`, which executes and renders all documentation notebooks and generates API pages in markdown.

```bash
./docs/scripts/build.sh           # build into site/
./docs/scripts/build.sh --serve   # serve with live updates on http://127.0.0.1:8000
```

---

## Citation

If you use *k*UPS in your research, please cite:

```bibtex
@software{kups2026,
  author = {Gao, Nicholas
    and K{\"o}hler, Jonas
    and Hanke, Felix
    and Ramanan, Anita
    and Moubarak, Elias
    and Morrow, Joe
    and de Haan, Pim
    and Openshaw, Hannah
    and Welling, Max
    and CuspAI Team},
  title = {kUPS - a universal particle simulation toolkit},
  year = {2026},
  url = {https://github.com/cusp-ai-oss/kups}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
