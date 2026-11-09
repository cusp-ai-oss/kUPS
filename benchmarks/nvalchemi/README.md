# kUPS on NVIDIA ALCHEMI ops

Does nvalchemi (`nvalchemi-toolkit-ops`, NVIDIA's Warp kernels for atomistic simulation)
speed up kUPS, and does launching its kernels through jax-ptx instead of
`warp.jax_kernel` remove Warp's per-launch dispatch cost? Measured on one H100 80GB
(JAX 0.12, Warp 1.19, nvalchemi 0.5.0), argon LJ MD at an 8.5 Å cutoff. Full tables are in
`results/h100.md`, raw rows in `results/*.csv`.

## Findings

- **Same answers.** Every nvalchemi variant matches kUPS main in energies, forces and
  cell gradients: about 1e-16 in float64, float32 rounding in float32 (`results/checks.csv`).
- **nvalchemi LJ** (its cell list plus fused energy/force/virial kernel, as a kUPS
  `Potential`): 10–25× faster than kUPS main for one system of 6,912–131,072 atoms,
  19–45× for 64–256 batched systems. It breaks even around 1,000–2,000 atoms in total;
  below that kUPS main is faster (0.46 ms against 0.83 ms per step at 256 atoms).
- **nvalchemi neighbor list only** (kUPS's LJ and masks kept): 2.3–3.2× from about 7,000
  atoms in one system, 1.5× batched. Most of kUPS main's large-system step is the autodiff
  scatter of per-edge forces onto atoms, which this variant keeps.
- **Launch path.** nvalchemi's cell list alone costs 0.86 ms per call through Warp's bridge
  against 0.49 ms through jax-ptx at 256 atoms when XLA does not use CUDA graphs; 0.54
  against 0.47 ms with graphs (XLA's default). Inside a jitted kUPS MD step the two are
  equal within a few percent: XLA captures Warp's FFI calls into CUDA graphs and replays
  them without the Python callback. jax-ptx matters where that capture breaks (an op that
  isn't capturable, eager calls), not in this MD loop.
- **Small systems.** nvalchemi's atom-centric kernels run one thread per atom, and its
  grid uses at least 4 cells per axis, so a small box gets a search radius of 2 cells
  (125 cells per atom, serially). Batching fills the GPU.
- **kUPS main side note.** Its adaptive neighbor list switches from dense to cell list at
  10,000 atoms per system; at 6,912 atoms the dense search is 2.4× slower than kUPS's own
  cell list on this GPU.

## Limits

- `NvalchemiLennardJones`: one species, plain truncated LJ, no exclusions, no per-pair
  mixing; cutoffs in the minimum-image regime.
- `NvalchemiNeighborList`: full self-graphs only; bipartite and `queried_keys` calls (MC)
  fall back to kUPS's cell list.
- nvalchemi 0.5.0's jitted `batch_cell_list` fails out of the box: its kernel-call wrapper
  marks generically annotated arrays static. `kups_nvalchemi._fix_static_arrays` patches it
  (an upstream bug, for both launch paths).
- nvalchemi's JAX cell list inverts the cell with `jnp.linalg.inv` (cuSOLVER), which breaks
  XLA's CUDA graph and costs about 0.25 ms of host time per call.

## Files

- `kups_nvalchemi.py`: `NvalchemiNeighborList`, `NvalchemiLennardJones`, `use_backend`.
- `md_bench.py`: one MD configuration (`--check` compares against kUPS main).
- `op_bench.py`: nvalchemi's cell list alone, stock against jax-ptx, graphs on and off.
- `sweep.py`: suites `checks`, `ops`, `md-size`, `md-batch`; `analyze.py` and
  `to_csv.py` turn its logs into tables.
- `cluster/`: run a suite on a Ray GPU cluster.

jax-ptx side: `jax_ptx_warp.jax_kernel` accepts `warp.jax_kernel`'s arguments
(`in_out_argnames`, per-call dims), so `use_backend("jax-ptx")` swaps it in before
nvalchemi imports Warp's.

## Reproduce

```bash
python md_bench.py --nl nvalchemi-lj-ptx --reps 8 --check     # correctness
python sweep.py checks,ops,md-size > size.log                  # local GPU
cluster/submit.sh <cluster> <wheels dir> -- python sweep.py md-batch
python analyze.py size.log
```

The wheels directory holds the jax-ptx and jax-ptx-warp wheels (jax-ptx branch
`warp-jax-kernel-dropin`).
