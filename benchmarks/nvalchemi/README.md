# kUPS on NVIDIA ALCHEMI ops (stack at #465)

Should kUPS take over anything from nvalchemi (`nvalchemi-toolkit-ops`, NVIDIA's Warp
kernels for atomistic simulation) for MD, now that kUPS steps over a Verlet skin list and
evaluates LJ over neighbor rows (float32 in a Pallas kernel)? Measured on one H100 80GB
(JAX 0.12, Warp 1.19, nvalchemi 0.5.0), argon LJ MD at an 8.5 Å cutoff with a 1 Å skin,
built as `kups_md` builds it. Full tables in `results/h100.md`, raw logs in `results/`.
The same benchmark against kUPS main is on branch `jonkhler/nvalchemi-bench`.

## Variants

- `kups`: the stack's MD path (skin list built by `AdaptiveNeighborList`, LJ over its rows).
- `nvalchemi-ptx|stock`: the same, with nvalchemi's cell list building the skin list.
- `nvalchemi-rows-ptx|stock`: nvalchemi's fused LJ kernel on kUPS's skin-list rows; pair
  kernel against pair kernel, same pairs and rebuild policy.
- `nvalchemi-lj-ptx`: nvalchemi's cell list every step plus its fused LJ kernel.

`--f32` is the app's fast setting (float32 state, Pallas row kernel, int32 indices);
otherwise float64 throughout. All variants match `kups` to about 1e-15 in float64 and to
float32 rounding in float32.

## Findings

- **float32: kUPS is faster everywhere.** The Pallas row kernel takes 228 µs at 131,072
  atoms against 2,010 µs for nvalchemi's kernel on the same rows (it accumulates in float64
  internally). MD step at 131,072 atoms: 1.21 ms against 3.66 ms; 256 × 864 atoms with
  20 steps per dispatch: 1.01 ms against 1.78 ms.
- **float64: nvalchemi's kernel is faster than kUPS's XLA row path.** 1.6 ms against about
  3.2 ms of device time per step at 131,072 atoms; up to 2.7× per step for 256 batched
  systems (221,184 atoms: 2.37 ms against 6.33 ms) and 2.9–3.3× with 20 steps per dispatch.
  kUPS has no float64 fused row kernel; that is the one gap.
- **Neighbor list: no gain from nvalchemi.** The skin list rebuilds rarely, so its builder
  barely shows, except at 6,912 atoms, where `AdaptiveNeighborList` rebuilds with the dense
  O(N²) search (1.88 ms against 0.77 ms with a cell-list builder): a kUPS dispatch setting.
- **nvalchemi's own cell list every step** is never faster than its kernel on kUPS's rows.
- **Launch path.** Stock `warp.jax_kernel` and jax-ptx are equal within noise, including
  inside the skin list's `lax.cond` rebuild.
- **Small systems are host-bound.** One step per dispatch costs about 0.7 ms of wall time
  against 0.08–0.13 ms of device time for every variant; 20 steps per dispatch bring kUPS to
  0.08–0.13 ms per step.

## What to take over

Nothing for LJ or neighbor lists. Close the float64 gap with a float64 row kernel in kUPS
(Pallas if it lowers well in float64, else loom), with nvalchemi's float64 kernel as the
bar, and fix the adaptive builder's crossover. nvalchemi remains a candidate for physics
kUPS lacks (DFT-D3, damped shifted force), wrapped as an optional backend reading kUPS's
skin-list rows, as `NvalchemiRowsLennardJones` does for LJ.

## Files

- `kups_nvalchemi.py`: `NvalchemiNeighborList` (skin-list builder),
  `NvalchemiRowsLennardJones`, `NvalchemiLennardJones`, `use_backend`.
- `md_bench.py`: one MD configuration (`--check` compares against `kups`, `--profile N`
  records device time and the top kernels).
- `sweep.py`: suites `checks`, `md-size`, `md-batch`, `md-batch-f32`, `profile`, `ops`;
  `analyze.py` turns their logs into tables.
- `cluster/`: run a suite on a Ray GPU cluster.

## Reproduce

```bash
python md_bench.py --nl nvalchemi-rows-ptx --reps 8 --check
cluster/submit.sh <cluster> <wheels dir> -- python sweep.py checks,md-size,profile
python analyze.py results/h100_*.log
```

The wheels directory holds the jax-ptx and jax-ptx-warp wheels (jax-ptx branch
`warp-jax-kernel-dropin`).
