# Streaming relaxation throughput — L4, 2026-09-08

Streaming helps when convergence times differ, not merely because input is
called a stream. This benchmark compares two refill policies over the same
numerical propagator, optimizer, and shuffled inputs:

- **Fixed:** retain a batch until every occupied slot completes.
- **Streaming:** replace completed slots at the next host boundary.

Both use JIT-compiled slot replacement, batched host snapshots, and bounded
`LoopPropagator` blocks. No device queue, threads, or alternate numerical kernel.
Thus the comparison isolates scheduling, rather than comparing against an
intentionally slow fixed-batch implementation or against the full logging CLI.

## Workload and validation

NVIDIA L4; JAX 0.10.0; FP32; 108-atom periodic FCC argon structures;
Lennard–Jones cutoff 5 Å; FIRE with a 0.2 Å `MaxStepSize`; force tolerance
0.01 eV/Å; maximum 400 updates. Seed 20260908 shuffles repeated copies of 32
templates. Their rattle amplitudes are 16 × 0.002, 8 × 0.03, 6 × 0.15, and
2 × 0.35 Å. This deliberately broad mix includes initially converged structures.

Each stream contains eight batches' worth of jobs. Every job must appear exactly
once, return finite outputs, and meet the common force tolerance or the reported
budget limit. In the measurements below **none hit the budget**, and energies,
positions, and update counts matched the fixed-batch reference exactly. The
update-count quantiles (min/median/p90/p99/max) were 0 / 7.5 / 42 / 123 / 123.

Timings synchronize GPU completion and include refill, host/device transfers,
and result snapshots. Initial CPU packing (~8–9 seconds), full-stream JIT warmup,
and final NumPy result comparison are excluded. No file I/O or ML potential is
timed. Template reuse is a controlled scheduling experiment, not a measurement
of loading thousands of unique structures.

## Follow-up: remove unnecessary image replication

The capacity estimator rounded and applied its safety multiplier to candidate
counts **before** weighting them by periodic images, then applied that multiplier
again to the final image estimate. With this fixture's multiplier of 2, that
produced 256 candidate rows but 512 image-candidate rows per particle, despite
every system needing only one image. Unequal capacities selected the general
replication pipeline instead of the existing minimum-image fast path.

The fix accumulates the unrounded `candidates * images` estimate, then applies
headroom and rounding once. Both capacities are now 256. This changes one
production expression, adds no streaming-specific optimization, and retains
runtime capacity assertions and repair for underestimated buffers.

Same fixture and two post-warmup trials; fixed uses 32-step blocks and streaming
uses eight-step blocks. "Before" is the initial measurement below, not a new
fixed-batch baseline:

| Slots | Streaming before (structures/s) | Streaming now (structures/s) | Improvement | Fixed now (structures/s) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 89.0 | 190.1 | 2.14× | 83.4 |
| 128 | 68.5 | 340.2 | 4.97× | 125.5 |
| 512 | 66.7 | 430.1 | 6.45× | 142.0 |

At 512 slots (55,296 atoms; 4,096 completed jobs), streaming now takes 9.52
seconds instead of 61.41. It remains 3.03× faster than the **also corrected**
fixed-batch path. Peak live JAX allocation fell from about 2.16 GB to 0.55 GB.

All jobs converged without hitting the budget. Within each rerun, fixed and
streaming positions, energies, and iteration counts matched exactly; the
iteration quantiles and evaluation counts also stayed the same as before.

At 128 slots, four-step streaming reached 298.8 structures/s and 16-step
streaming reached 348.0, versus 340.2 with eight steps. The small 16-step gain
is not enough evidence for a universal default; the measured tradeoff has
shifted as numerical work became cheaper. No scheduling defaults were changed.

Regression tests cover minimum-image capacity equality with heterogeneous
systems and safety multipliers both below and above one, plus single application
of headroom when multiple images are needed. The neighbor-list and streaming
tests also exercise periodic images, skewed cells, and capacity repair:
428 neighbor-list/streaming tests and 28 ASE comparison tests passed. Strict
Pyrefly checking passed for all changed Python files and the benchmark, without
new suppressions.

## Initial heterogeneous throughput (before capacity correction)

Two measured trials after warmup. Rates use the mean elapsed time. Fixed batches
use 32-step blocks; streaming uses eight-step blocks.

| Slots | Atoms in batch | Jobs | Fixed structures/s | Streaming structures/s | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3,456 | 256 | 30.1 | 89.0 | 2.96× |
| 128 | 13,824 | 1,024 | 20.8 | 68.5 | 3.30× |
| 512 | 55,296 | 4,096 | 19.7 | 66.7 | 3.39× |

At 128 slots, fixed batching performed 1,024 batch evaluations versus 296 for
streaming. Useful slot-evaluations rose from 15.6% to 53.9%; the remaining waste
includes within-block waiting and the final partially occupied tail. This is
the intended benefit, not a change in the convergence criterion.

More slots did not automatically improve absolute throughput for this particular
LJ/neighbor-list pipeline. GPU utilization was around 99–100% during numerical
work at 128 and 512 slots. The 512-slot process peaked at about 2.16 GB of live
JAX allocations (about 3.26 GB in its allocator pool). Utilization alone does not identify the limiting
kernel or establish an optimal batch size for another potential.

## Initial block-size and equal-work controls

At 32 slots, one-step streaming achieved 64.6 structures/s versus 89.0 with
eight-step blocks. Host refill time fell from about 0.815 to 0.257 seconds per
stream. At 128 slots, increasing streaming blocks from eight to 32 steps reduced
throughput from 68.5 to 46.6 structures/s: the saved host calls did not compensate
for keeping completed slots occupied longer. Eight is a useful measured setting
for this fixture, not a hard-coded optimum.

In the **equal-work control**, every structure was identical and needed 41
updates. Fixed and streaming policies both used eight-step blocks and both
performed 384 batch evaluations for 256 jobs: approximately 70.8 versus 70.6
structures/s. There was no meaningful streaming advantage without a long tail.

In an **all-inputs-need-work control** (`--minimum-rattle 0.03`), every structure
needed 15–123 updates. At 32 slots, fixed batching achieved 30.1 structures/s
versus 78.4 for streaming, a 2.61× gain. Results and iteration counts again
matched exactly, with no capped jobs. The speedup therefore does not depend on
including already-converged inputs.

## Implementation and correctness follow-up

The documented composition now uses bounded device blocks, skips numerical work
on repair-only attempts and after exhaustion, JIT-compiles pure replacement, and
transfers masks/snapshots in batches. These are existing kUPS abstractions.

Hard inputs exposed two important correctness checks: use appropriate step
control, and never count non-finite gradients as convergence. The convergence
helper now promotes non-finite row norms before segment reduction and raises a
runtime assertion for invalid active gradients. Padding stays excluded.
Tests cover this, blocked refill/exhaustion, and preserving committed progress
when a capacity repair occurs partway through a device block.

## Reproduction

From the repository root:

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false .venv/bin/python benchmarks/bench_relax_streaming.py --slots 128 --waves 8 --trials 2 --modes fixed:32,stream:8,stream:32
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false .venv/bin/python benchmarks/bench_relax_streaming.py --slots 128 --waves 8 --trials 2 --modes fixed:32,stream:8,stream:4,stream:16
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false .venv/bin/python benchmarks/bench_relax_streaming.py --slots 512 --waves 8 --trials 2 --modes fixed:32,stream:8
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false .venv/bin/python benchmarks/bench_relax_streaming.py --slots 32 --waves 8 --trials 2 --modes fixed:8,stream:8 --homogeneous
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=false .venv/bin/python benchmarks/bench_relax_streaming.py --slots 32 --waves 8 --trials 2 --modes fixed:32,stream:8 --minimum-rattle 0.03
```

The script emits JSONL containing timing, actual batch evaluations, useful-work
fraction, convergence counts, result differences, and device memory statistics.
For a real workflow, repeat with its potential, atom-count distribution, input
adapter, and output requirements; do not extrapolate these rates to MLFFs.
