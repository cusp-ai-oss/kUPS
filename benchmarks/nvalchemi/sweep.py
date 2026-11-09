"""Run a benchmark suite, one process per configuration; print ``RESULT {json}`` lines.

    sweep.py SUITE[,SUITE...] [--quick]

checks    energies, forces and cell gradients of each nvalchemi variant vs kUPS main.
ops       nvalchemi's JAX cell list alone: stock warp.jax_kernel vs jax-ptx, with and
          without XLA command buffers (CUDA graphs), single systems of N atoms and B
          batched 256-atom systems.
md-size   kUPS LJ MD step, one system of 4 R^3 argon atoms, R in 4..32, f64 and f32.
md-batch  kUPS LJ MD step, B systems of 256 or 864 atoms, steps dispatched one at a time
          and fused 20 per dispatch.

A configuration that fails (e.g. out of memory) or times out prints ``RESULT`` with its
error and the suite goes on.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time

IMPLS = ["kups", "nvalchemi-ptx", "nvalchemi-stock", "nvalchemi-lj-ptx",
         "nvalchemi-rows-ptx", "nvalchemi-rows-stock"]


def run(cmd: list[str], config: dict, timeout: float = 1500) -> None:
    t = time.monotonic()
    try:
        p = subprocess.run([sys.executable, *cmd], capture_output=True, text=True,
                           timeout=timeout)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        if p.returncode == 0 and lines:
            for ln in lines:
                print("RESULT", json.dumps({**config, **json.loads(ln)}), flush=True)
            return
        err = (p.stderr.strip().splitlines() or ["no output"])[-1]
    except subprocess.TimeoutExpired:
        err = f"timeout after {timeout:.0f} s"
    print("RESULT", json.dumps({**config, "error": err[:300],
                                "seconds": round(time.monotonic() - t)}), flush=True)


def checks(quick: bool) -> None:
    """Energies, forces and cell gradients of every nvalchemi variant against kUPS."""
    for impl, (r, b), f32 in itertools.product(IMPLS[1:], [(6, 1), (4, 8)], [False, True]):
        run(["md_bench.py", "--nl", impl, "--reps", str(r), "--batch", str(b), "--check"] +
            (["--f32"] if f32 else []), {"suite": "checks", "reps": r, "batch": b,
                                         "dtype": "f32" if f32 else "f64"})


def ops(quick: bool) -> None:
    for mode, graphs in itertools.product(["stock", "ptx"], [True, False]):
        cmd = ["op_bench.py", mode] + ([] if graphs else ["--nograph"]) + \
            (["--quick"] if quick else [])
        run(cmd, {"suite": "ops", "mode": mode, "graphs": graphs})


def md_size(quick: bool) -> None:
    reps = [4, 8] if quick else [4, 6, 8, 12, 16, 24, 32]
    for dtype, r, impl in itertools.product(["f64", "f32"], reps, IMPLS):
        cmd = ["md_bench.py", "--nl", impl, "--reps", str(r), "--steps", "100"] + \
            (["--f32"] if dtype == "f32" else [])
        run(cmd, {"suite": "md-size"})


def md_batch(quick: bool) -> None:
    reps = [4] if quick else [4, 6]
    batches = [1, 16] if quick else [1, 4, 16, 64, 256]
    for r, b, impl in itertools.product(reps, batches, IMPLS):
        run(["md_bench.py", "--nl", impl, "--reps", str(r), "--batch", str(b),
             "--steps", "100"], {"suite": "md-batch"})
    fused = ["kups", "nvalchemi-lj-ptx", "nvalchemi-rows-ptx", "nvalchemi-rows-stock"]
    for r, b, impl in itertools.product(reps, batches, fused):
        run(["md_bench.py", "--nl", impl, "--reps", str(r), "--batch", str(b),
             "--steps", "200", "--block", "20"], {"suite": "md-batch"})


def profile(quick: bool) -> None:
    """Device time per step and its largest kernels, single systems."""
    for dtype, r, impl in itertools.product(["f64", "f32"], [8, 20, 32],
                                            ["kups", "nvalchemi-lj-ptx", "nvalchemi-rows-ptx"]):
        run(["md_bench.py", "--nl", impl, "--reps", str(r), "--steps", "60",
             "--profile", "6"] + (["--f32"] if dtype == "f32" else []),
            {"suite": "profile"})


if __name__ == "__main__":
    suite, quick = sys.argv[1], "--quick" in sys.argv
    suites = {"checks": checks, "ops": ops, "md-size": md_size, "md-batch": md_batch,
              "profile": profile}
    for name in suite.split(","):
        suites[name](quick)
