"""nvalchemi's JAX cell list alone, jitted: wall time per call, one JSON line per size.

    op_bench.py stock|ptx [--nograph] [--quick]

``stock`` launches nvalchemi's Warp kernels through ``warp.jax_kernel`` (an XLA FFI
callback into Python per launch), ``ptx`` through ``jax_ptx_warp.jax_kernel``.
``--nograph`` disables XLA command buffers, which otherwise capture the call into CUDA
graphs and replay it without the per-launch callbacks. Single systems of N atoms use
``cell_list``; B systems of 256 atoms use ``batch_cell_list``. Argon-like density
(rho = 0.8 sigma^-3 in reduced units, cutoff 2.5), float32 positions, cell sized as
nvalchemi recommends (``estimate_cell_list_sizes``, outside the jit).
"""

from __future__ import annotations

import json
import os
import sys
import time

mode = sys.argv[1]
quick = "--quick" in sys.argv
if "--nograph" in sys.argv:
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

import kups_nvalchemi  # noqa: E402

kups_nvalchemi.use_backend("jax-ptx" if mode == "ptx" else "stock")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from nvalchemiops.jax.neighbors import batch_cell_list, cell_list  # noqa: E402
from nvalchemiops.jax.neighbors.cell_list import estimate_cell_list_sizes  # noqa: E402

kups_nvalchemi._fix_static_arrays()
RC, RHO, MAX_NEIGHBORS = 2.5, 0.8, 96


def timed(f, *a, reps: int = 50) -> float:
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def report(kind: str, n_sys: int, n_per: int, seconds: float) -> None:
    print(json.dumps({"op": kind, "mode": mode, "graphs": "--nograph" not in sys.argv,
                      "systems": n_sys, "atoms_per_system": n_per,
                      "atoms": n_sys * n_per, "ms": 1e3 * seconds,
                      "device": jax.devices()[0].device_kind}), flush=True)


def box(n: int) -> float:
    return (n / RHO) ** (1 / 3)


rng = np.random.default_rng(0)
for n in [256, 4096] if quick else [256, 1024, 4096, 16384, 65536, 262144]:
    L = box(n)
    pos = jnp.asarray(rng.uniform(0, L, (n, 3)), jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32)[None] * L
    pbc = jnp.ones((1, 3), bool)
    cells = int(estimate_cell_list_sizes(pos, cell, RC, pbc)[0])
    f = jax.jit(lambda p, c=cell, b=pbc, m=cells: cell_list(
        p, RC, cell=c, pbc=b, max_neighbors=MAX_NEIGHBORS, max_total_cells=m))
    report("cell_list", 1, n, timed(f, pos))

n = 256
L = box(n)
for b in [1, 16] if quick else [1, 4, 16, 64, 256, 1024]:
    pos = jnp.asarray(rng.uniform(0, L, (b * n, 3)), jnp.float32)
    cell = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32) * L, (b, 3, 3))
    pbc = jnp.ones((b, 3), bool)
    idx = jnp.repeat(jnp.arange(b, dtype=jnp.int32), n)
    ptr = jnp.arange(b + 1, dtype=jnp.int32) * n
    cells = b * int(estimate_cell_list_sizes(pos[:n], cell[:1], RC, pbc[:1])[0])
    f = jax.jit(lambda p, c=cell, s=pbc, i=idx, q=ptr, m=cells: batch_cell_list(
        p, RC, cell=c, pbc=s, batch_idx=i, batch_ptr=q, max_neighbors=MAX_NEIGHBORS,
        max_total_cells=m))
    report("batch_cell_list", b, n, timed(f, pos))
