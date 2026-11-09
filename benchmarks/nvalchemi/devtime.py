"""Device-side kernel times of a jitted function from a JAX profiler trace."""

from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import tempfile
from collections.abc import Callable
from typing import Any

import jax


def _trace(f: Callable[..., Any], args: tuple[Any, ...], reps: int) -> dict[str, Any]:
    jax.block_until_ready(f(*args))
    d = tempfile.mkdtemp()
    with jax.profiler.trace(d, create_perfetto_trace=True):
        for _ in range(reps):
            jax.block_until_ready(f(*args))
    (path,) = glob.glob(os.path.join(d, "**", "*.trace.json.gz"), recursive=True)
    with gzip.open(path) as fh:
        return json.load(fh)


def _gpu_pids(trace: dict[str, Any]) -> set[int]:
    return {e["pid"] for e in trace["traceEvents"] if e.get("ph") == "M"
            and e.get("name") == "process_name"
            and "/device:GPU" in str(e.get("args", {}).get("name", ""))}


def device_kernels(f: Callable[..., Any], *args: Any, reps: int = 10) -> dict[str, float]:
    """Mean device time per call (µs) of each kernel ``f(*args)`` launches."""
    trace = _trace(f, args, reps)
    pids = _gpu_pids(trace)
    times: dict[str, float] = collections.defaultdict(float)
    for e in trace["traceEvents"]:
        if e.get("ph") == "X" and e.get("pid") in pids and "dur" in e:
            times[e["name"]] += e["dur"] / reps
    return dict(sorted(times.items(), key=lambda kv: -kv[1]))


def timeline(f: Callable[..., Any], *args: Any) -> list[tuple[float, float, bool, str]]:
    """One call's events: (start µs from the first event, duration µs, on GPU, name)."""
    trace = _trace(f, args, 1)
    pids = _gpu_pids(trace)
    ev = [(e["ts"], e.get("dur", 0), e["pid"] in pids, e["name"])
          for e in trace["traceEvents"] if e.get("ph") == "X"]
    t0 = min(e[0] for e in ev)
    return sorted((t - t0, d, g, n) for t, d, g, n in ev)
