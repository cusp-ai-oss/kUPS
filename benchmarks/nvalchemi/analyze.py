"""Tables from sweep logs: ``analyze.py LOG [LOG...]`` (lines ``... RESULT {json}``).

Prints markdown tables: correctness checks; nvalchemi's cell list alone (stock vs
jax-ptx, with and without CUDA graphs); kUPS MD step time per implementation over system
size and batch size, with the speedup over kUPS main.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

IMPLS = ["kups", "kups-cell", "nvalchemi-ptx", "nvalchemi-stock", "nvalchemi-lj-ptx",
         "nvalchemi-lj-stock"]


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if "RESULT {" in line:
                    rows.append(json.loads(line.split("RESULT ", 1)[1]))
    return rows


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}g}"


def checks(rows: list[dict]) -> None:
    rs = [r for r in rows if r["suite"] == "checks"]
    if not rs:
        return
    print("\n### Correctness vs kUPS main (after 3 Verlet steps)\n")
    print(table(["impl", "systems×reps", "dtype", "energy rel", "force abs / scale",
                 "cell grad abs / scale"],
                [[r.get("check", r.get("error", "?")[:40]), f"{r['batch']}×{r['reps']}",
                  r["dtype"], fmt(r.get("energy_max_rel"), 2),
                  f"{fmt(r.get('force_max_abs'), 2)} / {fmt(r.get('force_scale'), 2)}",
                  f"{fmt(r.get('cell_grad_max_abs'), 2)} / {fmt(r.get('cell_grad_scale'), 2)}"]
                 for r in rs]))


def ops(rows: list[dict]) -> None:
    rs = [r for r in rows if r["suite"] == "ops" and "ms" in r]
    if not rs:
        return
    modes = [("stock", True), ("ptx", True), ("stock", False), ("ptx", False)]
    names = ["stock, graphs", "jax-ptx, graphs", "stock, no graphs", "jax-ptx, no graphs"]
    for op, key in [("cell_list", "atoms"), ("batch_cell_list", "systems")]:
        grid: dict[int, dict] = defaultdict(dict)
        for r in rs:
            if r["op"] == op:
                grid[r[key]][(r["mode"], r["graphs"])] = r["ms"]
        print(f"\n### nvalchemi `{op}` alone, ms per jitted call "
              f"({'atoms' if key == 'atoms' else 'systems of 256 atoms'})\n")
        print(table([key] + names,
                    [[str(k)] + [fmt(grid[k].get(m)) for m in modes] for k in sorted(grid)]))


def md(rows: list[dict], suite: str, key: str, block: int, dtype: str = "f64") -> None:
    rs = [r for r in rows if r["suite"] == suite and r.get("block", 1) == block
          and r.get("dtype", "f64") == dtype and "ms_per_step" in r]
    if not rs:
        return
    grid: dict[tuple, dict] = defaultdict(dict)
    for r in rs:
        grid[(r["atoms_per_system"], r["batch"])][r["nl"]] = r["ms_per_step"]
    impls = [i for i in IMPLS if any(i in g for g in grid.values())]
    print(f"\n### MD step, ms ({suite}, {dtype}, {block} step(s) per dispatch); "
          "× = speedup over kUPS main\n")
    body = []
    for (n, b) in sorted(grid, key=lambda k: (k[0], k[1])):
        g = grid[(n, b)]
        base = g.get("kups")
        cells = []
        for i in impls:
            t = g.get(i)
            sp = f" ({base / t:.1f}×)" if t and base and i != "kups" else ""
            cells.append(fmt(t) + sp)
        body.append([str(n), str(b), str(n * b)] + cells)
    print(table(["atoms/system", "systems", "atoms"] + impls, body))


def errors(rows: list[dict]) -> None:
    rs = [r for r in rows if "error" in r]
    if rs:
        print("\n### Failed configurations\n")
        for r in rs:
            print(f"- {json.dumps({k: v for k, v in r.items() if k != 'error'})}: "
                  f"{r['error'][:160]}")


if __name__ == "__main__":
    rows = load(sys.argv[1:])
    devices = {r["device"] for r in rows if "device" in r}
    print(f"Device: {', '.join(sorted(devices)) or '?'}; {len(rows)} results")
    checks(rows)
    ops(rows)
    for dtype in ("f64", "f32"):
        md(rows, "md-size", "atoms", 1, dtype)
    for block in (1, 20):
        md(rows, "md-batch", "batch", block)
    errors(rows)
