"""Sweep logs to CSVs: ``to_csv.py OUT_DIR LOG [LOG...]`` writes md_size.csv,
md_batch.csv, ops.csv and checks.csv (one row per configuration)."""

from __future__ import annotations

import csv
import os
import sys

from analyze import load

LABELS = {
    "kups": "kUPS (stack)",
    "kups-cell": "kUPS cell list",
    "nvalchemi-ptx": "nvalchemi cell list builds the skin list (jax-ptx)",
    "nvalchemi-stock": "nvalchemi cell list builds the skin list (stock)",
    "nvalchemi-rows-ptx": "nvalchemi LJ kernel on kUPS skin list (jax-ptx)",
    "nvalchemi-rows-stock": "nvalchemi LJ kernel on kUPS skin list (stock)",
    "nvalchemi-lj-ptx": "nvalchemi LJ (jax-ptx)",
    "nvalchemi-lj-stock": "nvalchemi LJ (stock)",
}


def write(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(path, len(rows))


def main(out: str, logs: list[str]) -> None:
    os.makedirs(out, exist_ok=True)
    rows = load(logs)
    md = [r for r in rows if "ms_per_step" in r]
    write(f"{out}/md_size.csv", ["impl", "label", "dtype", "atoms", "ms_per_step"],
          [[r["nl"], LABELS[r["nl"]], r["dtype"], r["atoms"], round(r["ms_per_step"], 4)]
           for r in md if r["suite"] == "md-size"])
    write(f"{out}/md_batch.csv",
          ["impl", "label", "atoms_per_system", "systems", "atoms", "steps_per_dispatch",
           "ms_per_step", "atom_steps_per_s"],
          [[r["nl"], LABELS[r["nl"]], r["atoms_per_system"], r["batch"], r["atoms"],
            r["block"], round(r["ms_per_step"], 4), round(r["atom_steps_per_s"])]
           for r in md if r["suite"] == "md-batch"])
    write(f"{out}/ops.csv", ["op", "launch", "graphs", "systems", "atoms_per_system",
                             "atoms", "ms"],
          [[r["op"], "jax-ptx" if r["mode"] == "ptx" else "stock",
            "on" if r["graphs"] else "off", r["systems"], r["atoms_per_system"], r["atoms"],
            round(r["ms"], 4)] for r in rows if r["suite"] == "ops" and "ms" in r])
    write(f"{out}/checks.csv",
          ["impl", "label", "dtype", "systems", "atoms_per_system", "energy_max_rel",
           "force_max_abs", "force_scale", "cell_grad_max_abs", "cell_grad_scale"],
          [[r["check"], LABELS[r["check"]], r["dtype"], r["batch"], 4 * r["reps"] ** 3,
            f"{r['energy_max_rel']:.2e}", f"{r['force_max_abs']:.2e}",
            f"{r['force_scale']:.3g}", f"{r['cell_grad_max_abs']:.2e}",
            f"{r['cell_grad_scale']:.3g}"]
           for r in rows if r["suite"] == "checks" and "check" in r])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
