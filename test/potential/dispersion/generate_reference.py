# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ``_reference_values.py`` from simple-dftd3.

simple-dftd3 is LGPL-3.0-or-later and is deliberately **not** declared as a kUPS
dependency — not in ``pyproject.toml`` and not in ``uv.lock`` — so that no
LGPL-licensed package is pulled into a kUPS install or published in its metadata.
It is imported here only to produce reference numbers, which are committed as
plain Python literals so the test suite never needs it.

Regeneration is a rare manual step. Install it into a throwaway environment::

    uv venv --python 3.13 /tmp/d3ref && VIRTUAL_ENV=/tmp/d3ref uv pip install dftd3 ase
    /tmp/d3ref/bin/python test/potential/dispersion/generate_reference.py

The generated values use simple-dftd3's own real-space cutoffs so the comparison
in ``test_d3_reference.py`` is exact rather than cutoff-limited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from test.potential.dispersion._systems import (  # noqa: E402
    PBE_D3BJ_BOHR,
    SDFTD3_CN_BOHR,
    SDFTD3_DISP2_BOHR,
    SYSTEMS,
)

BOHR = 0.5291772105638411  # Å, ASE/CODATA value also used by kups.core.constants
HARTREE = 27.211386024367243  # eV


def compute(system) -> dict[str, object]:
    from dftd3.interface import DispersionModel, RationalDampingParam

    model = DispersionModel(
        numbers=system.numbers,
        positions=system.positions / BOHR,
        lattice=None if system.cell is None else system.cell / BOHR,
        periodic=None if system.cell is None else np.array(system.pbc),
    )
    # match kUPS's masking exactly; disp3 is irrelevant (s9 = 0 below)
    model.set_realspace_cutoff(SDFTD3_DISP2_BOHR, 0.0, SDFTD3_CN_BOHR)
    param = RationalDampingParam(**PBE_D3BJ_BOHR, s9=0.0, alp=14.0)
    out = model.get_dispersion(param, grad=True)

    result: dict[str, object] = {
        "energy": float(out["energy"]) * HARTREE,
        "gradient": (np.asarray(out["gradient"]) * HARTREE / BOHR).tolist(),
    }
    if "virial" in out and out["virial"] is not None:
        result["virial"] = (np.asarray(out["virial"]) * HARTREE).tolist()
    return result


def _fmt(value: object, indent: int) -> str:
    pad = " " * indent
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        inner = ",\n".join(f"{pad}    {_fmt(v, indent + 4)}" for v in value)
        return f"[\n{inner},\n{pad}]"
    return repr(value)


def main() -> None:
    import dftd3

    results = {name: compute(system) for name, system in SYSTEMS.items()}

    body = [
        "# Copyright 2024-2026 Cusp AI",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        '"""Reference D3(BJ) values produced by simple-dftd3. GENERATED — do not edit.',
        "",
        f"Produced by ``generate_reference.py`` with dftd3 {dftd3.__version__}, PBE-D3(BJ)",
        f"damping, s9=0 (no ATM), real-space cutoffs disp2={SDFTD3_DISP2_BOHR} a0 and",
        f"cn={SDFTD3_CN_BOHR} a0.",
        "",
        "Energies are eV, gradients eV/Å, virials eV. The gradient is dE/dr, not the force.",
        '"""',
        "",
        "REFERENCE: dict[str, dict[str, object]] = {",
    ]
    for name, res in results.items():
        body.append(f"    {name!r}: {{")
        for key, value in res.items():
            body.append(f"        {key!r}: {_fmt(value, 8)},")
        body.append("    },")
    body.append("}")

    out_path = Path(__file__).parent / "_reference_values.py"
    out_path.write_text("\n".join(body) + "\n")
    print(f"wrote {out_path} ({len(results)} systems)")


if __name__ == "__main__":
    main()
