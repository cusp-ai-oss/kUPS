# D3 reference data — provenance and licensing

This directory embeds the element tables required by the D3 dispersion
correction. **All values originate from Apache-2.0 licensed upstreams** and are reproduced
here numerically unchanged, so that they can be diffed against the source. No
value is rescaled or re-derived on the way in.

Two caveats on "unchanged", for precision: the tables are *repacked* (truncated to
`Z <= 103`, padded to a fixed reference-slot count, and in the C6 case stored as
an upper triangle), and they are stored in whichever unit the upstream source
quotes — which is Å for the covalent radii and atomic units for the rest, not
atomic units throughout. Every conversion to kUPS units (Å, eV) and every derived
quantity (the `4/3` radius scaling, the `r4r2` pre-processing) happens at load
time in `__init__.py`, never in the checked-in data.

## Files

| File | Contents | Units as stored |
|---|---|---|
| `_tables.py` | `COVALENT_RADII_2009`, `R4_OVER_R2`, `REFERENCE_CN` | Å, atomic units, dimensionless |
| `d3_reference_c6.npz` | `c6_upper` — the reference C6 table, upper triangle only | Hartree·Bohr⁶ |

`d3_reference_c6.npz` stores only the `i <= j` block because the table satisfies
`C6[i, j, a, b] == C6[j, i, b, a]` exactly; `_load_reference_c6` restores the
dense `(104, 104, 7, 7)` array. This halves the element count and keeps the
committed file at ~332 kB, below the `check-added-large-files` pre-commit limit.

## Sources

| Table | Upstream | File | Licence |
|---|---|---|---|
| `COVALENT_RADII_2009` | [grimme-lab/mctc-lib](https://github.com/grimme-lab/mctc-lib) | `src/mctc/data/covrad.f90`, parameter `covalent_rad_2009` | Apache-2.0 |
| `R4_OVER_R2` | [dftd3/tad-dftd3](https://github.com/dftd3/tad-dftd3) | `src/tad_dftd3/data/r4r2.py`, list `_r4_over_r2` | Apache-2.0 |
| `REFERENCE_CN` | [dftd3/tad-dftd3](https://github.com/dftd3/tad-dftd3) | `src/tad_dftd3/reference.py`, function `_load_cn` | Apache-2.0 |
| `c6_upper` | [dftd3/tad-dftd3](https://github.com/dftd3/tad-dftd3) | `src/tad_dftd3/reference-c6.pt` | Apache-2.0 |

Both upstream repositories carry a full Apache-2.0 `LICENSE`, and the individual
source files carry `SPDX-Identifier: Apache-2.0` (tad-dftd3) or the Apache-2.0
header text (mctc-lib).

### Sources deliberately *not* used

- [dftd3/simple-dftd3](https://github.com/dftd3/simple-dftd3) — **LGPL-3.0-or-later**.
- [dftbplus/dftd3-lib](https://github.com/dftbplus/dftd3-lib), the repackaged
  original Grimme Fortran program — **GPL**.

Neither is compatible with kUPS's Apache-2.0 licence, so no code or data has been
taken from them. `simple-dftd3` is used only as an external *reference implementation*, to
generate the values committed in `test/potential/dispersion/_reference_values.py`.
It is deliberately **not** declared as a dependency in `pyproject.toml` or
`uv.lock`, so no LGPL package is installed with kUPS or named in its published
metadata. See `generate_reference.py` for how to run it from a throwaway
environment.

> **Note for reviewers.** The numeric content of these tables ultimately derives
> from Grimme's original DFT-D3 program, which is GPL-licensed. The Grimme group
> subsequently released the same data under Apache-2.0 through `mctc-lib` and
> `tad-dftd3`, which is the provenance relied on here. If kUPS's maintainers are
> not comfortable with that chain, this directory is self-contained and can be
> replaced without touching the potential implementation.

## Scientific provenance

- **Covalent radii** — Pyykkö & Atsumi, *Chem. Eur. J.* **15** (2009) 188–197,
  with radii of metals decreased by 10 %. The D3 coordination-number radii are
  `4/3` times these.
- **⟨r⁴⟩/⟨r²⟩ expectation values** — PBE0/def2-QZVP atomic values, S. Grimme
  (2010); rare gases recomputed at PBE0/aug-cc-pVQZ by J. Mewes (2018);
  super-heavy elements at 4c-PBE/Dyall-AE4Z (2022).
- **Reference coordination numbers and C6 coefficients** — Grimme, Antony,
  Ehrlich & Krieg, *J. Chem. Phys.* **132** (2010) 154104,
  [doi:10.1063/1.3382344](https://doi.org/10.1063/1.3382344), from TD-DFT
  Casimir–Polder integration of reference molecules.

The Becke–Johnson damping parameterisation is from Grimme, Ehrlich & Goerigk,
*J. Comput. Chem.* **32** (2011) 1456–1465,
[doi:10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759); the per-functional
values live in `../damping.py`, each annotated with its own source DOI.

## Coverage

The reference C6 table covers `Z = 1..103`, so all tables are truncated to 104
rows (row 0 is an unused placeholder so that atomic numbers index directly).
Elements use up to 7 reference systems; unused slots carry `-1` in `REFERENCE_CN`
and zero in the C6 table, and are masked by `D3Reference.reference_mask`.

No element has duplicate reference CN values, which is why a single Gaussian per
reference slot reproduces `simple-dftd3` exactly: its multi-width `ngw` weighting
collapses to one width when the reference CNs are distinct.

Numerical validation in `test/potential/dispersion` spans H, C, O, F, Na, Si, Cl,
Ar and the transition metals Cr, Ni, Cu, Zn, Zr — 13 of the 103 elements. The
metals are there deliberately: they are what MOF nodes are built from, and each
sits in a different awkward corner of the interpolation. `cu_fcc` has CN ≈ 8.7
against references `[0, 0.96]`, so every weight is extrapolated; `cr_bcc` sits
just above the widest reference gap in the table (`[0, 1.83, 10.62]`); and
`ni_fcc`'s reference CNs (`[0, 1.79, 6.55, 6.29]`) are not sorted, so slot order
cannot leak into the result. `zno` is the only cell pairing a metal with a light
element, which is what exercises the cross-element C6 block in both index orders.

The remaining 90 elements are **not** covered by a numerical test. The lanthanides
and actinides are the least exercised: they hold up to 7 reference slots, and Ce
is the only element whose single reference sits at a non-zero CN (2.7991).

## Regenerating

The tables were extracted from the upstream files listed above. `reference-c6.pt`
is a `torch.save` archive; its payload is a C-contiguous `(104, 104, 7, 7)`
little-endian float64 tensor stored at `archive/data/0`, which can be read with
`zipfile` + `numpy.frombuffer` without a torch dependency. The stored `_tables.py`
literals are verbatim copies of the corresponding upstream lists.
