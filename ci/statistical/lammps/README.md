# LAMMPS MD LJ Argon References

LAMMPS same-force-field references for the kUPS statistical MD LJ Argon cases:
NVE, NVT, isotropic NPT (`csvr_npt`), and fully flexible NPT
(`baoab_npt_langevin` cross-check).

Normal CI compares the committed YAML references in `expected/*.yaml` to the
kUPS expected YAMLs and does not require LAMMPS or `.dat` output tables.
The LAMMPS YAML references use the same expected-value
shape as `ci/statistical/expected/*.yaml`: `simulation_type`, `hdf5_output`,
and `observables` with `expected_mean`/`expected_sem`. SEMs are computed
with the same fixed 5-block average used by kUPS expected-value
generation. LAMMPS `press` is total
pressure; the analyzer subtracts `2K/(3V)` before writing `pressure`, so the
YAML matches the current kUPS MD configurational-pressure definition
(`trace(stress_tensor) / 3`, in `eV/A^3`).

To regenerate the YAML references on a machine with `lmp` installed, run:

```bash
cd ci/statistical/lammps
uv run python run_reference.py
uv run python validate.py

# To rewrite YAML stats from existing generated .dat tables without rerunning LAMMPS:
uv run python run_reference.py --analyze-only
```

The `.dat` tables written under `outputs/` are intermediate generated artifacts
and are ignored by git. The inputs use `units metal`, the CIF-derived
256-atom Argon structure,
unshifted `lj/cut 10.0`, `epsilon = 0.01032356174398622 eV`,
`sigma = 3.405 A`, and no analytical tail correction.

Notes:

- LAMMPS and kUPS do not share identical random number streams, so these are
  statistical ensemble checks, not trajectory matches.
- The NVT input uses LAMMPS Nose-Hoover thermostatting as a same-ensemble
  reference. The literal kUPS BAOAB-Langevin friction maps to a LAMMPS damping
  time smaller than the timestep and is not a stable LAMMPS setup.
- Both LAMMPS Nose-Hoover NVT and kUPS BAOAB Langevin keep zero center-of-mass
  momentum. kUPS projects active Langevin momentum updates back into the
  zero-total-momentum subspace used by the MD temperature analysis.
- The flexible NPT input is not Gao BAOAB. It is a same-state-point MTTK
  reference for the fully flexible-cell ensemble.
