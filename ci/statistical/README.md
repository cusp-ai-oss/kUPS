# Statistical Validation

Statistical validation tests that verify simulation outputs against known expected values. Tests run automatically on PRs marked ready for review and on merges to main.

## Directory Structure

```
ci/statistical/
├── inputs/
│   ├── ci/               # Short CI runs
│   ├── reference/         # Long runs for generating expected values
│   ├── host/              # CIF structure files
│   └── lennard_jones/     # Force field parameters
├── expected/              # Expected values for validation
├── validate.py            # Validation script (runs in CI)
├── run_reference.py       # Run all reference sims and generate expected values
├── generate_expected.py   # Generate expected values from a single HDF5 file
├── validate_nve_physics.py  # NVE physics validation
└── validate_nvt_physics.py  # NVT physics validation
```

## Adding a New Test

1. **Create the simulation input config** in both `inputs/ci/<name>.yaml` (short) and `inputs/reference/<name>.yaml` (long).

   For MD (Lennard-Jones), use the `kups_md_lj` config schema.
   For MCMC (NVT/GCMC), use the `kups_mcmc_rigid` config schema.

   The `out_file` should point to `../../<name>.h5` so the HDF5 output lands in `ci/statistical/`.

2. **Run all reference simulations and generate expected values**:

   ```bash
   uv run python ci/statistical/run_reference.py
   ```

   This runs all reference configs in parallel (with `XLA_PYTHON_CLIENT_PREALLOCATE=false`) and generates the expected YAML files automatically. All child processes are killed if the script is interrupted.

   To run a single reference simulation manually:

   ```bash
   cd ci/statistical/inputs/reference
   uv run kups_md_lj <name>.yaml       # for MD
   uv run kups_mcmc_rigid <name>.yaml   # for MCMC
   cd ../..
   uv run python generate_expected.py <name>_ref.h5 <sim_type>
   ```

   Where `<sim_type>` is `md`, `nvt`, or `gcmc`.

3. **Verify the test passes** with the CI config:

   ```bash
   uv run python ci/statistical/validate.py
   ```

## Validation Logic

The validator compares measured values against expected values using a statistical tolerance (default: 5 sigma). A test passes if:

```
|measured_mean - expected_mean| <= 5 * sqrt(measured_sem^2 + expected_sem^2)
```

This accounts for statistical uncertainty in both the reference and measured values.
