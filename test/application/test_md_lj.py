# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones MD entry point."""

import tempfile

import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import MdParameters, MdRunConfig
from kups.application.simulations import md
from kups.application.simulations.md import Config, run
from kups.application.simulations.potentials import LjPotentialConfig
from kups.core.data import Index


def _ar_cif(rattle: float = 0.0, repeat: int = 2) -> str:
    """Write ``repeat**3`` primitive fcc-argon cells (one atom each) as a P1 CIF.

    Labels are kept uniform (``Ar``) so they match the LJ parameter table;
    ASE's CIF writer would otherwise uniquify them to ``Ar1``, ``Ar2``, ...
    """
    atoms = ase.build.bulk("Ar", "fcc", a=5.3) * (repeat, repeat, repeat)
    if rattle:
        atoms.rattle(rattle, seed=1)
    a, b, c, al, be, ga = atoms.cell.cellpar()
    rows = "\n".join(
        f"Ar Ar {x:.6f} {y:.6f} {z:.6f}" for x, y, z in atoms.get_scaled_positions()
    )
    cif = f"""data_ar
_cell_length_a {a:.6f}
_cell_length_b {b:.6f}
_cell_length_c {c:.6f}
_cell_angle_alpha {al:.6f}
_cell_angle_beta {be:.6f}
_cell_angle_gamma {ga:.6f}
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
{rows}
"""
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.write(cif)
    f.close()
    return f.name


def _tmp_h5() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    f.close()
    return f.name


def _config(out_file: str, inp_file: str) -> Config:
    return Config(
        run=MdRunConfig(out_file=out_file, num_steps=5, num_warmup_steps=0, seed=42),
        md=MdParameters(
            temperature=100.0,
            time_step=2.0,
            friction_coefficient=1.0,
            thermostat_time_constant=100.0,
            target_pressure=1.0,
            pressure_coupling_time=1.0e10,
            compressibility=4.5e-5,
            minimum_scale_factor=1.0,
            integrator="baoab_langevin",
            initialize_momenta=True,
        ),
        potential=LjPotentialConfig(
            cutoff=5.0,
            parameters={"Ar": (3.405, 0.010326)},  # (sigma [Å], epsilon [eV])
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(inp_file,),
    )


class TestRun:
    """A short NVT MD run writes an HDF5 file the analyzer can read."""

    @pytest.fixture(scope="class")
    def out_file(self) -> str:
        out = _tmp_h5()
        run(_config(out, _ar_cif()))
        return out

    def test_analyzer_reads_back_physical_outputs(self, out_file: str) -> None:
        results = analyze_md_file(out_file, n_blocks=2)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(result.total_energy.mean).all().item()
        assert jnp.isfinite(result.temperature.mean).all().item()
        assert (result.temperature.mean >= 0.0).all().item()


def test_replicas_get_independent_momenta(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same input file passed twice must not start with identical momenta."""
    captured = {}
    monkeypatch.setattr(
        md, "run_md", lambda key, propagator, state, config: captured.update(s=state)
    )
    inp = _ar_cif()
    run(_config(_tmp_h5(), inp).model_copy(update={"inp_files": (inp, inp)}))
    first, second = np.split(np.asarray(captured["s"].particles.data.momenta), 2)
    assert not np.allclose(first, second)


def test_int32_indices_match_default() -> None:
    states = []
    for index_dtype in ("int64", "int32"):
        # fresh input files per run: particles_from_ase caches by path
        config = _config(_tmp_h5(), _ar_cif(rattle=0.1))
        config = config.model_copy(
            update={
                "run": config.run.model_copy(update={"index_dtype": index_dtype}),
                "inp_files": (*config.inp_files, _ar_cif(rattle=0.2)),
            }
        )
        states.append(run(config))
    leaves = jax.tree.leaves(states[1], is_leaf=lambda x: isinstance(x, Index))
    dtypes = {x.raw_indices.dtype for x in leaves if isinstance(x, Index)}
    assert dtypes == {jnp.dtype(jnp.int32)}
    np.testing.assert_allclose(
        states[1].particles.data.positions,
        states[0].particles.data.positions,
        rtol=1e-10,
        atol=1e-10,
    )


def test_skin_does_not_change_results() -> None:
    """``md.skin`` changes how often neighbors are listed, not the trajectory."""
    states = []
    for skin in (0.05, 1.5):
        # a fresh input file per run: particles_from_ase caches by path and the
        # MD cycle donates the cached buffers
        config = _config(_tmp_h5(), _ar_cif(rattle=0.1, repeat=4))
        md = config.md.model_copy(update={"skin": skin, "integrator": "verlet"})
        states.append(run(config.model_copy(update={"md": md})))
    small, large = (np.asarray(s.skin_list.reference.positions) for s in states)
    assert not np.array_equal(small, large)  # the small skin was listed again
    np.testing.assert_allclose(
        states[1].particles.data.positions,
        states[0].particles.data.positions,
        atol=1e-8,
    )
