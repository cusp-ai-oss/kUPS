# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Lennard-Jones plus Coulomb in kups_md and kups_relax: LJ over the skin list's
rows and Coulomb over its pairs follow per-step builds and store the same list, Ewald
and PME agree, a float32 state stays float32, and both apps read the charges."""

from functools import partial
from pathlib import Path

import ase
import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.application.md.data import (
    MdParameters,
    MdRunConfig,
    MdState,
    md_state_from_ase,
)
from kups.application.md.simulation import make_md_propagator
from kups.application.potential.classical.ewald import make_ewald_from_state
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.classical.pme import make_pme_from_state
from kups.application.potential.filter import (
    FRECHET_FILTER,
    POSITIONS_AND_CELL,
    POSITIONS_ONLY,
)
from kups.application.relaxation.data import (
    RelaxRunConfig,
    RelaxState,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import make_relax_propagator
from kups.application.simulations import md, relax
from kups.application.simulations.potentials import (
    BuiltPotential,
    CoulombConfig,
    LjCoulombPotentialConfig,
    LjPotentialConfig,
)
from kups.core.data import Table
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import AdaptiveNeighborList
from kups.core.potential import PotentialAsPropagator, sum_potentials
from kups.core.propagator import LoopPropagator, propagate_and_fix
from kups.core.result import as_result_function
from kups.core.utils.jax import astype_floating
from kups.potential.classical.ewald import EwaldParameters, PMEParameters
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import make_optimizer

# The parameters are synthetic, and the rock salt below is near zero pressure.
# 6.3 rounds up in float32, so a float32 state's Coulomb cutoff exceeds the
# Lennard-Jones one.
_LJ = LjPotentialConfig(
    cutoff=6.3,
    parameters={"Na": (2.25, 0.01), "Cl": (4.0, 0.01)},
    mixing_rule="lorentz_berthelot",
)
_LBFGS = [
    {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
    {"transform": "max_step_size", "max_step_size": 0.2},
    {"transform": "scale", "step_size": -1},
]


def _config(method: str) -> LjCoulombPotentialConfig:
    return LjCoulombPotentialConfig(lj=_LJ, coulomb=CoulombConfig(method=method))


def _salt(n: int, seed: int, cubic: bool = True) -> ase.Atoms:
    """Rattled rock salt with charges of ±1 e."""
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64, cubic=cubic) * (n, n, n)
    atoms.rattle(0.05, seed=seed)
    atoms.info["_atom_site_charge"] = np.where(atoms.numbers == 11, 1.0, -1.0)
    return atoms


def _batch() -> list[ase.Atoms]:
    """A cube and a skewed primitive cell narrower than twice the cutoff."""
    return [_salt(2, 0), _salt(3, 1, cubic=False)]


def _per_step(method: str, lens, gradient, particles, systems) -> BuiltPotential:
    """The same terms over per-step builds, LJ on the graph."""
    lj = LennardJonesParameters.from_dict(_LJ.cutoff, _LJ.parameters, _LJ.mixing_rule)
    make_parameters, make = {
        "ewald": (EwaldParameters.make, make_ewald_from_state),
        "pme": (PMEParameters.make, make_pme_from_state),
    }[method]
    coulomb = make_parameters(particles, systems, 1e-6, _LJ.cutoff)
    factory = AdaptiveNeighborList.from_state
    potential = sum_potentials(
        make_lennard_jones_from_state(
            lens, parameters=lj, gradient=gradient, neighborlist_factory=factory
        ),
        make(lens, parameters=coulomb, gradient=gradient, neighborlist_factory=factory),
    )
    return potential, lj.cutoff


def _md(integrator: str) -> MdParameters:
    return MdParameters(
        temperature=300.0,
        time_step=2.0,
        friction_coefficient=0.01,
        thermostat_time_constant=100.0,
        target_pressure=1e5,
        pressure_coupling_time=1000.0,
        compressibility=4e-11,
        minimum_scale_factor=0.9,
        integrator=integrator,
        initialize_momenta=True,
        skin=0.3,
    )


def _tables(md_parameters: MdParameters, atoms: list[ase.Atoms]) -> tuple:
    tables = [
        md_state_from_ase(a, md_parameters, key=jax.random.key(i))
        for i, a in enumerate(atoms)
    ]
    return Table.union([p for p, _ in tables], [s for _, s in tables])


def _md_state(md_parameters: MdParameters, build, tables) -> tuple:
    particles, systems = tables
    lens = identity_lens(MdState)
    potential, cutoff = build(lens, POSITIONS_AND_CELL, particles, systems)
    return MdState.create(particles, systems, cutoff, md_parameters.skin), potential


def _evaluate(state, potential):
    result = jax.jit(as_result_function(lambda s: potential(s).data))(state)
    result.raise_assertion()
    return result.value


@pytest.mark.parametrize("integrator", ["baoab_langevin", "baoab_npt_langevin"])
@pytest.mark.parametrize("method", ["ewald", "pme"])
def test_md_matches_per_step_builds(method, integrator):
    """Positions follow per-step builds; only the run that reads the list rebuilds
    it."""
    md_parameters = _md(integrator)
    positions = []
    for build, reads_the_list in (
        (_config(method).build, True),
        (partial(_per_step, method), False),
    ):
        state, potential = _md_state(
            md_parameters, build, _tables(md_parameters, _batch())
        )
        step = make_md_propagator(identity_lens(MdState), integrator, potential)
        reference = state.skin_list.reference.positions
        cycle = jax.jit(as_result_function(LoopPropagator(step, 10)))
        for i in range(6):
            state = propagate_and_fix(cycle, jax.random.key(i), state)
        rebuilt = not np.array_equal(reference, state.skin_list.reference.positions)
        assert rebuilt == reads_the_list
        positions.append(np.asarray(state.particles.data.positions))
    np.testing.assert_allclose(positions[0], positions[1], atol=1e-8)


@pytest.mark.parametrize("method", ["ewald", "pme"])
def test_lj_and_coulomb_store_the_same_list(method):
    """Within the skin neither term rebuilds the list, and beyond it both terms
    rebuild it to the same list."""
    md_parameters = _md("baoab_langevin")
    state, potential = _md_state(
        md_parameters, _config(method).build, _tables(md_parameters, _batch())
    )
    steps = [
        jax.jit(as_result_function(PotentialAsPropagator(term)))
        for term in potential.potentials
    ]
    built = state.skin_list.reference.positions
    direction = np.random.default_rng(0).normal(size=built.shape)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    # The skin of 0.3 Å allows moves of up to 0.15 Å.
    for distance, rebuilt in ((0.02, False), (0.6, True)):
        moved = bind(state, lambda s: s.particles.data.positions).apply(
            lambda x: x + distance * direction
        )
        stored = []
        for step in steps:
            result = step(jax.random.key(0), moved)
            result.raise_assertion()
            stored.append(result.value.skin_list)
            assert rebuilt != np.array_equal(built, stored[-1].reference.positions)
        for a, b in zip(*map(jax.tree.leaves, stored), strict=True):
            np.testing.assert_array_equal(a, b)


def test_pme_matches_ewald():
    """Energies, forces and cell gradients agree within the target error."""
    md_parameters = _md("baoab_npt_langevin")
    tables = _tables(md_parameters, _batch())
    ewald, pme = (
        _evaluate(*_md_state(md_parameters, _config(m).build, tables))
        for m in ("ewald", "pme")
    )
    np.testing.assert_allclose(
        pme.total_energies.data, ewald.total_energies.data, rtol=1e-6
    )
    forces = np.asarray(ewald.gradients.positions.data)
    np.testing.assert_allclose(
        pme.gradients.positions.data, forces, atol=1e-5 * np.abs(forces).max()
    )
    cell = np.asarray(ewald.gradients.cell.data.vectors)
    np.testing.assert_allclose(
        pme.gradients.cell.data.vectors, cell, atol=1e-5 * np.abs(cell).max()
    )


@pytest.mark.parametrize("method", ["ewald", "pme"])
def test_float32_state_stays_float32(method):
    md_parameters = _md("baoab_langevin")
    tables = _tables(md_parameters, _batch())
    out = {}
    for dtype in (jnp.float64, jnp.float32):
        state, potential = _md_state(
            md_parameters, _config(method).build, astype_floating(tables, dtype)
        )
        out[dtype] = _evaluate(state, potential)
    for leaf in jax.tree.leaves(out[jnp.float32]):
        assert leaf.dtype == jnp.float32
    a, b = out[jnp.float32], out[jnp.float64]
    np.testing.assert_allclose(a.total_energies.data, b.total_energies.data, rtol=1e-5)
    forces = np.asarray(b.gradients.positions.data)
    np.testing.assert_allclose(
        a.gradients.positions.data, forces, atol=1e-4 * np.abs(forces).max()
    )


def test_open_cells_raise():
    atoms = _salt(2, 0)
    atoms.pbc = False
    md_parameters = _md("verlet")
    with pytest.raises(TypeError, match="PeriodicCell"):
        _md_state(md_parameters, _config("pme").build, _tables(md_parameters, [atoms]))


def test_uncharged_systems_raise():
    atoms = _batch()
    del atoms[1].info["_atom_site_charge"]
    md_parameters = _md("verlet")
    with pytest.raises(ValueError, match=r"Systems \[1\] have no charges"):
        _md_state(md_parameters, _config("pme").build, _tables(md_parameters, atoms))


def _salt_cif(path: Path, seed: int) -> Path:
    """Rattled 2 x 2 x 2 rock salt as a P1 CIF with ``_atom_site_charge``."""
    atoms = _salt(2, seed)
    rows = "\n".join(
        f"{s} {s} {x:.6f} {y:.6f} {z:.6f} {q:+.1f}"
        for s, (x, y, z), q in zip(
            atoms.get_chemical_symbols(),
            atoms.get_scaled_positions(),
            atoms.info["_atom_site_charge"],
            strict=True,
        )
    )
    cif = f"""data_nacl
_cell_length_a {atoms.cell.lengths()[0]:.6f}
_cell_length_b {atoms.cell.lengths()[1]:.6f}
_cell_length_c {atoms.cell.lengths()[2]:.6f}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_charge
{rows}
"""
    path.write_text(cif)
    return path


def _inputs(directory: Path) -> tuple[Path, Path]:
    """Two rock salt CIFs, new for each run: particles_from_ase caches by path."""
    directory.mkdir()
    return _salt_cif(directory / "0.cif", 0), _salt_cif(directory / "1.cif", 1)


def _app(directory: Path, method: str, index_dtype: str = "int64") -> md.Config:
    return md.Config(
        run=MdRunConfig(
            out_file=directory / "md.h5",
            num_steps=4,
            num_warmup_steps=2,
            block_size=2,
            seed=42,
            index_dtype=index_dtype,
        ),
        md=_md("baoab_langevin"),
        potential=_config(method),
        inp_files=_inputs(directory),
    )


@pytest.mark.parametrize("method", ["ewald", "pme"])
def test_md_run_reads_the_charges(tmp_path: Path, method: str):
    state = md.run(_app(tmp_path / "run", method))
    charges = np.asarray(state.particles.data.charges)
    np.testing.assert_array_equal(np.abs(charges), 1.0)
    assert charges.sum() == 0.0
    assert np.all(np.isfinite(np.asarray(state.particles.data.positions)))


def test_md_run_int32_indices_match_default(tmp_path: Path):
    int64, int32 = (
        md.run(_app(tmp_path / dtype, "pme", dtype)) for dtype in ("int64", "int32")
    )
    assert int32.particles.data.system.raw_indices.dtype == jnp.int32
    np.testing.assert_allclose(
        int32.particles.data.positions,
        int64.particles.data.positions,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "gradient", [POSITIONS_ONLY, FRECHET_FILTER], ids=["positions", "cell"]
)
@pytest.mark.parametrize("method", ["ewald", "pme"])
def test_relaxation_matches_per_step_builds(method, gradient):
    """Positions and cells follow per-step builds; only the run that reads the list
    rebuilds it."""
    tables = [relax_state_from_ase(a) for a in _batch()]
    particles, systems = Table.union([p for p, _ in tables], [s for _, s in tables])
    lens = identity_lens(RelaxState)
    results = []
    for build, reads_the_list in (
        (_config(method).build, True),
        (partial(_per_step, method), False),
    ):
        potential, cutoff = build(lens, gradient, particles, systems)
        propagator, opt_init = make_relax_propagator(
            lens, potential, make_optimizer(_LBFGS), gradient
        )
        # A small skin makes the list rebuild even when only the positions relax.
        state = RelaxState.create(
            particles, systems, cutoff, 0.2, opt_init(particles, systems)
        )
        reference = state.skin_list.reference.positions
        step = jax.jit(as_result_function(propagator))
        energies = []
        for i in range(12):
            state = propagate_and_fix(step, jax.random.key(i), state)
            energies.append(np.asarray(state.systems.data.potential_energy))
        rebuilt = not np.array_equal(reference, state.skin_list.reference.positions)
        assert rebuilt == reads_the_list
        assert np.all(energies[-1] < energies[0])
        results.append(state)
    skin, per_step = results
    np.testing.assert_allclose(
        skin.particles.data.positions, per_step.particles.data.positions, atol=1e-8
    )
    np.testing.assert_allclose(
        skin.systems.data.cell.vectors, per_step.systems.data.cell.vectors, atol=1e-8
    )


@pytest.mark.parametrize("method", ["ewald", "pme"])
def test_relax_run_reads_the_charges(tmp_path: Path, method: str):
    config = relax.Config(
        run=RelaxRunConfig(
            out_file=tmp_path / "relax.h5",
            max_steps=6,
            seed=42,
            force_tolerance=1e-3,
            optimizer=_LBFGS,
            optimize_cell=True,
        ),
        potential=_config(method),
        inp_files=_inputs(tmp_path / "inputs"),
    )
    state = relax.run(config)
    np.testing.assert_array_equal(np.abs(state.particles.data.charges), 1.0)
    # The Madelung energy of rock salt, -8.9 eV per ion pair, dominates.
    pairs = state.particles.data.system.counts.data / 2
    assert np.all(np.asarray(state.systems.data.potential_energy) / pairs < -8.0)
