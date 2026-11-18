# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the Lennard-Jones relaxation entry point."""

import dataclasses
import tempfile

import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.analysis import (
    RelaxAnalysisResult,
    analyze_streaming_relax_file,
)
from kups.application.relaxation.data import (
    RelaxRunConfig,
    RelaxState,
    relax_gradients,
    relax_index_prefix,
    relax_state_from_ase,
)
from kups.application.relaxation.logging import RelaxLoggedData
from kups.application.relaxation.simulation import make_relax_propagator, run_relax
from kups.application.relaxation.streaming import StreamingRelaxLoggedData
from kups.application.simulations.potentials import LjPotentialConfig
from kups.application.simulations.relax import Config, run
from kups.core import units
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.storage import HDF5StorageReader
from kups.core.units import catalog as u
from kups.core.units import testing
from kups.observables.stress import stress_via_virial_theorem, total_lattice_gradient
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import make_optimizer
from kups.relaxation.convergence import fmax_per_system
from kups.relaxation.optimizer import ChainOptState
from test.units import FORCE, A


def _ar_cif(
    rattle: float, repeat: tuple[int, int, int] = (2, 2, 2), seed: int = 1
) -> str:
    """Write a rattled fcc-argon supercell as a P1 CIF with uniform ``Ar`` labels.

    The rattle gives the optimizer nonzero forces to act on; uniform labels
    keep them matching the LJ parameter table (ASE's writer would uniquify).
    """
    atoms = ase.build.bulk("Ar", "fcc", a=5.3) * repeat
    atoms.rattle(rattle, seed=seed)
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


def _config(
    out_file: str,
    *inp_files: str,
    optimize_cell: bool = False,
    max_steps: int = 5,
    force_tolerance: str = "0.5 eV/angstrom",
    batch_size: int = 16,
) -> Config:
    return Config(
        run=RelaxRunConfig(
            out_file=out_file,
            max_steps=max_steps,
            seed=42,
            force_tolerance=force_tolerance,
            optimizer=[
                {"transform": "scale_by_ase_lbfgs", "memory_size": 10, "alpha": 70},
                {"transform": "max_step_size", "max_step_size": 0.2},
                {"transform": "scale", "step_size": -1},
            ],
            optimize_cell=optimize_cell,
            batch_size=batch_size,
        ),
        potential=LjPotentialConfig(
            cutoff="5 angstrom",
            parameters={"Ar": ("3.405 angstrom", "0.010326 eV")},
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=inp_files,
    )


def _streamed(out_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every logged step's per-slot ``(ordinal, step, potential energy in eV)``."""
    with HDF5StorageReader[StreamingRelaxLoggedData](out_file) as reader:
        n_logged = int(reader.file.attrs["actual_steps"])
        ordinal, step, energy = reader.focus_group(lambda s: s.step).read(
            slice(n_logged), select=lambda d: (d.ordinal, d.step, d.potential_energy)
        )
    return ordinal, step, np.asarray(energy.to(u.eV))


class TestRun:
    """``run`` streams three structures of 8, 12 and 12 atoms through slots."""

    TOLERANCE = "0.01 eV/angstrom"

    @pytest.fixture(scope="class")
    def inp_files(self) -> tuple[str, ...]:
        repeats = ((2, 2, 2), (3, 2, 2), (2, 3, 2))
        return tuple(_ar_cif(0.1, r, seed) for seed, r in enumerate(repeats))

    def _run(self, inp_files: tuple[str, ...], batch_size: int) -> str:
        out = _tmp_h5()
        run(
            _config(
                out,
                *inp_files,
                max_steps=30,
                force_tolerance=self.TOLERANCE,
                batch_size=batch_size,
            )
        )
        return out

    @pytest.fixture(scope="class")
    def two_slots(self, inp_files: tuple[str, ...]) -> str:
        return self._run(inp_files, batch_size=2)

    @pytest.fixture(scope="class")
    def one_slot(self, inp_files: tuple[str, ...]) -> str:
        return self._run(inp_files, batch_size=1)

    def test_analyzer_reports_every_structure(self, two_slots: str):
        results = analyze_streaming_relax_file(two_slots)
        assert list(results) == [0, 1, 2]
        steps = [result.n_steps for result in results.values()]
        assert len(set(steps)) > 1  # the structures finish at different times
        for result in results.values():
            assert jnp.isfinite(result.final_energy.to(u.eV)).item()
            assert result.final_max_force.to(FORCE) < 0.01
            assert 1 <= result.n_steps < 30

    def test_log_names_the_structure_in_every_slot(self, two_slots: str):
        ordinal, step, _ = _streamed(two_slots)
        assert ordinal.shape[1] == 2
        npt.assert_array_equal(ordinal[0], [0, 1])
        npt.assert_array_equal(ordinal[-1], [-1, -1])
        for job, result in analyze_streaming_relax_file(two_slots).items():
            # One record per update, and a last one that only evaluates.
            npt.assert_array_equal(
                step[ordinal == job], [*range(1, result.n_steps + 1), result.n_steps]
            )

    def test_results_do_not_depend_on_batch_size(self, two_slots: str, one_slot: str):
        together = analyze_streaming_relax_file(two_slots)
        alone = analyze_streaming_relax_file(one_slot)
        assert list(alone) == [0, 1, 2]
        for job, result in together.items():
            _assert_same_result(result, alone[job])

    def test_first_structure_matches_fixed_batch(
        self, inp_files: tuple[str, ...], one_slot: str
    ):
        config = _config(
            _tmp_h5(), inp_files[0], max_steps=30, force_tolerance=self.TOLERANCE
        ).run
        propagator, state = _build_propagator(inp_files[0], optimize_cell=False)
        run_relax(jax.random.key(0), propagator, state, config)
        with HDF5StorageReader[RelaxLoggedData](config.out_file) as reader:
            n_logged = int(reader.file.attrs["actual_steps"])
            fixed = reader.focus_group(lambda s: s.step).read(
                slice(n_logged), select=lambda d: d.potential_energy
            )
        ordinal, _, energy = _streamed(one_slot)
        npt.assert_allclose(
            energy[ordinal == 0], fixed.to(u.eV)[:, 0], rtol=0, atol=1e-12
        )


def _assert_same_result(a: RelaxAnalysisResult, b: RelaxAnalysisResult) -> None:
    assert a.n_steps == b.n_steps
    testing.assert_allclose(a.final_energy, b.final_energy, u.eV, atol=1e-12)
    testing.assert_allclose(a.final_max_force, b.final_max_force, FORCE, atol=1e-12)


def _build_propagator(inp_file: str, optimize_cell: bool):
    """Build the fixed-batch LJ relaxation propagator and its initial state."""
    config = _config(_tmp_h5(), inp_file, optimize_cell=optimize_cell)
    assert isinstance(config.potential, LjPotentialConfig)
    lj = LennardJonesParameters.from_dict(
        cutoff=config.potential.cutoff,
        parameters=config.potential.parameters,
        mixing_rule=config.potential.mixing_rule,
    )
    state_lens = identity_lens(RelaxState[ChainOptState])
    optimizer = make_optimizer(config.run.optimizer)
    gradient = FRECHET_FILTER if optimize_cell else POSITIONS_ONLY
    potential = make_lennard_jones_from_state(
        state_lens, parameters=lj, gradient=gradient
    )
    propagator, opt_init = make_relax_propagator(
        state_lens, potential, optimizer, gradient
    )
    particles, systems = relax_state_from_ase(config.inp_files[0])
    nlp = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, lj.cutoff
    )
    opt_state = opt_init(particles, systems)
    state = RelaxState(particles, systems, nlp, opt_state, jnp.array([0]))
    return propagator, state


def test_nonfinite_gradients_abort_run() -> None:
    _, state = _build_propagator(_ar_cif(rattle=0.1), optimize_cell=False)
    state = (
        bind(state)
        .focus(lambda s: s.particles.data.position_gradients)
        .apply(lambda gradients: gradients * jnp.nan)
    )
    config = _config(_tmp_h5(), "unused").run
    with pytest.raises(AssertionError, match="Non-finite relaxation gradients"):
        run_relax(jax.random.key(0), lambda key, s: s, state, config)


class TestCellRelaxation:
    """``optimize_cell`` drives the lattice vectors via the total cell gradient."""

    @pytest.fixture(scope="class")
    def position_step(self) -> tuple[RelaxState, RelaxState]:
        prop, state = _build_propagator(_ar_cif(rattle=0.1), optimize_cell=False)
        return state, jax.jit(lambda s: prop(jax.random.key(0), s))(state)

    @pytest.fixture(scope="class")
    def cell_step(self) -> tuple[RelaxState, RelaxState]:
        prop, state = _build_propagator(_ar_cif(rattle=0.1), optimize_cell=True)
        return state, jax.jit(lambda s: prop(jax.random.key(0), s))(state)

    def test_cell_moves_only_when_optimizing_cell(self, cell_step, position_step):
        # optimize_cell=True: the cell relaxes (total lattice gradient is non-zero
        # for the off-equilibrium fcc-Ar cell).
        state, stepped = cell_step
        assert not jnp.allclose(
            stepped.systems.data.cell.vectors.to(A),
            state.systems.data.cell.vectors.to(A),
        )

        # optimize_cell=False: positions move but the cell is held fixed.
        state0, stepped0 = position_step
        npt.assert_array_equal(
            stepped0.systems.data.cell.vectors.to(A),
            state0.systems.data.cell.vectors.to(A),
        )
        assert not jnp.allclose(
            stepped0.particles.data.positions.to(A),
            state0.particles.data.positions.to(A),
        )

    def test_converged_value_matches_fmax(self, position_step):
        # POSITIONS_ONLY: the cached gradient is the optimizer DOF gradient ∂E/∂u,
        # which equals the physical ∂E/∂r, so max|∂E/∂u| is exactly ASE's fmax.
        _, stepped = position_step
        dof = stepped.particles.data.position_gradients  # ∂E/∂u == ∂E/∂r here
        fmax = units.max(units.norm(stepped.particles.data.forces, axis=-1))
        max_dof = units.max(units.norm(dof, axis=-1))
        testing.assert_allclose(max_dof, fmax, FORCE, atol=1e-12)

    def test_cell_gradients_enter_per_system_fmax(self, cell_step):
        _, stepped = cell_step
        # Zero the position gradients so only the cell gradient remains; otherwise
        # they dominate the maximum and hide a dropped cell term.
        stepped = (
            bind(stepped)
            .focus(lambda s: s.particles.data.position_gradients)
            .apply(units.zeros_like)
        )
        prefix = relax_index_prefix(stepped.particles, stepped.systems)
        # The only non-zero cell-gradient leaf is the matrix-log deformation; ASE's
        # rows are its columns.
        deformation = stepped.systems.data.cell_gradients.frame.deformation
        cell_max = jnp.max(jnp.linalg.norm(deformation.log_matrix.to(FORCE), axis=-2))
        assert cell_max > 0
        gradients = relax_gradients(stepped)
        npt.assert_allclose(
            fmax_per_system(gradients, prefix).data.to(FORCE), [cell_max]
        )
        npt.assert_array_equal(
            fmax_per_system(gradients, prefix, include_cell=False).data.to(FORCE), [0.0]
        )

    def test_stress_not_double_counted(self, cell_step):
        # cell_gradients caches the *partial* dE/dh|_r, the correct stress source.
        # Feeding the *total* lattice gradient to the virial theorem re-adds the
        # position virial -- the bug the filter design removes.
        _, stepped = cell_step
        sigma_partial = stress_via_virial_theorem(
            stepped.particles, stepped.systems
        ).data

        total = total_lattice_gradient(
            stepped.particles.data.positions,
            stepped.particles.data.position_gradients,
            stepped.systems.map_data(lambda s: s.cell),
            stepped.systems.map_data(lambda s: s.cell_gradients),
            stepped.particles.data.system,
        )
        sys_total = stepped.systems.set_data(
            dataclasses.replace(stepped.systems.data, cell_gradients=total.data)
        )
        sigma_total = stress_via_virial_theorem(stepped.particles, sys_total).data
        gpa = u.GPa
        assert not jnp.allclose(sigma_partial.to(gpa), sigma_total.to(gpa))

        # The difference is exactly the position-virial term (notebook section 9).
        positions = stepped.particles.data.positions
        g_r = stepped.particles.data.position_gradients
        system = stepped.particles.data.system
        volume = stepped.systems.data.cell.volume[:, None, None]
        outer = units.einsum("ni,nj->nij", positions, g_r)
        low = units.apply(jnp.tril, system.sum_over(outer).data)
        pos_term = -(low + low.swapaxes(-1, -2) - low * jnp.eye(3)) / volume
        testing.assert_allclose(sigma_total - sigma_partial, pos_term, gpa, atol=1e-10)
