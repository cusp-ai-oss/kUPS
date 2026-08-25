# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Lennard-Jones relaxation entry point."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import FRECHET_FILTER, POSITIONS_ONLY
from kups.application.relaxation.analysis import analyze_relax_file
from kups.application.relaxation.data import (
    RelaxRunConfig,
    RelaxState,
    relax_state_from_ase,
)
from kups.application.relaxation.simulation import make_relax_propagator
from kups.application.simulations.potentials import LjPotentialConfig
from kups.application.simulations.relax import Config, run
from kups.core.lens import identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.observables.stress import stress_via_virial_theorem, total_lattice_gradient
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.relaxation.config import make_optimizer

from ._builders import LBFGS_OPTIMIZER, ar_cif, tmp_h5


def _config(out_file: str, inp_file: str, *, optimize_cell: bool = False) -> Config:
    return Config(
        run=RelaxRunConfig(
            out_file=out_file,
            max_steps=5,
            seed=42,
            force_tolerance=0.5,
            optimizer=LBFGS_OPTIMIZER,
            optimize_cell=optimize_cell,
        ),
        potential=LjPotentialConfig(
            cutoff=5.0,
            parameters={"Ar": (3.405, 0.010326)},  # (sigma [Å], epsilon [eV])
            mixing_rule="lorentz_berthelot",
        ),
        inp_files=(inp_file,),
    )


class TestRun:
    """A short relaxation writes an HDF5 file the analyzer can read."""

    @pytest.fixture(scope="class")
    def out_file(self) -> str:
        out = tmp_h5()
        run(_config(out, ar_cif(rattle=0.1)))
        return out

    def test_analyzer_reads_back_physical_outputs(self, out_file: str):
        results = analyze_relax_file(out_file)
        assert len(results) == 1
        result = next(iter(results.values()))
        assert jnp.isfinite(jnp.asarray(result.final_energy)).item()
        assert jnp.isfinite(jnp.asarray(result.final_max_force)).item()
        assert result.n_steps >= 1


def _build_propagator(optimize_cell: bool):
    """Build an LJ relaxation propagator and its initial state."""
    config = _config(tmp_h5(), ar_cif(rattle=0.1), optimize_cell=optimize_cell)
    assert isinstance(config.potential, LjPotentialConfig)
    lj = LennardJonesParameters.from_dict(
        cutoff=config.potential.cutoff,
        parameters=config.potential.parameters,
        mixing_rule=config.potential.mixing_rule,
    )
    state_lens = identity_lens(RelaxState)
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


class TestCellRelaxation:
    """``optimize_cell`` drives the lattice vectors via the total cell gradient."""

    def test_cell_moves_only_when_optimizing_cell(self):
        # optimize_cell=True: the cell relaxes (total lattice gradient is non-zero
        # for the off-equilibrium fcc-Ar cell).
        prop, state = _build_propagator(optimize_cell=True)
        stepped = prop(jax.random.key(0), state)
        assert not jnp.allclose(
            stepped.systems.data.cell.vectors, state.systems.data.cell.vectors
        )

        # optimize_cell=False: positions move but the cell is held fixed.
        prop0, state0 = _build_propagator(optimize_cell=False)
        stepped0 = prop0(jax.random.key(0), state0)
        npt.assert_array_equal(
            stepped0.systems.data.cell.vectors, state0.systems.data.cell.vectors
        )
        assert not jnp.allclose(
            stepped0.particles.data.positions, state0.particles.data.positions
        )

    def test_converged_value_matches_fmax(self):
        # POSITIONS_ONLY: the cached gradient is the optimizer DOF gradient ∂E/∂u,
        # which equals the physical ∂E/∂r, so max|∂E/∂u| is exactly ASE's fmax.
        prop, state = _build_propagator(optimize_cell=False)
        stepped = prop(jax.random.key(0), state)
        dof = stepped.particles.data.position_gradients  # ∂E/∂u == ∂E/∂r here
        fmax = jnp.max(jnp.linalg.norm(stepped.particles.data.forces, axis=-1))
        max_dof = jnp.max(jnp.linalg.norm(dof, axis=-1))
        npt.assert_allclose(max_dof, fmax, atol=1e-12)

    def test_stress_not_double_counted(self):
        # cell_gradients caches the *partial* dE/dh|_r, the correct stress source.
        # Feeding the *total* lattice gradient to the virial theorem re-adds the
        # position virial -- the bug the filter design removes.
        prop, state = _build_propagator(optimize_cell=True)
        stepped = prop(jax.random.key(0), state)
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
        assert not jnp.allclose(sigma_partial, sigma_total)

        # The difference is exactly the position-virial term (notebook section 9).
        positions = stepped.particles.data.positions
        g_r = stepped.particles.data.position_gradients
        system = stepped.particles.data.system
        volume = stepped.systems.data.cell.volume[:, None, None]
        low = jnp.tril(system.sum_over(jnp.einsum("ni,nj->nij", positions, g_r)).data)
        pos_term = -(low + low.mT - low * jnp.eye(3)) / volume
        npt.assert_allclose(sigma_total - sigma_partial, pos_term, atol=1e-10)
