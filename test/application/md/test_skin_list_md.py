# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD on the skin list follows the same trajectories as with per-step builds."""

import ase
import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.application.md.data import MdParameters, MdState, md_state_from_ase
from kups.application.md.simulation import make_md_propagator
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.simulations.potentials import LjPotentialConfig
from kups.core.data import Table
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import AdaptiveNeighborList, SkinNeighborList
from kups.core.propagator import LoopPropagator, propagate_and_fix
from kups.core.result import as_result_function

_LJ = LjPotentialConfig(
    cutoff=8.0, parameters={"Ar": (3.405, 0.010326)}, mixing_rule="lorentz_berthelot"
)


def _md(integrator: str, skin: float = 0.3) -> MdParameters:
    # NPT settings of the CI argon runs (compressibility in 1/Pa)
    return MdParameters(
        temperature=120.0,
        time_step=4.0,
        friction_coefficient=0.01,
        thermostat_time_constant=100.0,
        target_pressure=4.2e7,
        pressure_coupling_time=1000.0,
        compressibility=5.4e-10,
        minimum_scale_factor=0.9,
        integrator=integrator,
        initialize_momenta=True,
        skin=skin,
    )


def _argon(n: int, seed: int) -> ase.Atoms:
    atoms = ase.build.bulk("Ar", "fcc", a=5.3, cubic=True) * (n, n, n)
    atoms.rattle(0.1, seed=seed)
    return atoms


def _state(md: MdParameters, factory, atoms: list[ase.Atoms]) -> tuple:
    tables = [
        md_state_from_ase(a, md, key=jax.random.key(i)) for i, a in enumerate(atoms)
    ]
    particles, systems = Table.union([p for p, _ in tables], [s for _, s in tables])
    potential, cutoff = _LJ.build(
        identity_lens(MdState), POSITIONS_AND_CELL, neighborlist_factory=factory
    )
    step = make_md_propagator(identity_lens(MdState), md.integrator, potential)
    return MdState.create(particles, systems, cutoff, md.skin), step


@pytest.mark.parametrize(
    "integrator",
    ["verlet", "baoab_langevin", "csvr", "csvr_npt", "baoab_npt_langevin"],
)
def test_trajectory_matches_per_step_builds(integrator):
    """Two systems; the 15.9 Å one is narrower than twice the cutoff, so pairs
    appear through several images. A small skin makes the run rebuild often."""
    positions = []
    for factory in (SkinNeighborList.from_state, AdaptiveNeighborList.from_state):
        state, step = _state(_md(integrator), factory, [_argon(4, 0), _argon(3, 1)])
        built = state.skin_list.reference.positions
        cycle = jax.jit(as_result_function(LoopPropagator(step, 10)))
        for i in range(6):
            state = propagate_and_fix(cycle, jax.random.key(i), state)
        assert not np.array_equal(built, state.skin_list.reference.positions)
        positions.append(np.asarray(state.particles.data.positions))
    np.testing.assert_allclose(positions[0], positions[1], atol=1e-8)


def test_margin_is_checked_where_forces_are_evaluated():
    """In one step two atoms close from beyond cutoff + skin to inside the cutoff."""
    atoms = ase.Atoms(
        "Ar2", positions=[[10.0, 15.0, 15.0], [18.5, 15.0, 15.0]], cell=[30.0] * 3
    )
    atoms.pbc = True
    forces = []
    for factory in (SkinNeighborList.from_state, AdaptiveNeighborList.from_state):
        md = _md("verlet").model_copy(update={"initialize_momenta": False})
        state, step = _state(md, factory, [atoms])
        masses = state.particles.data.masses[:, None]
        time_step = state.systems.data.integrator_params.time_step[0]
        velocity = jnp.array([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]]) / time_step
        state = bind(state, lambda s: s.particles.data.momenta).set(masses * velocity)
        result = jax.jit(as_result_function(step))(jax.random.key(0), state)
        result.raise_assertion()
        forces.append(np.asarray(result.value.particles.data.forces))
    assert np.abs(forces[1]).max() > 0
    np.testing.assert_allclose(forces[0], forces[1], rtol=1e-10)
