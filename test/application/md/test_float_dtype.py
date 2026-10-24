# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD steps keep the dtype of every state leaf, for every integrator and potential
config of the MD app, and a float32 step follows float64 within float32 round-off."""

from functools import partial
from typing import get_args

import ase
import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from kups.application.md.data import MdParameters, MdState
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.simulations.potentials import (
    CoulombConfig,
    LjCoulombPotentialConfig,
    TojaxPotentialConfig,
)
from kups.core.constants import BAR
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import SkinNeighborList
from kups.core.propagator import Propagator, propagate_and_fix
from kups.core.result import as_result_function
from kups.core.utils.jax import astype_floating
from kups.md.integrators import Integrator
from kups.md.observables import instantaneous_pressure, system_kinetic_energy
from kups.observables.stress import stress_via_virial_theorem

from .._builders import (
    ROWS,
    Build,
    graph_lj,
    md_parameters,
    md_tables,
    state_and_step,
    write_pair_model,
)

_POTENTIALS = ["lj", "lj_float32", "graph", "tojax", "ewald", "pme"]


@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Build]:
    """Every potential config of the MD app: LJ over rows with float64 and float32
    pair terms, graph LJ as the MLFF backends pair atoms, a float64 jaxified model,
    and LJ plus Ewald or PME."""
    path = tmp_path_factory.mktemp("tojax") / "pair.zip"
    write_pair_model(path)
    configs = {
        "lj": ROWS,
        "lj_float32": ROWS.model_copy(update={"kernel_dtype": "float32"}),
        "tojax": TojaxPotentialConfig(model_path=path),
        "ewald": LjCoulombPotentialConfig(
            lj=ROWS, coulomb=CoulombConfig(method="ewald")
        ),
        "pme": LjCoulombPotentialConfig(lj=ROWS, coulomb=CoulombConfig(method="pme")),
    }
    lens = identity_lens(MdState)
    return {
        "graph": graph_lj(SkinNeighborList.from_state),
        **{k: partial(c.build, lens, POSITIONS_AND_CELL) for k, c in configs.items()},
    }


def _atoms() -> list[ase.Atoms]:
    """Rattled argon in two random skewed cells, both narrower than twice the cutoff,
    with charges of ±0.1 e for the Coulomb configs."""
    rng = np.random.default_rng(0)
    atoms = []
    for n in (3, 2):
        crystal = ase.build.bulk("Ar", "fcc", a=5.3, cubic=True) * (n, n, n)
        strain = np.eye(3) + rng.uniform(-0.1, 0.1, (3, 3))
        crystal.set_cell(crystal.cell[:] @ strain, scale_atoms=True)
        crystal.rattle(0.1, seed=n)
        crystal.set_initial_charges(np.resize([0.1, -0.1], len(crystal)))
        atoms.append(crystal)
    return atoms


def _state(
    md: MdParameters, build: Build, dtype: DTypeLike
) -> tuple[MdState, Propagator[MdState]]:
    """State with int32 indices and floats in ``dtype``, both set before the potential
    and the skin list are built, and its MD step."""
    tables = md_tables(md, _atoms(), index_dtype=jnp.int32)
    return state_and_step(md, build, *astype_floating(tables, dtype))


def _dtypes(tree: object) -> list[jnp.dtype]:
    return [x.dtype for x in jax.tree.leaves(tree)]


@pytest.mark.parametrize("potential", _POTENTIALS)
@pytest.mark.parametrize("integrator", get_args(Integrator.__value__))
def test_step_keeps_dtypes(integrator: str, potential: str, builds: dict[str, Build]):
    """A float32 state with int32 indices keeps every leaf's dtype over a step."""
    state, step = _state(md_parameters(integrator), builds[potential], jnp.float32)
    step = as_result_function(step)
    out = jax.eval_shape(lambda key, s: step(key, s).value, jax.random.key(0), state)
    assert _dtypes(out) == _dtypes(state)


def test_capacity_growth_keeps_dtypes(builds: dict[str, Build]):
    """Pair and skin-list capacities that grow on the host keep every dtype."""
    md = md_parameters("baoab_langevin")
    state, step = _state(md, builds["graph"], jnp.float32)
    dtypes, params = _dtypes(state), state.skin_list.params
    # compressed to a third of the volume, the next build outgrows both capacities
    state = bind(state, lambda s: s.systems.data.cell).apply(lambda c: c * 0.7)
    state = bind(state, lambda s: s.particles.data.positions).apply(lambda x: x * 0.7)
    state = propagate_and_fix(
        jax.jit(as_result_function(step)), jax.random.key(0), state
    )
    assert state.skin_list.params.avg_edges > params.avg_edges
    assert state.neighborlist_params.avg_edges > params.avg_edges
    assert _dtypes(state) == dtypes


@pytest.mark.parametrize("potential", _POTENTIALS)
def test_float32_matches_float64(potential: str, builds: dict[str, Build]):
    """A Verlet step from the same state gives the same energies, forces and
    pressure (kinetic plus virial, as the barostats use) in float32 and float64."""
    results = []
    for dtype in (jnp.float64, jnp.float32):
        state, step = _state(md_parameters("verlet"), builds[potential], dtype)
        state = propagate_and_fix(
            jax.jit(as_result_function(step)), jax.random.key(0), state
        )
        particles, systems = state.particles, state.systems
        kinetic = system_kinetic_energy(
            particles.data.momenta, particles.data.masses, particles.data.system
        )
        stress = stress_via_virial_theorem(particles, systems).data
        pressure = instantaneous_pressure(kinetic, stress, systems.data.cell.volume)
        results.append((systems.data.potential_energy, particles.data.forces, pressure))
    (energy, forces, pressure), (energy32, forces32, pressure32) = results
    assert energy32.dtype == forces32.dtype == pressure32.dtype == jnp.float32
    np.testing.assert_allclose(energy32, energy, rtol=1e-5)
    np.testing.assert_allclose(forces32, forces, atol=1e-5)
    np.testing.assert_allclose(pressure32, pressure, atol=BAR)
