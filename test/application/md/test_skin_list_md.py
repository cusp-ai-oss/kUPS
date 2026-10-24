# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD on the skin list follows the same trajectories as with per-step builds."""

import json
import zipfile

import ase
import ase.build
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export

from kups.application.md.data import MdParameters, MdState, md_state_from_ase
from kups.application.md.simulation import make_md_propagator
from kups.application.potential.classical.lennard_jones import (
    make_lennard_jones_from_state,
)
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.potential.mliap.tojax import make_tojaxed_from_state
from kups.application.simulations.potentials import (
    BuiltPotential,
    TojaxPotentialConfig,
)
from kups.core.data import Table
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import AdaptiveNeighborList, SkinNeighborList
from kups.core.propagator import LoopPropagator, propagate_and_fix
from kups.core.result import as_result_function
from kups.core.utils.msgpack import serialize as msgpack_serialize
from kups.potential.classical.lennard_jones import LennardJonesParameters
from kups.potential.mliap.tojax import TojaxedMliap

_LJ = LennardJonesParameters.from_dict(
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


def _graph(factory) -> BuiltPotential:
    """Graph LJ over ``factory``'s pairs."""
    potential = make_lennard_jones_from_state(
        identity_lens(MdState),
        parameters=_LJ,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=factory,
    )
    return potential, _LJ.cutoff


def _state(md: MdParameters, built: BuiltPotential, atoms: list[ase.Atoms]) -> tuple:
    potential, cutoff = built
    tables = [
        md_state_from_ase(a, md, key=jax.random.key(i)) for i, a in enumerate(atoms)
    ]
    particles, systems = Table.union([p for p, _ in tables], [s for _, s in tables])
    step = make_md_propagator(identity_lens(MdState), md.integrator, potential)
    return MdState.create(particles, systems, cutoff, md.skin), step


def _run(
    md: MdParameters, built: BuiltPotential, atoms: list[ase.Atoms], blocks: int = 6
) -> np.ndarray:
    state, step = _state(md, built, atoms)
    reference = state.skin_list.reference.positions
    cycle = jax.jit(as_result_function(LoopPropagator(step, 10)))
    for i in range(blocks):
        state = propagate_and_fix(cycle, jax.random.key(i), state)
    assert not np.array_equal(reference, state.skin_list.reference.positions)
    return np.asarray(state.particles.data.positions)


@pytest.mark.parametrize(
    "integrator",
    ["verlet", "baoab_langevin", "csvr", "csvr_npt", "baoab_npt_langevin"],
)
def test_trajectory_matches_per_step_builds(integrator):
    """Two systems; the 15.9 Å one is narrower than twice the cutoff, so pairs
    appear through several images. A small skin makes the run rebuild often."""
    atoms = [_argon(4, 0), _argon(3, 1)]
    per_step, skin = (
        _run(_md(integrator), _graph(factory), atoms)
        for factory in (AdaptiveNeighborList.from_state, SkinNeighborList.from_state)
    )
    np.testing.assert_allclose(skin, per_step, atol=1e-8)


def test_margin_is_checked_where_forces_are_evaluated():
    """In one step two atoms close from beyond cutoff + skin to inside the cutoff."""
    atoms = ase.Atoms(
        "Ar2", positions=[[10.0, 15.0, 15.0], [18.5, 15.0, 15.0]], cell=[30.0] * 3
    )
    atoms.pbc = True
    forces = []
    for factory in (SkinNeighborList.from_state, AdaptiveNeighborList.from_state):
        md = _md("verlet").model_copy(update={"initialize_momenta": False})
        state, step = _state(md, _graph(factory), [atoms])
        masses = state.particles.data.masses[:, None]
        time_step = state.systems.data.integrator_params.time_step[0]
        velocity = jnp.array([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]]) / time_step
        state = bind(state, lambda s: s.particles.data.momenta).set(masses * velocity)
        result = jax.jit(as_result_function(step))(jax.random.key(0), state)
        result.raise_assertion()
        forces.append(np.asarray(result.value.particles.data.forces))
    assert np.abs(forces[1]).max() > 0
    np.testing.assert_allclose(forces[0], forces[1], rtol=1e-10)


def _pair_model(path) -> None:
    """Exported model with energy sum_ij (rc^2 - r_ij^2)^2 over its edges, rc 5 Å."""

    def energy(params, x):
        i, j = x["edge_index"]
        cell = x["cell"][x["batch"][i]]
        shift = jnp.einsum("ek,ekl->el", x["cell_offsets"], cell)
        r = x["pos"][j] - x["pos"][i] + shift
        pair = params[0] * (25.0 - (r**2).sum(-1)) ** 2
        return jnp.zeros_like(x["charge"]).at[x["batch"][i]].add(pair)

    f64, i64 = np.float64, np.int64
    concrete = {
        "pos": np.zeros((4, 3), f64),
        "atomic_numbers": np.zeros(4, i64),
        "cell": np.zeros((2, 3, 3), f64),
        "pbc": np.ones((2, 3), bool),
        "edge_index": np.zeros((2, 6), i64),
        "cell_offsets": np.zeros((6, 3), f64),
        "charge": np.zeros(2, f64),
        "spin": np.zeros(2, f64),
        "batch": np.zeros(4, i64),
    }
    shapes = {
        "pos": "n, _",
        "atomic_numbers": "n",
        "cell": "b, _, _",
        "pbc": "b, _",
        "edge_index": "_, e",
        "cell_offsets": "e, _",
        "charge": "b",
        "spin": "b",
        "batch": "n",
    }
    params = [np.array(1e-4, f64)]
    specs = export.symbolic_args_specs((params, concrete), shapes_specs=(None, shapes))
    exported = export.export(jax.jit(energy))(*specs)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("model.jax", exported.serialize(vjp_order=1))
        zf.writestr("metadata.json", json.dumps({"cutoff": 5.0}))
        zf.writestr("params.msgpack", msgpack_serialize(params))


def test_tojax_config_reads_the_skin_list(tmp_path):
    """The jaxified-model config builds over the state's skin list."""
    path = tmp_path / "pair.zip"
    _pair_model(path)
    model = TojaxedMliap.from_zip_file(path)
    per_step = make_tojaxed_from_state(
        identity_lens(MdState),
        parameters=model,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=AdaptiveNeighborList.from_state,
    )
    config = TojaxPotentialConfig(model_path=path)
    md, atoms = _md("baoab_langevin"), [_argon(3, 0)]
    reference = _run(md, (per_step, model.cutoff), atoms, blocks=4)
    built = config.build(identity_lens(MdState), POSITIONS_AND_CELL)
    np.testing.assert_allclose(_run(md, built, atoms, blocks=4), reference, atol=1e-8)
