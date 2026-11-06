# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD on the skin list follows the same trajectories as with per-step builds, through
the graph path of the MLFF backends and through LJ's neighbor rows."""

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
    LjPotentialConfig,
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

_ROWS = LjPotentialConfig(
    cutoff=8.0, parameters={"Ar": (3.405, 0.010326)}, mixing_rule="lorentz_berthelot"
)
_LJ = LennardJonesParameters.from_dict(
    _ROWS.cutoff, _ROWS.parameters, _ROWS.mixing_rule
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
    """Graph LJ over ``factory``'s pairs, as the MLFF backends evaluate them."""
    potential = make_lennard_jones_from_state(
        identity_lens(MdState),
        parameters=_LJ,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=factory,
    )
    return potential, _LJ.cutoff


def _rows(dtype: str = "float64") -> BuiltPotential:
    """LJ over the skin list's rows, as the app config builds it."""
    config = _ROWS.model_copy(update={"kernel_dtype": dtype})
    return config.build(identity_lens(MdState), POSITIONS_AND_CELL)


def _paths() -> list[tuple[BuiltPotential, bool]]:
    """Per-step builds (the reference), the skin list's pairs and its rows, each with
    whether it reads the skin list."""
    return [
        (_graph(AdaptiveNeighborList.from_state), False),
        (_graph(SkinNeighborList.from_state), True),
        (_rows(), True),
    ]


def _state(md: MdParameters, built: BuiltPotential, atoms: list[ase.Atoms]) -> tuple:
    potential, cutoff = built
    tables = [
        md_state_from_ase(a, md, key=jax.random.key(i)) for i, a in enumerate(atoms)
    ]
    particles, systems = Table.union([p for p, _ in tables], [s for _, s in tables])
    step = make_md_propagator(identity_lens(MdState), md.integrator, potential)
    return MdState.create(particles, systems, cutoff, md.skin), step


def _run(
    md: MdParameters,
    built: BuiltPotential,
    atoms: list[ase.Atoms],
    reads_the_list: bool,
    blocks: int = 6,
) -> np.ndarray:
    state, step = _state(md, built, atoms)
    reference = state.skin_list.reference.positions
    cycle = jax.jit(as_result_function(LoopPropagator(step, 10)))
    for i in range(blocks):
        state = propagate_and_fix(cycle, jax.random.key(i), state)
    rebuilt = not np.array_equal(reference, state.skin_list.reference.positions)
    assert rebuilt == reads_the_list
    return np.asarray(state.particles.data.positions)


@pytest.mark.parametrize(
    "integrator",
    ["verlet", "baoab_langevin", "csvr", "csvr_npt", "baoab_npt_langevin"],
)
def test_trajectory_matches_per_step_builds(integrator):
    """Two systems; the 15.9 Å one is narrower than twice the cutoff, so pairs
    appear through several images. A small skin makes rebuilds frequent, and only
    the runs that read the list rebuild it."""
    atoms = [_argon(4, 0), _argon(3, 1)]
    per_step, *skin = (
        _run(_md(integrator), built, atoms, reads_the_list=reads_the_list)
        for built, reads_the_list in _paths()
    )
    for positions in skin:
        np.testing.assert_allclose(positions, per_step, atol=1e-8)


def test_margin_is_checked_where_forces_are_evaluated():
    """In one step two atoms close from beyond cutoff + skin to inside the cutoff."""
    atoms = ase.Atoms(
        "Ar2", positions=[[10.0, 15.0, 15.0], [18.5, 15.0, 15.0]], cell=[30.0] * 3
    )
    atoms.pbc = True
    forces = []
    for built, _ in _paths():
        md = _md("verlet").model_copy(update={"initialize_momenta": False})
        state, step = _state(md, built, [atoms])
        masses = state.particles.data.masses[:, None]
        time_step = state.systems.data.integrator_params.time_step[0]
        velocity = jnp.array([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]]) / time_step
        state = bind(state, lambda s: s.particles.data.momenta).set(masses * velocity)
        result = jax.jit(as_result_function(step))(jax.random.key(0), state)
        result.raise_assertion()
        forces.append(np.asarray(result.value.particles.data.forces))
    assert np.abs(forces[0]).max() > 0
    for skin in forces[1:]:
        np.testing.assert_allclose(skin, forces[0], rtol=1e-10)


def test_float32_rows_follow_float64(row_kernel):
    """Float32 pair terms, in the Pallas kernel or in XLA, stay within float32
    round-off of float64 over a block."""
    md, atoms = _md("baoab_langevin"), [_argon(3, 0), _argon(2, 1)]
    float64, float32 = (
        _run(md, _rows(dtype), atoms, reads_the_list=True, blocks=1)
        for dtype in ("float64", "float32")
    )
    assert row_kernel is None or row_kernel
    assert not np.array_equal(float32, float64)
    np.testing.assert_allclose(float32, float64, atol=1e-6)


def test_failed_step_keeps_the_skin_list():
    """A step that fails after its force evaluation rebuilt the list is undone,
    except for the list."""
    md = _md("verlet").model_copy(update={"initialize_momenta": False})
    state, step = _state(md, _graph(SkinNeighborList.from_state), [_argon(3, 0)])
    masses = state.particles.data.masses[:, None]
    time_step = state.systems.data.integrator_params.time_step[0]
    # Every atom moves by twice the skin, so the force evaluation rebuilds the list.
    velocity = jnp.array([2 * md.skin, 0.0, 0.0]) / time_step
    state = bind(state, lambda s: s.particles.data.momenta).set(masses * velocity)
    # One edge per particle is too few, so the step fails after the rebuild.
    state = bind(state, lambda s: s.neighborlist_params.avg_edges).set(1)
    result = jax.jit(as_result_function(step))(jax.random.key(0), state)
    assert result.failed_assertions
    undone = result.value
    np.testing.assert_array_equal(undone.step, state.step)
    np.testing.assert_array_equal(
        undone.particles.data.positions, state.particles.data.positions
    )
    built = state.skin_list.reference.positions
    assert not np.array_equal(built, undone.skin_list.reference.positions)


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
    reference = _run(
        md, (per_step, model.cutoff), atoms, reads_the_list=False, blocks=4
    )
    built = config.build(identity_lens(MdState), POSITIONS_AND_CELL)
    skin = _run(md, built, atoms, reads_the_list=True, blocks=4)
    np.testing.assert_allclose(skin, reference, atol=1e-8)
