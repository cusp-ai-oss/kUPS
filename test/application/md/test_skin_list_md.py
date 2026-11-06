# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD on the skin list follows the same trajectories as with per-step builds, through
the graph path of the MLFF backends and through LJ's neighbor rows."""

from functools import partial

import ase
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.application.md.data import MdParameters, MdState
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.potential.mliap.tojax import make_tojaxed_from_state
from kups.application.simulations.potentials import TojaxPotentialConfig
from kups.core.lens import bind, identity_lens
from kups.core.neighborlist import AdaptiveNeighborList, SkinNeighborList
from kups.core.propagator import LoopPropagator, propagate_and_fix
from kups.core.result import as_result_function
from kups.potential.mliap.tojax import TojaxedMliap

from .._builders import (
    ROWS,
    Build,
    argon,
    graph_lj,
    md_parameters,
    md_tables,
    state_and_step,
    write_pair_model,
)


def _rows(dtype: str = "float64") -> Build:
    """LJ over the skin list's rows, as the app config builds it."""
    config = ROWS.model_copy(update={"kernel_dtype": dtype})
    return partial(config.build, identity_lens(MdState), POSITIONS_AND_CELL)


def _paths() -> list[tuple[Build, bool]]:
    """Per-step builds (the reference), the skin list's pairs and its rows, each with
    whether it reads the skin list."""
    return [
        (graph_lj(AdaptiveNeighborList.from_state), False),
        (graph_lj(SkinNeighborList.from_state), True),
        (_rows(), True),
    ]


def _run(
    md: MdParameters,
    build: Build,
    atoms: list[ase.Atoms],
    reads_the_list: bool,
    blocks: int = 6,
) -> np.ndarray:
    state, step = state_and_step(md, build, *md_tables(md, atoms))
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
    atoms = [argon(4, 0), argon(3, 1)]
    per_step, *skin = (
        _run(md_parameters(integrator), build, atoms, reads_the_list=reads_the_list)
        for build, reads_the_list in _paths()
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
    for build, _ in _paths():
        md = md_parameters("verlet").model_copy(update={"initialize_momenta": False})
        state, step = state_and_step(md, build, *md_tables(md, [atoms]))
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
    md, atoms = md_parameters("baoab_langevin"), [argon(3, 0), argon(2, 1)]
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
    md = md_parameters("verlet").model_copy(update={"initialize_momenta": False})
    build = graph_lj(SkinNeighborList.from_state)
    state, step = state_and_step(md, build, *md_tables(md, [argon(3, 0)]))
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


def test_tojax_config_reads_the_skin_list(tmp_path):
    """The jaxified-model config builds over the state's skin list."""
    path = tmp_path / "pair.zip"
    write_pair_model(path)
    model = TojaxedMliap.from_zip_file(path)
    per_step = make_tojaxed_from_state(
        identity_lens(MdState),
        parameters=model,
        gradient=POSITIONS_AND_CELL,
        neighborlist_factory=AdaptiveNeighborList.from_state,
    )
    config = TojaxPotentialConfig(model_path=path)
    md, atoms = md_parameters("baoab_langevin"), [argon(3, 0)]
    reference = _run(
        md, lambda *_: (per_step, model.cutoff), atoms, reads_the_list=False, blocks=4
    )
    build = partial(config.build, identity_lens(MdState), POSITIONS_AND_CELL)
    skin = _run(md, build, atoms, reads_the_list=True, blocks=4)
    np.testing.assert_allclose(skin, reference, atol=1e-8)
