# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""MD steps keep the dtype of every state leaf, for every integrator, and a float32
step follows float64 within float32 round-off."""

from typing import get_args

import jax
import jax.numpy as jnp
import pytest
from jax.typing import DTypeLike

from kups.application.md.data import MdState
from kups.application.simulations import md
from kups.core.propagator import Propagator, propagate_and_fix
from kups.core.result import as_result_function
from kups.core.units import catalog as u
from kups.core.units import testing
from kups.core.utils.jax import astype_floating
from kups.md.integrators import Integrator
from kups.md.observables import instantaneous_pressure, system_kinetic_energy
from kups.observables.stress import stress_via_virial_theorem
from test.application.test_md_lj import _ar_cif, _config, _tmp_h5
from test.units import EV, FORCE


def _state_and_step(
    integrator: str, dtype: DTypeLike, monkeypatch: pytest.MonkeyPatch
) -> tuple[MdState, Propagator[MdState]]:
    """The state and MD step that `kups_md` builds for two rattled argon systems,
    with the state's floats cast to ``dtype``. The potential keeps its float64
    parameters."""
    captured = {}
    monkeypatch.setattr(
        md, "run_md", lambda key, step, state, config: captured.update(s=(state, step))
    )
    inp_files = (_ar_cif(rattle=0.1), _ar_cif(rattle=0.2))
    config = _config(_tmp_h5(), inp_files[0])
    config = config.model_copy(
        update={
            "md": config.md.model_copy(update={"integrator": integrator}),
            "inp_files": inp_files,
        }
    )
    md.run(config)
    state, step = captured["s"]
    return astype_floating(state, dtype), step


def _dtypes(tree: object) -> list[jnp.dtype]:
    return [x.dtype for x in jax.tree.leaves(tree)]


@pytest.mark.parametrize("integrator", get_args(Integrator.__value__))
def test_step_keeps_dtypes(integrator: str, monkeypatch: pytest.MonkeyPatch):
    """A float32 state keeps every leaf's dtype over a step."""
    state, step = _state_and_step(integrator, jnp.float32, monkeypatch)
    step = as_result_function(step)
    out = jax.eval_shape(lambda key, s: step(key, s).value, jax.random.key(0), state)
    assert _dtypes(out) == _dtypes(state)


def test_float32_matches_float64(monkeypatch: pytest.MonkeyPatch):
    """A Verlet step from the same state gives the same energies, forces and
    pressure (kinetic plus virial, as the barostats use) in float32 and float64."""
    results = []
    for dtype in (jnp.float64, jnp.float32):
        state, step = _state_and_step("verlet", dtype, monkeypatch)
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
    testing.assert_allclose(energy32, energy, EV, rtol=1e-5)
    testing.assert_allclose(forces32, forces, FORCE, rtol=0, atol=1e-5)
    testing.assert_allclose(pressure32, pressure, u.bar, rtol=0, atol=1.0)
