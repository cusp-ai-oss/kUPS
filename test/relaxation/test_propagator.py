# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for relaxation propagators."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.core.data import Table
from kups.core.lens import lens
from kups.core.patch import ExplicitPatch, IdPatch, WithPatch
from kups.core.potential import PotentialOut
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass
from kups.relaxation.propagator import RelaxationPropagator

from ..clear_cache import clear_cache  # noqa: F401


@dataclass
class PotentialState:
    """State for potential-based propagator tests."""

    positions: jax.Array
    opt_state: optax.OptState


@dataclass
class PotentialStateWithCounter:
    """State with a counter to track patch applications."""

    positions: jax.Array
    opt_state: optax.OptState
    patch_count: jax.Array


class QuadraticPotential:
    """Mock potential: E = 0.5 * ||x||^2, grad = x."""

    def __call__(self, state: PotentialState, patch=None):
        del patch
        energy = 0.5 * jnp.sum(state.positions**2)
        gradients = state.positions
        return WithPatch(
            PotentialOut(
                total_energies=Table.arange(jnp.array([energy]), label=SystemId),
                gradients=gradients,
                hessians=(),
            ),
            IdPatch(),
        )


class QuadraticPotentialWithPatch:
    """Mock potential that increments a counter via patch."""

    def __call__(self, state: PotentialStateWithCounter, patch=None):
        del patch
        energy = 0.5 * jnp.sum(state.positions**2)
        gradients = state.positions

        def apply_fn(state, payload, accept):
            del accept
            return PotentialStateWithCounter(
                positions=state.positions,
                opt_state=state.opt_state,
                patch_count=state.patch_count + payload,
            )

        return WithPatch(
            PotentialOut(
                total_energies=Table.arange(jnp.array([energy]), label=SystemId),
                gradients=gradients,
                hessians=(),
            ),
            ExplicitPatch(payload=jnp.array(1), apply_fn=apply_fn),
        )


class TestRelaxationPropagator:
    """Tests for unified RelaxationPropagator."""

    def test_sgd_single_step(self):
        """SGD should take a single gradient step."""
        optimizer = optax.sgd(learning_rate=0.1)
        potential = QuadraticPotential()

        initial_pos = jnp.array([1.0, 2.0, 3.0])
        state = PotentialState(
            positions=initial_pos,
            opt_state=optimizer.init(initial_pos),
        )

        propagator = RelaxationPropagator(
            potential=potential,
            property=lens(lambda s: s.positions, cls=PotentialState),
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optimizer,
        )

        key = jax.random.key(0)
        new_state = propagator(key, state)

        expected = initial_pos - 0.1 * initial_pos
        npt.assert_allclose(new_state.positions, expected)

    def test_lbfgs_converges(self):
        """L-BFGS with linesearch should converge."""
        optimizer = optax.lbfgs()
        potential = QuadraticPotential()

        initial_pos = jnp.array([5.0, -3.0])
        state = PotentialState(
            positions=initial_pos,
            opt_state=optimizer.init(initial_pos),
        )

        propagator = RelaxationPropagator(
            potential=potential,
            property=lens(lambda s: s.positions, cls=PotentialState),
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optimizer,
        )
        propagator = jax.jit(propagator)

        key = jax.random.key(0)

        for _ in range(10):
            state = propagator(key, state)

        npt.assert_allclose(state.positions, jnp.zeros(2), atol=1e-6)

    def test_backtracking_linesearch(self):
        """Backtracking line search works."""
        optimizer = optax.chain(
            optax.sgd(learning_rate=1.0),
            optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=15,
                store_grad=True,
            ),
        )
        potential = QuadraticPotential()

        initial_pos = jnp.array([3.0, 4.0])
        state = PotentialState(
            positions=initial_pos,
            opt_state=optimizer.init(initial_pos),
        )

        propagator = RelaxationPropagator(
            potential=potential,
            property=lens(lambda s: s.positions, cls=PotentialState),
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optimizer,
        )

        key = jax.random.key(42)
        propagator = jax.jit(propagator)

        for _ in range(10):
            state = propagator(key, state)

        npt.assert_allclose(state.positions, jnp.zeros(2), atol=1e-6)

    def test_applies_patch_each_step(self):
        """Potential's patch should be applied after each relaxation step."""
        optimizer = optax.sgd(learning_rate=0.1)
        potential = QuadraticPotentialWithPatch()

        initial_pos = jnp.array([1.0, 2.0, 3.0])
        state = PotentialStateWithCounter(
            positions=initial_pos,
            opt_state=optimizer.init(initial_pos),
            patch_count=jnp.array(0),
        )

        propagator = RelaxationPropagator(
            potential=potential,
            property=lens(lambda s: s.positions, cls=PotentialStateWithCounter),
            opt_state=lens(lambda s: s.opt_state),
            optimizer=optimizer,
        )

        key = jax.random.key(0)

        state = propagator(key, state)

        assert state.patch_count == 1
