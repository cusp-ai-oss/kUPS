# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for relaxation propagators."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.core import units
from kups.core.data import Table
from kups.core.lens import lens
from kups.core.patch import ExplicitPatch, IdPatch, WithPatch
from kups.core.potential import PotentialOut
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass
from kups.core.utils.units import magnitudes_of
from kups.relaxation.optimizer import Optimizer, apply_updates, chain, relabel_like
from kups.relaxation.propagator import RelaxationPropagator
from kups.relaxation.transforms import (
    ScaleByAseLbfgs,
    ScaleByBacktrackingLinesearch,
    ScaleByMoreThuenteLinesearch,
)
from test.units import EV, A, assert_close


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


class DimensionedQuadraticPotential:
    """Mock potential: E = 0.5 * k * ||x||^2 with x a length, grad = k * x."""

    def __call__(self, state: PotentialState, patch=None):
        del patch
        k = 1.0 * EV / A**2
        energy = 0.5 * k * units.sum(state.positions**2)
        return WithPatch(
            PotentialOut(
                total_energies=Table.arange(units.stack([energy]), label=SystemId),
                gradients=k * state.positions,
                hessians=(),
            ),
            IdPatch(),
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

    def test_kups_more_thuente_linesearch_converges(self):
        """The per-system More-Thuente search drives the quadratic to its minimum."""
        optimizer: Optimizer[Any, Any] = chain(
            optax.scale(-1.0), ScaleByMoreThuenteLinesearch()
        )
        state = PotentialState(
            positions=jnp.array([5.0, -3.0]),
            opt_state=optimizer.init(jnp.array([5.0, -3.0])),
        )
        propagator = jax.jit(
            RelaxationPropagator(
                potential=QuadraticPotential(),
                property=lens(lambda s: s.positions, cls=PotentialState),
                opt_state=lens(lambda s: s.opt_state),
                optimizer=optimizer,
            )
        )
        key = jax.random.key(0)
        for _ in range(5):
            state = propagator(key, state)
        npt.assert_allclose(state.positions, jnp.zeros(2), atol=1e-6)

    def test_kups_lbfgs_with_backtracking_converges(self):
        """L-BFGS direction + per-system backtracking (the documented chain)."""
        optimizer: Optimizer[Any, Any] = chain(
            ScaleByAseLbfgs(memory_size=10, alpha=1.0),
            optax.scale(-1.0),
            ScaleByBacktrackingLinesearch(),
        )
        state = PotentialState(
            positions=jnp.array([5.0, -3.0]),
            opt_state=optimizer.init(jnp.array([5.0, -3.0])),
        )
        propagator = jax.jit(
            RelaxationPropagator(
                potential=QuadraticPotential(),
                property=lens(lambda s: s.positions, cls=PotentialState),
                opt_state=lens(lambda s: s.opt_state),
                optimizer=optimizer,
            )
        )
        key = jax.random.key(0)
        for _ in range(8):
            state = propagator(key, state)
        npt.assert_allclose(state.positions, jnp.zeros(2), atol=1e-6)

    def test_dimensioned_positions_converge(self):
        """Length positions relax through a line search and stay lengths."""
        optimizer: Optimizer[Any, Any] = chain(
            optax.scale(-1.0), ScaleByMoreThuenteLinesearch()
        )
        positions = jnp.array([5.0, -3.0]) * A
        state = PotentialState(
            positions=positions, opt_state=optimizer.init(magnitudes_of(positions))
        )
        propagator = jax.jit(
            RelaxationPropagator(
                potential=DimensionedQuadraticPotential(),
                property=lens(lambda s: s.positions, cls=PotentialState),
                opt_state=lens(lambda s: s.opt_state),
                optimizer=optimizer,
            )
        )
        for _ in range(5):
            state = propagator(jax.random.key(0), state)
        assert_close(state.positions, jnp.zeros(2) * A, atol=1e-6)


def test_relabel_and_apply_updates_keep_dimensions():
    """Steps on magnitudes are relabelled per leaf; logs keep their reference."""
    params = (
        jnp.array([1.0, 2.0]) * A,
        units.Log.from_logs(jnp.zeros(2), 1 / A**3),
    )
    steps = jax.tree.map(lambda x: jnp.full_like(x, 0.5), magnitudes_of(params))
    new_x, new_log = apply_updates(params, relabel_like(params, steps))
    assert_close(new_x, jnp.array([1.5, 2.5]) * A)
    npt.assert_allclose(new_log.log, jnp.full(2, 0.5))
    assert new_log.ref == params[1].ref
