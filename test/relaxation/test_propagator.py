# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for relaxation propagators."""

from typing import Literal, assert_type, overload

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax

from kups.core.data import Index, Table
from kups.core.lens import bind, lens
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import (
    CompensatedPotentialResult,
    Potential,
    PotentialOut,
    PotentialResult,
)
from kups.core.typing import SystemId
from kups.core.utils.jax import dataclass, field
from kups.core.utils.kahan import KahanSummand
from kups.relaxation.optimizer import ChainOptimizer, ChainOptState, chain
from kups.relaxation.propagator import RelaxationPropagator, UpdateMask
from kups.relaxation.transforms import (
    ScaleByAseLbfgs,
    ScaleByBacktrackingLinesearch,
    ScaleByMoreThuenteLinesearch,
)


@dataclass
class PotentialState:
    """State for potential-based propagator tests."""

    positions: jax.Array
    opt_state: ChainOptState


@dataclass
class PotentialStateWithCounter(PotentialState):
    """State with a counter to track patch applications."""

    patch_count: jax.Array


@dataclass
class QuadraticPotential[State: PotentialState](
    Potential[State, jax.Array, tuple[()], Patch[State]]
):
    """Mock potential: E = 0.5 * ||x||^2, grad = x."""

    cache_patch: Patch[State] = field(static=True, default_factory=IdPatch)

    @overload
    def __call__(
        self,
        state: State,
        patch: Patch[State] | None = None,
        *,
        include_compensate: Literal[False] = False,
    ) -> PotentialResult[State, jax.Array, tuple[()]]: ...
    @overload
    def __call__(
        self,
        state: State,
        patch: Patch[State] | None = None,
        *,
        include_compensate: Literal[True],
    ) -> CompensatedPotentialResult[State, jax.Array, tuple[()]]: ...
    def __call__(
        self,
        state: State,
        patch: Patch[State] | None = None,
        *,
        include_compensate: bool = False,
    ) -> (
        PotentialResult[State, jax.Array, tuple[()]]
        | CompensatedPotentialResult[State, jax.Array, tuple[()]]
    ):
        del patch
        energy = 0.5 * jnp.sum(state.positions**2)
        out = PotentialOut(
            total_energies=Table.arange(jnp.array([energy]), label=SystemId),
            gradients=state.positions,
            hessians=(),
        )
        if include_compensate:
            return WithPatch(KahanSummand.init(out), self.cache_patch)
        return WithPatch(out, self.cache_patch)


def _counted_quadratic() -> QuadraticPotential[PotentialStateWithCounter]:
    return QuadraticPotential(
        cache_patch=lambda s, accept: (
            bind(s).focus(lambda x: x.patch_count).apply(lambda count: count + 1)
        )
    )


class TestRelaxationPropagator:
    """Tests for unified RelaxationPropagator."""

    def test_sgd_single_step(self):
        """SGD should take a single gradient step."""
        optimizer: ChainOptimizer[jax.Array] = chain(optax.sgd(learning_rate=0.1))
        potential = QuadraticPotential[PotentialState]()

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
        optimizer: ChainOptimizer[jax.Array] = chain(optax.sgd(learning_rate=0.1))
        potential = _counted_quadratic()

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

    def test_mask_reads_patched_state_and_preserves_rejected_rows(self) -> None:
        optimizer: ChainOptimizer[jax.Array] = chain(optax.sgd(learning_rate=0.1))
        positions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        state = PotentialStateWithCounter(
            positions=positions,
            opt_state=optimizer.init(positions),
            patch_count=jnp.array(0),
        )
        system_index = Index.integer(jnp.array([0, 1]), n=2, label=SystemId)
        mask: UpdateMask[PotentialStateWithCounter, Index[SystemId]] = UpdateMask(
            accept=lambda s: Table(
                system_index.keys, jnp.array([s.patch_count > 0, False])
            ),
            system_index=lambda s: system_index,
        )
        assert_type(mask.system_index(state), Index[SystemId])
        step = jax.jit(
            RelaxationPropagator(
                potential=_counted_quadratic(),
                property=lens(lambda s: s.positions, cls=PotentialStateWithCounter),
                opt_state=lens(lambda s: s.opt_state),
                optimizer=optimizer,
                mask=mask,
            )
        )
        updated = step(jax.random.key(0), state)
        npt.assert_allclose(updated.positions[0], 0.9 * positions[0])
        npt.assert_array_equal(updated.positions[1], positions[1])
        assert updated.patch_count == 1

    def test_kups_more_thuente_linesearch_converges(self):
        """The per-system More-Thuente search drives the quadratic to its minimum."""
        optimizer: ChainOptimizer[jax.Array] = chain(
            optax.scale(-1.0), ScaleByMoreThuenteLinesearch()
        )
        state = PotentialState(
            positions=jnp.array([5.0, -3.0]),
            opt_state=optimizer.init(jnp.array([5.0, -3.0])),
        )
        propagator = jax.jit(
            RelaxationPropagator(
                potential=QuadraticPotential[PotentialState](),
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
        optimizer: ChainOptimizer[jax.Array] = chain(
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
                potential=QuadraticPotential[PotentialState](),
                property=lens(lambda s: s.positions, cls=PotentialState),
                opt_state=lens(lambda s: s.opt_state),
                optimizer=optimizer,
            )
        )
        key = jax.random.key(0)
        for _ in range(8):
            state = propagator(key, state)
        npt.assert_allclose(state.positions, jnp.zeros(2), atol=1e-6)
