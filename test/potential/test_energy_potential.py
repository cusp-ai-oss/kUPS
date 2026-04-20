# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PotentialFromEnergy class from kups.potential.common.energy."""

from typing import NamedTuple

import jax.numpy as jnp
import pytest
from jax import Array

from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import lens, view
from kups.core.patch import IdPatch, Patch, WithPatch
from kups.core.potential import Energy, PotentialOut
from kups.core.typing import SystemId
from kups.potential.common.energy import (
    IdentityComposer,
    PotentialFromEnergy,
    Sum,
    Summand,
)


class ExampleState(NamedTuple):
    """Test state containing positions."""

    positions: Array
    total_potential: PotentialOut | None = None


class ExampleInput(NamedTuple):
    """Test input for energy function."""

    positions: Array


class ExampleGradients(NamedTuple):
    """Test gradients structure."""

    positions: Array


class ExampleHessians(NamedTuple):
    """Test Hessians structure."""

    positions: Array


class MockEnergyFunction:
    """Mock energy function: E = 0.5 * sum(x^2)."""

    def __call__(
        self, inp: ExampleInput
    ) -> WithPatch[Table[SystemId, Energy], Patch[ExampleState]]:
        energy = 0.5 * jnp.sum(inp.positions**2)
        return WithPatch(
            Table.arange(jnp.atleast_1d(energy), label=SystemId), IdPatch()
        )


class MockSumComposer:
    def __call__(self, state: ExampleState, patch=None) -> Sum[ExampleInput]:
        return Sum(
            Summand(ExampleInput(positions=state.positions), weight=1.0),
            add_previous_total=False,
        )


class MockLenses:
    @staticmethod
    def gradient_lens():
        return lens(lambda inp: ExampleGradients(positions=inp.positions))

    @staticmethod
    def hessian_lens():
        return lens(lambda grad: ExampleHessians(positions=grad.positions))

    @staticmethod
    def potential_lens():
        def get_potential(state):
            if state.total_potential is None:
                energy = Table.arange(jnp.array([0.0]), label=SystemId)
                gradients = ExampleGradients(positions=jnp.zeros_like(state.positions))
                n_particles, n_dims = state.positions.shape
                hessians = ExampleHessians(positions=jnp.zeros(n_particles * n_dims))
                return PotentialOut(energy, gradients, hessians)
            return state.total_potential

        return lens(get_potential)


class MockViews:
    @staticmethod
    def hessian_idx_view():
        def view_fn(state):
            n = state.positions.shape[0] * state.positions.shape[1]
            indices = jnp.array([[i, i] for i in range(n)])
            indices = indices.reshape(-1, 1, 2).transpose(0, 2, 1)
            return ExampleHessians(positions=indices)

        return view(view_fn)

    @staticmethod
    def patch_idx_view():
        def view_fn(state):
            sys_keys = (SystemId(0),)
            energy = Table(sys_keys, Index.new(sys_keys), _cls=SystemId)
            idx = Index(sys_keys, jnp.array([0]))
            return PotentialOut(
                energy, ExampleGradients(positions=idx), ExampleHessians(positions=idx)
            )

        return view(view_fn)


@pytest.fixture
def sample_state():
    return ExampleState(positions=jnp.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.fixture
def potential_from_energy():
    lenses = MockLenses()
    views = MockViews()
    return PotentialFromEnergy(
        energy_fn=MockEnergyFunction(),
        composer=MockSumComposer(),
        gradient_lens=lenses.gradient_lens(),
        hessian_lens=lenses.hessian_lens(),
        hessian_idx_view=views.hessian_idx_view(),
        cache_lens=lenses.potential_lens(),
        patch_idx_view=views.patch_idx_view(),
    )


class TestPotentialFromEnergy:
    """Test the PotentialFromEnergy class."""

    @classmethod
    def setup_class(cls):
        cls._jit_potential = None

    def test_call_outputs(self, potential_from_energy, sample_state):
        """Merged: basic + gradients + hessians (call once, check all)."""
        result = potential_from_energy(sample_state)

        assert isinstance(result, WithPatch)

        # Energy: 0.5 * (1^2 + 2^2 + 3^2 + 4^2) = 15.0
        expected_energy = 0.5 * (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2)
        assert jnp.isclose(result.data.total_energies.data, expected_energy)

        # Gradients: dE/dx = x
        assert jnp.allclose(result.data.gradients.positions, sample_state.positions)

        # Hessians: identity matrix diagonal -> all 1.0
        assert jnp.allclose(result.data.hessians.positions, 1.0)

    def test_empty_plan_assertion(self):
        """Test that empty plans raise assertion error."""
        lenses = MockLenses()
        views = MockViews()

        class EmptyComposer:
            def __call__(self, state, patch=None):
                return Sum(add_previous_total=False)

        potential = PotentialFromEnergy(
            energy_fn=MockEnergyFunction(),
            composer=EmptyComposer(),
            gradient_lens=lenses.gradient_lens(),
            hessian_lens=lenses.hessian_lens(),
            hessian_idx_view=views.hessian_idx_view(),
            cache_lens=lenses.potential_lens(),
            patch_idx_view=views.patch_idx_view(),
        )
        state = ExampleState(positions=jnp.array([[1.0, 2.0]]))
        with pytest.raises(
            AssertionError, match="At least one configuration must be added"
        ):
            potential(state)

    def test_multiple_summands(self, sample_state):
        """Weighted sum: 1.0*E(x) + 0.5*E(2x) = 15.0 + 30.0 = 45.0."""
        lenses = MockLenses()
        views = MockViews()

        class MultiSumComposer:
            def __call__(self, state, patch=None):
                return Sum(
                    Summand(ExampleInput(positions=state.positions), weight=1.0),
                    Summand(ExampleInput(positions=state.positions * 2), weight=0.5),
                    add_previous_total=False,
                )

        potential = PotentialFromEnergy(
            energy_fn=MockEnergyFunction(),
            composer=MultiSumComposer(),
            gradient_lens=lenses.gradient_lens(),
            hessian_lens=lenses.hessian_lens(),
            hessian_idx_view=views.hessian_idx_view(),
            cache_lens=lenses.potential_lens(),
            patch_idx_view=views.patch_idx_view(),
        )
        result = potential(sample_state)
        expected = 0.5 * jnp.sum(sample_state.positions**2) + 0.5 * 0.5 * jnp.sum(
            (sample_state.positions * 2) ** 2
        )
        assert jnp.isclose(result.data.total_energies.data, expected)

    def test_add_previous_total(self):
        """Current E(x)=15.0 + cached=5.0 -> 20.0."""
        lenses = MockLenses()
        views = MockViews()

        class PreviousTotalComposer:
            def __call__(self, state, patch=None):
                return Sum(
                    Summand(ExampleInput(positions=state.positions), weight=1.0),
                    add_previous_total=True,
                )

        potential = PotentialFromEnergy(
            energy_fn=MockEnergyFunction(),
            composer=PreviousTotalComposer(),
            gradient_lens=lenses.gradient_lens(),
            hessian_lens=lenses.hessian_lens(),
            hessian_idx_view=views.hessian_idx_view(),
            cache_lens=lenses.potential_lens(),
            patch_idx_view=views.patch_idx_view(),
        )
        existing_potential = PotentialOut(
            total_energies=Table.arange(jnp.array([5.0]), label=SystemId),
            gradients=ExampleGradients(positions=jnp.array([[1.0, 1.0], [1.0, 1.0]])),
            hessians=ExampleHessians(positions=jnp.array([1.0, 1.0, 1.0, 1.0])),
        )
        state = ExampleState(
            positions=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            total_potential=existing_potential,
        )
        result = potential(state)
        expected = 0.5 * jnp.sum(state.positions**2) + 5.0
        assert jnp.isclose(result.data.total_energies.data, expected)

    def test_different_input_shapes(self):
        """Test energy is correct for 2 different position shapes."""
        lenses = MockLenses()

        def simple_hessian_view(state):
            return ExampleHessians(positions=jnp.array([[0, 0]]).reshape(1, 2, 1))

        def simple_patch_view(state):
            energy = Table.arange(jnp.array([0]), label=SystemId)
            gradients = ExampleGradients(positions=jnp.zeros_like(state.positions))
            hessians = ExampleHessians(positions=jnp.array([0.0]))
            return PotentialOut(energy, gradients, hessians)

        potential = PotentialFromEnergy(
            energy_fn=MockEnergyFunction(),
            composer=MockSumComposer(),
            gradient_lens=lenses.gradient_lens(),
            hessian_lens=lenses.hessian_lens(),
            hessian_idx_view=view(simple_hessian_view),
            cache_lens=lenses.potential_lens(),
            patch_idx_view=view(simple_patch_view),
        )
        for positions in [
            jnp.array([[1.0, 2.0]]),
            jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        ]:
            state = ExampleState(positions=positions)
            result = potential(state)
            assert jnp.isclose(
                result.data.total_energies.data, 0.5 * jnp.sum(positions**2)
            )


class TestIdentityComposer:
    def test_returns_single_summand(self):
        composer = IdentityComposer()
        inp = ExampleInput(positions=jnp.array([[1.0, 2.0]]))
        result = composer(inp, None)
        assert len(result) == 1
        assert result[0].weight == 1
        assert jnp.array_equal(result[0].inp.positions, inp.positions)
        assert result.add_previous_total is False

    def test_rejects_patch(self):
        composer = IdentityComposer()
        inp = ExampleInput(positions=jnp.array([[1.0, 2.0]]))
        with pytest.raises(ValueError, match="does not support patches"):
            composer(inp, IdPatch())


class TestSum:
    def test_empty_sum(self):
        s = Sum(add_previous_total=False)
        assert len(s) == 0
        assert s.add_previous_total is False

    def test_sum_with_previous_total(self):
        inp = ExampleInput(positions=jnp.array([[1.0]]))
        s = Sum(Summand(inp, -1), Summand(inp, 1), add_previous_total=True)
        assert len(s) == 2
        assert s.add_previous_total is True
        assert s[0].weight == -1
        assert s[1].weight == 1
