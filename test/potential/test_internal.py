# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the InternalEnergies potential implementation."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.assertion import runtime_assert
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import view
from kups.core.patch import Accept, IdPatch, WithPatch
from kups.core.potential import EmptyType, PotentialOut
from kups.core.result import as_result_function
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import dataclass, field
from kups.potential.classical.internal import InternalEnergies, MotifData


def _make_motifs(data: Array, system_ids: list[int]) -> Table[ParticleId, MotifData]:
    system = Index.new([SystemId(i) for i in system_ids])
    return Table.arange(MotifData(data, system), label=ParticleId)


@dataclass
class MockState:
    """Test state containing motifs and potential outputs."""

    motifs: Table[ParticleId, MotifData]
    motif_potential_out: Array
    other_data: float = field(default=1.0)


class TestInternalEnergies:
    """Tests for InternalEnergies potential class."""

    @pytest.fixture
    def internal_energies_potential(self):
        """Create an InternalEnergies potential instance."""
        return InternalEnergies(
            motifs=view(lambda state: state.motifs),
            motif_potential_out=view(
                lambda state: state.motif_potential_out, cls=MockState
            ),
        )

    def test_basic_calculation_and_structure(self, internal_energies_potential):
        """Test basic internal energy calculation and result structure."""
        data = jnp.array([0, 1, 2, 3, 4])
        motifs = _make_motifs(data, [0, 0, 0, 1, 1])
        energies = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        state = MockState(motifs=motifs, motif_potential_out=energies)

        result = internal_energies_potential(state)

        # Check structure
        assert isinstance(result, WithPatch)
        assert isinstance(result.data, PotentialOut)
        assert isinstance(result.patch, IdPatch)
        assert isinstance(result.data.gradients, EmptyType)
        assert isinstance(result.data.hessians, EmptyType)
        assert result.data.total_energies.data.ndim == 1

        # Check values: system 0 (10+20+30=60), system 1 (40+50=90)
        npt.assert_array_equal(result.data.total_energies.data, jnp.array([60.0, 90.0]))

    @pytest.mark.parametrize(
        "data,system_ids,energies,expected",
        [
            pytest.param(
                jnp.array([0, 1]),
                [0, 1],
                jnp.array([0.0, 0.0]),
                jnp.array([0.0, 0.0]),
                id="zero_energies",
            ),
            pytest.param(
                jnp.array([0, 1, 2]),
                [0, 0, 1],
                jnp.array([10.0, -20.0, -5.0]),
                jnp.array([-10.0, -5.0]),
                id="negative_energies",
            ),
            pytest.param(
                jnp.array([0, 1, 2]),
                [0, 0, 0],
                jnp.array([5.0, 10.0, 15.0]),
                jnp.array([30.0]),
                id="single_system",
            ),
            pytest.param(
                jnp.array([0, 1, 2, 3, 4, 5]),
                [0, 1, 1, 1, 2, 2],
                jnp.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
                jnp.array([100.0, 900.0, 1100.0]),
                id="multiple_different_sizes",
            ),
        ],
    )
    def test_energy_values(
        self, internal_energies_potential, data, system_ids, energies, expected
    ):
        """Test internal energy computation for various configurations."""
        motifs = _make_motifs(data, system_ids)
        state = MockState(motifs=motifs, motif_potential_out=energies)
        result = internal_energies_potential(state)
        npt.assert_array_equal(result.data.total_energies.data, expected)

    def test_empty_motifs(self, internal_energies_potential):
        """Test behavior with empty motifs."""
        motifs = Table(
            (),
            MotifData(
                jnp.array([], dtype=int),
                Index((), jnp.array([], dtype=int), _cls=SystemId),
            ),
            _cls=ParticleId,
        )
        state = MockState(motifs=motifs, motif_potential_out=jnp.array([]))
        result = internal_energies_potential(state)
        npt.assert_array_equal(result.data.total_energies.data, jnp.array([]))

    def test_jax_transformations(self, internal_energies_potential):
        """Test that InternalEnergies works with JAX transformations."""
        data = jnp.array([0, 1, 2, 3, 4])
        motifs = _make_motifs(data, [0, 0, 0, 1, 1])
        energies = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        state = MockState(motifs=motifs, motif_potential_out=energies)

        def compute_energy(s):
            return internal_energies_potential(s).data.total_energies.data.sum()

        # JIT
        npt.assert_allclose(jax.jit(compute_energy)(state), compute_energy(state))

        # Grad
        def energy_fn(e):
            return compute_energy(MockState(motifs=motifs, motif_potential_out=e))

        gradients = jax.grad(energy_fn)(energies)
        npt.assert_array_equal(gradients, jnp.ones_like(energies))

    def test_patch_application(self, internal_energies_potential):
        """Test that the patch= code path works with IdPatch."""
        data = jnp.array([0, 1, 2, 3, 4])
        motifs = _make_motifs(data, [0, 0, 0, 1, 1])
        energies = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        state = MockState(motifs=motifs, motif_potential_out=energies)

        result_no_patch = internal_energies_potential(state)
        result_with_patch = internal_energies_potential(state, patch=IdPatch())
        npt.assert_array_equal(
            result_no_patch.data.total_energies.data,
            result_with_patch.data.total_energies.data,
        )

    def test_with_assertion_propagation(self, internal_energies_potential):
        """Test that assertions from patches are properly propagated."""
        data = jnp.array([0, 1, 2, 3, 4])
        motifs = _make_motifs(data, [0, 0, 0, 1, 1])
        energies = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        state = MockState(motifs=motifs, motif_potential_out=energies)

        @dataclass
        class AssertionPatch:
            def __call__(self, state: MockState, accept: Accept) -> MockState:
                sys_idx = Index.new(accept.keys)
                runtime_assert(
                    predicate=jnp.array(True),
                    message="Test assertion from patch",
                    fmt_args={"accept": accept[sys_idx]},
                    fix_args=accept,
                )
                return state

        result = as_result_function(internal_energies_potential)(
            state, patch=AssertionPatch()
        )
        assert len(result.assertions) == 1
        assert "Test assertion from patch" in result.assertions[0].message
