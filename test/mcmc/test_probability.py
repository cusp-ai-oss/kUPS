# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax import Array

from kups.core.constants import BOLTZMANN_CONSTANT
from kups.core.data.index import Index
from kups.core.data.table import Table
from kups.core.lens import lens
from kups.core.patch import Accept, ComposedPatch, WithPatch
from kups.core.potential import CachedPotential, Potential, PotentialOut
from kups.core.typing import GroupId, MotifId, SystemId
from kups.core.utils.jax import dataclass, field
from kups.mcmc.probability import BoltzmannLogProbabilityRatio, motif_counts

# -- helpers for motif_counts tests --


@dataclass
class _GroupData:
    system: Index[SystemId]
    motif: Index[MotifId]


def _make_groups(
    motif_ids: list[int],
    system_ids: list[int],
    n_systems: int,
    n_motifs: int,
) -> Table[GroupId, _GroupData]:
    sys_keys = tuple(SystemId(i) for i in range(n_systems))
    motif_keys = tuple(MotifId(i) for i in range(n_motifs))
    data = _GroupData(
        system=Index(sys_keys, jnp.array(system_ids, dtype=int)),
        motif=Index(motif_keys, jnp.array(motif_ids, dtype=int)),
    )
    return Table.arange(data, label=GroupId)


# -- motif_counts tests --


class TestMotifCounts:
    @pytest.mark.parametrize(
        "motif_ids,system_ids,n_systems,n_motifs,expected",
        [
            pytest.param(
                [0, 0, 0],
                [0, 0, 0],
                1,
                1,
                [[3]],
                id="single_system_single_species",
            ),
            pytest.param(
                [0, 1, 0, 1, 1],
                [0, 0, 0, 0, 0],
                1,
                2,
                [[2, 3]],
                id="single_system_multiple_species",
            ),
            pytest.param(
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                2,
                1,
                [[2], [2]],
                id="multi_system_single_species",
            ),
            pytest.param(
                [0, 1, 0, 1, 1],
                [0, 0, 1, 1, 1],
                2,
                2,
                [[1, 1], [1, 2]],
                id="multi_system_multi_species",
            ),
            pytest.param(
                [0, 1, 0],
                [0, 0, 0],
                2,
                2,
                [[2, 1], [0, 0]],
                id="empty_system",
            ),
            pytest.param(
                [0, 1, 5],
                [0, 0, 0],
                1,
                6,
                [[1, 1, 0, 0, 0, 1]],
                id="out_of_bounds_species",
            ),
            pytest.param(
                [],
                [],
                3,
                4,
                jnp.zeros((3, 4), dtype=int).tolist(),
                id="no_groups",
            ),
        ],
    )
    def test_motif_counts(self, motif_ids, system_ids, n_systems, n_motifs, expected):
        """Verify motif_counts for various configurations."""
        groups = _make_groups(motif_ids, system_ids, n_systems, n_motifs)
        result = motif_counts(groups)
        assert result.shape == (n_systems, n_motifs)
        npt.assert_array_equal(result, expected)


@dataclass
class MockState:
    """Mock state class for testing."""

    temperature: Array
    last_potential_out: PotentialOut


@dataclass
class MockPatch:
    """Mock patch class for testing."""

    name: str = field(default="test_patch", static=True)

    def __call__(self, state: MockState, accept: Accept) -> MockState:
        return state


class TestNVTProbabilityRatio:
    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.n_systems = 3
        cls.temperature = jnp.array([300.0, 350.0, 400.0])
        cls.old_energies = jnp.array([-100.0, -150.0, -200.0])
        cls.last_potential_out = PotentialOut(
            total_energies=Table.arange(cls.old_energies, label=SystemId),
            gradients=(),
            hessians=(),
        )
        cls.state = MockState(
            temperature=cls.temperature, last_potential_out=cls.last_potential_out
        )
        cls.temperature_view = staticmethod(lambda state: state.temperature)
        cls.test_patch = MockPatch("test_move")

    def create_mock_potential(
        self, new_energies: Array, patch_result: MockPatch | None = None
    ):
        """Create a mock potential that returns specified energies."""
        result_patch: MockPatch = (
            patch_result if patch_result is not None else MockPatch()
        )

        def mock_potential(
            state: MockState, patch: MockPatch | None = None
        ) -> WithPatch[PotentialOut[tuple[()], tuple[()]], MockPatch]:
            new_potential_out: PotentialOut[tuple[()], tuple[()]] = PotentialOut(
                total_energies=Table.arange(new_energies, label=SystemId),
                gradients=(),
                hessians=(),
            )
            return WithPatch(new_potential_out, result_patch)

        potential = CachedPotential(
            cast(Potential[MockState, Any, Any, MockPatch], mock_potential),
            lens(lambda x: x.last_potential_out, cls=MockState),
            lambda state: PotentialOut(
                Table.arange(jnp.arange(3), label=SystemId), (), ()
            ),
        )
        return potential

    def test_result_and_patch_structure(self):
        """Test return type is WithPatch wrapping a ComposedPatch with correct patch."""
        new_energies = jnp.array([-105.0, -155.0, -205.0])
        potential_patch = MockPatch("potential_patch")
        mock_potential = self.create_mock_potential(new_energies, potential_patch)
        probability_calc = BoltzmannLogProbabilityRatio(
            temperature=self.temperature_view, potential=mock_potential
        )
        result = probability_calc(self.state, self.test_patch)
        assert isinstance(result, WithPatch)
        assert isinstance(result.patch, ComposedPatch)
        assert result.patch.patches[0] == potential_patch  # type: ignore

    @pytest.mark.parametrize(
        "delta_e",
        [-10.0, -5.0, 0.0, 5.0, 10.0],
        ids=[
            "large_decrease",
            "small_decrease",
            "no_change",
            "small_increase",
            "large_increase",
        ],
    )
    def test_exact_boltzmann_ratio(self, delta_e):
        """Verify (E_old - E_new) / (T * kB) for various energy differences."""
        new_energies = self.old_energies + delta_e
        mock_potential = self.create_mock_potential(new_energies)
        probability_calc = BoltzmannLogProbabilityRatio(
            temperature=self.temperature_view, potential=mock_potential
        )
        result = probability_calc(self.state, self.test_patch)
        expected = (self.old_energies - new_energies) / (
            self.temperature * BOLTZMANN_CONSTANT
        )
        npt.assert_allclose(result.data.data, expected, rtol=1e-10)

    def test_temperature_dependence_exact(self):
        """Test exact temperature dependence of probability ratio."""
        energy_increase = 5.0
        new_energies = self.old_energies + energy_increase

        low_temp = jnp.array([100.0, 100.0, 100.0])
        high_temp = jnp.array([1000.0, 1000.0, 1000.0])

        mock_potential = self.create_mock_potential(new_energies)

        low_temp_state = MockState(low_temp, self.last_potential_out)
        probability_calc_low = BoltzmannLogProbabilityRatio(
            temperature=self.temperature_view,
            potential=mock_potential,
        )
        result_low = probability_calc_low(low_temp_state, self.test_patch)

        high_temp_state = MockState(high_temp, self.last_potential_out)
        probability_calc_high = BoltzmannLogProbabilityRatio(
            temperature=self.temperature_view,
            potential=mock_potential,
        )
        result_high = probability_calc_high(high_temp_state, self.test_patch)

        expected_log_low = -energy_increase / (low_temp * BOLTZMANN_CONSTANT)
        expected_log_high = -energy_increase / (high_temp * BOLTZMANN_CONSTANT)

        npt.assert_allclose(result_low.data.data, expected_log_low, rtol=1e-10)
        npt.assert_allclose(result_high.data.data, expected_log_high, rtol=1e-10)

    def test_boltzmann_constant_scaling_exact(self):
        """Test exact Boltzmann constant scaling."""
        old_energy = jnp.array([0.0])
        new_energy = jnp.array([BOLTZMANN_CONSTANT])
        temperature = jnp.array([1.0])

        simple_potential_out = PotentialOut(
            total_energies=Table.arange(old_energy, label=SystemId),
            gradients=(),
            hessians=(),
        )
        simple_state = MockState(temperature, simple_potential_out)

        mock_potential = self.create_mock_potential(new_energy)

        probability_calc = BoltzmannLogProbabilityRatio(
            temperature=self.temperature_view,
            potential=mock_potential,
        )

        result = probability_calc(simple_state, self.test_patch)

        expected_log_ratio = jnp.array([-1.0])
        npt.assert_allclose(result.data.data, expected_log_ratio, rtol=1e-15)

    def test_scalar_temperature_broadcasting(self):
        """Test scalar temperature broadcasts against vector energies."""
        scalar_temp = jnp.array([500.0])
        scalar_potential_out = PotentialOut(
            total_energies=Table.arange(jnp.array([-100.0]), label=SystemId),
            gradients=(),
            hessians=(),
        )
        scalar_state = MockState(scalar_temp, scalar_potential_out)
        new_energies = jnp.array([-105.0])
        mock_potential = self.create_mock_potential(new_energies)
        probability_calc = BoltzmannLogProbabilityRatio(
            temperature=self.temperature_view, potential=mock_potential
        )
        result = probability_calc(scalar_state, self.test_patch)
        expected = (jnp.array([-100.0]) - new_energies) / (
            scalar_temp * BOLTZMANN_CONSTANT
        )
        npt.assert_allclose(result.data.data, expected, rtol=1e-10)

    def test_single_system(self):
        """Test with a single system."""
        single_temp = jnp.array([300.0])
        single_old_energy = jnp.array([-100.0])
        single_new_energy = jnp.array([-105.0])

        single_potential_out = PotentialOut(
            total_energies=Table.arange(single_old_energy, label=SystemId),
            gradients=(),
            hessians=(),
        )
        single_state = MockState(single_temp, single_potential_out)

        mock_potential = self.create_mock_potential(single_new_energy)

        probability_calc = BoltzmannLogProbabilityRatio(
            temperature=self.temperature_view,
            potential=mock_potential,
        )

        probability_ratio_and_patch = probability_calc(single_state, self.test_patch)

        assert probability_ratio_and_patch.data.data.shape == (1,)
        expected_log_ratio = (single_old_energy - single_new_energy) / (
            single_temp * BOLTZMANN_CONSTANT
        )
        npt.assert_allclose(
            probability_ratio_and_patch.data.data,
            expected_log_ratio,
            rtol=1e-10,
        )
