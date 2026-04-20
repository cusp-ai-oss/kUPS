# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest

from kups.core.assertion import runtime_assert
from kups.core.data import Table
from kups.core.data.index import Index
from kups.core.lens import SimpleLens, view
from kups.core.patch import IdPatch, WithPatch
from kups.core.potential import (
    CachedPotential,
    MappedPotential,
    PotentialOut,
    sum_potentials,
)
from kups.core.result import as_result_function
from kups.core.typing import SystemId


# Test fixtures and helper classes
@pytest.fixture
def sample_energy():
    """Sample energy array for testing."""
    return jnp.array([1.0, 2.0, 3.0])


@pytest.fixture
def sample_gradients():
    """Sample gradients for testing."""
    return {"pos": jnp.array([[1.0, 2.0], [3.0, 4.0]])}


@pytest.fixture
def sample_hessians():
    """Sample hessians for testing."""
    return {"pos": jnp.array([[[1.0, 0.0], [0.0, 1.0]]])}


class MockPotential:
    """Mock potential for testing."""

    def __init__(
        self, energy_multiplier=1.0, gradient_multiplier=1.0, hessian_multiplier=1.0
    ):
        self.energy_multiplier = energy_multiplier
        self.gradient_multiplier = gradient_multiplier
        self.hessian_multiplier = hessian_multiplier

    def __call__(self, state, patch=None):
        # Mock implementation that returns predictable values
        energy = jnp.array([1.0, 2.0, 3.0]) * self.energy_multiplier
        gradients = {
            "pos": jnp.array([[1.0, 2.0], [3.0, 4.0]]) * self.gradient_multiplier
        }
        hessians = {
            "pos": jnp.array([[[1.0, 0.0], [0.0, 1.0]]]) * self.hessian_multiplier
        }

        potential_out = PotentialOut(
            total_energies=Table.arange(energy, label=SystemId),
            gradients=gradients,
            hessians=hessians,
        )

        return WithPatch(potential_out, IdPatch())


class TestPotentialOut:
    """Test the PotentialOut class."""

    def test_init(self, sample_energy, sample_gradients, sample_hessians):
        """Test PotentialOut initialization."""
        pot_out = PotentialOut(
            total_energies=Table.arange(sample_energy, label=SystemId),
            gradients=sample_gradients,
            hessians=sample_hessians,
        )

        assert jnp.array_equal(pot_out.total_energies.data, sample_energy)
        assert jnp.array_equal(pot_out.gradients["pos"], sample_gradients["pos"])
        assert jnp.array_equal(pot_out.hessians["pos"], sample_hessians["pos"])

    def test_add_operation(self, sample_energy, sample_gradients, sample_hessians):
        """Test addition of PotentialOut objects."""
        pot_out1 = PotentialOut(
            total_energies=Table.arange(sample_energy, label=SystemId),
            gradients=sample_gradients,
            hessians=sample_hessians,
        )

        pot_out2 = PotentialOut(
            total_energies=Table.arange(sample_energy * 2, label=SystemId),
            gradients={"pos": sample_gradients["pos"] * 2},
            hessians={"pos": sample_hessians["pos"] * 2},
        )

        result = pot_out1 + pot_out2

        expected_energy = sample_energy + sample_energy * 2
        expected_gradients = {
            "pos": sample_gradients["pos"] + sample_gradients["pos"] * 2
        }
        expected_hessians = {"pos": sample_hessians["pos"] + sample_hessians["pos"] * 2}

        assert jnp.array_equal(result.total_energies.data, expected_energy)
        assert jnp.array_equal(result.gradients["pos"], expected_gradients["pos"])
        assert jnp.array_equal(result.hessians["pos"], expected_hessians["pos"])

    def test_sub_operation(self, sample_energy, sample_gradients, sample_hessians):
        """Test subtraction of PotentialOut objects."""
        pot_out1 = PotentialOut(
            total_energies=Table.arange(sample_energy * 3, label=SystemId),
            gradients={"pos": sample_gradients["pos"] * 3},
            hessians={"pos": sample_hessians["pos"] * 3},
        )

        pot_out2 = PotentialOut(
            total_energies=Table.arange(sample_energy, label=SystemId),
            gradients=sample_gradients,
            hessians=sample_hessians,
        )

        result = pot_out1 - pot_out2

        expected_energy = sample_energy * 3 - sample_energy
        expected_gradients = {
            "pos": sample_gradients["pos"] * 3 - sample_gradients["pos"]
        }
        expected_hessians = {"pos": sample_hessians["pos"] * 3 - sample_hessians["pos"]}

        assert jnp.array_equal(result.total_energies.data, expected_energy)
        assert jnp.array_equal(result.gradients["pos"], expected_gradients["pos"])
        assert jnp.array_equal(result.hessians["pos"], expected_hessians["pos"])

    def test_mul_operation(self, sample_energy, sample_gradients, sample_hessians):
        """Test multiplication of PotentialOut by scalar."""
        pot_out = PotentialOut(
            total_energies=Table.arange(sample_energy, label=SystemId),
            gradients=sample_gradients,
            hessians=sample_hessians,
        )

        scalar = 2.5
        result = pot_out * scalar

        expected_energy = sample_energy * scalar
        expected_gradients = {"pos": sample_gradients["pos"] * scalar}
        expected_hessians = {"pos": sample_hessians["pos"] * scalar}

        assert jnp.array_equal(result.total_energies.data, expected_energy)
        assert jnp.array_equal(result.gradients["pos"], expected_gradients["pos"])
        assert jnp.array_equal(result.hessians["pos"], expected_hessians["pos"])

    def test_rmul_operation(self, sample_energy, sample_gradients, sample_hessians):
        """Test right multiplication of PotentialOut by scalar."""
        pot_out = PotentialOut(
            total_energies=Table.arange(sample_energy, label=SystemId),
            gradients=sample_gradients,
            hessians=sample_hessians,
        )

        scalar = 3.0
        result = scalar * pot_out

        expected_energy = sample_energy * scalar
        expected_gradients = {"pos": sample_gradients["pos"] * scalar}
        expected_hessians = {"pos": sample_hessians["pos"] * scalar}

        assert jnp.array_equal(result.total_energies.data, expected_energy)
        assert jnp.array_equal(result.gradients["pos"], expected_gradients["pos"])
        assert jnp.array_equal(result.hessians["pos"], expected_hessians["pos"])

    def test_as_tuple_property(self, sample_energy, sample_gradients, sample_hessians):
        """Test the as_tuple property."""
        pot_out = PotentialOut(
            total_energies=Table.arange(sample_energy, label=SystemId),
            gradients=sample_gradients,
            hessians=sample_hessians,
        )

        energy, gradients, hessians = pot_out.as_tuple

        assert jnp.array_equal(energy.data, sample_energy)
        assert jnp.array_equal(gradients["pos"], sample_gradients["pos"])
        assert jnp.array_equal(hessians["pos"], sample_hessians["pos"])


class TestComposePotentials:
    """Test the compose_potentials function."""

    def test_compose_single_potential(self):
        """Test composing a single potential."""
        potential = MockPotential()
        composed = sum_potentials(potential)

        mock_state = {"positions": jnp.array([[0.0, 0.0]])}
        result = composed(mock_state)

        assert isinstance(result, WithPatch)
        assert jnp.array_equal(
            result.data.total_energies.data, jnp.array([1.0, 2.0, 3.0])
        )

    def test_compose_multiple_potentials(self):
        """Test composing multiple potentials."""
        potential1 = MockPotential(energy_multiplier=1.0, gradient_multiplier=1.0)
        potential2 = MockPotential(energy_multiplier=2.0, gradient_multiplier=2.0)
        potential3 = MockPotential(energy_multiplier=3.0, gradient_multiplier=3.0)

        composed = sum_potentials(potential1, potential2, potential3)

        mock_state = {"positions": jnp.array([[0.0, 0.0]])}
        result = composed(mock_state)

        # Energy should be sum of all potentials: 1*[1,2,3] + 2*[1,2,3] + 3*[1,2,3] = 6*[1,2,3]
        expected_energy = jnp.array([6.0, 12.0, 18.0])
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)

        # Gradients should also be summed
        expected_gradients = {"pos": jnp.array([[6.0, 12.0], [18.0, 24.0]])}
        assert jnp.array_equal(result.data.gradients["pos"], expected_gradients["pos"])

    def test_compose_potentials_with_patch(self):
        """Test composing potentials with patch argument."""
        potential1 = MockPotential()
        potential2 = MockPotential()

        composed = sum_potentials(potential1, potential2)

        mock_state = {"positions": jnp.array([[0.0, 0.0]])}
        mock_patch = IdPatch()

        result = composed(mock_state, mock_patch)

        assert isinstance(result, WithPatch)
        # Energy should be sum of two potentials: 2*[1,2,3]
        expected_energy = jnp.array([2.0, 4.0, 6.0])
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)

    def test_compose_empty_potentials_raises_error(self):
        """Test that composing empty potentials raises ValueError."""
        with pytest.raises(ValueError, match="At least one potential must be provided"):
            sum_potentials()

    def test_compose_potentials_linearity(self):
        """Test that composition preserves linearity."""
        # Create potentials with different multipliers
        potential1 = MockPotential(energy_multiplier=2.0, gradient_multiplier=2.0)
        potential2 = MockPotential(energy_multiplier=3.0, gradient_multiplier=3.0)

        composed = sum_potentials(potential1, potential2)

        mock_state = {"positions": jnp.array([[0.0, 0.0]])}
        result = composed(mock_state)

        # Check that the result is linear combination
        expected_energy = jnp.array([5.0, 10.0, 15.0])  # 2*[1,2,3] + 3*[1,2,3]
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)

        expected_gradients = {"pos": jnp.array([[5.0, 10.0], [15.0, 20.0]])}
        assert jnp.array_equal(result.data.gradients["pos"], expected_gradients["pos"])


# Integration tests
class TestIntegration:
    """Integration tests for the potential module."""

    def test_full_workflow(self):
        """Test a complete workflow with multiple potentials."""
        # Create multiple potentials
        coulomb = MockPotential(energy_multiplier=1.0, gradient_multiplier=1.0)
        lj = MockPotential(energy_multiplier=2.0, gradient_multiplier=2.0)

        # Compose them
        total_potential = sum_potentials(coulomb, lj)

        # Create mock state
        mock_state = {"positions": jnp.array([[0.0, 0.0], [1.0, 1.0]])}

        # Calculate potential
        result = total_potential(mock_state)

        # Verify result structure
        assert isinstance(result, WithPatch)

        # Verify energy is sum of components
        expected_energy = jnp.array([3.0, 6.0, 9.0])  # 1*[1,2,3] + 2*[1,2,3]
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)

        # Test arithmetic operations on results
        scaled_result = result.data * 2.0
        assert jnp.array_equal(scaled_result.total_energies.data, expected_energy * 2.0)

        # Test tuple conversion
        energy, gradients, hessians = result.data.as_tuple
        assert jnp.array_equal(energy.data, expected_energy)
        assert "pos" in gradients
        assert "pos" in hessians

    def test_jax_transformations(self):
        """Test that PotentialOut works with JAX transformations."""
        potential = MockPotential()
        mock_state = {"positions": jnp.array([[0.0, 0.0]])}

        # Test JIT compilation
        jitted_potential = jax.jit(potential)
        result = jitted_potential(mock_state)

        assert isinstance(result, WithPatch)

        # Test that energies are computed correctly
        expected_energy = jnp.array([1.0, 2.0, 3.0])
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)


class TestCachedPotential:
    """Tests for CachedPotential class."""

    @pytest.fixture
    def test_state(self):
        """Create a test state with cache."""
        return {
            "positions": jnp.array([[0.0, 0.0], [1.0, 1.0]]),
            "cached_energy": PotentialOut(
                total_energies=Table.arange(jnp.array([0.0, 0.0, 0.0]), label=SystemId),
                gradients={"pos": jnp.zeros((2, 2))},
                hessians={"pos": jnp.zeros((1, 2, 2))},
            ),
        }

    @pytest.fixture
    def mock_potential(self):
        """Create a mock potential for testing."""
        return MockPotential(energy_multiplier=2.0, gradient_multiplier=2.0)

    @pytest.fixture
    def cache_lens(self):
        """Create a lens for the cache."""
        return SimpleLens(view(lambda state: state["cached_energy"], cls=dict))

    @pytest.fixture
    def patch_idx_view(self):
        """Create a patch index view."""
        system_keys = (SystemId(0), SystemId(1), SystemId(2))
        return view(
            lambda state: PotentialOut(
                total_energies=Table(
                    system_keys, Index.new(system_keys), _cls=SystemId
                ),
                gradients={"pos": Index(system_keys, jnp.array([0, 1]))},
                hessians={"pos": Index(system_keys, jnp.array([0, 1]))},
            )
        )

    @pytest.fixture
    def cached_potential(self, mock_potential, cache_lens, patch_idx_view):
        """Create a CachedPotential instance."""
        return CachedPotential(
            potential=mock_potential, cache=cache_lens, patch_idx_view=patch_idx_view
        )

    def test_basic_functionality(self, cached_potential, test_state):
        """Test basic CachedPotential functionality."""
        result = cached_potential(test_state)

        # Should return the same energy as the underlying potential
        expected_energy = jnp.array([2.0, 4.0, 6.0])  # 2 * [1, 2, 3]
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)

        # Should have composed patch that includes caching
        assert hasattr(result.patch, "__call__")

    def test_caching_behavior(self, cached_potential, test_state):
        """Test that CachedPotential properly sets up caching."""
        result = cached_potential(test_state)

        # The potential output should be available for caching
        potential_out = result.data
        assert isinstance(potential_out, PotentialOut)
        assert potential_out.total_energies.data.shape == (3,)
        assert "pos" in potential_out.gradients
        assert "pos" in potential_out.hessians

    def test_with_patch(self, cached_potential, test_state):
        """Test CachedPotential with patch argument."""
        patch = IdPatch()
        result = cached_potential(test_state, patch=patch)

        # Should still work with patches
        expected_energy = jnp.array([2.0, 4.0, 6.0])
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)

    def test_cached_value_access(self, cached_potential, test_state):
        """Test accessing the cached value."""
        # Get the cached value (should be the initial state)
        cached_value = cached_potential.cached_value(test_state)

        assert isinstance(cached_value, PotentialOut)
        assert jnp.array_equal(
            cached_value.total_energies.data, jnp.array([0.0, 0.0, 0.0])
        )

    def test_composed_patch_functionality(self, cached_potential, test_state):
        """Test that the composed patch works correctly."""
        result = cached_potential(test_state)

        # Apply the patch to see if it updates the cache
        accept = Table.arange(jnp.array([True, False, True]), label=SystemId)
        patched_result = result.patch(test_state, accept)

        # Should return a new state
        assert isinstance(patched_result, dict)

        # The new state should have updated cached values where accept is True
        # This tests that the IndexLensPatch is working
        assert "cached_energy" in patched_result

    def test_result_propagation(self, cached_potential, test_state):
        """Test that assertions and results are properly propagated."""

        class MockPotentialWithAssertions:
            """Mock potential that returns assertions."""

            def __call__(self, state, patch=None):
                potential_out = PotentialOut(
                    total_energies=Table.arange(
                        jnp.array([1.0, 2.0, 3.0]), label=SystemId
                    ),
                    gradients={"pos": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
                    hessians={"pos": jnp.array([[[1.0, 0.0], [0.0, 1.0]]])},
                )

                runtime_assert(
                    jnp.array(True),
                    "Test assertion",
                    fmt_args={"value": jnp.array(1.0)},
                    fix_args=jnp.array(1.0),
                )
                return WithPatch(potential_out, IdPatch())

        # Create cached potential with assertion-generating mock
        cache_lens = SimpleLens(view(lambda state: state["cached_energy"], cls=dict))
        patch_idx_view = view(
            lambda state: PotentialOut(
                total_energies=Table.arange(jnp.array([0, 1, 2]), label=SystemId),
                gradients={"pos": jnp.array([[0, 1], [0, 1]])},
                hessians={"pos": jnp.array([[[0, 1], [0, 1]]])},
            )
        )

        cached_pot = CachedPotential(
            potential=MockPotentialWithAssertions(),
            cache=cache_lens,
            patch_idx_view=patch_idx_view,
        )

        result = as_result_function(cached_pot)(test_state)

        # Should propagate assertions from the underlying potential
        assert len(result.assertions) == 1
        assert "Test assertion" in result.assertions[0].message

    def test_jax_compatibility(self, cached_potential, test_state):
        """Test that CachedPotential works with JAX transformations."""

        def compute_total_energy(state):
            result = cached_potential(state)
            return result.data.total_energies.data.sum()

        # Test JIT compilation
        jitted_compute = jax.jit(compute_total_energy)
        total_energy = jitted_compute(test_state)
        expected_total = jnp.array(12.0)  # sum of [2, 4, 6]

        assert jnp.allclose(total_energy, expected_total)

    def test_lens_and_view_integration(self, mock_potential):
        """Test integration between lens and view components."""
        # Create a more complex state structure
        complex_state = {
            "system": {
                "positions": jnp.array([[0.0, 0.0]]),
                "cache": {
                    "energy_cache": PotentialOut(
                        total_energies=Table.arange(
                            jnp.array([10.0, 20.0, 30.0]), label=SystemId
                        ),
                        gradients={"pos": jnp.array([[5.0, 6.0], [7.0, 8.0]])},
                        hessians={"pos": jnp.array([[[2.0, 0.0], [0.0, 2.0]]])},
                    )
                },
            }
        }

        # Create lens for nested cache
        nested_cache_lens = SimpleLens(
            view(lambda state: state["system"]["cache"]["energy_cache"], cls=dict)
        )

        # Create view for patch indices
        nested_patch_view = view(
            lambda state: PotentialOut(
                total_energies=Table.arange(jnp.array([0, 1, 2]), label=SystemId),
                gradients={"pos": jnp.array([[0, 1], [0, 1]])},
                hessians={"pos": jnp.array([[[0, 1], [0, 1]]])},
            )
        )

        cached_pot = CachedPotential(
            potential=mock_potential,
            cache=nested_cache_lens,
            patch_idx_view=nested_patch_view,
        )

        result = cached_pot(complex_state)

        # Should work with nested structures
        assert isinstance(result.data, PotentialOut)
        expected_energy = jnp.array([2.0, 4.0, 6.0])
        assert jnp.array_equal(result.data.total_energies.data, expected_energy)

        # Test cached value access with nested structure
        cached_value = cached_pot.cached_value(complex_state)
        assert jnp.array_equal(
            cached_value.total_energies.data, jnp.array([10.0, 20.0, 30.0])
        )

    def test_empty_gradients_and_hessians(self, cache_lens, patch_idx_view):
        """Test CachedPotential with empty gradients and hessians."""

        class SimpleEnergyPotential:
            """Potential that only returns energies."""

            def __call__(self, state, patch=None):
                potential_out = PotentialOut(
                    total_energies=Table.arange(
                        jnp.array([1.0, 2.0, 3.0]), label=SystemId
                    ),
                    gradients=(),
                    hessians=(),
                )
                return WithPatch(potential_out, IdPatch())

        # Update fixtures for empty gradients/hessians
        simple_cache_lens = SimpleLens(
            view(
                lambda state: PotentialOut(
                    total_energies=Table.arange(
                        jnp.array([0.0, 0.0, 0.0]), label=SystemId
                    ),
                    gradients=(),
                    hessians=(),
                )
            )
        )

        simple_patch_view = view(
            lambda state: PotentialOut(
                total_energies=Table.arange(jnp.array([0, 1, 2]), label=SystemId),
                gradients=(),
                hessians=(),
            )
        )

        cached_pot = CachedPotential(
            potential=SimpleEnergyPotential(),
            cache=simple_cache_lens,
            patch_idx_view=simple_patch_view,
        )

        test_state = {"cached_energy": None}
        result = cached_pot(test_state)

        # Should work with empty gradients and hessians
        assert isinstance(result.data, PotentialOut)
        assert jnp.array_equal(
            result.data.total_energies.data, jnp.array([1.0, 2.0, 3.0])
        )
        assert result.data.gradients == ()
        assert result.data.hessians == ()


class TestMappedPotential:
    """Tests for MappedPotential class."""

    def test_gradient_mapping(self):
        """Test that gradients are correctly mapped."""
        potential = MockPotential()

        mapped = MappedPotential(
            potential=potential,
            gradient_map=lambda g: g["pos"][0],
            hessian_map=lambda h: h,
        )

        result = mapped({"positions": jnp.zeros((2, 2))})

        assert jnp.array_equal(
            result.data.total_energies.data, jnp.array([1.0, 2.0, 3.0])
        )
        assert jnp.array_equal(result.data.gradients, jnp.array([1.0, 2.0]))
        assert "pos" in result.data.hessians

    def test_hessian_mapping(self):
        """Test that hessians are correctly mapped."""
        potential = MockPotential()

        mapped = MappedPotential(
            potential=potential,
            gradient_map=lambda g: g,
            hessian_map=lambda h: h["pos"][0, 0],
        )

        result = mapped({"positions": jnp.zeros((2, 2))})

        assert jnp.array_equal(result.data.hessians, jnp.array([1.0, 0.0]))

    def test_both_mappings(self):
        """Test mapping both gradients and hessians."""
        potential = MockPotential()

        mapped = MappedPotential(
            potential=potential,
            gradient_map=lambda g: g["pos"].sum(),
            hessian_map=lambda h: h["pos"].sum(),
        )

        result = mapped({"positions": jnp.zeros((2, 2))})

        assert jnp.array_equal(result.data.gradients, jnp.array(10.0))
        assert jnp.array_equal(result.data.hessians, jnp.array(2.0))

    def test_energy_preserved(self):
        """Test that energy is not affected by mapping."""
        potential = MockPotential(energy_multiplier=5.0)

        mapped = MappedPotential(
            potential=potential,
            gradient_map=lambda g: jnp.array(0.0),
            hessian_map=lambda h: jnp.array(0.0),
        )

        result = mapped({"positions": jnp.zeros((2, 2))})

        assert jnp.array_equal(
            result.data.total_energies.data, jnp.array([5.0, 10.0, 15.0])
        )

    def test_patch_passthrough(self):
        """Test that patch is passed through unchanged."""
        potential = MockPotential()

        mapped = MappedPotential(
            potential=potential,
            gradient_map=lambda g: g,
            hessian_map=lambda h: h,
        )

        result = mapped({"positions": jnp.zeros((2, 2))})

        assert isinstance(result.patch, IdPatch)

    def test_jit_compatible(self):
        """Test that MappedPotential works with JIT."""
        potential = MockPotential()

        mapped = MappedPotential(
            potential=potential,
            gradient_map=lambda g: g["pos"][0],
            hessian_map=lambda h: h["pos"][0],
        )

        jitted = jax.jit(mapped)
        result = jitted({"positions": jnp.zeros((2, 2))})

        assert jnp.array_equal(result.data.gradients, jnp.array([1.0, 2.0]))

    def test_composition_with_sum(self):
        """Test MappedPotential composed with SummedPotential."""
        p1 = MockPotential(energy_multiplier=1.0, gradient_multiplier=1.0)
        p2 = MockPotential(energy_multiplier=2.0, gradient_multiplier=2.0)

        summed = sum_potentials(p1, p2)
        mapped = MappedPotential(
            potential=summed,
            gradient_map=lambda g: g["pos"][0],
            hessian_map=lambda h: h,
        )

        result = mapped({"positions": jnp.zeros((2, 2))})

        assert jnp.array_equal(
            result.data.total_energies.data, jnp.array([3.0, 6.0, 9.0])
        )
        assert jnp.array_equal(result.data.gradients, jnp.array([3.0, 6.0]))
