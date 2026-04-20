# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from kups.core.utils.block_average import (
    BlockAverageResult,
    BlockTransformResult,
    block_average,
    block_transform,
    optimal_block_average,
)


class TestBlockAverage:
    """Test the block_average function."""

    def test_basic_1d(self):
        """Test basic block averaging on 1D data."""
        data = jnp.arange(100.0)
        result = block_average(data, n_blocks=10)

        assert isinstance(result, BlockAverageResult)
        assert result.n_blocks == 10
        npt.assert_allclose(result.mean, jnp.mean(data), rtol=1e-10)

    def test_basic_2d(self):
        """Test block averaging on 2D data along axis 0."""
        data = jnp.ones((100, 3)) * jnp.arange(3)
        result = block_average(data, n_blocks=10, axis=0)

        assert result.mean.shape == (3,)
        assert result.sem.shape == (3,)
        npt.assert_allclose(result.mean, jnp.arange(3), rtol=1e-10)
        npt.assert_allclose(result.sem, jnp.zeros(3), atol=1e-10)

    def test_axis_parameter(self):
        """Test block averaging along different axes."""
        data = jnp.ones((5, 100, 3))
        result = block_average(data, n_blocks=10, axis=1)

        assert result.mean.shape == (5, 3)
        assert result.sem.shape == (5, 3)

    def test_incomplete_blocks_discarded(self):
        """Test that incomplete blocks are discarded."""
        data = jnp.arange(105.0)
        result = block_average(data, n_blocks=10)

        expected_mean = jnp.mean(data[:100])
        npt.assert_allclose(result.mean, expected_mean, rtol=1e-10)

    def test_sem_calculation(self):
        """Test SEM calculation with known variance."""
        block_means = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        block_size = 10
        data = jnp.repeat(block_means, block_size)

        result = block_average(data, n_blocks=5)

        expected_mean = jnp.mean(block_means)
        expected_sem = jnp.std(block_means, ddof=1) / jnp.sqrt(5)

        npt.assert_allclose(result.mean, expected_mean, rtol=1e-10)
        npt.assert_allclose(result.sem, expected_sem, rtol=1e-10)

    def test_single_block(self):
        """Test with single block (edge case)."""
        data = jnp.arange(100.0)
        result = block_average(data, n_blocks=1)

        assert result.n_blocks == 1
        npt.assert_allclose(result.mean, jnp.mean(data), rtol=1e-10)
        assert jnp.isnan(result.sem)

    def test_n_blocks_equals_n_samples(self):
        """Test when n_blocks equals n_samples (block_size=1)."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = block_average(data, n_blocks=5)

        npt.assert_allclose(result.mean, jnp.mean(data), rtol=1e-10)
        expected_sem = jnp.std(data, ddof=1) / jnp.sqrt(5)
        npt.assert_allclose(result.sem, expected_sem, rtol=1e-10)

    def test_multidimensional_sem(self):
        """Test SEM computed element-wise for multidimensional data."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (1000, 3)) * jnp.array([1.0, 2.0, 3.0])

        result = block_average(data, n_blocks=10, axis=0)

        assert result.sem.shape == (3,)
        assert result.sem[1] > result.sem[0]
        assert result.sem[2] > result.sem[1]


class TestBlockTransform:
    """Test the block_transform function."""

    def test_basic_output_structure(self):
        """Test basic output structure of block_transform."""
        data = jnp.arange(1000.0)
        result = block_transform(data)

        assert isinstance(result, BlockTransformResult)
        assert result.block_sizes.ndim == 1
        assert result.sems.ndim == 1
        assert len(result.block_sizes) == len(result.sems)

    def test_custom_n_blocks_range(self):
        """Test block_transform with custom n_blocks range."""
        data = jnp.arange(100.0)
        n_blocks_range = [5, 10, 20]
        result = block_transform(data, n_blocks_range=n_blocks_range)

        expected_block_sizes = [20, 10, 5]
        npt.assert_array_equal(result.block_sizes, jnp.array(expected_block_sizes))

    def test_mean_consistency(self):
        """Test that mean is consistent across all block sizes."""
        data = jnp.arange(100.0)
        result = block_transform(data, n_blocks_range=[5, 10, 20])

        expected_mean = jnp.mean(data)
        npt.assert_allclose(result.mean, expected_mean, rtol=1e-10)

    def test_sem_increases_with_block_size_for_correlated_data(self):
        """Test that SEM generally increases with block size for correlated data."""
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (10000,))
        data = jnp.cumsum(noise)

        result = block_transform(data, min_blocks=4)

        sems = np.asarray(result.sems)
        block_sizes = np.asarray(result.block_sizes)

        order = np.argsort(block_sizes)
        sorted_sems = sems[order]

        assert sorted_sems[0] < sorted_sems[-1]

    def test_auto_range_generation(self):
        """Test automatic n_blocks range generation."""
        data = jnp.arange(1024.0)
        result = block_transform(data, min_blocks=4)

        block_sizes = np.asarray(result.block_sizes)
        assert np.all(np.diff(block_sizes) > 0)

    def test_min_blocks_respected(self):
        """Test that min_blocks parameter is respected."""
        data = jnp.arange(100.0)
        result = block_transform(data, min_blocks=10)

        max_block_size = 100 // 10
        assert np.max(np.asarray(result.block_sizes)) <= max_block_size

    def test_multidimensional_data(self):
        """Test block_transform with multidimensional data."""
        data = jnp.ones((1000, 3)) * jnp.arange(3)
        result = block_transform(data, n_blocks_range=[5, 10, 20])

        assert result.mean.shape == (3,)
        assert result.sems.shape == (3, 3)


class TestOptimalBlockAverage:
    """Test the optimal_block_average function."""

    def test_basic_output_structure(self):
        """Test basic output structure of optimal_block_average."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (1000,))
        result = optimal_block_average(data)

        assert isinstance(result, BlockAverageResult)
        assert result.n_blocks > 0

    def test_uncorrelated_data(self):
        """Test with uncorrelated data - should find reasonable block size."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (10000,))
        result = optimal_block_average(data)

        expected_mean = jnp.mean(data)
        npt.assert_allclose(result.mean, expected_mean, rtol=1e-5)

    def test_correlated_data_larger_block_size(self):
        """Test that correlated data results in larger optimal block size."""
        key = jax.random.PRNGKey(42)

        uncorr_data = jax.random.normal(key, (10000,))
        corr_data = jnp.cumsum(jax.random.normal(key, (10000,)))

        result_uncorr = optimal_block_average(uncorr_data)
        result_corr = optimal_block_average(corr_data)

        assert result_corr.n_blocks >= 1
        assert result_uncorr.n_blocks >= 1

    def test_rtol_parameter(self):
        """Test that rtol parameter affects block size selection."""
        key = jax.random.PRNGKey(42)
        data = jnp.cumsum(jax.random.normal(key, (10000,)))

        result_strict = optimal_block_average(data, rtol=0.01)
        result_loose = optimal_block_average(data, rtol=0.20)

        assert result_strict.n_blocks >= 1
        assert result_loose.n_blocks >= 1

    def test_min_blocks_parameter(self):
        """Test that min_blocks parameter is respected."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (1000,))

        result = optimal_block_average(data, min_blocks=10)

        assert result.n_blocks >= 10

    def test_multidimensional_data(self):
        """Test optimal_block_average with multidimensional data."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (1000, 3))
        result = optimal_block_average(data)

        assert result.mean.shape == (3,)
        assert result.sem.shape == (3,)

    def test_axis_parameter(self):
        """Test optimal_block_average along different axes."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (5, 1000, 3))
        result = optimal_block_average(data, axis=1)

        assert result.mean.shape == (5, 3)
        assert result.sem.shape == (5, 3)

    def test_no_plateau_uses_largest_block(self):
        """Test that when no plateau is found, largest block size is used."""
        data = jnp.cumsum(jnp.ones(100))
        result = optimal_block_average(data, min_blocks=4, rtol=0.001)

        assert result.n_blocks >= 4


class TestBlockAverageEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.parametrize(
        "data,n_blocks,check_mean,check_sem,mean_atol,sem_atol",
        [
            pytest.param(
                jnp.ones(100) * 5.0,
                10,
                5.0,
                0.0,
                1e-10,
                1e-10,
                id="constant_data",
            ),
            pytest.param(
                jnp.arange(-50.0, 50.0),
                10,
                None,
                None,
                1e-10,
                None,
                id="negative_values",
            ),
            pytest.param(
                jnp.arange(100.0, dtype=jnp.float32),
                10,
                None,
                None,
                1e-5,
                None,
                id="float32_precision",
            ),
        ],
    )
    def test_edge_cases(
        self, data, n_blocks, check_mean, check_sem, mean_atol, sem_atol
    ):
        """Test edge cases: constant data, negative values, float32 precision."""
        result = block_average(data, n_blocks=n_blocks)
        expected_mean = check_mean if check_mean is not None else float(jnp.mean(data))
        npt.assert_allclose(result.mean, expected_mean, atol=mean_atol)
        if check_sem is not None and sem_atol is not None:
            npt.assert_allclose(result.sem, check_sem, atol=sem_atol)

    def test_two_blocks_and_large_array(self):
        """Test minimum meaningful blocks and larger array for coverage."""
        # Two blocks
        data = jnp.array([1.0, 1.0, 2.0, 2.0])
        result = block_average(data, n_blocks=2)
        assert result.n_blocks == 2
        npt.assert_allclose(result.mean, 1.5, rtol=1e-10)
        expected_sem = jnp.std(jnp.array([1.0, 2.0]), ddof=1) / jnp.sqrt(2)
        npt.assert_allclose(result.sem, expected_sem, rtol=1e-10)

        # Larger array (1K instead of 100K)
        key = jax.random.PRNGKey(42)
        large_data = jax.random.normal(key, (1000,))
        large_result = block_average(large_data, n_blocks=100)
        assert large_result.n_blocks == 100

        # 3D data
        data_3d = jnp.ones((100, 5, 3))
        result_3d = block_average(data_3d, n_blocks=10, axis=0)
        assert result_3d.mean.shape == (5, 3)
        assert result_3d.sem.shape == (5, 3)


class TestBlockAverageStatisticalProperties:
    """Test statistical properties of block averaging."""

    def test_sem_consistent_for_uncorrelated_data(self):
        """Test that SEM is consistent for uncorrelated data regardless of block size."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (10000,))

        result_10 = block_average(data, n_blocks=10)
        result_100 = block_average(data, n_blocks=100)

        ratio = float(result_10.sem / result_100.sem)
        npt.assert_allclose(ratio, 1.0, rtol=0.5)

    def test_mean_unbiased(self):
        """Test that mean estimate is unbiased regardless of block size."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (1000,)) + 5.0

        results = [block_average(data, n_blocks=n) for n in [5, 10, 20, 50]]
        means = [float(r.mean) for r in results]

        npt.assert_allclose(means, means[0], rtol=1e-10)

    def test_block_transform_monotonicity(self):
        """Test that block sizes in transform are monotonic."""
        data = jnp.arange(1000.0)
        result = block_transform(data, min_blocks=4)

        block_sizes = np.asarray(result.block_sizes)
        assert np.all(np.diff(block_sizes) > 0)
