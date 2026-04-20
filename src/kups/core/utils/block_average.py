# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Block averaging for error estimation in correlated time series.

This module implements block averaging to estimate the standard error of the mean
for time series data with autocorrelation. By dividing the time series into
independent blocks, the variance of block means provides an unbiased estimate
of the standard error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from kups.core.utils.jax import jit, no_jax_tracing


class BlockAverageResult(NamedTuple):
    """Result of block averaging analysis.

    Attributes:
        mean: Mean value over the entire time series. Shape matches input with
            time axis removed.
        sem: Standard error of the mean estimated from block variance.
            Shape matches `mean`.
        n_blocks: Number of blocks used in the analysis.
    """

    mean: Array
    sem: Array
    n_blocks: int


def compute_block_means(data: Array, n_blocks: int, axis: int = 0) -> Array:
    """Compute per-block means without aggregating across blocks.

    This is the core building block for block averaging. Use this when you need
    per-block values for computing derived quantities before averaging.

    Note:
        This function is not JIT-compiled directly. When used inside a JIT-compiled
        function, the compilation will include this code. When used standalone,
        consider wrapping the calling code with JIT for performance.

    Args:
        data: Input array with time series data along `axis`.
        n_blocks: Number of blocks to divide the data into. Samples that don't
            fit into complete blocks are discarded from the end.
        axis: Axis along which to compute block means. Defaults to 0.

    Returns:
        Array of per-block means. The `axis` dimension becomes `n_blocks`.
        E.g., input shape (1000, 3) with n_blocks=10, axis=0 → output (10, 3).

    Example:
        ```python
        # Compute block means for multiple quantities
        U_blocks = compute_block_means(energy, n_blocks=10)      # (10,)
        N_blocks = compute_block_means(counts, n_blocks=10)      # (10, 3)
        UN_blocks = compute_block_means(energy[:, None] * counts, n_blocks=10)
        ```
    """
    n_samples = data.shape[axis]
    block_size = n_samples // n_blocks
    n_used = block_size * n_blocks

    # Slice to use only complete blocks
    slices: list[slice | int] = [slice(None)] * data.ndim
    slices[axis] = slice(0, n_used)
    data = data[tuple(slices)]

    # Reshape to (n_blocks, block_size, ...)
    new_shape = list(data.shape)
    new_shape[axis : axis + 1] = [n_blocks, block_size]
    data = data.reshape(new_shape)

    # Block means along the block_size axis (axis + 1)
    return jnp.mean(data, axis=axis + 1)


def block_average_from_blocks(block_values: Array, axis: int = 0) -> BlockAverageResult:
    """Compute block average statistics from pre-computed block values.

    Use this when you have already computed per-block values (e.g., derived
    quantities like heat of adsorption computed per block).

    Args:
        block_values: Array where `axis` indexes blocks.
        axis: Axis along which blocks are indexed. Defaults to 0.

    Returns:
        BlockAverageResult with mean and SEM computed from block values.

    Example:
        ```python
        # Compute derived quantity per block, then get statistics
        hoa_blocks = compute_hoa_per_block(...)  # shape (n_blocks, n_species)
        result = block_average_from_blocks(hoa_blocks)
        print(f"HoA: {result.mean} ± {result.sem}")
        ```
    """
    n_blocks = block_values.shape[axis]
    mean = jnp.mean(block_values, axis=axis)
    sem = jnp.std(block_values, axis=axis, ddof=1) / jnp.sqrt(n_blocks)
    return BlockAverageResult(mean, sem, n_blocks)


@jit(static_argnames=("n_blocks", "axis"))
def block_average(
    data: Array,
    n_blocks: int,
    axis: int = 0,
) -> BlockAverageResult:
    """Compute block average statistics for error estimation.

    Divides the time series into blocks and uses the variance of block means
    to estimate the standard error of the overall mean. This accounts for
    autocorrelation in the data.

    The standard error is computed as:

    $$
    \\text{SEM} = \\frac{\\sigma_{\\text{blocks}}}{\\sqrt{n_{\\text{blocks}}}}
    $$

    where $\\sigma_{\\text{blocks}}$ is the standard deviation of block means.

    Args:
        data: Input array with time series data along `axis`.
        n_blocks: Number of blocks to divide the data into. Samples that don't
            fit into complete blocks are discarded from the end.
        axis: Axis along which to perform block averaging. Defaults to 0.

    Returns:
        BlockAverageResult containing mean, standard error, and number of blocks.

    Example:
        ```python
        # Time series of 1000 samples
        data = jnp.array([...])  # shape (1000,)
        result = block_average(data, n_blocks=10)
        print(f"Mean: {result.mean:.3f} +/- {result.sem:.3f}")

        # Multidimensional data: 1000 timesteps, 3 components
        data = jnp.array([...])  # shape (1000, 3)
        result = block_average(data, n_blocks=10, axis=0)
        # result.mean has shape (3,), result.sem has shape (3,)
        ```

    Note:
        For reliable error estimates, choose `n_blocks` such that each block
        is longer than the autocorrelation time of the data.
    """
    block_means = compute_block_means(data, n_blocks, axis)
    return block_average_from_blocks(block_means, axis)


class BlockTransformResult(NamedTuple):
    """Result of block transformation analysis.

    Attributes:
        block_sizes: Array of block sizes tested.
        sems: Standard error estimates for each block size. Shape is
            `(n_sizes,) + data_shape` where `data_shape` is the input shape
            with the time axis removed.
        mean: Overall mean of the data.
    """

    block_sizes: Array
    sems: Array
    mean: Array


def block_transform(
    data: Array,
    n_blocks_range: Sequence[int] | None = None,
    axis: int = 0,
    min_blocks: int = 4,
) -> BlockTransformResult:
    """Compute SEM across a range of block sizes for plateau detection.

    Performs block averaging for multiple block counts to enable identification
    of the plateau where SEM stabilizes. The plateau value corresponds to the
    true standard error when blocks are longer than the autocorrelation time.

    Args:
        data: Input array with time series data along `axis`.
        n_blocks_range: Sequence of block counts to test. If None, automatically
            generates a geometric series from `min_blocks` to `n_samples // 2`.
        axis: Axis along which to perform block averaging. Defaults to 0.
        min_blocks: Minimum number of blocks when auto-generating range.
            Defaults to 4.

    Returns:
        BlockTransformResult containing block sizes, corresponding SEMs, and
        the overall mean.

    Example:
        ```python
        result = block_transform(data)

        # Plot to find plateau
        import matplotlib.pyplot as plt
        plt.semilogx(result.block_sizes, result.sems)
        plt.xlabel("Block size")
        plt.ylabel("SEM")
        ```
    """
    n_samples = data.shape[axis]

    if n_blocks_range is None:
        # Generate geometric series: many blocks (small size) to few blocks (large size)
        max_blocks = n_samples // 2
        n_steps = int(np.log2(max_blocks / min_blocks)) + 1
        n_blocks_range = [
            int(max_blocks / (2**i))
            for i in range(n_steps)
            if max_blocks // (2**i) >= min_blocks
        ]

    block_sizes = []
    sems = []
    mean = None

    for n_blocks in n_blocks_range:
        result = block_average(data, n_blocks=n_blocks, axis=axis)
        block_sizes.append(n_samples // n_blocks)
        sems.append(result.sem)
        if mean is None:
            mean = result.mean

    return BlockTransformResult(
        block_sizes=jnp.array(block_sizes),
        sems=jnp.stack(sems),
        mean=mean,  # type: ignore[arg-type]
    )


@no_jax_tracing
def optimal_block_average(
    data: Array,
    axis: int = 0,
    min_blocks: int = 4,
    rtol: float = 0.05,
) -> BlockAverageResult:
    """Automatically select optimal block size and compute block average.

    Performs block transformation analysis and detects the plateau where SEM
    stabilizes. Returns the block average result at the optimal block size.

    The plateau is detected by finding where the relative change in SEM falls
    below `rtol`. For multidimensional data, the maximum relative change across
    all elements is used.

    Args:
        data: Input array with time series data along `axis`.
        axis: Axis along which to perform block averaging. Defaults to 0.
        min_blocks: Minimum number of blocks to consider. Defaults to 4.
        rtol: Relative tolerance for plateau detection. The plateau is reached
            when the relative change in SEM is below this threshold. Defaults
            to 0.05 (5%).

    Returns:
        BlockAverageResult at the optimal block size.

    Example:
        ```python
        result = optimal_block_average(data)
        print(f"Mean: {result.mean} +/- {result.sem}")
        ```
    """
    transform = block_transform(data, axis=axis, min_blocks=min_blocks)

    # SEMs are ordered from small to large block sizes
    sems = np.asarray(transform.sems)
    block_sizes = np.asarray(transform.block_sizes)

    # Compute relative change in SEM between consecutive block sizes
    # sems[i] corresponds to block_sizes[i], ordered small -> large block size
    rel_change = np.abs(np.diff(sems, axis=0)) / (np.abs(sems[:-1]) + 1e-10)

    # For multidimensional data, take max relative change across elements
    if rel_change.ndim > 1:
        rel_change = np.max(rel_change, axis=tuple(range(1, rel_change.ndim)))

    # Find first index where relative change drops below tolerance
    plateau_indices = np.where(rel_change < rtol)[0]
    if len(plateau_indices) > 0:
        # Use the block size after the first plateau point
        optimal_idx = plateau_indices[0] + 1
    else:
        # No clear plateau found, use largest block size (most conservative)
        optimal_idx = len(block_sizes) - 1

    optimal_block_size = int(block_sizes[optimal_idx])
    n_samples = data.shape[axis]
    optimal_n_blocks = n_samples // optimal_block_size

    return block_average(data, n_blocks=optimal_n_blocks, axis=axis)
