# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Random sampling utilities for PyTree structures."""

from collections.abc import Callable

import jax
from jax import Array

type Key = Array
type Shape = tuple[int, ...]
type DType = jax.numpy.dtype
type Sampler = Callable[[Key, Shape, DType], Array]


def sample_like[PyTree](sampler: Sampler, key: Key, tree: PyTree) -> PyTree:
    """Generate random arrays matching the structure and shapes of a PyTree.

    Args:
        sampler: Function ``(key, shape, dtype) -> Array`` that produces samples.
        key: JAX PRNG key (split internally for each leaf).
        tree: Template PyTree; each leaf's shape and dtype are matched.

    Returns:
        PyTree with the same structure as ``tree``, filled with random samples.

    Example:
        ```python
        import jax.random as jr
        template = {"a": jnp.zeros((3,)), "b": jnp.zeros((2, 4))}
        result = sample_like(jr.normal, jr.PRNGKey(0), template)
        # result["a"].shape == (3,), result["b"].shape == (2, 4)
        ```
    """
    leaves, treedef = jax.tree.flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten(
        [sampler(key, ref.shape, ref.dtype) for key, ref in zip(keys, leaves)]
    )
