# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from kups.core.data import Index
from kups.core.typing import SystemId
from kups.core.utils.random import segment_chi_squared


def test_segment_chi_squared_moments():
    counts = np.array([1, 2, 5, 40])
    dof = jnp.array([3, 1, 7, 116])  # any value up to 3 * count
    rows = np.random.default_rng(0).permutation(np.repeat(np.arange(4), counts))
    rows = np.append(rows, 4)  # out-of-range row: ignored
    system = Index(tuple(SystemId(i) for i in range(4)), jnp.asarray(rows))
    keys = jax.random.split(jax.random.key(0), 20_000)
    samples = np.asarray(jax.vmap(lambda k: segment_chi_squared(k, system, dof))(keys))
    np.testing.assert_allclose(samples.mean(0), dof, rtol=0.03)
    np.testing.assert_allclose(samples.var(0), 2 * dof, rtol=0.06)


def test_segment_chi_squared_zero_dof():
    system = Index((SystemId(0), SystemId(1)), jnp.array([0, 1, 1]))
    out = segment_chi_squared(jax.random.key(1), system, jnp.array([0, -1]))
    np.testing.assert_array_equal(out, [0.0, 0.0])
