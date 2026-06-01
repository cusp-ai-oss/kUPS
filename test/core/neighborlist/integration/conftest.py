# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def clear_cache():
    """Module-scoped cache clearing so JAX compilation caches persist across
    the test classes within an integration file (the algorithm suites recompile
    heavily; per-class clearing would thrash the compile cache)."""
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()
