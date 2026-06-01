# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def clear_cache():
    """Clear JAX caches once per unit-test module.

    Module scope (rather than per-class) lets the small primitive kernels these
    largely-eager unit tests dispatch stay cached across classes within a file,
    while still bounding host-memory growth between files."""
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()
