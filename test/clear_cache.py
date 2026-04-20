# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="class")
def clear_cache():
    # Ensure that before and after every module we clear JAX's caches
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()
