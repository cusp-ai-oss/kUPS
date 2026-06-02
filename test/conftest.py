# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def clear_cache():
    """Clear JAX caches around every test module.

    Module scope lets JIT-compiled functions be reused across the classes
    within a file, while bounding host-memory growth between files. A nearer
    ``conftest.py`` may override this fixture to change the scope.
    """
    jax.clear_caches()
    gc.collect()
    yield
    jax.clear_caches()
    gc.collect()
