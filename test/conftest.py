# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import gc
from collections.abc import Iterator

import jax
import pytest

from kups.potential.common import rows as rows_module


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


@pytest.fixture(params=[True, False], ids=["kernel", "xla"])
def row_kernel(request, monkeypatch) -> Iterator[list[int] | None]:
    """Float32 row pair terms in the Pallas kernel or in XLA; yields the kernel's
    traces."""
    traced = []
    pallas = rows_module._rows_pallas
    monkeypatch.setattr(rows_module, "_use_kernel", lambda: request.param)
    monkeypatch.setattr(
        rows_module, "_rows_pallas", lambda *a: traced.append(1) or pallas(*a)
    )
    jax.clear_caches()
    yield traced if request.param else None
    jax.clear_caches()
