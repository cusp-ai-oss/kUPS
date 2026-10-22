# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import jax
import pytest

from kups.potential.common import rows as rows_module


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
