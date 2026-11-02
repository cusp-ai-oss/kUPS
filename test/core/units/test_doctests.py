# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Run the doctests of every kups.core.units module."""

import doctest
import importlib
import inspect
import pkgutil

import pytest

from kups.core import units

MODULES = [
    name
    for name in sorted(
        m.name
        for m in pkgutil.iter_modules(units.__path__, prefix=f"{units.__name__}.")
    )
    + [units.__name__]
    if ">>>" in inspect.getsource(importlib.import_module(name))
]


@pytest.mark.parametrize("name", MODULES)
def test_doctests(name: str):
    result = doctest.testmod(importlib.import_module(name))
    assert result.attempted > 0, f"no doctest examples collected in {name}"
    assert result.failed == 0, f"{result.failed} doctest(s) failed in {name}"
