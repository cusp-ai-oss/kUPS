# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import pytest

from ._builders import HandlerFactory, MockContext


@pytest.fixture
def empty_context() -> MockContext:
    return MockContext((), ())


@pytest.fixture
def handler_factory() -> HandlerFactory:
    return HandlerFactory()
