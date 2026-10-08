# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import pytest

from kups.core.interpreter.interpreter import (
    Dispatcher,
    Interpreter,
    InterpreterPolicy,
    contains_subjaxprs,
)

from ._builders import HandlerFactory, MockContext


@pytest.fixture
def empty_context() -> MockContext:
    return MockContext((), ())


@pytest.fixture
def base_dispatcher() -> Dispatcher[MockContext]:
    return Dispatcher({}).register_custom_matching_rule(contains_subjaxprs)


@pytest.fixture
def base_interpreter(
    base_dispatcher: Dispatcher[MockContext],
) -> Interpreter[MockContext]:
    return Interpreter(dispatcher=base_dispatcher, policy=InterpreterPolicy.WARN)


@pytest.fixture
def handler_factory() -> HandlerFactory:
    return HandlerFactory()
