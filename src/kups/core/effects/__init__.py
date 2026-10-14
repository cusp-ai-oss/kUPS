# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Effect handlers for JAX.

Code performs an effect by binding a custom JAX primitive. An
[Interpreter][kups.core.effects.interpreter.Interpreter] re-traces a function
and passes each primitive matched by a
[Dispatcher][kups.core.effects.interpreter.Dispatcher] to its handler. Handlers
read and update a context that is carried through the whole program, including
`jit`, `scan`, `while_loop`, `cond` and `shard_map`. All other primitives are
bound unchanged.

[kups.core.assertion][kups.core.assertion] is built on this: `runtime_assert`
performs an effect and `with_runtime_assertions` handles it.
"""

from kups.core.effects.environment import InterpreterEnvironment
from kups.core.effects.handlers import (
    ScanSemantics,
    Uninitialized,
    default_checkpoint_handler,
    default_cond_handler,
    default_jit_handler,
    default_primitive_handler,
    default_scan_handler,
    default_shard_map_handler,
    default_while_handler,
)
from kups.core.effects.interpreter import (
    Dispatcher,
    Handler,
    HandlerResult,
    Interpreter,
    InterpreterContext,
    InterpreterPolicy,
    JaxprInterpreterResult,
    MatchingRule,
    TracerValue,
    contains_subjaxprs,
    reinterpret,
)

__all__ = [
    "Dispatcher",
    "Handler",
    "HandlerResult",
    "Interpreter",
    "InterpreterContext",
    "InterpreterEnvironment",
    "InterpreterPolicy",
    "JaxprInterpreterResult",
    "MatchingRule",
    "ScanSemantics",
    "TracerValue",
    "Uninitialized",
    "contains_subjaxprs",
    "default_checkpoint_handler",
    "default_cond_handler",
    "default_jit_handler",
    "default_primitive_handler",
    "default_scan_handler",
    "default_shard_map_handler",
    "default_while_handler",
    "reinterpret",
]
