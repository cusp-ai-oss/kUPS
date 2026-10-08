# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Jaxpr interpreter that threads a custom context through traced computations.

A [Dispatcher][kups.core.interpreter.interpreter.Dispatcher] maps primitives to
handlers; an [Interpreter][kups.core.interpreter.interpreter.Interpreter] walks a
jaxpr, re-binding unhandled primitives and routing handled ones (including
control flow) through their handlers. The runtime assertion system in
[kups.core.assertion][kups.core.assertion] is built on top of it.
"""
