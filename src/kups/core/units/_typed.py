# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared IntVar type variables, so each symbol means the same exponent everywhere.

At runtime they are plain ``TypeVar``s, which is all ``Generic[...]`` needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntVar
else:
    from typing import TypeVar as IntVar

# E_*: receiver exponents; F_*/G_*: other operands'; K: an integer power.
E_M = IntVar("E_M")
E_KG = IntVar("E_KG")
E_S = IntVar("E_S")
E_A = IntVar("E_A")
E_K = IntVar("E_K")
F_M = IntVar("F_M")
F_KG = IntVar("F_KG")
F_S = IntVar("F_S")
F_A = IntVar("F_A")
F_K = IntVar("F_K")
G_M = IntVar("G_M")
G_KG = IntVar("G_KG")
G_S = IntVar("G_S")
G_A = IntVar("G_A")
G_K = IntVar("G_K")
K = IntVar("K")
