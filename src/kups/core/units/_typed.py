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

# _E_*: receiver exponents; _F_*/_G_*: other operands'; _K: an integer power.
_E_M = IntVar("_E_M")
_E_KG = IntVar("_E_KG")
_E_S = IntVar("_E_S")
_E_A = IntVar("_E_A")
_E_K = IntVar("_E_K")
_F_M = IntVar("_F_M")
_F_KG = IntVar("_F_KG")
_F_S = IntVar("_F_S")
_F_A = IntVar("_F_A")
_F_K = IntVar("_F_K")
_G_M = IntVar("_G_M")
_G_KG = IntVar("_G_KG")
_G_S = IntVar("_G_S")
_G_A = IntVar("_G_A")
_G_K = IntVar("_G_K")
_K = IntVar("_K")
