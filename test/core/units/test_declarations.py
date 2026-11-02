# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The class declarations the checker sees must match the runtime surface.

``Quantity`` and ``Unit`` are single classes; names with no
meaning are defined under ``if not TYPE_CHECKING`` so the checker flags them
while the runtime raises with guidance, and those hidden blocks are pinned here.
``Dimensionless`` stays an alias for the checker with a runtime subclass twin.
"""

import ast
import pathlib

from kups.core import units

PKG = pathlib.Path(units.__file__).parent


def names_in(cls: ast.ClassDef) -> set[str]:
    """Return the methods and annotated fields bound in a class body."""
    out: set[str] = set()
    for n in cls.body:
        if isinstance(n, ast.FunctionDef):
            out.add(n.name)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def test_no_pyi_stubs():
    # a stub would silently shadow its module
    assert not list(PKG.glob("*.pyi")), (
        f"{[p.name for p in PKG.glob('*.pyi')]} shadow the sources they sit next to"
    )


def test_dimensionless_twins_match():
    # the checker sees the Unit[0, ...] alias; the runtime subclass may only
    # override reflected operators, whose types the self-typed overloads carry
    tree = ast.parse((PKG / "_unit.py").read_text())
    runtime = next(
        cls
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)
        for cls in node.orelse
        if isinstance(cls, ast.ClassDef) and cls.name == "Dimensionless"
    )
    extra = names_in(runtime) - {"__rmul__", "__rtruediv__"}
    assert not extra, f"Dimensionless overrides {sorted(extra)}, which the alias hides"
