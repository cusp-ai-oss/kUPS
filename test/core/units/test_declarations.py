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
from typing import TypeAliasType

from kups.core import units

PKG = pathlib.Path(units.__file__).parent

# dunders kups.core.units installs on Quantity
DUNDERS = (
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "truediv",
    "rtruediv",
    "matmul",
    "rmatmul",
    "mod",
    "rmod",
    "floordiv",
    "rfloordiv",
    "pow",
    "neg",
    "abs",
    "getitem",
    "iter",
    "len",
    "float",
    "int",
    "format",
    "lt",
    "le",
    "gt",
    "ge",
    "eq",
    "ne",
)

# defined only to raise with guidance; hidden from the checker so it flags them
QUANTITY_HIDDEN = ("__rpow__",)


def names_in(cls: ast.ClassDef) -> set[str]:
    """Return the methods and annotated fields bound in a class body."""
    out: set[str] = set()
    for n in cls.body:
        if isinstance(n, ast.FunctionDef):
            out.add(n.name)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def split_body(module: str, name: str) -> tuple[set[str], set[str]]:
    """A unified class's (checker-visible, hidden) names.

    Hidden names are the ones defined under ``if not TYPE_CHECKING`` in the
    class body; everything else the checker sees.
    """
    tree = ast.parse((PKG / module).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    visible: set[str] = set()
    hidden: set[str] = set()
    for n in cls.body:
        if isinstance(n, ast.If) and "TYPE_CHECKING" in ast.dump(n.test):
            for inner in n.body:
                if isinstance(inner, ast.FunctionDef):
                    hidden.add(inner.name)
        elif isinstance(n, ast.FunctionDef):
            visible.add(n.name)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            visible.add(n.target.id)
        elif isinstance(n, ast.Assign):
            visible.update(t.id for t in n.targets if isinstance(t, ast.Name))
    return visible, hidden


def aliases() -> dict[tuple[int, ...], str]:
    """Map dimension exponents to alias names, read from _core.py's type aliases."""
    out: dict[tuple[int, ...], str] = {}
    for node in ast.walk(ast.parse((PKG / "_core.py").read_text())):
        if not isinstance(node, ast.TypeAlias) or not isinstance(node.name, ast.Name):
            continue
        v = node.value
        if (
            isinstance(v, ast.Subscript)
            and isinstance(v.slice, ast.Tuple)
            and ast.unparse(v.value) == "Quantity"
        ):
            try:
                out[tuple(ast.literal_eval(e) for e in v.slice.elts)] = node.name.id
            except ValueError:  # parameterised by something other than ints
                continue
    return out


def test_no_pyi_stubs():
    # a stub would silently shadow its module
    assert not list(PKG.glob("*.pyi")), (
        f"{[p.name for p in PKG.glob('*.pyi')]} shadow the sources they sit next to"
    )


def test_quantity_surface_declared():
    visible, hidden = split_body("_core.py", "Quantity")
    surface = {n for n in dir(units.Quantity) if not n.startswith("_")}
    surface |= {f"__{n}__" for n in DUNDERS}
    missing = surface - visible
    assert not missing, f"the Quantity class is missing {sorted(missing)}"
    assert hidden == set(QUANTITY_HIDDEN), f"unexpected hidden names: {sorted(hidden)}"


def test_unit_surface_declared():
    visible, hidden = split_body("_unit.py", "Unit")
    surface = {n for n in dir(units.Unit) if not n.startswith("_")}
    surface |= {
        "__mul__",
        "__truediv__",
        "__pow__",
        "__rmul__",
        "__rtruediv__",
        "__repr__",
    }
    missing = surface - visible
    assert not missing, f"the Unit class is missing {sorted(missing)}"
    assert not hidden, f"unexpected hidden names: {sorted(hidden)}"


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


def test_dimension_aliases_exported():
    named = set(aliases().values())
    assert len(named) == 27, f"{len(named)} dimensions found"
    for name in sorted(named):
        for spelling in (name, f"{name}Unit"):
            assert spelling in units.__all__, (
                f"kups.core.units does not export {spelling}"
            )
            assert isinstance(getattr(units, spelling), TypeAliasType)
    exported = {
        n for n in units.__all__ if isinstance(getattr(units, n), TypeAliasType)
    }
    extra = exported - named - {f"{n}Unit" for n in named}
    assert not extra, f"kups.core.units exports {sorted(extra)} as well"


def test_annotations_match_dimensions():
    # units and constants annotate their own members, so the checker cannot catch
    # a wrong alias; compare against the dimension read from the value instead
    table = aliases()
    for mod, suffix in ((units.catalog, "Unit"), (units.constants, "")):
        source = (PKG / f"{mod.__name__.split('.')[-1]}.py").read_text()
        seen = 0
        tree = ast.parse(source)
        # constants declares under TYPE_CHECKING, catalog at module level
        top = list(tree.body)
        for node in tree.body:
            if isinstance(node, ast.If):
                top += node.body + node.orelse
        for node in top:
            if not (
                isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            ):
                continue
            value = getattr(mod, node.target.id)
            if not hasattr(value, "dim"):  # N_A and alpha are plain floats
                continue
            dim = tuple(value.dim)
            want = "Dimensionless" if not any(dim) else table[dim] + suffix
            got = ast.unparse(node.annotation).removeprefix("Final[").removesuffix("]")
            assert got == want, (
                f"{mod.__name__}.{node.target.id} is annotated {got} but measures "
                f"{want}"
            )
            seen += 1
        assert seen > 10, f"only {seen} annotations in {mod.__name__}"
