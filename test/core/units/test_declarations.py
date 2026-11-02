# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The class declarations the checker sees must match the runtime surface.

``Quantity`` and ``Unit`` are single classes; names with no
meaning are defined under ``if not TYPE_CHECKING`` so the checker flags them
while the runtime raises with guidance, and those hidden blocks are pinned here.
"""

import ast
import pathlib
from typing import TypeAliasType, cast

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
        if isinstance(v, ast.Subscript) and ast.unparse(v.value) == "Annotated":
            v = cast(ast.Tuple, v.slice).elts[0]  # the metadata is pydantic's
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


def unit_aliases() -> dict[tuple[int, ...], str]:
    """Map dimension exponents to the Unit alias names declared in _unit.py."""
    out: dict[tuple[int, ...], str] = {}
    for node in ast.walk(ast.parse((PKG / "_unit.py").read_text())):
        if not isinstance(node, ast.TypeAlias) or not isinstance(node.name, ast.Name):
            continue
        v = node.value
        if (
            isinstance(v, ast.Subscript)
            and isinstance(v.slice, ast.Tuple)
            and ast.unparse(v.value) == "Unit"
        ):
            try:
                out[tuple(ast.literal_eval(e) for e in v.slice.elts)] = node.name.id
            except ValueError:  # parameterised by something other than ints
                continue
    return out


def test_catalogue_annotations_match_dimensions():
    # the catalogue annotates its own members, so the checker cannot catch a wrong
    # alias; compare against the dimension read from the value instead
    table = unit_aliases()
    seen = 0
    for node in ast.parse((PKG / "catalog.py").read_text()).body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        value = getattr(units.catalog, node.target.id)
        if not hasattr(value, "dim"):
            continue
        want = table[tuple(value.dim)]
        got = ast.unparse(node.annotation).removeprefix("Final[").removesuffix("]")
        assert got == want, f"catalog.{node.target.id} is {got} but measures {want}"
        seen += 1
    assert seen > 10, f"only {seen} annotations in catalog"


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
    extra = exported - named - {f"{n}Unit" for n in named} - {"Measured"}
    assert not extra, f"kups.core.units exports {sorted(extra)} as well"


def test_constant_annotations_match_dimensions():
    # constants annotate their own members under TYPE_CHECKING, so the checker
    # cannot catch a wrong alias; compare against the dimension read from the value
    table = aliases()
    seen = 0
    for node in ast.parse((PKG / "constants.py").read_text()).body:
        if not isinstance(node, ast.If):
            continue
        for decl in node.body:
            if not (
                isinstance(decl, ast.AnnAssign) and isinstance(decl.target, ast.Name)
            ):
                continue
            value = getattr(units.constants, decl.target.id)
            if not hasattr(value, "dim"):  # N_A and alpha are plain floats
                continue
            dim = tuple(value.dim)
            want = "Dimensionless" if not any(dim) else table[dim]
            got = ast.unparse(decl.annotation)
            assert got == want, (
                f"constants.{decl.target.id} is {got} but measures {want}"
            )
            seen += 1
    assert seen > 10, f"only {seen} annotations in constants"
