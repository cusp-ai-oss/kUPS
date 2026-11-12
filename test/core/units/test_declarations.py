# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""The declarations the checker sees must match the runtime values."""

import ast
import pathlib

from kups.core import units

PKG = pathlib.Path(units.__file__).parent


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
