import importlib.util
import textwrap
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "kups_lint", Path(__file__).parents[2] / "ci" / "kups_lint.py"
)
assert _SPEC is not None and _SPEC.loader is not None
kups_lint = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kups_lint)


def codes(source: str, path: str = "src/kups/core/mod.py") -> list[str]:
    findings = kups_lint.check_source(Path(path), textwrap.dedent(source))
    return [f.code for f in findings]


JAX_DC = "from kups.core.utils.jax import dataclass, field\n"
PLAIN_DC = "from dataclasses import dataclass\n"


@pytest.mark.parametrize("header", [JAX_DC, PLAIN_DC])
def test_dataclass_init(header: str):
    src = header + "@dataclass\nclass A:\n    x: int\n    def __init__(self): ...\n"
    assert codes(src) == ["KUP001"]


def test_init_on_plain_class_is_fine():
    assert codes("class A:\n    def __init__(self): ...\n") == []


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("x: Array | None", ["KUP002"]),
        ("x: Optional[Table[K, Array]] = None", ["KUP002"]),
        ("x: Array | None = field(static=True, default=None)", []),
        ("x: float | None = None", []),
        ("x: Array", []),
    ],
)
def test_optional_array_field(annotation: str, expected: list[str]):
    src = JAX_DC + f"@dataclass\nclass A:\n    {annotation}\n"
    assert codes(src) == expected


def test_optional_array_field_on_plain_dataclass_is_fine():
    assert codes(PLAIN_DC + "@dataclass\nclass A:\n    x: Array | None\n") == []


def test_long_docstring():
    body = "\n    ".join(["line"] * (kups_lint.MAX_DOCSTRING_LINES + 1))
    assert codes(f'def f():\n    """{body}"""\n') == ["KUP003"]
    assert codes(f'"""{body}"""\n') == []


def test_long_comment_block():
    block = "# c\n" * (kups_lint.MAX_COMMENT_LINES + 1)
    assert codes(block) == ["KUP004"]
    assert codes("# c\n" * kups_lint.MAX_COMMENT_LINES) == []
    assert codes("# c\n" * 3 + "x = 1\n" + "# c\n" * 4) == []


@pytest.mark.parametrize(
    "text", ["# previously this was a loop", '"""Legacy entry point."""']
)
def test_history(text: str):
    assert codes(text + "\n") == ["KUP005"]


def test_private_import():
    assert codes("from kups.core.lens import _helper\n") == ["KUP006"]
    assert codes("from kups.core.mod import _helper\n") == []
    assert codes("from jax._src import core\n") == []
    assert codes("from kups.core.lens import __version__\n") == []


def test_private_submodule_of_own_package(tmp_path: Path):
    pkg = tmp_path / "src" / "kups" / "core" / "units"
    pkg.mkdir(parents=True)
    (pkg / "_unit.py").write_text("")
    catalog = pkg / "catalog.py"
    imports = "from kups.core.units import _unit\n"
    assert codes(imports, str(catalog)) == []
    assert codes("from kups.core.units import _helper\n", str(catalog)) == ["KUP006"]
    assert codes(imports, str(tmp_path / "src" / "kups" / "md" / "x.py")) == ["KUP006"]
    assert codes("from kups.core.units._unit import _private\n", str(catalog)) == [
        "KUP006"
    ]


def test_type_alias_docstring():
    assert codes('type X = int\n"""An int."""\n') == ["KUP007"]


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("x = f()  # type: ignore", ["KUP008"]),
        ("x = f()  # type: ignore[arg-type]", ["KUP008"]),
        ("x = f()  # pyrefly: ignore", ["KUP008"]),
        ("x = f()  # pyrefly: ignore[bad-return]", []),
        ("x = f()  # pyrefly: ignore [bad-return]  # reason", []),
    ],
)
def test_suppression_names_pyrefly_kind(comment: str, expected: list[str]):
    assert codes(comment + "\n") == expected


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("# ----- setup -----", ["KUP009"]),
        ("# ==========", ["KUP009"]),
        ("#### Section ####", ["KUP009"]),
        ("# x -> y, a - b", []),
    ],
)
def test_banner_comment(comment: str, expected: list[str]):
    assert codes(comment + "\n") == expected


def test_tracer_isinstance():
    assert codes("isinstance(x, jax.core.Tracer)\n") == ["KUP010"]
    assert codes("isinstance(x, jax.core.Tracer)\n", "src/kups/core/utils/jax.py") == []
    assert codes("isinstance(x, Array)\n") == []


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ('"""A."""', ["KUP011"]),
        ('"""A.\n\n    Attributes:\n        x: An int.\n    """', ["KUP011"]),
        (
            '"""A.\n\n    Attributes:\n        x: An int.\n        y: A str.\n    """',
            [],
        ),
        (
            '"""A.\n\n    Attributes:\n        x (int): An int.\n        y: A str.\n    """',
            [],
        ),
    ],
)
def test_dataclass_attributes(doc: str, expected: list[str]):
    fields = "    x: int\n    y: str\n    _z: int\n    n: ClassVar[int] = 0\n"
    assert codes(JAX_DC + f"@dataclass\nclass A:\n    {doc}\n{fields}") == expected


def test_dataclass_attributes_skip_private_and_plain_classes():
    assert codes(JAX_DC + '@dataclass\nclass _A:\n    """A."""\n    x: int\n') == []
    assert codes('class A:\n    """A."""\n    x: int\n') == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from kups.core.units import catalog as u\n", ["KUP012"]),
        ("from kups.core.units.catalog import eV\n", ["KUP012"]),
        ("import kups.core.units.catalog\n", ["KUP012"]),
        ("from kups.core.units import ATOMISTIC\n", ["KUP012"]),
        ("x = units.catalog.angstrom\n", ["KUP012"]),
        ("with units.base(units.SI):\n    pass\n", ["KUP012", "KUP012"]),
        ("x = units.catalog.angstrom  # noqa: KUP012\n", []),
        ("from kups.core import units\nx = units.constants.k_B\n", []),
        ("x = units.from_dim(0.0, units.ENERGY)\n", []),
    ],
)
def test_explicit_units(source: str, expected: list[str]):
    assert codes(source, "src/kups/potential/classical/mod.py") == expected


@pytest.mark.parametrize(
    "path",
    [
        "src/kups/core/units/mod.py",
        "src/kups/application/mcmc/mod.py",
        "src/kups/potential/mliap/mod.py",
    ],
)
def test_explicit_units_allowed_at_the_boundary(path: str):
    assert codes("from kups.core.units import catalog as u\n", path) == []
