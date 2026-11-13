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
