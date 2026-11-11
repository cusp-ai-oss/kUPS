"""kUPS-specific static checks that ruff and pyrefly do not cover.

Run as ``python ci/kups_lint.py [paths...]``; exits non-zero on findings.
Silence a finding with ``# noqa: KUPxxx`` on the reported line; ruff treats
``KUP`` as an external code, so the directive is not flagged as unknown.

Codes:
    KUP001  ``__init__`` defined on a dataclass.
"""

import ast
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

NOQA = re.compile(r"#\s*noqa:\s*(?P<codes>[A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*)")

MESSAGES = {
    "KUP001": "dataclass defines `__init__`; use fields, `__post_init__` or a classmethod constructor",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def _finding(path: Path, line: int, code: str, **fmt: object) -> Finding:
    return Finding(path, line, code, MESSAGES[code].format(**fmt))


def _dataclass_origins(tree: ast.Module) -> dict[str, str]:
    """Map each decorator spelling of ``dataclass`` to its defining module."""
    origins = {"dataclasses.dataclass": "dataclasses"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "dataclass":
                    origins[alias.asname or alias.name] = node.module
    return origins


def _dataclass_origin(cls: ast.ClassDef, origins: dict[str, str]) -> str | None:
    for deco in cls.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if (origin := origins.get(ast.unparse(target))) is not None:
            return origin
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, tree: ast.Module) -> None:
        self.path = path
        self.origins = _dataclass_origins(tree)
        self.findings: list[Finding] = []

    def report(self, node: ast.stmt | ast.expr, code: str, **fmt: object) -> None:
        self.findings.append(_finding(self.path, node.lineno, code, **fmt))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        origin = _dataclass_origin(node, self.origins)
        for stmt in node.body if origin is not None else ():
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                self.report(stmt, "KUP001")
        self.generic_visit(node)


def _suppressed(lines: list[str], finding: Finding) -> bool:
    match = NOQA.search(lines[finding.line - 1])
    return match is not None and finding.code in {
        code.strip() for code in match["codes"].split(",")
    }


def check_source(path: Path, source: str) -> list[Finding]:
    """Return every unsuppressed finding in ``source``, sorted by line."""
    tree = ast.parse(source, filename=str(path))
    visitor = _Visitor(path, tree)
    visitor.visit(tree)
    lines = source.splitlines()
    findings = visitor.findings
    return sorted(
        (f for f in findings if not _suppressed(lines, f)),
        key=lambda f: (f.line, f.code),
    )


def _iter_files(paths: list[str]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        elif path.suffix == ".py":
            yield path


def main(argv: list[str]) -> int:
    findings = [
        finding
        for path in _iter_files(argv or ["src"])
        for finding in check_source(path, path.read_text())
    ]
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
