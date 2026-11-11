"""kUPS-specific static checks that ruff and pyrefly do not cover.

Run as ``python ci/kups_lint.py [paths...]``; exits non-zero on findings.
Silence a finding with ``# noqa: KUPxxx`` on the reported line; ruff treats
``KUP`` as an external code, so the directive is not flagged as unknown.

Docstring findings are reported on the ``def``/``class`` line.

Codes:
    KUP001  ``__init__`` defined on a dataclass.
    KUP002  Optional array-valued field on a JAX dataclass (``Array | None``).
    KUP003  Function or class docstring longer than ``MAX_DOCSTRING_LINES``.
    KUP004  Comment block longer than ``MAX_COMMENT_LINES``.
    KUP005  Comment or docstring narrating code history.
    KUP006  Private name imported from another kUPS module (a private submodule
            of the importer's own package is fine).
    KUP007  Docstring attached to a type alias.
    KUP008  Type-check suppression without a pyrefly error kind.
"""

import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MAX_DOCSTRING_LINES = 40
MAX_COMMENT_LINES = 6

ARRAY_TYPES = re.compile(r"\b(Array|ArrayLike|Table|Index)\b")
HISTORY = re.compile(
    r"\b(previously|no longer|used to be|backwards?[- ]compat\w*|legacy|"
    r"was (changed|renamed|moved|removed)|old (implementation|version|behaviou?r)|"
    r"originally|we now|now uses?)\b",
    re.IGNORECASE,
)
SUPPRESSION = re.compile(r"#\s*(type:\s*ignore|pyrefly:\s*ignore(?!\s*\[))")
NOQA = re.compile(r"#\s*noqa:\s*(?P<codes>[A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*)")

MESSAGES = {
    "KUP001": "dataclass defines `__init__`; use fields, `__post_init__` or a classmethod constructor",
    "KUP002": "optional array field; None -> Array retraces, initialise a real (possibly empty) array",
    "KUP003": "docstring has {n} lines (max {max}); cut it down",
    "KUP004": "{n}-line comment block (max {max}); shorten it or move it to the docstring",
    "KUP005": "narrates history ({word!r}); describe the current code only",
    "KUP006": "private `{name}` imported from `{module}`; make it public or keep it local",
    "KUP007": "type alias docstring; the alias name should say enough",
    "KUP008": "suppress with `# pyrefly: ignore[kind]`; pyrefly treats other forms as blanket ignores",
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


def _is_optional(annotation: ast.expr) -> bool:
    match annotation:
        case ast.BinOp(op=ast.BitOr(), left=left, right=right):
            return _is_optional(left) or _is_optional(right)
        case ast.Constant(value=None):
            return True
        case ast.Subscript(
            value=ast.Name(id="Optional") | ast.Attribute(attr="Optional")
        ):
            return True
    return False


def _is_static(stmt: ast.AnnAssign) -> bool:
    return isinstance(stmt.value, ast.Call) and any(
        kw.arg == "static"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
        for kw in stmt.value.keywords
    )


def _is_optional_array(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.AnnAssign)
        and _is_optional(stmt.annotation)
        and not _is_static(stmt)
        and ARRAY_TYPES.search(ast.unparse(stmt.annotation)) is not None
    )


def _module_name(path: Path) -> str | None:
    parts = path.with_suffix("").parts
    if "kups" not in parts:
        return None
    parts = parts[parts.index("kups") :]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _is_submodule(path: Path, package: str, name: str) -> bool:
    """Whether ``package.name`` is a module next to the file at ``path``."""
    root = Path(*path.parts[: path.parts.index("kups")])
    target = root.joinpath(*package.split("."), name)
    return target.with_suffix(".py").exists() or (target / "__init__.py").exists()


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, tree: ast.Module) -> None:
        self.path = path
        self.module = _module_name(path)
        self.origins = _dataclass_origins(tree)
        self.findings: list[Finding] = []

    def report(self, node: ast.stmt | ast.expr, code: str, **fmt: object) -> None:
        self.findings.append(_finding(self.path, node.lineno, code, **fmt))

    def visit_Module(self, node: ast.Module) -> None:
        self._check_history(node)
        for prev, stmt in zip(node.body, node.body[1:]):
            if isinstance(prev, ast.TypeAlias) and _is_string(stmt):
                self.report(stmt, "KUP007")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        origin = _dataclass_origin(node, self.origins)
        for stmt in node.body if origin is not None else ():
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                self.report(stmt, "KUP001")
            if origin != "dataclasses" and _is_optional_array(stmt):
                self.report(stmt, "KUP002")
        self._check_docstring(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._check_docstring(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or module.split(".")[0] != "kups" or module == self.module:
            return
        own_package = self.module is not None and self.module.startswith(f"{module}.")
        for alias in node.names:
            name = alias.name
            if not name.startswith("_") or name.startswith("__"):
                continue
            if own_package and _is_submodule(self.path, module, name):
                continue
            self.report(node, "KUP006", name=name, module=module)

    def _check_docstring(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        doc = ast.get_docstring(node)
        if doc is not None and (n := len(doc.splitlines())) > MAX_DOCSTRING_LINES:
            self.report(node, "KUP003", n=n, max=MAX_DOCSTRING_LINES)
        self._check_history(node)

    def _check_history(
        self, node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        doc = ast.get_docstring(node)
        if doc is not None and (match := HISTORY.search(doc)):
            line = 1 if isinstance(node, ast.Module) else node.lineno
            self.findings.append(
                _finding(self.path, line, "KUP005", word=match.group(0))
            )


def _is_string(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _comment_findings(path: Path, source: str) -> Iterator[Finding]:
    blocks: list[list[int]] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT:
            continue
        line = tok.start[0]
        if SUPPRESSION.search(tok.string):
            yield _finding(path, line, "KUP008")
        if match := HISTORY.search(tok.string):
            yield _finding(path, line, "KUP005", word=match.group(0))
        if not tok.line.lstrip().startswith("#"):
            continue
        if blocks and blocks[-1][-1] == line - 1:
            blocks[-1].append(line)
        else:
            blocks.append([line])
    for block in blocks:
        if len(block) > MAX_COMMENT_LINES:
            yield _finding(
                path, block[0], "KUP004", n=len(block), max=MAX_COMMENT_LINES
            )


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
    findings = [*visitor.findings, *_comment_findings(path, source)]
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
