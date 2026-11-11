"""Facts about a PR that the reviewer would otherwise have to guess.

Reads the PR's unified diff and its head checkout and prints a Markdown list:
changed source lines that no test ran (from ``coverage json`` of the CI test
run) and definitions the PR adds that nothing references. For a re-review,
``--since`` (old head to new head) and ``--moved`` (old base to new base)
restrict both lists to lines changed since the last review.

Usage: ``review_facts.py ROOT DIFF [--coverage FILE] [--since FILE --moved FILE
[--files]]``; ``--files`` prints the files a re-review covers instead.
"""

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

SOURCES = ("*.py", "*.md", "*.ipynb", "*.yaml", "*.yml", "*.toml")
SEARCH = ("src", "test", "docs", "ci", "examples")
# Lines whose only job is to defer or to refuse a type, not behaviour.
PLUMBING = re.compile(r"return NotImplemented|raise NotImplementedError|\.\.\.|pass")


def added_lines(diff: str) -> dict[str, dict[int, str]]:
    """Map each file to the line numbers and text the diff adds to it."""
    added: dict[str, dict[int, str]] = defaultdict(dict)
    path, line = "", 0
    for row in diff.splitlines():
        if row.startswith("+++ "):
            path = row[6:] if row.startswith("+++ b/") else ""
        elif row.startswith("@@"):
            match = re.match(r"@@ -\S+ \+(\d+)", row)
            line = int(match.group(1)) if match else 0
        elif path and row.startswith("+"):
            added[path][line] = row[1:].strip()
            line += 1
        elif path and not row.startswith("-"):
            line += 1
    return added


def contents(diff: str) -> set[tuple[str, str]]:
    """Return the (file, stripped text) pairs the diff adds."""
    return {(p, t) for p, rows in added_lines(diff).items() for t in rows.values()}


def scopes(tree: ast.Module) -> dict[int, str]:
    """Map each line to the qualified name of its innermost def or class."""
    names: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                for n in range(child.lineno, (child.end_lineno or child.lineno) + 1):
                    names[n] = name
                visit(child, f"{name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return names


def ranges(lines: list[int]) -> str:
    """Format sorted line numbers as compact ranges."""
    out: list[str] = []
    for n in lines:
        if out and n == int(out[-1].split("-")[-1]) + 1:
            out[-1] = f"{out[-1].split('-')[0]}-{n}"
        else:
            out.append(str(n))
    return ", ".join(out)


def untested(
    root: Path, added: dict[str, dict[int, str]], coverage: dict[str, set[int]]
) -> list[str]:
    """List added source lines that coverage marks as never run."""
    facts: list[str] = []
    for path, rows in sorted(added.items()):
        if not path.startswith("src/") or not path.endswith(".py"):
            continue
        key = next((k for k in coverage if k.endswith(path)), None)
        if key is None:
            facts.append(f"`{path}`: no test imports this file.")
            continue
        missed = sorted(
            n for n in set(rows) & coverage[key] if not PLUMBING.fullmatch(rows[n])
        )
        if not missed:
            continue
        names = scopes(ast.parse((root / path).read_text()))
        by_scope: dict[str, list[int]] = defaultdict(list)
        for n in missed:
            by_scope[names.get(n, "<module>")].append(n)
        for scope, lines in by_scope.items():
            facts.append(f"`{path}` `{scope}`: lines {ranges(lines)}")
    return facts


def unreferenced(root: Path, added: dict[str, dict[int, str]]) -> list[str]:
    """List definitions the diff adds whose name appears nowhere else."""
    corpus = "\n".join(
        f.read_text(errors="ignore")
        for d in SEARCH
        if (root / d).is_dir()
        for pattern in SOURCES
        for f in (root / d).rglob(pattern)
    )
    facts: list[str] = []
    for path, rows in sorted(added.items()):
        if not path.startswith("src/") or not path.endswith(".py"):
            continue
        for node in ast.walk(ast.parse((root / path).read_text())):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if node.lineno not in rows or node.name.startswith("__"):
                continue
            if len(re.findall(rf"\b{re.escape(node.name)}\b", corpus)) <= 1:
                facts.append(f"`{path}:{node.lineno}` `{node.name}`")
    return facts


def main() -> None:
    """Print the facts for one PR."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("diff", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--since", type=Path)
    parser.add_argument("--moved", type=Path)
    parser.add_argument("--files", action="store_true", help="print in-scope files")
    args = parser.parse_args()

    added = added_lines(args.diff.read_text())
    if args.since and args.moved:
        fresh = contents(args.since.read_text()) - contents(args.moved.read_text())
        added = {
            p: {n: t for n, t in rows.items() if (p, t) in fresh}
            for p, rows in added.items()
        }
        added = {p: rows for p, rows in added.items() if rows}
        if args.files:
            print(" ".join(p for p in added if p != "uv.lock"))
            return
        changed = sum(len(rows) for rows in added.values())
        print(f"Re-review: {changed} lines changed since the last review are in scope:")
        for path, rows in sorted(added.items()):
            print(f"- `{path}`: lines {ranges(sorted(rows))}")

    print("\n## Changed lines no test runs")
    if args.coverage and args.coverage.exists():
        report = json.loads(args.coverage.read_text())["files"]
        coverage = {k: set(v["missing_lines"]) for k, v in report.items()}
        print(
            "\n".join(f"- {f}" for f in untested(args.root, added, coverage)) or "None."
        )
    else:
        print("Unknown: no coverage report for this commit.")

    print("\n## New definitions nothing references")
    print("\n".join(f"- {f}" for f in unreferenced(args.root, added)) or "None.")


if __name__ == "__main__":
    main()
