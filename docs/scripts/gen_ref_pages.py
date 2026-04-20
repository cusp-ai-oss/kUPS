#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generate API reference pages and a build-ready mkdocs config.

Walks the source tree under src/kups/, generates mkdocstrings-compatible
markdown files under docs/reference/, and writes a temporary mkdocs config
with the API Reference nav section injected.

The original mkdocs.yml is never modified.

Usage:
    uv run python docs/scripts/gen_ref_pages.py [-f mkdocs.yml]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PACKAGE = ROOT / "src" / "kups"
DOCS_DIR = ROOT / "docs"
REF_DIR = DOCS_DIR / "reference"

# Module paths to exclude (dotted notation). Supports exact match and prefix match.
EXCLUDE: list[str] = []


class _NavNode:
    """Tree node for building the mkdocs nav structure."""

    def __init__(self) -> None:
        self.index_path: str | None = None
        self.children: dict[str, _NavNode] = {}
        self.leaf_path: str | None = None

    def add(self, parts: tuple[str, ...], doc_path: str, *, is_package: bool) -> None:
        if len(parts) == 1:
            name = parts[0]
            if is_package:
                self.children.setdefault(name, _NavNode()).index_path = doc_path
            else:
                self.children.setdefault(name, _NavNode()).leaf_path = doc_path
        else:
            first = parts[0]
            self.children.setdefault(first, _NavNode()).add(
                parts[1:], doc_path, is_package=is_package
            )

    def to_list(self) -> list:
        """Convert tree to mkdocs nav list format."""
        result: list = []
        if self.index_path:
            result.append(self.index_path)
        for name, child in self.children.items():
            if child.leaf_path and not child.children:
                result.append({name: child.leaf_path})
            else:
                result.append({name: child.to_list()})
        return result


def _iter_modules(package_root: Path):
    """Yield (dotted_parts, is_package) for each public module."""
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root.parent)
        parts = list(rel.with_suffix("").parts)

        if any(p.startswith("_") and p != "__init__" for p in parts):
            continue

        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]

        parts_tuple = tuple(parts)
        mod_path = ".".join(parts_tuple)

        if any(mod_path == ex or mod_path.startswith(ex + ".") for ex in EXCLUDE):
            continue

        yield parts_tuple, is_package


def _module_markdown(parts: tuple[str, ...]) -> str:
    """Generate mkdocstrings markdown content for a module."""
    identifier = ".".join(parts)
    title = parts[-1]
    return (
        f"---\ntitle: {title}\n---\n\n"
        f"::: {identifier}\n"
        f"    options:\n"
        f"      heading_level: 1\n"
        f"      show_root_heading: true\n"
    )


def _generate_reference_pages() -> list:
    """Generate .md files under docs/reference/ and return the nav list."""
    if REF_DIR.exists():
        shutil.rmtree(REF_DIR)
    REF_DIR.mkdir(parents=True)

    nav = _NavNode()
    count = 0

    for parts, is_package in _iter_modules(SRC_PACKAGE):
        if is_package:
            doc_file = REF_DIR.joinpath(*parts, "index.md")
        elif parts[-1] == "index":
            doc_file = REF_DIR.joinpath(*parts[:-1], "index_py.md")
        else:
            doc_file = REF_DIR.joinpath(*parts).with_suffix(".md")

        doc_file.parent.mkdir(parents=True, exist_ok=True)
        doc_file.write_text(_module_markdown(parts))

        rel_path = str(doc_file.relative_to(DOCS_DIR))
        nav.add(parts, rel_path, is_package=is_package)
        count += 1

    print(f"Generated {count} reference pages in {REF_DIR.relative_to(ROOT)}/")
    return nav.to_list()


def _find_nav_span(text: str) -> tuple[int, int] | None:
    """Return (start, end) char offsets of the nav section in a YAML file."""
    lines = text.splitlines(keepends=True)
    nav_start = None
    nav_end = None
    for i, line in enumerate(lines):
        if nav_start is None:
            if line.rstrip() == "nav:" or line.startswith("nav:"):
                nav_start = i
        else:
            stripped = line.strip()
            if stripped and not line[0].isspace() and not line.startswith("-"):
                nav_end = i
                break
    if nav_start is None:
        return None
    if nav_end is None:
        nav_end = len(lines)
    start = sum(len(lines[j]) for j in range(nav_start))
    end = sum(len(lines[j]) for j in range(nav_end))
    return start, end


def _build_config(config_path: Path, api_nav: list) -> Path:
    """Read config_path, inject API Reference nav, write to a temp file.

    Returns the path to the generated temporary config file.
    """
    text = config_path.read_text()

    # Remove api-autonav plugin reference if present
    text = text.replace("  - api-autonav:\n", "")
    # Remove any indented options under it (like modules: ...)
    text = re.sub(r"      modules:.*\n", "", text)

    span = _find_nav_span(text)
    if span is not None:
        start, end = span
        nav_text = text[start:end]
        existing_nav: list = yaml.safe_load(nav_text).get("nav") or []

        existing_nav = [
            e
            for e in existing_nav
            if not (isinstance(e, dict) and "API Reference" in e)
        ]
        existing_nav.append({"API Reference": api_nav})

        new_nav = yaml.dump(
            {"nav": existing_nav},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        text = text[:start] + new_nav + "\n" + text[end:]

    # Write to a temp file next to the original (so relative paths work)
    out = config_path.with_suffix(".build.yml")
    out.write_text(text)
    print(f"Wrote build config to {out.relative_to(ROOT)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--config-file",
        default="mkdocs.yml",
        help="Source mkdocs config file (default: mkdocs.yml)",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config_file
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    api_nav = _generate_reference_pages()
    _build_config(config_path, api_nav)


if __name__ == "__main__":
    main()
