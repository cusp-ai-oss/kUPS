#!/usr/bin/env python
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Convert Jupyter notebooks to markdown and tag links for hover previews.

Runs ``jupyter nbconvert`` on every notebook under ``docs/notebooks/`` and
appends ``{ data-preview }`` to every markdown link in the resulting markdown
files. The attribute enables zensical/Material hover-previews for symbol
references (e.g. ``[Lens][kups.core.lens.Lens]``) and other links.

Usage:
    uv run python docs/scripts/convert_notebooks.py [--no-execute] [PATHS...]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_DIR = ROOT / "docs" / "notebooks"

# Match a markdown reference-style link [text][ref] that is neither a
# reference image (preceded by `!`) nor already followed by an attribute list
# (`{...}`). The link text allows up to two levels of balanced brackets so
# that nested generic type signatures like
# `[Dict[str, Lens[S, A]]][kups.core.lens.Lens]` match. Inline links
# `[text](url)` are intentionally excluded.
_LINK_RE = re.compile(
    r"(?<!!)"
    r"(\[(?:[^\[\]\n]|\[(?:[^\[\]\n]|\[[^\]\n]*\])*\])*\]"
    r"\[[^\]\n]*\])"
    r"(?!\s*\{)"
)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _tag_links(md_path: Path) -> int:
    """Append ``{ data-preview }`` to every markdown link in ``md_path``.

    Lines inside fenced code blocks are left untouched. Returns the number of
    links tagged.
    """
    text = md_path.read_text()
    out: list[str] = []
    in_fence = False
    count = 0
    for line in text.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        new_line, n = _LINK_RE.subn(r"\1{ data-preview }", line)
        out.append(new_line)
        count += n
    md_path.write_text("".join(out))
    return count


def _convert(notebooks: list[Path], *, execute: bool) -> None:
    """Run ``jupyter nbconvert`` on the given notebooks."""
    cmd = ["jupyter", "nbconvert", "--to", "markdown"]
    if execute:
        cmd.append("--execute")
    cmd.extend(str(nb) for nb in notebooks)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Convert notebooks without executing them.",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths (default: all .ipynb files under docs/notebooks/).",
    )
    args = parser.parse_args()

    notebooks: list[Path] = args.notebooks or sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    _convert(notebooks, execute=not args.no_execute)

    total = 0
    for nb in notebooks:
        md = nb.with_suffix(".md")
        if md.exists():
            total += _tag_links(md)
    print(f"Tagged {total} link(s) across {len(notebooks)} notebook(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
