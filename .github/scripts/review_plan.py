"""Split a PR review into reviewer scopes.

Reads ``additions deletions path`` lines (``git diff --numstat`` or the PR
files API) on stdin and prints a JSON list of ``{"lens", "files"}`` scopes:
one ``gate`` reviewer, plus one ``correctness`` reviewer per ``CHUNK``
changed lines outside ``test/`` and ``uv.lock``. Above ``LIMIT`` changed
lines of Python outside ``test/``, the PR is too large to review in one
pass, so only the gate reviewer runs and asks for a split.
"""

import json
import sys

CHUNK = 600
MAX_CHUNKS = 6
LIMIT = 1500


def plan(lines: list[str]) -> list[dict[str, str]]:
    """Return the reviewer scopes for the given numstat lines."""
    code: list[tuple[int, str]] = []
    for line in lines:
        added, deleted, path = line.split(maxsplit=2)
        if added != "-" and not path.startswith("test/") and path != "uv.lock":
            code.append((int(added) + int(deleted), path))
    total = sum(size for size, path in code if path.endswith(".py"))
    if total > LIMIT:
        return [
            {"lens": "gate", "files": f"all; {total} changed lines (limit {LIMIT})"}
        ]
    count = max(1, min(MAX_CHUNKS, round(total / CHUNK)))
    groups: list[list[str]] = [[] for _ in range(count)]
    loads = [0] * count
    for size, path in sorted(code, reverse=True):
        emptiest = loads.index(min(loads))
        groups[emptiest].append(path)
        loads[emptiest] += size
    return [
        {"lens": "gate", "files": "all"},
        *({"lens": "correctness", "files": " ".join(g)} for g in groups if g),
    ]


if __name__ == "__main__":
    print(json.dumps(plan(sys.stdin.read().splitlines())))
