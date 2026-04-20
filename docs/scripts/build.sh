#!/usr/bin/env bash
# Build or serve documentation: generate API reference pages, create a build
# config with the nav injected, and run zensical.
#
# Usage:
#   ./docs/scripts/build.sh                        # build with mkdocs.yml
#   ./docs/scripts/build.sh --serve                # serve with mkdocs.yml
#   ./docs/scripts/build.sh -f mkdocs.yml.dev      # build with a different config
#   ./docs/scripts/build.sh --serve -f mkdocs.yml.dev
set -euo pipefail

CONFIG_FILE="mkdocs.yml"
MODE="build"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve) MODE="serve"; shift ;;
    -f)      CONFIG_FILE="$2"; shift 2 ;;
    *)       echo "Usage: $0 [--serve] [-f CONFIG_FILE]" >&2; exit 1 ;;
  esac
done

BUILD_CONFIG="${CONFIG_FILE%.yml}.build.yml"

# 1. Convert notebooks to markdown
JAX_PLATFORMS="cpu" uv run jupyter nbconvert --execute --to markdown docs/notebooks/*.ipynb

# 2. Generate reference .md files and the build config
uv run python docs/scripts/gen_ref_pages.py -f "$CONFIG_FILE"

# 3. Build or serve
cleanup() { rm -f "$BUILD_CONFIG"; }
trap cleanup EXIT

if [[ "$MODE" == "serve" ]]; then
  uv run python -m zensical serve -f "$BUILD_CONFIG"
else
  uv run python -m zensical build -f "$BUILD_CONFIG"
fi
