#!/usr/bin/env bash
# Build or serve documentation: generate API reference pages, create a build
# config with the nav injected, and run zensical.
#
# Usage:
#   ./docs/scripts/build.sh                        # build with mkdocs.yml
#   ./docs/scripts/build.sh --serve                # serve with mkdocs.yml
#   ./docs/scripts/build.sh -f mkdocs.yml.dev      # build with a different config
#   ./docs/scripts/build.sh --no-execute           # convert notebooks without executing them
#   ./docs/scripts/build.sh --serve -f mkdocs.yml.dev
set -euo pipefail

CONFIG_FILE="mkdocs.yml"
MODE="build"
EXECUTE_NOTEBOOKS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve)      MODE="serve"; shift ;;
    --no-execute) EXECUTE_NOTEBOOKS=0; shift ;;
    -f)           CONFIG_FILE="$2"; shift 2 ;;
    *)            echo "Usage: $0 [--serve] [--no-execute] [-f CONFIG_FILE]" >&2; exit 1 ;;
  esac
done

BUILD_CONFIG="${CONFIG_FILE%.yml}.build.yml"

# 1. Convert notebooks to markdown (optionally executing them first) and
#    tag links with `{ data-preview }` for zensical hover previews.
CONVERT_ARGS=()
if [[ "$EXECUTE_NOTEBOOKS" == "0" ]]; then
  CONVERT_ARGS=(--no-execute)
fi
uv run python docs/scripts/convert_notebooks.py "${CONVERT_ARGS[@]}"

# 2. Generate reference .md files and the build config
uv run python docs/scripts/gen_ref_pages.py -f "$CONFIG_FILE"

# 3. Build or serve
cleanup() { rm -f "$BUILD_CONFIG"; }
trap cleanup EXIT

if [[ "$MODE" == "serve" ]]; then
  uv run python -m zensical serve -f "$BUILD_CONFIG" --strict
else
  uv run python -m zensical build -f "$BUILD_CONFIG" --strict
fi
