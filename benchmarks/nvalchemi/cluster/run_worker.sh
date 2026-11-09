#!/bin/bash
# Worker side: a venv with JAX, Warp, nvalchemi, the shipped jax-ptx wheels and kUPS's
# dependencies, cached on the node (keyed by requirements, wheels and kUPS's pyproject);
# the shipped kUPS source comes first on PYTHONPATH. Runs <cmd...> from bench/.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
export UV_CACHE_DIR=/tmp/nvalchemi-uvcache
UV=$(command -v uv || true)
if [ -z "$UV" ]; then
  python3 -m pip install -q --user uv >/dev/null 2>&1 || python3 -m pip install -q uv >/dev/null 2>&1
  UV=$(command -v uv || echo "$HOME/.local/bin/uv")
fi
KEY=$(cat "$root/bench/cluster/requirements.txt" "$root/wheels/"*.whl "$root/kups/pyproject.toml" |
  sha1sum | cut -c1-12)
VENV=/tmp/nvalchemi-venv-$KEY
if [ ! -x "$VENV/bin/python" ]; then
  echo "== building venv $VENV"
  t0=$SECONDS
  "$UV" venv -q -p 3.12 "$VENV.tmp"
  VIRTUAL_ENV="$VENV.tmp" "$UV" pip install -q -r "$root/bench/cluster/requirements.txt" \
    "$root/wheels/"*.whl "$root/kups"
  mv "$VENV.tmp" "$VENV"
  echo "== venv ready in $((SECONDS - t0)) s"
fi
export PYTHONPATH="$root/kups/src:$root/bench"
export JAX_PTX_CACHE_DIR=/tmp/nvalchemi-jax-ptx-cache
export PATH="$VENV/bin:$PATH"
cd "$root/bench"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
"$VENV/bin/python" -c "import jax, warp, nvalchemiops, jax_ptx_warp; print('jax', jax.__version__, 'warp', warp.__version__, jax.devices())"
# The worker's BASH_ENV can reset PATH in child shells: run python as the venv's.
case "${1:-}" in python|python3) shift; set -- "$VENV/bin/python" "$@" ;; esac
exec "$@"
