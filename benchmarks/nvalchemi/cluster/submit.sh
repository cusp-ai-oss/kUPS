#!/bin/bash
# Run a command on one GPU worker of a Ray cluster:
#   cluster/submit.sh CLUSTER WHEELS_DIR [--log FILE] -- <cmd...>
#
# Ships a snapshot of this kUPS checkout (pyproject + src/) as kups/ and of
# benchmarks/nvalchemi/ as bench/, plus the wheels in WHEELS_DIR (jax-ptx, jax-ptx-warp);
# runs <cmd> from bench/ on a worker (cluster/run_worker.sh builds and caches the venv),
# streaming output here and to --log. Blocks until the job ends.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bench=$(dirname "$here")
kups=$(cd "$bench/../.." && pwd)
cluster=${1:?cluster}
wheels=${2:?wheels dir}
shift 2
log=/dev/null
while [ $# -gt 0 ] && [ "$1" != -- ]; do
  case "$1" in
    --log) log=$2; mkdir -p "$(dirname "$log")"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ "${1:-}" = -- ] && shift
[ $# -gt 0 ] || { echo "no command" >&2; exit 2; }

pf=/home/coder/dev/ray-cluster-validation-offload/scripts/pf.sh
port=$("$pf" "$cluster" 2>&1 | grep -o 'port [0-9]*' | tail -1 | cut -d' ' -f2)
[ -n "$port" ] || { echo "port-forward for $cluster failed (not admitted yet?)" >&2; exit 69; }
rayver=$(curl -sf "http://127.0.0.1:$port/api/version" |
  python3 -c "import json,sys; print(json.load(sys.stdin)['ray_version'])")

stage=$(mktemp -d "${TMPDIR:-/tmp}/nvalchemi-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/kups" "$stage/wheels"
cp "$kups/pyproject.toml" "$kups/README.md" "$stage/kups/"
rsync -a --exclude '__pycache__/' "$kups/src" "$stage/kups/"
rsync -a --exclude '__pycache__/' --exclude 'results/' "$bench/" "$stage/bench/"
cp "$wheels"/*.whl "$stage/wheels/"
find "$stage" -name .gitignore -delete

echo "== $cluster (ray $rayver, port $port): $*" | tee -a "$log"
uvx -q --from "ray[default]==$rayver" ray job submit --address "http://127.0.0.1:$port" \
  --working-dir "$stage" --entrypoint-num-cpus 0 -- \
  python bench/cluster/remote_entry.py 1 8 8 bash bench/cluster/run_worker.sh "$@" 2>&1 | tee -a "$log"
