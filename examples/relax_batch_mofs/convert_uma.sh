#!/usr/bin/env bash
# Convert a UMA PyTorch checkpoint to a tojaxed .zip consumable by kUPS.
# Requires a checkout of https://github.com/cusp-ai-oss/tojax.
#
# Usage: ./convert_uma.sh <uma.pt> [<dataset>] [<output.zip>]
#
# Defaults: dataset=odac (DAC / MOFs), output=mlffs/uma-s-1p2_<dataset>.zip
set -euo pipefail

CHECKPOINT="${1:?missing UMA .pt path}"
DATASET="${2:-odac}"
OUT="${3:-mlffs/uma-s-1p2_${DATASET}.zip}"
TOJAX_DIR="${TOJAX_DIR:-$HOME/tojax}"

if [[ ! -d "$TOJAX_DIR" ]]; then
    echo "tojax not found at $TOJAX_DIR — clone https://github.com/cusp-ai-oss/tojax first" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
cd "$TOJAX_DIR/examples/mlff"
uv run --script export_uma.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --output "$(realpath -m "$OLDPWD/$OUT")" \
    --symbolic NSE \
    --multi-system \
    --min-edges-per-system 64 \
    --merge-mole \
    --skip-verify
