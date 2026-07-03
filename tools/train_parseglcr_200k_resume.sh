#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-4}"
if [[ $# -gt 0 ]]; then
  shift
fi

CONFIG="${CONFIG:-local_configs/offseg2/Base/parseglcr_ade20k_200k-512x512.py}"
CKPT="${CKPT:-work_dirs/parseglcr_ade20k_160k-512x512_4x4_try1/iter_160000.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/parseglcr_ade20k_200k-512x512_4x4_try1}"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  echo "Run this script from the offseg2 repo root, or set CKPT=/path/to/iter_160000.pth." >&2
  exit 1
fi

bash tools/dist_train.sh "$CONFIG" "$GPUS" \
  --work-dir "$WORK_DIR" \
  --resume \
  "$@" \
  --cfg-options load_from="$CKPT"
