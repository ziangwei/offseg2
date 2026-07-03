#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-4}"
if [[ $# -gt 0 ]]; then
  shift
fi

CONFIG="${CONFIG:-local_configs/offseg2/Base/parseglcr_ade20k_200k-512x512.py}"
CKPT="${CKPT:-work_dirs/parseglcr_ade20k_160k-512x512_4x4_try1/iter_160000.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/parseglcr_ade20k_200k-512x512_4x4_try1}"

if [[ "${STRICT_RESUME:-0}" != "1" ]]; then
  echo "Strict resume asks MMEngine to advance the dataloader by 160000 steps before training." >&2
  echo "This can take a long time while GPUs are reserved. Prefer:" >&2
  echo "  bash tools/train_parseglcr_post40k_from160k.sh $GPUS" >&2
  echo "Set STRICT_RESUME=1 only if you need exact dataloader/optimizer/scheduler resume." >&2
  exit 2
fi

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
