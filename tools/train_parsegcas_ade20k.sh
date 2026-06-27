#!/usr/bin/env bash
set -euo pipefail

CONFIG="local_configs/offseg2/Base/parsegcas_ade20k_160k-512x512.py"
WORK_DIR="${1:-work_dirs/parsegcas_ade20k_160k-512x512_4x4_try1}"
GPUS="${2:-4}"

MAX_IMAGES="${MAX_IMAGES:-250}" \
ANALYZE_DEVICE="${ANALYZE_DEVICE:-cuda:0}" \
bash tools/train_test_analyze.sh "${CONFIG}" "${WORK_DIR}" "${GPUS}"
